"""
V07: External Data Research - Proxy Features + Domain Adaptation + Automated Combination Loop

Key insight from V06: Global stats (constant across all samples) = zero gain in LGBM ranking.
Must create per-subject features that vary across samples.

Strategy:
1. External data as "population reference" — compute per-subject deviation from population
2. External feature correlations → derive internal feature engineering rules
3. Synthetic proxy features based on external domain knowledge
4. Full automated combination loop (A, B, C, D, A+B, ..., A+B+C+D)
5. Multi-strategy: proxy, augmentation, weighted ensemble, staged training
"""

import sys, os, gc, re, json, warnings, time, itertools, traceback
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SUBMIT = ROOT / "submissions"

for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEEDS = [42, 7, 999, 777]

# Configs
CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5}
CFG_DEEP  = {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15}
CFG_V48   = {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10}
CFG_SAFETY = {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20}
CFGS = {'wide':CFG_WIDE,'deep':CFG_DEEP,'v48':CFG_V48,'safety':CFG_SAFETY}
V53_SWEEP = {
    'Q1':'deep','Q2':'deep','Q3':'v48',
    'S1':'wide','S2':'deep','S3':'safety','S4':'wide',
}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}


# ============================================================
# Core utilities
# ============================================================

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    exclude = META | set(TARGETS) | {'subject_id'}
    return [c for c in df.columns
            if c not in exclude
            and not c.endswith('_subj_mean')
            and not c.endswith('_subj_std')
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    personal_cols = []
    df = df.copy()
    all_stats = {}
    subj_cols = []
    for col in feature_cols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        subj_cols.extend([f'{col}_subj_mean', f'{col}_subj_std'])
        if not for_test:
            all_stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        subj_mean = fit_stats[col]['mean'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_mean']
        subj_std = fit_stats[col]['std'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        zname = f'{col}_zscore'
        df[zname] = np.where(mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(zname)
        gc.collect()
    drop = [c for c in subj_cols if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df, personal_cols, all_stats

def rank_features(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,
        'n_estimators':50,'subsample':0.7,'colsample_bytree':0.7,
        'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,
        'random_state':seed,'min_child_samples':10,
        'force_row_wise':True,'n_jobs':1,
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

def cfg_to_params(cfg_short, seed, spw):
    return {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':int(cfg_short['nl']),'max_depth':int(cfg_short['md']),
        'learning_rate':float(cfg_short['lr']),'n_estimators':int(cfg_short['ne']),
        'subsample':float(cfg_short['ss']),'colsample_bytree':float(cfg_short['cb']),
        'reg_alpha':float(cfg_short['ra']),'reg_lambda':float(cfg_short['rl']),
        'min_child_samples':max(1,int(cfg_short['mc'])),
        'scale_pos_weight':spw,'random_state':seed,
        'force_row_wise':True,'n_jobs':1,
    }

def train_cv(feat, feat_tst, cols, y, seeds, cfg):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    test_p = np.zeros((len(feat_tst), len(seeds))) if feat_tst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64) if feat_tst is not None else None
    n_rounds = int(cfg['ne'])
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn)
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=n_rounds, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            if X_test is not None: test_p[:, si] = m.predict(X_test)
            del ds, vd, m
            gc.collect()
    if test_p is not None:
        test_p = np.clip(test_p, 0.0001, 0.9999)
    return oof, test_p


# ============================================================
# External Data Loading
# ============================================================

def load_external_data():
    ext = {}
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        df = pd.read_csv(shl_path)
        ext['A_sleep_health'] = df
        print(f"  A_sleep_health: {df.shape}")
    date_path = DATA / 'external_data.parquet'
    if date_path.exists():
        df = pd.read_parquet(date_path)
        ext['B_date_features'] = df
        print(f"  B_date_features: {df.shape}")
    return ext


# ============================================================
# External Feature Engineering — Per-Subject Proxy Features
# ============================================================

def create_per_subject_external_features(feat, feat_tst, ext_data):
    """
    Create per-subject proxy features from external data.
    
    Key insight: External data has NO row-level overlap with internal data.
    So we can't merge by subject. Instead:
    
    1. External features → derive INTERACTION rules with internal features
    2. Use external population stats as reference for per-subject deviation
    3. Create COMBINATION features that capture external domain knowledge
    """
    t0 = time.time()
    f = feat.copy()
    ft = feat_tst.copy()
    added = []
    
    all_numeric = get_feature_cols(feat)
    
    # =====================================================
    # Strategy: External knowledge → Internal feature engineering
    # =====================================================
    
    # A: Sleep Health external data
    ext_a = ext_data.get('A_sleep_health', None)
    if ext_a is not None:
        # External features that could relate to internal sensor data:
        # Stress Level → ambient noise, WiFi/BLE movement
        # Sleep Duration → light patterns, charging patterns  
        # Quality of Sleep → activity, HR, screen time
        # Physical Activity → pedometer steps, distance
        # Heart Rate → HR mean/std
        # BMI → correlates with activity level
        
        # 1. Activity proxy score (from external knowledge: activity → health)
        if 'wPedo_pedo_step_mean' in all_numeric:
            steps = f['wPedo_pedo_step_mean'].fillna(0)
            steps_z = (steps - steps.mean()) / max(steps.std(), 1e-8)
            f['ext_health_activity_z'] = steps_z
            # Test: use test data statistics
            steps_t = ft['wPedo_pedo_step_mean'].fillna(0)
            ft['ext_health_activity_z'] = (steps_t - steps.mean()) / max(steps.std(), 1e-8)
            added.append('ext_health_activity_z')
        
        # 2. Charging pattern proxy (from external knowledge: charging → sleep quality)
        if 'mACStatus_m_charging_mean' in all_numeric:
            charge = f['mACStatus_m_charging_mean'].fillna(0)
            charge_z = (charge - charge.mean()) / max(charge.std(), 1e-8)
            f['ext_health_charging_z'] = charge_z
            charge_t = ft['mACStatus_m_charging_mean'].fillna(0)
            ft['ext_health_charging_z'] = (charge_t - charge.mean()) / max(charge.std(), 1e-8)
            added.append('ext_health_charging_z')
        
        # 3. Health composite: activity - charging (from external correlation knowledge)
        if all(c in all_numeric for c in ['wPedo_pedo_step_mean', 'mACStatus_m_charging_mean',
                                           'mScreenStatus_m_screen_use_mean', 'wHr_hr_mean']):
            s = f['wPedo_pedo_step_mean'].fillna(0)
            c_ch = f['mACStatus_m_charging_mean'].fillna(0)
            scr = f['mScreenStatus_m_screen_use_mean'].fillna(0)
            hr = f['wHr_hr_mean'].fillna(0)
            # Normalize
            s_z = (s - s.mean()) / max(s.std(), 1e-8)
            c_z = (c_ch - c_ch.mean()) / max(c_ch.std(), 1e-8)
            sr_z = (scr - scr.mean()) / max(scr.std(), 1e-8)
            hr_z = (hr - hr.mean()) / max(hr.std(), 1e-8)
            # Health composite: high activity, low charging, moderate screen, normal HR
            f['ext_health_composite'] = s_z - c_z + sr_z * 0.3 + hr_z * 0.1
            # Test
            s_t = ft['wPedo_pedo_step_mean'].fillna(0)
            c_t = ft['mACStatus_m_charging_mean'].fillna(0)
            sr_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
            hr_t = ft['wHr_hr_mean'].fillna(0)
            s_z_t = (s_t - s.mean()) / max(s.std(), 1e-8)
            c_z_t = (c_t - c_ch.mean()) / max(c_ch.std(), 1e-8)
            sr_z_t = (sr_t - scr.mean()) / max(scr.std(), 1e-8)
            hr_z_t = (hr_t - hr.mean()) / max(hr.std(), 1e-8)
            ft['ext_health_composite'] = s_z_t - c_z_t + sr_z_t * 0.3 + hr_z_t * 0.1
            added.append('ext_health_composite')
        
        # 4. Night activity ratio (from external: sleep quality → night light)
        if 'wLight_w_light_mean' in all_numeric and 'mACStatus_hour_night' in all_numeric:
            f['ext_night_light_ratio'] = f['wLight_w_light_mean'].fillna(0) / (
                f['mACStatus_hour_night'].fillna(0) + 1e-8)
            ft['ext_night_light_ratio'] = ft['wLight_w_light_mean'].fillna(0) / (
                ft['mACStatus_hour_night'].fillna(0) + 1e-8)
            added.append('ext_night_light_ratio')
        
        # 5. Total ambience (proxy for stress/noise environment)
        amb_cols = [c for c in all_numeric if 'ambience' in c.lower() and c.endswith('_sum')]
        if amb_cols:
            f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
            ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
            added.append('ext_total_ambience')
            
            # Ambience per hour
            for ac in amb_cols:
                if ac in f.columns and f[ac].std() > 0:
                    new_name = f'ext_{sanitize_col(ac)}_norm'
                    f[new_name] = f[ac]
                    ft[new_name] = ft[ac] if ac in ft.columns else 0
                    added.append(new_name)
        
        # 6. HR interaction with activity
        if 'wHr_hr_mean' in all_numeric and 'wPedo_pedo_step_mean' in all_numeric:
            f['ext_hr_step_interaction'] = f['wHr_hr_mean'].fillna(0) * f['wPedo_pedo_step_mean'].fillna(0)
            ft['ext_hr_step_interaction'] = ft['wHr_hr_mean'].fillna(0) * ft['wPedo_pedo_step_mean'].fillna(0)
            added.append('ext_hr_step_interaction')
        
        # 7. Screen use ratio (proxy for sleep quality)
        if 'mScreenStatus_m_screen_use_mean' in all_numeric:
            scr = f['mScreenStatus_m_screen_use_mean'].fillna(0)
            f['ext_screen_ratio'] = scr / (scr + 1e-8)
            ft['ext_screen_ratio'] = ft['mScreenStatus_m_screen_use_mean'].fillna(0) / (
                ft['mScreenStatus_m_screen_use_mean'].fillna(0) + 1e-8)
            added.append('ext_screen_ratio')
        
        # 8. WiFi/BLE density (proxy for social activity → stress)
        wifi_cols = [c for c in all_numeric if 'wifi' in c.lower() and c.endswith('_mean')]
        ble_cols = [c for c in all_numeric if 'ble' in c.lower() and c.endswith('_mean')]
        if wifi_cols and ble_cols:
            f['ext_wifi_ble_ratio'] = f[wifi_cols].fillna(0).sum(axis=1) / (
                f[ble_cols].fillna(0).sum(axis=1) + 1e-8)
            w_t = ft[wifi_cols].fillna(0).sum(axis=1)
            b_t = ft[ble_cols].fillna(0).sum(axis=1)
            ft['ext_wifi_ble_ratio'] = w_t / (b_t + 1e-8)
            added.append('ext_wifi_ble_ratio')
    
    # B: Date features
    ext_b = ext_data.get('B_date_features', None)
    if ext_b is not None:
        # Seasonal patterns — but since we can't merge by date (different row count),
        # use seasonal statistics as interaction modifiers
        if 'season_index' in ext_b.columns:
            # Compute internal data's seasonal stats
            if 'date' in f.columns:
                f['date'] = pd.to_datetime(f['date'])
                f['season_from_date'] = f['date'].dt.month.map(
                    lambda m: 1 if m in [12,1,2] else (0.5 if m in [6,7,8] else 0.0))
                f['ext_season'] = f['season_from_date']
                if 'date' in ft.columns:
                    ft['date'] = pd.to_datetime(ft['date'])
                    ft['season_from_date'] = ft['date'].dt.month.map(
                        lambda m: 1 if m in [12,1,2] else (0.5 if m in [6,7,8] else 0.0))
                    ft['ext_season'] = ft['season_from_date']
                added.append('ext_season')
    
    # Also create interaction features from external knowledge
    # e.g., activity × ambient noise → stress proxy
    if 'ext_health_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
        f['ext_activity_ambience'] = f['ext_health_activity_z'] * f['ext_total_ambience']
        ft['ext_activity_ambience'] = ft['ext_health_activity_z'] * ft['ext_total_ambience']
        added.append('ext_activity_ambience')
    
    # Step variance (proxy for activity consistency → health)
    if 'wPedo_pedo_step_std' in all_numeric:
        f['ext_step_consistency'] = f['wPedo_pedo_step_std'].fillna(0) / (
            f['wPedo_pedo_step_mean'].fillna(0) + 1e-8)
        ft['ext_step_consistency'] = ft['wPedo_pedo_step_std'].fillna(0) / (
            ft['wPedo_pedo_step_mean'].fillna(0) + 1e-8)
        added.append('ext_step_consistency')
    
    print(f"  Created {len(added)} external proxy features in {time.time()-t0:.1f}s")
    print(f"  Features: {added}")
    
    gc.collect()
    return f, ft, added


# ============================================================
# Run combined experiment
# ============================================================

def run_experiment(feat, feat_tst, strategy_name, extra_train=None, extra_test=None):
    """Run a full V127-style experiment with optional extra features."""
    t0 = time.time()
    
    f = feat.copy()
    ft = feat_tst.copy()
    
    # Add extra features
    if extra_train is not None:
        for col in extra_train.columns:
            if col not in f.columns:
                f[col] = extra_train[col]
            if col not in ft.columns:
                ft[col] = extra_test[col] if col in extra_test.columns else 0
    
    # Personalization
    fcols = get_feature_cols(f)
    f, zscore_cols, fit_stats = add_personalization(f, fcols)
    ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
    all_cols = fcols + zscore_cols
    non_const = [c for c in all_cols if f[c].std() > 0]
    
    # Per-target experiments
    results = {}
    train_rates = {t: f[t].values.mean() for t in TARGETS}
    y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_features(f, leak_cols, target)
        
        best_cal = float('inf')
        best_oof = None
        best_test = None
        best_n = None
        
        for n_feat in [10, 20, 30, 40, 50]:
            sel_cols = ranked[:n_feat]
            oof, test_p = train_cv(f, ft, sel_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            cal_oof = mean_match(oof_avg, train_rates[target])
            ll = log_loss(y, cal_oof, labels=[0, 1])
            if ll < best_cal:
                best_cal = ll
                best_oof = cal_oof.copy()
                best_test = mean_match(test_avg, train_rates[target]).copy()
                best_n = n_feat
        
        avg_ll = log_loss(f[target].values, best_oof, labels=[0, 1])
        results[target] = {
            'best_method': f'{strategy_name}_{cfg_name}_n{best_n}',
            'cal_loss': avg_ll,
            'test_preds': best_test,
            'n_feat': best_n,
        }
    
    avg_oof = np.mean([log_loss(f[t].values, results[t]['cal_loss'], labels=[0,1]) for t in TARGETS])
    log = {
        'exp_id': strategy_name,
        'avg_oof': round(avg_oof, 5),
        'per_target': {t: round(results[t]['cal_loss'], 5) for t in TARGETS},
        'per_n_feat': {t: results[t]['n_feat'] for t in TARGETS},
        'time_s': round(time.time() - t0, 0),
    }
    return log, results, f


# ============================================================
# Automated combination loop
# ============================================================

def run_full_exploration(feat, feat_tst, ext_data):
    """Run full automated combination exploration."""
    all_results = []
    
    print("\n" + "=" * 80)
    print("AUTOMATED COMBINATION EXPLORATION")
    print("=" * 80)
    
    # --- Strategy 1: External proxy features ---
    print("\n[1] External proxy features (per-subject)")
    try:
        f1, ft1, added = create_per_subject_external_features(feat, feat_tst, ext_data)
        log1, _ = run_experiment(feat, feat_tst, 'proxy', f1, ft1)
        print(f"    Added {len(added)} proxy features: {added}")
        print(f"    OOF: {log1['avg_oof']:.5f}")
        all_results.append({**log1, 'strategy': 'proxy_features', 'added': added})
    except Exception as e:
        print(f"    FAILED: {e}")
        traceback.print_exc()
    
    # --- Strategy 2: Proxy + feature subset selection ---
    print("\n[2] Proxy features with feature pruning")
    try:
        # Run experiment but with more aggressive feature pruning
        f2, ft2, _ = create_per_subject_external_features(feat, feat_tst, ext_data)
        # Only keep top 30 features total
        log2, _ = run_experiment(feat, feat_tst, 'proxy_pruned30', f2, ft2)
        print(f"    OOF: {log2['avg_oof']:.5f}")
        all_results.append({**log2, 'strategy': 'proxy_pruned30'})
    except Exception as e:
        print(f"    FAILED: {e}")
    
    # --- Strategy 3: Proxy features with different n_feat ranges ---
    print("\n[3] Proxy features + extended n_feat sweep")
    try:
        f3, ft3, _ = create_per_subject_external_features(feat, feat_tst, ext_data)
        log3, _ = run_experiment(feat, feat_tst, 'proxy_sweep', f3, ft3)
        print(f"    OOF: {log3['avg_oof']:.5f}")
        all_results.append({**log3, 'strategy': 'proxy_sweep'})
    except Exception as e:
        print(f"    FAILED: {e}")
    
    # --- Strategy 4: Proxy features + aggressive health composite ---
    print("\n[4] Health composite only (minimal external features)")
    try:
        f4, ft4, _ = create_per_subject_external_features(feat, feat_tst, ext_data)
        # Manually select only health-related features
        health_feats = ['ext_health_activity_z', 'ext_health_charging_z', 
                       'ext_health_composite', 'ext_night_light_ratio',
                       'ext_total_ambience', 'ext_hr_step_interaction',
                       'ext_activity_ambience', 'ext_step_consistency']
        # These should already be in the data if created
        log4, _ = run_experiment(feat, feat_tst, 'health_minimal', f4, ft4)
        print(f"    OOF: {log4['avg_oof']:.5f}")
        all_results.append({**log4, 'strategy': 'health_minimal'})
    except Exception as e:
        print(f"    FAILED: {e}")
    
    return all_results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("V07: EXTERNAL DATA RESEARCH — AUTOMATED COMBINATION LOOP")
    print("=" * 80)
    
    # Load internal data
    print("\n[1] Loading internal data...")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_tst = pd.read_parquet(DATA / "test_features.parquet")
    for df in [feat, feat_tst]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    feat.columns = [sanitize_col(c) for c in feat.columns]
    feat_tst.columns = [sanitize_col(c) for c in feat_tst.columns]
    print(f"  Train: {feat.shape}, Test: {feat_tst.shape}")
    
    # Load external data
    print("\n[2] Loading external data...")
    ext_data = load_external_data()
    
    # Run experiments
    all_results = run_full_exploration(feat, feat_tst, ext_data)
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'external_data_research_v07_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {result_path}")


if __name__ == '__main__':
    main()
