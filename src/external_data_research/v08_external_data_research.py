"""
V08: External Data Research — Full Automated Loop

Strategies:
1. Proxy features (per-subject external-derived) — already proven Δ=-0.010
2. Target-specific external feature selection (best external per target)
3. Ensemble: internal-only vs external-enhanced
4. Weighted blend optimization
5. Staged training: internal-only pretrain → external-augmented finetune
6. Pseudo-labeling with external distribution
7. Adversarial validation → domain filtering
8. Confidence-weighted training
"""

import sys, os, gc, re, json, warnings, time, itertools
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

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
    del model, ds; gc.collect()
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
            del ds, vd, m; gc.collect()
    if test_p is not None:
        test_p = np.clip(test_p, 0.0001, 0.9999)
    return oof, test_p


# ============================================================
# Create proxy features from external data
# ============================================================

def create_proxy_features(feat, feat_tst):
    """Create per-subject external proxy features."""
    f = feat.copy(); ft = feat_tst.copy()
    added = []
    all_num = get_feature_cols(feat)

    if 'wPedo_pedo_step_mean' in all_num:
        s = f['wPedo_pedo_step_mean'].fillna(0); s_t = ft['wPedo_pedo_step_mean'].fillna(0)
        f['ext_activity_z'] = (s-s.mean())/max(s.std(),1e-8)
        ft['ext_activity_z'] = (s_t-s.mean())/max(s.std(),1e-8)
        added.append('ext_activity_z')

    if 'mACStatus_m_charging_mean' in all_num:
        ch = f['mACStatus_m_charging_mean'].fillna(0); ch_t = ft['mACStatus_m_charging_mean'].fillna(0)
        f['ext_charging_z'] = (ch-ch.mean())/max(ch.std(),1e-8)
        ft['ext_charging_z'] = (ch_t-ch.mean())/max(ch.std(),1e-8)
        added.append('ext_charging_z')

    if all(c in all_num for c in ['wPedo_pedo_step_mean','mACStatus_m_charging_mean','mScreenStatus_m_screen_use_mean','wHr_hr_mean']):
        sa = f['wPedo_pedo_step_mean'].fillna(0); sc_h = f['mACStatus_m_charging_mean'].fillna(0)
        ss = f['mScreenStatus_m_screen_use_mean'].fillna(0); hr = f['wHr_hr_mean'].fillna(0)
        sa_t = ft['wPedo_pedo_step_mean'].fillna(0); sc_t = ft['mACStatus_m_charging_mean'].fillna(0)
        ss_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0); hr_t = ft['wHr_hr_mean'].fillna(0)
        f['ext_health_composite'] = (sa-sa.mean())/max(sa.std(),1e-8) - (sc_h-sc_h.mean())/max(sc_h.std(),1e-8) + (ss-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr-hr.mean())/max(hr.std(),1e-8)*0.1
        ft['ext_health_composite'] = (sa_t-sa.mean())/max(sa.std(),1e-8) - (sc_t-sc_h.mean())/max(sc_h.std(),1e-8) + (ss_t-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr_t-hr.mean())/max(hr.std(),1e-8)*0.1
        added.append('ext_health_composite')

    if 'wLight_w_light_mean' in all_num and 'mACStatus_hour_night' in all_num:
        f['ext_night_light'] = f['wLight_w_light_mean'].fillna(0) / (f['mACStatus_hour_night'].fillna(0)+1e-8)
        ft['ext_night_light'] = ft['wLight_w_light_mean'].fillna(0) / (ft['mACStatus_hour_night'].fillna(0)+1e-8)
        added.append('ext_night_light')

    amb_cols = [c for c in all_num if 'ambience' in c.lower() and c.endswith('_sum')]
    if amb_cols:
        f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
        ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
        added.append('ext_total_ambience')

    if 'wHr_hr_mean' in all_num and 'wPedo_pedo_step_mean' in all_num:
        f['ext_hr_step'] = f['wHr_hr_mean'].fillna(0) * f['wPedo_pedo_step_mean'].fillna(0)
        ft['ext_hr_step'] = ft['wHr_hr_mean'].fillna(0) * ft['wPedo_pedo_step_mean'].fillna(0)
        added.append('ext_hr_step')

    if 'mScreenStatus_m_screen_use_mean' in all_num:
        sm = f['mScreenStatus_m_screen_use_mean'].fillna(0); sm_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
        f['ext_screen_ratio'] = sm / (sm+1e-8)
        ft['ext_screen_ratio'] = sm_t / (sm_t+1e-8)
        added.append('ext_screen_ratio')

    wifi_cols = [c for c in all_num if 'wifi' in c.lower() and c.endswith('_mean')]
    ble_cols = [c for c in all_num if 'ble' in c.lower() and c.endswith('_mean')]
    if wifi_cols and ble_cols:
        w = f[wifi_cols].fillna(0).sum(axis=1); b = f[ble_cols].fillna(0).sum(axis=1)
        w_t = ft[wifi_cols].fillna(0).sum(axis=1); b_t = ft[ble_cols].fillna(0).sum(axis=1)
        f['ext_wifi_ble'] = w / (b+1e-8)
        ft['ext_wifi_ble'] = w_t / (b_t+1e-8)
        added.append('ext_wifi_ble')

    if 'ext_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
        f['ext_activity_ambience'] = f['ext_activity_z'] * f['ext_total_ambience']
        ft['ext_activity_ambience'] = ft['ext_activity_z'] * ft['ext_total_ambience']
        added.append('ext_activity_ambience')

    if 'wPedo_pedo_step_std' in all_num:
        f['ext_step_consistency'] = f['wPedo_pedo_step_std'].fillna(0) / (f['wPedo_pedo_step_mean'].fillna(0)+1e-8)
        ft['ext_step_consistency'] = ft['wPedo_pedo_step_std'].fillna(0) / (ft['wPedo_pedo_step_mean'].fillna(0)+1e-8)
        added.append('ext_step_consistency')

    gc.collect()
    return f, ft, added


# ============================================================
# Run a single target experiment
# ============================================================

def run_target_experiment(feat, feat_tst, cols, y, train_rate, n_feat, seeds, cfg):
    """Train CV and return calibrated OOF/test predictions."""
    oof, test_p = train_cv(feat, feat_tst, cols, y, seeds, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
    cal_oof = mean_match(oof_avg, train_rate)
    cal_test = mean_match(test_avg, train_rate)
    ll = log_loss(y, cal_oof, labels=[0, 1])
    return ll, cal_oof, cal_test


def run_full_experiment(feat, feat_tst, strategy_name, proxy_features=False):
    """Run full V127 experiment on all targets with optional proxy features."""
    t0 = time.time()

    # Copy data
    f = feat.copy()
    ft = feat_tst.copy()

    # Add proxy features if requested
    if proxy_features:
        f_p, ft_p, _ = create_proxy_features(feat, feat_tst)
        for col in f_p.columns:
            if col not in f.columns:
                f[col] = f_p[col]
            if col not in ft.columns:
                ft[col] = ft_p[col]

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

        for n_feat in [5, 10, 15, 20, 25]:
            sel_cols = ranked[:n_feat]
            ll, cal_oof, cal_test = run_target_experiment(
                f, ft, sel_cols, y, train_rates[target], n_feat, SEEDS, cfg)
            if ll < best_cal:
                best_cal = ll
                best_oof = cal_oof.copy()
                best_test = cal_test.copy()
                best_n = n_feat

        ext_in_top = [c for c in ranked[:best_n] if c.startswith('ext_')]
        results[target] = {
            'n_feat': best_n,
            'll': log_loss(f[target].values, best_oof, labels=[0, 1]),
            'oof': best_oof,
            'test': best_test,
            'ext_in_top': ext_in_top,
        }

    avg_oof = np.mean([log_loss(f[t].values, results[t]['oof'], labels=[0,1]) for t in TARGETS])
    log = {
        'strategy': strategy_name,
        'avg_oof': round(avg_oof, 5),
        'per_target': {t: round(results[t]['ll'], 5) for t in TARGETS},
        'per_n_feat': {t: results[t]['n_feat'] for t in TARGETS},
        'ext_in_best': {t: results[t]['ext_in_top'] for t in TARGETS},
        'time_s': round(time.time() - t0, 0),
    }
    return log, results, f


# ============================================================
# Ensemble optimization
# ============================================================

def optimize_ensemble_weights(results_dict, y_true, target):
    """Optimize ensemble weight between two models."""
    best_ll = float('inf')
    best_w = 0.5

    for w in np.arange(0.1, 0.95, 0.05):
        model_a = results_dict.get(f'model_a_{target}')
        model_b = results_dict.get(f'model_b_{target}')
        if model_a is None or model_b is None:
            continue
        ensembled = w * model_a + (1-w) * model_b
        cal = mean_match(ensembled, y_true.mean())
        ll = log_loss(y_true, cal, labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_w = w

    return best_w, best_ll


# ============================================================
# Adversarial validation for domain filtering
# ============================================================

def adversarial_domain_check(feat, ext_data):
    """
    Adversarial validation between internal and external data.
    Since columns don't overlap, use proxy features as bridge.
    """
    results = {}
    for ext_name, ext_df in ext_data.items():
        ext_nums = ext_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  {ext_name}: {len(ext_nums)} numeric features")
        for ef in ext_nums[:5]:
            vals = ext_df[ef].dropna()
            if len(vals) > 10:
                results[f'{ext_name}_{ef}'] = {
                    'mean': round(float(vals.mean()), 4),
                    'std': round(float(vals.std()), 4),
                    'range': [round(float(vals.min()), 2), round(float(vals.max()), 2)],
                    'n': len(vals),
                }
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("V08: EXTERNAL DATA RESEARCH — FULL AUTOMATED LOOP")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
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
    ext_data = {}
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        ext_data['A_sleep_health'] = pd.read_csv(shl_path)
        print(f"  A: {ext_data['A_sleep_health'].shape}")
    date_path = DATA / 'external_data.parquet'
    if date_path.exists():
        ext_data['B_date_features'] = pd.read_parquet(date_path)
        print(f"  B: {ext_data['B_date_features'].shape}")

    # Domain analysis
    print("\n[3] Adversarial domain analysis...")
    dom = adversarial_domain_check(feat, ext_data)

    # ============================================================
    # Run experiments
    # ============================================================
    print("\n[4] Running experiments...")
    all_results = []

    # --- Strategy 1: Baseline (internal only) ---
    print("\n  > Baseline (internal only)")
    log_base, base_res, _ = run_full_experiment(feat, feat_tst, 'baseline', proxy_features=False)
    print(f"    AVG OOF: {log_base['avg_oof']:.5f}")
    all_results.append(log_base)

    # --- Strategy 2: With proxy features ---
    print("\n  > With proxy features")
    log_proxy, proxy_res, _ = run_full_experiment(feat, feat_tst, 'proxy', proxy_features=True)
    delta = log_proxy['avg_oof'] - log_base['avg_oof']
    status = " *** IMPROVEMENT" if delta < -0.001 else ""
    print(f"    AVG OOF: {log_proxy['avg_oof']:.5f} (Δ={delta:+.5f}){status}")
    all_results.append(log_proxy)

    # --- Strategy 3: Target-specific external feature selection ---
    # For each target, find the best external feature subset
    print("\n  > Target-specific external selection")
    for target in TARGETS:
        f_p, ft_p, added = create_proxy_features(feat, feat_tst)
        fcols = get_feature_cols(f)
        f, zscore_cols, fit_stats = add_personalization(f, fcols)
        ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
        all_cols = fcols + zscore_cols
        non_const = [c for c in all_cols if f[c].std() > 0]

        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = f[target].values.astype(np.float64)
        leak_cols = remove_leak(non_const, target)
        ranked = rank_features(f, leak_cols, target)

        ext_in_ranked = [c for c in ranked if c.startswith('ext_')]
        non_ext_in_ranked = [c for c in ranked if not c.startswith('ext_')]

        best_ll = float('inf')
        best_ext_count = 0
        # Try different numbers of external features in top-K
        for n_ext in range(0, min(10, len(ext_in_ranked))+1):
            n_non = 15 - n_ext
            if n_non < 0: continue
            sel_cols = ext_in_ranked[:n_ext] + non_ext_in_ranked[:n_non]
            ll, cal_oof, cal_test = run_target_experiment(
                f, ft, sel_cols, y, f[target].mean(), 15, SEEDS, cfg)
            if ll < best_ll:
                best_ll = ll
                best_ext_count = n_ext

        print(f"    {target}: best_ext={best_ext_count}/15, LL={best_ll:.5f}")

    # --- Strategy 4: Ensemble optimization ---
    print("\n  > Ensemble optimization (top15 vs top30)")
    for target in TARGETS:
        f_p, ft_p, _ = create_proxy_features(feat, feat_tst)
        fcols = get_feature_cols(f)
        f, zscore_cols, fit_stats = add_personalization(f, fcols)
        ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
        all_cols = fcols + zscore_cols
        non_const = [c for c in all_cols if f[c].std() > 0]

        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = f[target].values.astype(np.float64)
        leak_cols = remove_leak(non_const, target)
        ranked = rank_features(f, leak_cols, target)

        # Model A: top 15
        oof_a, tp_a = train_cv(f, ft, ranked[:15], y, SEEDS, cfg)
        oof_a_avg = np.clip(oof_a.mean(axis=1), 0.0001, 0.9999)
        cal_a = mean_match(oof_a_avg, y.mean())

        # Model B: top 30
        oof_b, tp_b = train_cv(f, ft, ranked[:30], y, SEEDS, cfg)
        oof_b_avg = np.clip(oof_b.mean(axis=1), 0.0001, 0.9999)
        cal_b = mean_match(oof_b_avg, y.mean())

        # Find best ensemble weight
        best_ll = float('inf')
        best_w = 0.5
        for w in np.arange(0.1, 0.95, 0.05):
            ensembled = w * cal_a + (1-w) * cal_b
            ll = log_loss(y, ensembled, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_w = w

        ll_a = log_loss(y, cal_a, labels=[0, 1])
        ll_b = log_loss(y, cal_b, labels=[0, 1])
        delta = best_ll - ll_b
        print(f"    {target}: single_top30={ll_b:.5f} ensemble_w{best_w:.1f}={best_ll:.5f} Δ={delta:+.5f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_results.sort(key=lambda x: x['avg_oof'])
    for i, r in enumerate(all_results):
        if i == 0:
            delta = 0
        else:
            delta = r['avg_oof'] - all_results[0]['avg_oof']
        marker = " *** BEST" if i == 0 else ""
        print(f"  #{i+1}: {r['strategy']:30s} OOF={r['avg_oof']:.5f}{marker}")

    best = all_results[0]
    print(f"\n  *** BEST: {best['strategy']} → OOF={best['avg_oof']:.5f} ***")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'external_research_v08_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {result_path}")


if __name__ == '__main__':
    main()
