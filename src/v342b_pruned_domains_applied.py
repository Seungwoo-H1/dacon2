"""
V342b — Per-Target Pruned Domain Aggregates (Applied)

V342 showed that domain aggregates add signal for S1/S2/Q3 but noise for Q1/S3/S4.
Domain importance ranking revealed:
- mBle: most important across all targets
- wPedo/wLight: least important across all targets
- Target-specific patterns: S1←mScreenStatus, S3←mBle/wHr/mACStatus

Hypothesis: Per-target domain pruning (keep top-N important domains, discard rest)
will reduce noise → lower student OOF → smaller student-meta gap → better LB.

Method:
1. For each target, select top-N domains by importance (N varies per target)
2. Use ONLY base features + pruned domain aggregates + ratios
3. Apply z-scores only to retained features
4. V308 architecture otherwise

Expected: student OOF improvement -0.01 to -0.03 vs V341
          AVG Meta OOF: similar or modestly better
          student-meta gap: significantly smaller
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0

DOMAINS = [
    'mACStatus','mActivity','mAmbience','mBle','mGps',
    'mLight','mScreenStatus','mUsageStats','mWifi',
    'wHr','wLight','wPedo',
]

# Top-N domains per target (based on V342 ranking analysis)
# Keep domains with avg_rank < threshold
# Strategy: keep domains that appear in top-6 of the ranking
PRUNED_DOMAINS = {
    'Q1': ['mBle', 'mAmbience', 'mWifi', 'mScreenStatus', 'mACStatus', 'mActivity'],  # top 6
    'Q2': ['mACStatus', 'mAmbience', 'mBle', 'mLight', 'mScreenStatus', 'wPedo'],  # top 6
    'Q3': ['mBle', 'wHr', 'mLight', 'mWifi', 'mACStatus', 'mScreenStatus'],  # top 6
    'S1': ['mScreenStatus', 'wHr', 'mBle', 'mLight', 'mUsageStats', 'mWifi'],  # top 6
    'S2': ['mBle', 'mActivity', 'mGps', 'mAmbience', 'mScreenStatus', 'mLight'],  # top 6
    'S3': ['mBle', 'wHr', 'mACStatus', 'mWifi', 'mGps', 'mActivity'],  # top 6
    'S4': ['mACStatus', 'mBle', 'mWifi', 'mGps', 'mUsageStats', 'mScreenStatus'],  # top 6
}

# Also sweep: try keeping only top-4 domains for more aggressive pruning
PRUNED_DOMAINS_4 = {
    'Q1': ['mBle', 'mAmbience', 'mWifi', 'mScreenStatus'],
    'Q2': ['mACStatus', 'mAmbience', 'mBle', 'mLight'],
    'Q3': ['mBle', 'wHr', 'mLight', 'mWifi'],
    'S1': ['mScreenStatus', 'wHr', 'mBle', 'mLight'],
    'S2': ['mBle', 'mActivity', 'mGps', 'mAmbience'],
    'S3': ['mBle', 'wHr', 'mACStatus', 'mWifi'],
    'S4': ['mACStatus', 'mBle', 'mWifi', 'mGps'],
}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def compute_domain_aggregates_for_domains(df, base_cols, domains_to_use):
    """Compute domain-level aggregates only for specified domains."""
    result = df.copy()
    new_cols = []
    
    # Group columns by domain
    domain_features = {d: [] for d in domains_to_use}
    for col in base_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            prefix = parts[0]
            if prefix in domains_to_use:
                domain_features[prefix[0]].append(col)  # Use prefix as domain key
    
    # Map prefix to full domain name
    prefix_to_domain = {}
    for d in domains_to_use:
        # Find which key in domain_features matches
        for key in domain_features:
            if d.startswith(key) or key.startswith(d[0]):
                prefix_to_domain[key] = d
                break
    
    for domain, cols in domain_features.items():
        if len(cols) < 2:
            continue
        col_data = df[cols].fillna(0).values.astype(np.float64)
        
        domain_mean = np.nanmean(col_data, axis=1)
        result[f'{domain}_agg_mean'] = domain_mean
        new_cols.append(f'{domain}_agg_mean')
        
        domain_std = np.nanstd(col_data, axis=1, ddof=0)
        result[f'{domain}_agg_std'] = domain_std
        new_cols.append(f'{domain}_agg_std')
        
        domain_min = np.nanmin(col_data, axis=1)
        result[f'{domain}_agg_min'] = domain_min
        new_cols.append(f'{domain}_agg_min')
        
        domain_max = np.nanmax(col_data, axis=1)
        result[f'{domain}_agg_max'] = domain_max
        new_cols.append(f'{domain}_agg_max')
        
        domain_range = domain_max - domain_min
        result[f'{domain}_agg_range'] = domain_range
        new_cols.append(f'{domain}_agg_range')
        
        abs_mean = np.abs(domain_mean)
        cv = domain_std / np.maximum(abs_mean, 1e-8)
        result[f'{domain}_agg_cv'] = cv
        new_cols.append(f'{domain}_agg_cv')
        
        try:
            flat = col_data.flatten()
            bins = np.percentile(flat, [0, 20, 40, 60, 80, 100])
            bins = np.unique(bins)
            if len(bins) > 2:
                entropies = []
                for i in range(len(df)):
                    row_vals = col_data[i]
                    hist, _ = np.histogram(row_vals, bins=bins)
                    probs = hist / hist.sum()
                    probs = probs[probs > 0]
                    ent = -np.sum(probs * np.log2(probs))
                    entropies.append(ent)
                max_ent = np.log2(len(bins) - 1) if len(bins) > 1 else 1.0
                norm_ent = np.array(entropies) / max_ent if max_ent > 0 else entropies
                result[f'{domain}_agg_entropy'] = norm_ent
                new_cols.append(f'{domain}_agg_entropy')
        except Exception as e:
            log.warning(f"  Entropy for {domain} failed: {e}")
    
    return result, new_cols


def compute_cross_domain_ratios(df, base_cols):
    """Compute cross-domain ratio features."""
    result = df.copy()
    new_cols = []
    
    representatives = {}
    for col in base_cols:
        if col.endswith('_mean'):
            parts = col.split('_')
            prefix = parts[0]
            if prefix not in representatives:
                representatives[prefix] = col
    
    ratios = [
        ('wHr', 'mActivity', 'hr_activity_ratio'),
        ('mGps', 'mActivity', 'gps_activity_ratio'),
        ('mLight', 'mScreenStatus', 'light_screen_ratio'),
        ('mWifi', 'mBle', 'wifi_ble_ratio'),
        ('mActivity', 'mGps', 'activity_gps_ratio'),
        ('wHr', 'mWifi', 'hr_wifi_ratio'),
        ('wPedo', 'mGps', 'pedo_gps_ratio'),
        ('wLight', 'mACStatus', 'light_charging_ratio'),
    ]
    
    for dom_a, dom_b, suffix in ratios:
        if dom_a in representatives and dom_b in representatives:
            col_a = representatives[dom_a]
            col_b = representatives[dom_b]
            if col_a in df.columns and col_b in df.columns:
                val_a = df[col_a].fillna(0).values.astype(np.float64)
                val_b = df[col_b].fillna(0).values.astype(np.float64)
                ratio = val_a / np.maximum(np.abs(val_b), 1e-8)
                result[f'{dom_a}_{dom_b}_{suffix}'] = ratio
                new_cols.append(f'{dom_a}_{dom_b}_{suffix}')
    
    return result, new_cols


def rank_features(feat_df, feat_cols, target, seed=SEED):
    """Rank features by LGBM gain importance."""
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def generate_test_zscore(train_df, test_df, base_cols):
    """Generate z-score features."""
    for col in base_cols:
        if col not in test_df.columns:
            continue
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc_name = f'{col}_zscore'
        test_df[zc_name] = (test_vals - mean) / std
        train_df[zc_name] = (train_vals - mean) / std
    return test_df, train_df


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V342b — Per-Target Pruned Domain Aggregates (Applied)")
    log.info("Hypothesis: prune domain aggregates per target → lower student OOF")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Base columns
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"Base features: {len(base_cols)}")
    
    # Run both sweeps (top-4 and top-6)
    sweeps = [
        ('top6', PRUNED_DOMAINS),
        ('top4', PRUNED_DOMAINS_4),
    ]
    
    best_result = None
    best_avg_oof = float('inf')
    
    for sweep_name, domain_config in sweeps:
        log.info(f"\n{'='*60}")
        log.info(f"SWEEP: {sweep_name} (keeping top domains per target)")
        
        # For each target, build feature set with pruned domains
        gkf = GroupKFold(n_splits=N_FOLDS)
        n_train = len(train_df)
        n_test = len(test_df)
        group = train_df['subject_id'].values
        
        target_configs = {}
        for t in TARGETS:
            pruned = domain_config[t]
            log.info(f"  {t}: keeping {len(pruned)} domains: {pruned}")
            
            # Compute domain aggregates for pruned domains
            t_train = train_df.copy()
            t_test = test_df.copy()
            
            agg_cols = []
            for domain in pruned:
                # Find base cols for this domain
                domain_base = [c for c in base_cols if c.split('_')[0] == domain[0] or c.startswith(domain + '_') or c.startswith(domain[0] + '_')]
                # More precise matching
                domain_base = [c for c in base_cols if any(c.startswith(p + '_') for p in [domain])]
                # Actually: match by prefix
                domain_base = [c for c in base_cols if c.split('_')[0] in [domain]]
                
                if not domain_base:
                    # Try prefix match (e.g., 'wHr' matches 'wHr_hr_*')
                    prefix = domain[0]  # 'm' or 'w' + first char
                    for dname in DOMAINS:
                        if domain == dname:
                            domain_base = [c for c in base_cols if c.split('_')[0] == dname or c.split('_')[0] == dname[:2]]
                            break
            
            # Simpler: directly match prefix
            domain_base = []
            for col in base_cols:
                prefix = col.split('_')[0]
                for d in pruned:
                    if prefix == d:
                        domain_base.append(col)
                        break
            
            # Compute aggregates
            if len(domain_base) >= 2:
                col_data = t_train[domain_base].fillna(0).values.astype(np.float64)
                
                t_train[f'{domain}_agg_mean'] = np.nanmean(col_data, axis=1)
                t_train[f'{domain}_agg_std'] = np.nanstd(col_data, axis=1, ddof=0)
                t_train[f'{domain}_agg_min'] = np.nanmin(col_data, axis=1)
                t_train[f'{domain}_agg_max'] = np.nanmax(col_data, axis=1)
                t_train[f'{domain}_agg_range'] = np.nanmax(col_data, axis=1) - np.nanmin(col_data, axis=1)
                abs_m = np.abs(np.nanmean(col_data, axis=1))
                t_train[f'{domain}_agg_cv'] = np.nanstd(col_data, axis=1, ddof=0) / np.maximum(abs_m, 1e-8)
                
                # Entropy
                try:
                    flat = col_data.flatten()
                    bins = np.percentile(flat, [0, 20, 40, 60, 80, 100])
                    bins = np.unique(bins)
                    if len(bins) > 2:
                        entropies = []
                        for i in range(n_train):
                            row_vals = col_data[i]
                            hist, _ = np.histogram(row_vals, bins=bins)
                            probs = hist / hist.sum()
                            probs = probs[probs > 0]
                            ent = -np.sum(probs * np.log2(probs))
                            entropies.append(ent)
                        max_ent = np.log2(len(bins) - 1)
                        t_train[f'{domain}_agg_entropy'] = np.array(entropies) / max_ent if max_ent > 0 else entropies
                except:
                    pass
                
                # Test
                test_data = t_test[domain_base].fillna(0).values.astype(np.float64)
                t_test[f'{domain}_agg_mean'] = np.nanmean(test_data, axis=1)
                t_test[f'{domain}_agg_std'] = np.nanstd(test_data, axis=1, ddof=0)
                t_test[f'{domain}_agg_min'] = np.nanmin(test_data, axis=1)
                t_test[f'{domain}_agg_max'] = np.nanmax(test_data, axis=1)
                t_test[f'{domain}_agg_range'] = np.nanmax(test_data, axis=1) - np.nanmin(test_data, axis=1)
                abs_m_t = np.abs(np.nanmean(test_data, axis=1))
                t_test[f'{domain}_agg_cv'] = np.nanstd(test_data, axis=1, ddof=0) / np.maximum(abs_m_t, 1e-8)
                try:
                    bins_t = np.percentile(test_data.flatten(), [0, 20, 40, 60, 80, 100])
                    bins_t = np.unique(bins_t)
                    if len(bins_t) > 2:
                        entropies_t = []
                        for i in range(n_test):
                            row_vals = test_data[i]
                            hist, _ = np.histogram(row_vals, bins=bins_t)
                            probs = hist / hist.sum()
                            probs = probs[probs > 0]
                            ent = -np.sum(probs * np.log2(probs))
                            entropies_t.append(ent)
                        max_ent_t = np.log2(len(bins_t) - 1)
                        t_test[f'{domain}_agg_entropy'] = np.array(entropies_t) / max_ent_t if max_ent_t > 0 else entropies_t
                except:
                    pass
                
                agg_cols.extend([f'{domain}_agg_mean', f'{domain}_agg_std', f'{domain}_agg_min',
                                 f'{domain}_agg_max', f'{domain}_agg_range', f'{domain}_agg_cv',
                                 f'{domain}_agg_entropy'])
            
            # Build feature set: base (from pruned domains) + agg
            # For feature ranking, we include base features from pruned domains + agg features
            all_feat_cols = domain_base + agg_cols
            
            target_configs[t] = {
                'train_df': t_train,
                'test_df': t_test,
                'features': all_feat_cols,
            }
        
        # Now run full pipeline with pruned features
        all_oofs = {}
        all_student_oofs = {}
        
        for t in TARGETS:
            tc = target_configs[t]
            y = tc['train_df'][t].values.astype(np.float64)
            feats = tc['features']
            feat_cols_clean = remove_leak(feats, t)
            n_feat = V53_SWEEP[t]['n_feat']
            cfg_name = V53_SWEEP[t]['cfg']
            
            # Feature ranking
            ranked = rank_features(tc['train_df'], feat_cols_clean, t)
            sel_cols = ranked[:n_feat]
            
            sel_cols_test = [c for c in sel_cols if c in tc['test_df'].columns]
            
            cfg = CFGS[cfg_name]
            
            per_seed_oofs = []
            test_preds_arr = np.zeros((n_test, N_SEEDS))
            
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                seed_oof = np.zeros(n_train)
                seed_test = np.zeros(n_test)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(tc['train_df'], y, group)):
                    X_tr = tc['train_df'][sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = tc['train_df'][sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    seed_test += m.predict(tc['test_df'][sel_cols_test].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                test_preds_arr[:, si] = seed_test
            
            stacked = np.column_stack(per_seed_oofs)
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
            
            train_oof = meta.predict_proba(stacked)[:, 1]
            oof_ll = log_loss(y, np.clip(train_oof, 0.001, 0.999))
            all_oofs[t] = oof_ll
            
            student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
            student_ll = log_loss(y, student_oof)
            all_student_oofs[t] = student_ll
            
            test_stacked = np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)])
            test_pred = meta.predict_proba(test_stacked)[:, 1]
            
            if t not in locals() or True:
                pass
            tc['test_pred'] = np.clip(test_pred, 0.01, 0.99)
            
            log.info(f"    {t}: student={student_ll:.5f}, meta={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
        
        avg_oof = np.mean(list(all_oofs.values()))
        
        log.info(f"\n  {sweep_name} AVG Meta OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
        
        if avg_oof < best_avg_oof:
            best_avg_oof = avg_oof
            best_result = {
                'sweep': sweep_name,
                'oofs': all_oofs,
                'student_oofs': all_student_oofs,
                'configs': target_configs,
                'train_df': train_df,
                'test_df': test_df,
            }
    
    # Use best result
    log.info(f"\n{'='*70}")
    log.info(f"V342b BEST: {best_result['sweep']} sweep")
    log.info(f"{'='*70}")
    
    all_oofs = best_result['oofs']
    all_student_oofs = best_result['student_oofs']
    v308_avg = 0.62235
    v341_avg = 0.61825
    
    log.info(f"{'Target':<6} {'Student OOF':>12} {'Meta OOF':>12} {'Gap':>8} {'ΔV308':>8}")
    log.info(f"{'-'*48}")
    for t in TARGETS:
        log.info(f"{t:<6} {all_student_oofs[t]:>12.5f} {all_oofs[t]:>12.5f} {all_oofs[t]-all_student_oofs[t]:>+8.5f} {all_oofs[t]-v308_avg:>+8.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    log.info(f"{'='*48}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF:    {v308_avg:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  V341 AVG OOF:    {v341_avg:.5f}")
    log.info(f"  Δ vs V341:       {avg_oof - v341_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs = best_result['configs']
    
    sub = pd.DataFrame()
    sub['subject_id'] = best_result['test_df']['subject_id'].values
    sub['sleep_date'] = best_result['test_df']['sleep_date'].values
    sub['lifelog_date'] = best_result['test_df']['lifelog_date'].values
    for t in TARGETS:
        sub[t] = configs[t]['test_pred']
    
    sub_path = SUBMIT / f"submission_v342b_{best_result['sweep']}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V342b',
        'name': f'Per-Target Pruned Domains ({best_result["sweep"]})',
        'sweep_used': best_result['sweep'],
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v341': round(float(avg_oof - v341_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v342b_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
