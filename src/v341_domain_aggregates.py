"""
V341 — Domain-Level Aggregated Features + Cross-Domain Ratios

Hypothesis: V308의 top-K feature selection이 특정 도메인의 signal을 놓치고 있음.
각 도메인의 전체 aggregated features (mean, std, entropy)와
도메인 간 비율 features가 추가 signal을 제공할 것.

Changes from V308:
1. Add domain-level aggregates: for each domain, compute mean/std/min/max of all its features
2. Add cross-domain ratios: ratios of representative features between domains
3. Add domain entropy: per-domain feature distribution entropy
4. Keep V308 architecture (15 seeds, GroupKFold 5, LR meta C=10)
5. Append new features to V308's 282 features → ~350 total

Key insight: Domain aggregates capture holistic sensor behavior that individual
features might miss. Cross-domain ratios capture inter-device relationships.
Domain entropy captures behavioral regularity.

Expected: modest OOF improvement (-0.003 to -0.008) with low risk.
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

# Domain definitions based on feature prefix analysis
DOMAINS = {
    'mACStatus': ['mACStatus'],
    'mActivity': ['mActivity'],
    'mAmbience': ['mAmbience'],
    'mBle': ['mBle'],
    'mGps': ['mGps'],
    'mLight': ['mLight'],
    'mScreenStatus': ['mScreenStatus'],
    'mUsageStats': ['mUsageStats'],
    'mWifi': ['mWifi'],
    'wHr': ['wHr'],
    'wLight': ['wLight'],
    'wPedo': ['wPedo'],
}

# Map prefix to domain
PREFIX_TO_DOMAIN = {}
for domain, prefixes in DOMAINS.items():
    for p in prefixes:
        PREFIX_TO_DOMAIN[p] = domain


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


def compute_domain_aggregates(df, base_cols):
    """Compute domain-level aggregated features.
    
    For each domain, compute:
    - mean of all features in domain
    - std of all features in domain
    - min of all features in domain
    - max of all features in domain
    - entropy of feature distribution (Shannon-like)
    - range (max - min)
    """
    result = df.copy()
    new_cols = []
    
    # Group columns by domain
    domain_features = {d: [] for d in DOMAINS}
    for col in base_cols:
        # Extract prefix (before first underscore after 'm' or 'w')
        parts = col.split('_')
        if len(parts) >= 2:
            prefix = parts[0]  # e.g., 'wHr', 'mLight'
            if prefix in PREFIX_TO_DOMAIN:
                domain = PREFIX_TO_DOMAIN[prefix]
                domain_features[domain].append(col)
    
    for domain, cols in domain_features.items():
        if len(cols) < 2:
            continue
        
        # Get numeric values
        col_data = df[cols].fillna(0).values.astype(np.float64)
        
        # Mean
        domain_mean = np.nanmean(col_data, axis=1)
        result[f'{domain}_agg_mean'] = domain_mean
        new_cols.append(f'{domain}_agg_mean')
        
        # Std
        domain_std = np.nanstd(col_data, axis=1, ddof=0)
        result[f'{domain}_agg_std'] = domain_std
        new_cols.append(f'{domain}_agg_std')
        
        # Min
        domain_min = np.nanmin(col_data, axis=1)
        result[f'{domain}_agg_min'] = domain_min
        new_cols.append(f'{domain}_agg_min')
        
        # Max
        domain_max = np.nanmax(col_data, axis=1)
        result[f'{domain}_agg_max'] = domain_max
        new_cols.append(f'{domain}_agg_max')
        
        # Range
        domain_range = domain_max - domain_min
        result[f'{domain}_agg_range'] = domain_range
        new_cols.append(f'{domain}_agg_range')
        
        # Coefficient of variation (std/mean, with protection)
        abs_mean = np.abs(domain_mean)
        cv = domain_std / np.maximum(abs_mean, 1e-8)
        result[f'{domain}_agg_cv'] = cv
        new_cols.append(f'{domain}_agg_cv')
        
        # Entropy (Shannon-like): bin features and compute distribution entropy
        # Use percentile-based bins
        try:
            # Flatten all values, compute histogram
            flat = col_data.flatten()
            # 5-bin histogram per row
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
    
    log.info(f"  Added {len(new_cols)} domain aggregate features")
    return result, new_cols


def compute_cross_domain_ratios(df, base_cols):
    """Compute cross-domain ratio features.
    
    Key domain pairs that might have meaningful ratios:
    - Heart rate / activity (efficiency)
    - GPS / activity (mobility vs exertion)
    - Light / screen use (screen time in light vs dark)
    - WiFi / BLE (network environment)
    """
    result = df.copy()
    new_cols = []
    
    # Representative features per domain (use mean features)
    representatives = {}
    for col in base_cols:
        if col.endswith('_mean'):
            parts = col.split('_')
            prefix = parts[0]
            if prefix not in representatives:
                representatives[prefix] = col
    
    # Define meaningful ratios
    ratios = [
        # HR / Activity efficiency
        ('wHr', 'mActivity', 'hr_activity_ratio'),
        # GPS / Activity mobility
        ('mGps', 'mActivity', 'gps_activity_ratio'),
        # Light / Screen (screen usage in light environment)
        ('mLight', 'mScreenStatus', 'light_screen_ratio'),
        # WiFi / BLE (network context)
        ('mWifi', 'mBle', 'wifi_ble_ratio'),
        # Acceleration / GPS (movement type)
        ('mActivity', 'mGps', 'activity_gps_ratio'),
        # Heart rate / WiFi (resting in different environments)
        ('wHr', 'mWifi', 'hr_wifi_ratio'),
        # Pedo / GPS (step vs distance correlation)
        ('wPedo', 'mGps', 'pedo_gps_ratio'),
        # Light / AC charging (charging behavior in light)
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
    
    log.info(f"  Added {len(new_cols)} cross-domain ratio features")
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


def generate_test_zscore(train_df, test_df, zscore_base_cols):
    """Generate z-score features for test set using training data statistics."""
    log.info("Generating z-score features...")
    for col in zscore_base_cols:
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
    log.info("V341 — Domain-Level Aggregated Features + Cross-Domain Ratios")
    log.info("Hypothesis: domain aggregates capture signal missed by top-K selection")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Get base columns
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"Base features: {len(base_cols)}")
    
    # Step 1: Compute domain aggregates
    log.info("Computing domain aggregates...")
    train_df, domain_agg_cols = compute_domain_aggregates(train_df, base_cols)
    test_df, _ = compute_domain_aggregates(test_df, base_cols)
    
    # Step 2: Compute cross-domain ratios
    log.info("Computing cross-domain ratios...")
    train_df, ratio_cols = compute_cross_domain_ratios(train_df, base_cols)
    test_df, _ = compute_cross_domain_ratios(test_df, base_cols)
    
    # Step 3: Generate z-scores for ALL features (base + domain agg + ratios)
    all_base_cols = base_cols + domain_agg_cols + ratio_cols
    test_df, train_df = generate_test_zscore(train_df, test_df, all_base_cols)
    
    # Get final feature columns
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"\nFinal features:")
    log.info(f"  Train: {len(train_feat_cols)} (base={len(base_cols)}, domain_agg={len(domain_agg_cols)}, ratio={len(ratio_cols)}, zscore={len([c for c in train_feat_cols if '_zscore' in c])})")
    log.info(f"  Test:  {len(test_feat_cols)}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    all_oofs = {}
    all_test_preds = {}
    all_student_oofs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        # Ensure test consistency
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test: {missing}")
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        
        cfg = CFGS[cfg_name]
        
        # Level 0: N_SEEDS LGBM models
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            all_test_preds.setdefault(t, np.zeros((n_test, N_SEEDS)))[:, si] = seed_test
        
        # Level 1: LR meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof = meta.predict_proba(stacked)[:, 1]
        oof_ll = log_loss(y, np.clip(train_oof, 0.001, 0.999))
        all_oofs[t] = oof_ll
        
        # Student OOF (mean of seeds)
        student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        # Meta prediction on test
        test_stacked = np.column_stack([all_test_preds[t][:, i] for i in range(N_SEEDS)])
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"    {t}: student_OOF={student_ll:.5f}, meta_OOF={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
    
    # Compute results
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    log.info(f"\n{'='*70}")
    log.info(f"V341 RESULTS (Domain Aggregates + Cross-Domain Ratios)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Student OOF':>12} {'Meta OOF':>12} {'Gap':>8}")
    log.info(f"{'-'*40}")
    for t in TARGETS:
        log.info(f"{t:<6} {all_student_oofs[t]:>12.5f} {all_oofs[t]:>12.5f} {all_oofs[t]-all_student_oofs[t]:>+8.5f}")
    log.info(f"{'='*40}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF:    {v308_avg:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v341_domain_agg_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V341',
        'name': 'Domain-Level Aggregated Features + Cross-Domain Ratios',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_base_features': len(base_cols),
        'n_domain_agg_features': len(domain_agg_cols),
        'n_ratio_features': len(ratio_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v341_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
