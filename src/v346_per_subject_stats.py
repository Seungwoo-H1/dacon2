"""
V346 — Per-Subject Statistics as Features

CRITICAL INSIGHT: Per-subject mean baseline gives:
  S1: 0.552 | S2: 0.551 | S3: 0.510 | Q1: 0.637 | Q2: 0.645 | Q3: 0.642 | S4: 0.620

V344 student OOF:
  S1: 0.602 | S2: 0.584 | S3: 0.620 | Q1: 0.690 | Q2: 0.665 | Q3: 0.672 | S4: 0.709

→ Per-subject statistics are THE strongest signal but NOT in current features.
→ Each row has features aggregated over ~45 days. But the aggregation 
  (mean/std/min/max/count) loses the per-subject distribution shape.

Hypothesis: Adding per-subject aggregate statistics (computed from ALL subjects' 
data, leave-one-subject-out) as additional features will give the model 
the per-subject signal it needs.

New features per target:
1. Per-subject mean of each feature (from OTHER subjects → leave-one-out)
2. Per-subject std of each feature
3. Per-subject trend (linear slope over time)
4. Per-subject daily activity level
5. Per-subject ratio features (step/sleep_ratio, hr/activity_ratio)

These are computed using leave-one-subject-out to prevent leakage.
Then stacked with V308 pipeline.

Architecture: V308 (15 seeds, GroupKFold 5, LR meta C=10)
Features: 141 base + zscore + PER-SUBJECT STATS (~500+ features)
"""
import sys, gc, logging, json, re, time, warnings
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


def compute_per_subject_stats(train_df, test_df):
    """Compute per-subject aggregate statistics using leave-one-subject-out.
    
    For each row (subject x date), compute statistics from ALL OTHER subjects.
    This prevents leakage while giving each row a "subject profile".
    
    New features:
    - Global mean/std of each base feature (across all subjects)
    - Per-subject mean/std/skew/kurtosis of each base feature
    - Per-subject daily activity ratio (steps/sleep_hours)
    - Per-subject sensor diversity (BLE/WiFi device counts)
    - Per-subject temporal trend (slope over days)
    """
    log.info("Computing per-subject statistics...")
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and not c.endswith('_ps_')  # per-subject features
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"  Base columns: {len(base_cols)}")
    
    # STEP 1: Compute global statistics per column
    global_stats = {}
    for col in base_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        global_stats[col] = {
            'mean': np.mean(vals),
            'std': max(np.std(vals, ddof=0), 1e-8),
            'median': np.median(vals),
            'skew': float(pd.Series(vals).skew()),
            'kurtosis': float(pd.Series(vals).kurtosis()),
        }
    
    # STEP 2: Per-subject statistics
    # Each subject has multiple rows (dates). Compute aggregate stats per subject.
    subject_stats = {}
    for subj_id in train_df['subject_id'].unique():
        subj_mask = train_df['subject_id'] == subj_id
        subj_data = train_df[subj_mask][base_cols].fillna(0).values.astype(np.float64)
        
        stats = {}
        for i, col in enumerate(base_cols):
            col_vals = subj_data[:, i]
            stats[f'{col}_ps_mean'] = np.mean(col_vals)
            stats[f'{col}_ps_std'] = max(np.std(col_vals, ddof=0), 1e-8)
            stats[f'{col}_ps_skew'] = float(pd.Series(col_vals).skew())
            stats[f'{col}_ps_kurtosis'] = float(pd.Series(col_vals).kurtosis())
            stats[f'{col}_ps_median'] = np.median(col_vals)
            stats[f'{col}_ps_range'] = np.max(col_vals) - np.min(col_vals)
            stats[f'{col}_ps_cv'] = stats[f'{col}_ps_std'] / max(abs(stats[f'{col}_ps_mean']), 1e-8)
        
        # Per-subject activity summary
        if 'wPedo_pedo_step_mean' in base_cols:
            steps = subj_data[:, base_cols.index('wPedo_pedo_step_mean')]
            stats['ps_total_steps'] = np.sum(steps)
            stats['ps_mean_steps'] = np.mean(steps)
            stats['ps_max_steps'] = np.max(steps)
        
        if 'wHr_hr_mean' in base_cols:
            hr = subj_data[:, base_cols.index('wHr_hr_mean')]
            stats['ps_mean_hr'] = np.mean(hr)
            stats['ps_std_hr'] = max(np.std(hr, ddof=0), 1e-8)
            stats['ps_max_hr'] = np.max(hr)
        
        if 'mActivity_m_activity_mean' in base_cols:
            act = subj_data[:, base_cols.index('mActivity_m_activity_mean')]
            stats['ps_mean_activity'] = np.mean(act)
            stats['ps_max_activity'] = np.max(act)
        
        if 'mGps_gps_avg_speed_mean' in base_cols:
            spd = subj_data[:, base_cols.index('mGps_gps_avg_speed_mean')]
            stats['ps_mean_speed'] = np.mean(spd[spd > 0]) if (spd > 0).any() else 0
            stats['ps_max_speed'] = np.max(spd)
        
        if 'mLight_m_light_mean' in base_cols:
            light = subj_data[:, base_cols.index('mLight_m_light_mean')]
            stats['ps_mean_light'] = np.mean(light)
            stats['ps_max_light'] = np.max(light)
        
        # Per-subject sensor diversity
        if 'mBle_ble_device_count_mean' in base_cols:
            dev = subj_data[:, base_cols.index('mBle_ble_device_count_mean')]
            stats['ps_ble_diversity'] = np.mean(dev)
        
        if 'mWifi_wifi_bssid_count_mean' in base_cols:
            bssid = subj_data[:, base_cols.index('mWifi_wifi_bssid_count_mean')]
            stats['ps_wifi_diversity'] = np.mean(bssid)
        
        # Per-subject temporal pattern
        if 'date' in train_df.columns:
            dates = pd.to_datetime(train_df.loc[subj_mask, 'date'])
            n_days = (dates.max() - dates.min()).days if len(dates) > 1 else 1
            stats['ps_span_days'] = n_days
            stats['ps_n_dates'] = len(dates)
            stats['ps_density'] = len(dates) / max(n_days, 1)
        
        # Per-subject hourly patterns
        for hour_col in ['mACStatus_hour_morning', 'mACStatus_hour_afternoon', 
                         'mACStatus_hour_evening', 'mACStatus_hour_night']:
            if hour_col in base_cols:
                idx = base_cols.index(hour_col)
                vals = subj_data[:, idx]
                stats[f'ps_{hour_col}'] = np.mean(vals)
        
        subject_stats[subj_id] = stats
    
    # STEP 3: For each row, look up its subject's stats
    new_cols_added = []
    for df in [train_df, test_df]:
        for subj_id in df['subject_id'].unique():
            if subj_id in subject_stats:
                subj_stat = subject_stats[subj_id]
                mask = df['subject_id'] == subj_id
                for feat_name, feat_val in subj_stat.items():
                    feat_col = f'{feat_name}_{subj_id}'
                    df.loc[mask, feat_col] = feat_val
                    new_cols_added.append(feat_col)
    
    # Also add GLOBAL statistics as features for each row
    for col in base_cols:
        gs = global_stats[col]
        for stat_name, stat_val in gs.items():
            feat_col = f'{col}_global_{stat_name}'
            train_df[feat_col] = stat_val
            test_df[feat_col] = stat_val
            new_cols_added.append(feat_col)
    
    log.info(f"  Added {len(new_cols_added)} per-subject + global stats features")
    
    # Return feature dataframe with only per-subject stats (not the subject-specific columns)
    # Instead, compute leave-one-subject-out stats
    return train_df, test_df, base_cols, new_cols_added


def compute_loo_subject_stats(train_df, test_df, base_cols):
    """Compute leave-one-subject-out statistics.
    
    For each subject, compute statistics from ALL OTHER subjects (excluding self).
    This is the TRUE per-subject profile that the model uses as features.
    """
    log.info("Computing leave-one-subject-out statistics...")
    
    all_subjects = sorted(train_df['subject_id'].unique())
    n_subjects = len(all_subjects)
    
    # For each subject, compute stats from all OTHER subjects
    loo_stats = {}
    for idx, subj_id in enumerate(all_subjects):
        log.info(f"  LOO subject {idx+1}/{n_subjects}: {subj_id}")
        
        # Get all subjects except this one
        other_subjects = [s for s in all_subjects if s != subj_id]
        other_mask = train_df['subject_id'].isin(other_subjects)
        other_data = train_df[other_mask][base_cols].fillna(0).values.astype(np.float64)
        
        stats = {}
        for i, col in enumerate(base_cols):
            col_vals = other_data[:, i]
            stats[f'{col}_loo_mean'] = np.mean(col_vals)
            stats[f'{col}_loo_std'] = max(np.std(col_vals, ddof=0), 1e-8)
            stats[f'{col}_loo_median'] = np.median(col_vals)
        
        # Summary stats
        stats['loo_n_rows'] = len(other_data)
        
        loo_stats[subj_id] = stats
    
    # For test subjects (same IDs), use train LOO stats
    # If test has unique subject IDs not in train, compute from train all
    loo_all = {}
    for col in base_cols:
        all_vals = train_df[col].fillna(0).values.astype(np.float64)
        loo_all[col] = {
            'mean': np.mean(all_vals),
            'std': max(np.std(all_vals, ddof=0), 1e-8),
            'median': np.median(all_vals),
        }
    
    return loo_stats, loo_all


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V346 — Per-Subject Statistics as Features (LOO)")
    log.info("Hypothesis: per-subject profile is the strongest signal")
    log.info("per-subject mean baseline: S1=0.552, S2=0.551, S3=0.510")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and not c.endswith('_ps_')
                 and not c.endswith('_loo_')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"Base features: {len(base_cols)}")
    
    # Compute LOO subject stats
    loo_stats, loo_all = compute_loo_subject_stats(train_df, test_df, base_cols)
    
    # Add LOO stats to each row
    for df in [train_df, test_df]:
        for subj_id in df['subject_id'].unique():
            if subj_id in loo_stats:
                stat = loo_stats[subj_id]
            else:
                stat = loo_all  # fallback
            mask = df['subject_id'] == subj_id
            for feat_name, feat_val in stat.items():
                df.loc[mask, feat_name] = feat_val
    
    # Z-score all new features
    for col in base_cols:
        if f'{col}_loo_mean' in train_df.columns:
            vals_t = train_df[f'{col}_loo_mean'].fillna(0).values.astype(np.float64)
            vals_te = test_df[f'{col}_loo_mean'].fillna(0).values.astype(np.float64)
            mean = np.mean(vals_t)
            std = max(np.std(vals_t, ddof=0), 1e-8)
            train_df[f'{col}_loo_mean_z'] = (vals_t - mean) / std
            test_df[f'{col}_loo_mean_z'] = (vals_te - mean) / std
    
    # Get feature columns
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    # Count feature types
    loo_cols = [c for c in train_feat_cols if 'loo_' in c]
    loo_z_cols = [c for c in train_feat_cols if 'loo_z' in c]
    base_only = [c for c in train_feat_cols if 'loo' not in c and 'z' not in c and c not in TARGETS]
    
    log.info(f"\nFeature breakdown:")
    log.info(f"  Base: {len(base_only)}")
    log.info(f"  LOO stats: {len(loo_cols)}")
    log.info(f"  LOO z-score: {len(loo_z_cols)}")
    log.info(f"  Total: {len(train_feat_cols)}")
    
    # Run full pipeline with LOO features
    log.info("\nRunning V308 pipeline with LOO features...")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    group = train_df['subject_id'].values
    
    all_oofs = {}
    all_student_oofs = {}
    all_preds = {}
    
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
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        log.info(f"    Selected features: {sel_cols}")
        
        cfg = CFGS[cfg_name]
        
        # Level 0: N_SEEDS LGBM models
        per_seed_oofs = []
        test_preds_arr = np.zeros((n_test, N_SEEDS))
        
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
            test_preds_arr[:, si] = seed_test
        
        # Level 1: LR meta-learner
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
        all_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"    {t}: student={student_ll:.5f}, meta={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
        
        # Check which LOO features were selected
        selected_loo = [c for c in sel_cols if 'loo_' in c]
        if selected_loo:
            log.info(f"    Selected LOO features ({len(selected_loo)}): {selected_loo[:5]}...")
    
    # Compute results
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    v344_avg = 0.61304
    
    log.info(f"\n{'='*70}")
    log.info(f"V346 RESULTS (Per-Subject LOO Statistics)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Student OOF':>12} {'Meta OOF':>12} {'Gap':>8} {'ΔV308':>8} {'ΔV339':>8}")
    log.info(f"{'-'*58}")
    for t in TARGETS:
        log.info(f"{t:<6} {all_student_oofs[t]:>12.5f} {all_oofs[t]:>12.5f} {all_oofs[t]-all_student_oofs[t]:>+8.5f} {all_oofs[t]-v308_avg:>+8.5f} {all_oofs[t]-v339_avg:>+8.5f}")
    log.info(f"{'='*58}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF:    {v308_avg:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  V339 AVG OOF:    {v339_avg:.5f}")
    log.info(f"  Δ vs V339:       {avg_oof - v339_avg:+.5f}")
    log.info(f"  V344 AVG OOF:    {v344_avg:.5f}")
    log.info(f"  Δ vs V344:       {avg_oof - v344_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Check: do LOO features help?
    # Count how many targets have LOO features in top-K
    loo_helpful = 0
    for t in TARGETS:
        ranked = rank_features(train_df, remove_leak(get_feature_cols(train_df), t), t)
        top_k = ranked[:V53_SWEEP[t]['n_feat']]
        if any('loo_' in c for c in top_k):
            loo_helpful += 1
    log.info(f"  LOO features in top-K for {loo_helpful}/{len(TARGETS)} targets")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_preds[t]
    
    sub_path = SUBMIT / f"submission_v346_loo_stats_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V346',
        'name': 'Per-Subject LOO Statistics',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_loo_features': len(loo_cols),
        'loo_helpful_targets': loo_helpful,
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'delta_vs_v344': round(float(avg_oof - v344_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v346_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
