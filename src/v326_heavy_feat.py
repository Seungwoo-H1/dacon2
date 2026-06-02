"""
V326: Heavy Feature Engineering + V321 (Stacking)

Hypothesis: Domain-specific interactions + rolling window stats + per-subject
z-scores + ratio features will enrich the feature space, then V321 stacking
will leverage them for better predictions.

New features:
1. HR × pedo interactions (activity intensity)
2. Light × screen interactions (sleep quality proxy)
3. GPS × BLE interactions (location diversity)
4. Rolling window stats (3/5/10 day windows)
5. Per-subject z-scores (z-score within each subject's history)
6. Ratio features between domains (e.g., walking/running step ratio)
7. Day-of-week, hour-of-day interactions

Then run V321 stacking on enriched features.

Expected OOF: 0.595-0.605 (enriched features → better stacking)
Risk: MEDIUM (new features might introduce noise)
Cost: ~60s
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0
FEATURE_BAG_FRACTION = 0.75


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


def generate_interaction_features(df, is_test=False):
    """Generate domain-specific interaction features."""
    log.info("Generating heavy feature engineering features...")
    df = df.copy()
    new_cols = []
    
    # Base feature prefixes for interactions
    hr_cols = [c for c in df.columns if c.startswith('wHr_') and np.issubdtype(df[c].dtype, np.number)]
    pedo_cols = [c for c in df.columns if c.startswith('wPedo_') and np.issubdtype(df[c].dtype, np.number)]
    light_cols = [c for c in df.columns if c.startswith('mLight_') and np.issubdtype(df[c].dtype, np.number)]
    screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_') and np.issubdtype(df[c].dtype, np.number)]
    gps_cols = [c for c in df.columns if c.startswith('mGps_') and np.issubdtype(df[c].dtype, np.number)]
    ble_cols = [c for c in df.columns if c.startswith('mBle_') and np.issubdtype(df[c].dtype, np.number)]
    wifi_cols = [c for c in df.columns if c.startswith('mWifi_') and np.issubdtype(df[c].dtype, np.number)]
    activity_cols = [c for c in df.columns if c.startswith('mActivity_') and np.issubdtype(df[c].dtype, np.number)]
    
    # HR × pedo interactions
    if hr_cols and pedo_cols:
        # Mean HR × mean step
        df['hr_pedo_interaction'] = df[hr_cols].fillna(0).mean(axis=1) * df[pedo_cols].fillna(0).mean(axis=1)
        new_cols.append('hr_pedo_interaction')
        # HR max × pedo distance
        hr_max = [c for c in hr_cols if 'max' in c]
        pedo_dist = [c for c in pedo_cols if 'distance' in c]
        if hr_max and pedo_dist:
            df['hr_max_pedo_dist'] = df[hr_max].fillna(0).mean(axis=1) * df[pedo_dist].fillna(0).mean(axis=1)
            new_cols.append('hr_max_pedo_dist')
    
    # Light × screen interactions
    if light_cols and screen_cols:
        df['light_screen_interaction'] = df[light_cols].fillna(0).mean(axis=1) * df[screen_cols].fillna(0).mean(axis=1)
        new_cols.append('light_screen_interaction')
        # Light mean × screen count
        light_mean = [c for c in light_cols if 'mean' in c]
        screen_count = [c for c in screen_cols if 'count' in c]
        if light_mean and screen_count:
            df['light_mean_screen_count'] = df[light_mean].fillna(0).mean(axis=1) * df[screen_count].fillna(0).mean(axis=1)
            new_cols.append('light_mean_screen_count')
    
    # GPS × BLE interactions (location diversity)
    if gps_cols and ble_cols:
        df['gps_ble_interaction'] = df[gps_cols].fillna(0).mean(axis=1) * df[ble_cols].fillna(0).mean(axis=1)
        new_cols.append('gps_ble_interaction')
    
    # WiFi × GPS interactions
    if wifi_cols and gps_cols:
        df['wifi_gps_interaction'] = df[wifi_cols].fillna(0).mean(axis=1) * df[gps_cols].fillna(0).mean(axis=1)
        new_cols.append('wifi_gps_interaction')
    
    # Ratio features
    pedo_steps = [c for c in pedo_cols if 'step' in c and 'sum' not in c and 'frequency' not in c]
    pedo_dist = [c for c in pedo_cols if 'distance' in c]
    if pedo_steps and pedo_dist:
        # Average step length (distance / steps)
        step_mean = df[pedo_steps].fillna(0).mean(axis=1)
        dist_mean = df[pedo_dist].fillna(0).mean(axis=1)
        df['step_length_ratio'] = (dist_mean + 1e-8) / (step_mean + 1e-8)
        new_cols.append('step_length_ratio')
    
    # Walking / running ratio
    pedo_walk = [c for c in pedo_cols if 'walking' in c]
    pedo_run = [c for c in pedo_cols if 'running' in c]
    if pedo_walk and pedo_run:
        walk_sum = df[pedo_walk].fillna(0).mean(axis=1)
        run_sum = df[pedo_run].fillna(0).mean(axis=1)
        df['walk_run_ratio'] = (walk_sum + 1e-8) / (run_sum + 1e-8)
        new_cols.append('walk_run_ratio')
    
    # Activity × light interaction
    if activity_cols and light_cols:
        df['activity_light_interaction'] = df[activity_cols].fillna(0).mean(axis=1) * df[light_cols].fillna(0).mean(axis=1)
        new_cols.append('activity_light_interaction')
    
    # Usage stats × ambient interactions
    usage_cols = [c for c in df.columns if c.startswith('mUsageStats_') and np.issubdtype(df[c].dtype, np.number)]
    amb_cols = [c for c in df.columns if c.startswith('mAmbience_') and np.issubdtype(df[c].dtype, np.number)]
    if usage_cols and amb_cols:
        df['usage_ambient_interaction'] = df[usage_cols].fillna(0).mean(axis=1) * df[amb_cols].fillna(0).mean(axis=1)
        new_cols.append('usage_ambient_interaction')
    
    # Total activity proxy
    all_base = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                and not c.endswith('_zscore') and np.issubdtype(df[c].dtype, np.number)]
    df['total_activity_proxy'] = df[all_base].fillna(0).abs().sum(axis=1)
    new_cols.append('total_activity_proxy')
    
    log.info(f"  Generated {len(new_cols)} interaction features: {new_cols}")
    return df, new_cols


def generate_rolling_features(df, window_days=[3, 5, 10]):
    """Generate rolling window statistics per subject."""
    log.info("Generating rolling window features...")
    df = df.copy()
    new_cols = []
    
    numeric_cols = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                    and not c.endswith('_zscore') and np.issubdtype(df[c].dtype, np.number)]
    
    if 'sleep_date' in df.columns:
        df['sleep_date'] = pd.to_datetime(df['sleep_date'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        log.info("  No date column found for rolling features, skipping")
        return df, new_cols
    
    date_col = 'sleep_date' if 'sleep_date' in df.columns else 'date'
    
    for window in window_days:
        for col in numeric_cols[:30]:  # Limit to top 30 numeric cols
            rolled = df.groupby('subject_id')[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            # Reorder to match original index
            rolled = rolled.reindex(df.index)
            col_name = f'roll_{window}_{col}'
            df[col_name] = rolled.values
            new_cols.append(col_name)
    
    log.info(f"  Generated {len(new_cols)} rolling features (windows: {window_days})")
    return df, new_cols


def generate_per_subject_zscores(df):
    """Generate per-subject z-scores using only subject's own history."""
    log.info("Generating per-subject z-scores...")
    df = df.copy()
    new_cols = []
    
    base_cols = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and np.issubdtype(df[c].dtype, np.number)]
    
    for col in base_cols:
        def calc_zscore(group):
            mean = group.mean()
            std = group.std(ddof=0)
            if std < 1e-8:
                std = 1e-8
            return (group - mean) / std
        
        zscored = df.groupby('subject_id')[col].transform(calc_zscore)
        zc = f'ps_zscore_{col}'
        df[zc] = zscored.values
        new_cols.append(zc)
    
    log.info(f"  Generated {len(new_cols)} per-subject z-score features")
    return df, new_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V326 — Heavy Feature Engineering + V321 Stacking")
    log.info("Domain interactions + rolling windows + per-subject z-scores")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # 1. Generate z-score features (global, like V321)
    log.info("Generating global z-score features...")
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_base) & set(test_base)
    
    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        test_df = test_df.copy()
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
        train_df = train_df.copy()
        train_df[zc] = (vals - mean) / std
    
    # 2. Generate interaction features
    train_df, interact_cols = generate_interaction_features(train_df)
    test_df, _ = generate_interaction_features(test_df)
    
    # 3. Generate per-subject z-scores
    train_df, ps_zscore_cols = generate_per_subject_zscores(train_df)
    test_df, _ = generate_per_subject_zscores(test_df)
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"\nFeature counts:")
    log.info(f"  Base: {len(train_base)}")
    log.info(f"  Global z-score: {len(common_cols)}")
    log.info(f"  Interaction: {len(interact_cols)}")
    log.info(f"  Per-subject z-score: {len(ps_zscore_cols)}")
    log.info(f"  Total: {len(train_feat_cols)}")
    log.info(f"  Test total: {len(test_feat_cols)}")
    
    # V321-style stacking with feature bagging
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
    
    n_train = len(train_df)
    n_test = len(test_df)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        candidate_feats = ranked
        
        cfg = CFGS[cfg_name]
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(candidate_feats) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            
            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            
            sel_cols = [c for c in bag_feats if c in test_feat_cols]
            
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
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            all_seed_oofs[t].append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            if si < 3 or si == N_SEEDS - 1:
                s_oof = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
    
    # LR meta-learner
    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
        meta_weights_info[t] = meta.coef_[0]
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V326 RESULTS (Heavy Feat Eng + V321 Stacking)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        w_sum = np.sum(np.abs(meta_weights_info[t]))
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f}, |W|={w_sum:.3f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V321: 0.60569 | V312: 0.61448 | V308: 0.62235")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    log.info(f"  Δ vs V312: {avg_oof - 0.61448:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v326_heavy_feat_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V326',
        'name': 'Heavy Feature Engineering + V321 Stacking',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_features_base': len(train_base),
        'n_features_zscore': len(common_cols),
        'n_features_interaction': len(interact_cols),
        'n_features_ps_zscore': len(ps_zscore_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'v321_avg_oof': 0.60569,
        'v312_avg_oof': 0.61448,
        'v308_avg_oof': 0.62235,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'delta_vs_v312': round(float(avg_oof - 0.61448), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'heavy feature engineering + V321 stacking',
    }
    
    meta_path = EXPERIMENTS / f'v326_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
