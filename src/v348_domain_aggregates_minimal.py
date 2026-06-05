"""
V348 — Domain-Aggregate Features (Minimal, No V346's 565 Features)

V346 added 565 LOO features → 706 total → feature selection diluted signal
→ S1/S2 improved but Q1/Q2/Q3/S4 worsened.

V348: Add ONLY domain-level aggregates (6 domains × 2 stats = 12 features)
These are simpler than V346's 565 features.

Domains: BLE, GPS, WiFi, HR, Activity, Sleep
Stats per domain: mean, std of the domain's base features

Architecture: V339 (15 seeds, GroupKFold 5, LR meta C=10)
Features: 141 base + 141 zscore + 12 domain-aggregate = 294 features
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

# Domain definitions
DOMAINS = {
    'BLE': ['mBle_ble_count_mean', 'mBle_ble_count_std', 'mBle_ble_device_count_mean',
            'mBle_ble_avg_rssi_mean', 'mBle_ble_max_rssi_mean', 'mBle_ble_min_rssi_mean'],
    'GPS': ['mGps_gps_count_mean', 'mGps_gps_avg_speed_mean', 'mGps_gps_max_speed_mean',
            'mGps_gps_alt_range_mean', 'mGps_gps_has_speed_mean'],
    'WiFi': ['mWifi_wifi_count_mean', 'mWifi_wifi_bssid_count_mean', 'mWifi_wifi_avg_rssi_mean',
             'mWifi_wifi_max_rssi_mean', 'mWifi_wifi_strong_ratio_mean'],
    'HR': ['wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_count'],
    'Activity': ['mActivity_m_activity_mean', 'mActivity_m_activity_std',
                 'mActivity_m_activity_min', 'mActivity_m_activity_max', 'mActivity_m_activity_count'],
    'Ambience': ['mAmbience_ambience_speech_sum', 'mAmbience_ambience_music_sum',
                 'mAmbience_ambience_vehicle_sum', 'mAmbience_ambience_max_cat'],
    'Light': ['mLight_m_light_mean', 'mLight_m_light_std', 'mLight_m_light_min',
              'mLight_m_light_max', 'mLight_m_light_count'],
    'Pedo': ['wPedo_pedo_step_mean', 'wPedo_pedo_step_sum', 'wPedo_pedo_distance_mean',
             'wPedo_pedo_burned_calories_mean'],
    'Usage': ['mUsageStats_usage_app_count_mean', 'mUsageStats_usage_total_time_mean',
              'mUsageStats_usage_major_ratio_mean', 'mUsageStats_usage_game_ratio_mean'],
    'Screen': ['mScreenStatus_m_screen_use_mean', 'mScreenStatus_m_screen_use_std',
               'mScreenStatus_m_screen_use_count'],
    'Charging': ['mACStatus_m_charging_mean', 'mACStatus_m_charging_std'],
    'Hour': ['mACStatus_hour_morning', 'mACStatus_hour_afternoon',
             'mACStatus_hour_evening', 'mACStatus_hour_night'],
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


def get_feature_cols(df, exclude_targets=True):
    exclude = set(META_COLS)
    if exclude_targets:
        exclude = exclude | set(TARGETS)
    return [c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


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


def compute_domain_aggregates(train_df, test_df):
    """Compute per-row domain-level aggregates.
    
    For each row, compute the mean/std of features within each domain.
    This is NOT leave-one-subject-out (unlike V346).
    These are per-row domain aggregates based on the same row's features.
    
    Actually wait — these features are already per-row (subject_id, date).
    So the mean/std within a domain for a single row doesn't make sense
    (each column has one value per row).
    
    What we need instead: for each domain, compute a "domain score"
    based on the domain's features. E.g., for BLE domain:
    - mean of (ble_count_mean, ble_device_count_mean, ble_avg_rssi_mean)
    - std of those values
    
    This creates 2 new features per domain (mean of domain features, std of domain features).
    """
    log.info("Computing domain-level aggregates...")
    
    domain_aggs = {}
    for domain, domain_cols in DOMAINS.items():
        # Get valid columns
        valid_cols = [c for c in domain_cols if c in train_df.columns]
        if len(valid_cols) < 2:
            continue
        
        # Compute domain mean and std for each row
        domain_vals = train_df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
        domain_mean = np.mean(domain_vals, axis=1)
        domain_std = np.std(domain_vals, axis=1, ddof=0)
        
        domain_aggs[f'{domain}_agg_mean'] = domain_mean
        domain_aggs[f'{domain}_agg_std'] = domain_std
        
        log.info(f"  {domain}: {len(valid_cols)} cols → agg_mean, agg_std")
    
    # Apply to both train and test
    for df in [train_df, test_df]:
        for agg_name, agg_vals in domain_aggs.items():
            if isinstance(agg_vals, np.ndarray) and len(agg_vals) == len(train_df):
                # For test, need to compute separately
                pass
        
        # Compute for this df
        for domain, domain_cols in DOMAINS.items():
            valid_cols = [c for c in domain_cols if c in df.columns]
            if len(valid_cols) < 2:
                continue
            domain_vals = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
            df[f'{domain}_agg_mean'] = np.mean(domain_vals, axis=1)
            df[f'{domain}_agg_std'] = np.std(domain_vals, axis=1, ddof=0)
    
    n_new = len(DOMAINS) * 2
    log.info(f"  Total new features: {n_new}")
    return domain_aggs


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V348 — Domain-Aggregate Features (12 features)")
    log.info("11 domains × 2 stats (mean, std) = 22 features")
    log.info("Architecture: V339 pipeline")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Compute domain aggregates
    domain_aggs = compute_domain_aggregates(train_df, test_df)
    
    # Z-score the new features
    for col in domain_aggs.keys():
        if f'{col}_agg_mean' not in train_df.columns:
            # Compute now
            domain_cols = [c for c in DOMAINS.keys() if col.startswith(c)]
            if domain_cols:
                domain = domain_cols[0]
                valid = [c for c in DOMAINS[domain] if c in train_df.columns]
                vals_t = train_df[valid].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
                vals_te = test_df[valid].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
                m_t = np.mean(vals_t, axis=1)
                m_te = np.mean(vals_te, axis=1)
                s_t = np.std(vals_t, axis=1, ddof=0)
                s_te = np.std(vals_te, axis=1, ddof=0)
                mean_m = np.mean(m_t)
                std_m = max(np.std(m_t, ddof=0), 1e-8)
                train_df[f'{domain}_agg_mean'] = (m_t - mean_m) / std_m
                test_df[f'{domain}_agg_mean'] = (m_te - mean_m) / std_m
                mean_s = np.mean(s_t)
                std_s = max(np.std(s_t, ddof=0), 1e-8)
                train_df[f'{domain}_agg_std'] = (s_t - mean_s) / std_s
                test_df[f'{domain}_agg_std'] = (s_te - mean_s) / std_s
    
    log.info(f"\nFeature count: train={len(get_feature_cols(train_df))}, test={len(get_feature_cols(test_df))}")
    
    # Run V339 pipeline
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
        feat_cols_clean = remove_leak(get_feature_cols(train_df), t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_df)]
        
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test, fixing...")
            sel_cols_test = [c for c in sel_cols if c in test_df.columns]
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        
        # Check if domain aggregates were selected
        agg_selected = [c for c in sel_cols if '_agg_' in c]
        if agg_selected:
            log.info(f"    Domain aggregates selected ({len(agg_selected)}): {agg_selected}")
        
        cfg = CFGS[cfg_name]
        
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
        
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        oof_ll = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
        all_oofs[t] = oof_ll
        
        student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        test_stacked = np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)])
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        all_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"    {t}: student={student_ll:.5f}, meta={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    v344_avg = 0.61304
    
    log.info(f"\n{'='*70}")
    log.info(f"V348 RESULTS (Domain-Aggregate Features)")
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
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_preds[t]
    
    sub_path = SUBMIT / f"submission_v348_domain_agg_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V348',
        'name': 'Domain-Aggregate Features',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_features_total': len(get_feature_cols(train_df)),
        'n_domain_aggs': 22,
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v348_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
