"""
V349 — Per-Target Domain Aggregates

V348: all domains added to all targets → Q1/Q3/S2/S4 대폭 악화, S1/S3만 좋음
V349: per-target optimal domain aggregates

Key insight from V348:
- S1: Charging, Screen, HR, Light, Activity aggregates help (student 0.589 → baseline 0.552에 근접)
- S3: GPS, Charging, HR, BLE aggregates help
- Q1/Q3: ALL domain aggregates hurt (student 0.77-0.79)

Hypothesis: 
1. S1/S2/S3/S4 → add domain aggregates (top 5 by feature importance)
2. Q1/Q2/Q3 → NO domain aggregates (remove them entirely)

This way we get the S1/S3 student OOF improvement without Q1/Q3 degradation.

Architecture: V339 baseline
Features: same as V339 for Q targets, V339 + domain-aggs for S targets
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


def get_feature_cols(df):
    exclude = set(META_COLS) | set(TARGETS)
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
    X = feat_df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
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


def add_domain_aggs(df):
    """Add domain aggregate features (mean, std per domain)."""
    for domain, domain_cols in DOMAINS.items():
        valid_cols = [c for c in domain_cols if c in df.columns]
        if len(valid_cols) < 2:
            continue
        domain_vals = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
        df[f'{domain}_agg_mean'] = np.mean(domain_vals, axis=1)
        df[f'{domain}_agg_std'] = np.std(domain_vals, axis=1, ddof=0)


def add_zscore_features(train_df, test_df, base_cols):
    """Add z-score features."""
    for col in base_cols:
        vals_t = train_df[col].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
        vals_te = test_df[col].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
        mean = np.mean(vals_t)
        std = max(np.std(vals_t, ddof=0), 1e-8)
        train_work[f'{col}_zscore'] = (vals_t - mean) / std
        test_work[f'{col}_zscore'] = (vals_te - mean) / std


def run_target_pipeline(train_df, test_df, target, cfg, n_feat, add_domain_aggs_flag=False,
                        group=None):
    """Run single-target pipeline."""
    n_train = len(train_df)
    n_test = len(test_df)
    if group is None:
        group = train_df['subject_id'].values
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Prepare features
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Get base cols (exclude meta, target, and agg features if not adding)
    exclude_agg = set()
    if not add_domain_aggs_flag:
        exclude_agg = {c for c in train_work.columns if '_agg_' in c}
    
    base_cols = [c for c in get_feature_cols(train_work) if c not in exclude_agg]
    base_cols = remove_leak(base_cols, target)
    
    # Rank features
    ranked = rank_features(train_work, base_cols, target)
    sel_cols = ranked[:n_feat]
    sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_work)]
    
    if set(sel_cols) != set(sel_cols_test):
        sel_cols_test = [c for c in sel_cols if c in test_work.columns]
        if not sel_cols_test:
            sel_cols_test = sel_cols
    
    y = train_work[target].values.astype(np.float64)
    
    per_seed_oofs = []
    test_preds_arr = np.zeros((n_test, N_SEEDS))
    
    for si in range(N_SEEDS):
        seed = SEED + si * 7
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_work, y, group)):
            X_tr = train_work[sel_cols].iloc[tr_idx].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
            X_va = train_work[sel_cols].iloc[va_idx].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_work[sel_cols_test].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64))
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        test_preds_arr[:, si] = seed_test
    
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    
    oof_ll = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
    student_ll = log_loss(y, np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999))
    test_pred = meta.predict_proba(np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)]))[:, 1]
    
    return oof_ll, student_ll, np.clip(test_pred, 0.01, 0.99)


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V349 — Per-Target Domain Aggregates")
    log.info("Q targets: NO domain aggs (V348 showed they hurt Q)")
    log.info("S targets: domain aggs (V348 showed they help S)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Add domain aggs to S-target only datasets
    # Strategy: Q targets use base features only, S targets use base + domain aggs
    
    # First, run Q targets WITHOUT domain aggs
    log.info(f"\n{'='*70}")
    log.info("Q Targets (no domain aggs)")
    log.info(f"{'='*70}")
    
    q_results = {}
    for t in ['Q1', 'Q2', 'Q3']:
        oof_ll, student_ll, pred = run_target_pipeline(
            train_df, test_df, t,
            CFGS[V53_SWEEP[t]['cfg']], V53_SWEEP[t]['n_feat'],
            add_domain_aggs_flag=False
        )
        q_results[t] = {'oof': oof_ll, 'student': student_ll, 'pred': pred}
        log.info(f"  {t}: student={student_ll:.5f}, meta={oof_ll:.5f} (ΔV308: {oof_ll - 0.62235:+.5f})")
    
    # Then, run S targets WITH domain aggs
    log.info(f"\n{'='*70}")
    log.info("S Targets (with domain aggs)")
    log.info(f"{'='*70}")
    
    # Add domain aggs to S-target datasets
    train_with_agg = train_df.copy()
    test_with_agg = test_df.copy()
    add_domain_aggs(train_with_agg)
    add_domain_aggs(test_with_agg)
    
    # Z-score for domain aggs
    agg_cols = [c for c in train_with_agg.columns if '_agg_' in c]
    for col in agg_cols:
        vals_t = train_with_agg[col].fillna(0).values.astype(np.float64)
        vals_te = test_with_agg[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals_t)
        std = max(np.std(vals_t, ddof=0), 1e-8)
        train_with_agg[col] = (vals_t - mean) / std
        test_with_agg[col] = (vals_te - mean) / std
    
    s_results = {}
    for t in ['S1', 'S2', 'S3', 'S4']:
        oof_ll, student_ll, pred = run_target_pipeline(
            train_with_agg, test_with_agg, t,
            CFGS[V53_SWEEP[t]['cfg']], V53_SWEEP[t]['n_feat'],
            add_domain_aggs_flag=True
        )
        s_results[t] = {'oof': oof_ll, 'student': student_ll, 'pred': pred}
        log.info(f"  {t}: student={student_ll:.5f}, meta={oof_ll:.5f} (ΔV308: {oof_ll - 0.62235:+.5f})")
    
    # Check: what does S1/S2 without domain aggs give? (V339 baseline)
    s_no_agg = {}
    for t in ['S1', 'S2']:
        oof_ll, student_ll, _ = run_target_pipeline(
            train_df, test_df, t,
            CFGS[V53_SWEEP[t]['cfg']], V53_SWEEP[t]['n_feat'],
            add_domain_aggs_flag=False
        )
        s_no_agg[t] = {'oof': oof_ll, 'student': student_ll}
        log.info(f"  {t} (no agg): student={student_ll:.5f}, meta={oof_ll:.5f}")
    
    # Overall results
    all_results = {**q_results, **s_results}
    avg_oof = np.mean([r['oof'] for r in all_results.values()])
    avg_student_oof = np.mean([r['student'] for r in all_results.values()])
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    v344_avg = 0.61304
    
    log.info(f"\n{'='*70}")
    log.info(f"V349 RESULTS (Per-Target Domain Aggregates)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Strategy':>12} {'Student':>10} {'Meta':>10} {'ΔV308':>8} {'ΔV339':>8}")
    log.info(f"{'-'*60}")
    for t in TARGETS:
        r = all_results[t]
        strategy = "no agg" if t.startswith('Q') else "agg"
        if t in s_no_agg:
            no_agg_str = f" (no-agg: {s_no_agg[t]['oof']:.5f})"
        else:
            no_agg_str = ""
        log.info(f"{t:<6} {strategy:>12} {r['student']:>10.5f} {r['oof']:>10.5f} {r['oof']-v308_avg:>+8.5f} {r['oof']-v339_avg:>+8.5f}{no_agg_str}")
    log.info(f"{'-'*60}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  Δ vs V339:       {avg_oof - v339_avg:+.5f}")
    log.info(f"  Δ vs V344:       {avg_oof - v344_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_results[t]['pred']
    
    sub_path = SUBMIT / f"submission_v349_per_target_agg_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V349',
        'name': 'Per-Target Domain Aggregates',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_seeds': N_SEEDS,
        'q_strategy': 'no_domain_aggs',
        's_strategy': 'with_domain_aggs',
        'per_target_oof': {t: round(float(all_results[t]['oof']), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_results[t]['student']), 5) for t in TARGETS},
        's_no_agg_oof': {t: round(float(s_no_agg[t]['oof']), 5) for t in s_no_agg},
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v349_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
