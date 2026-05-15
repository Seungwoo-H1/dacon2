"""
V37_fix — Per-Target Hyper + V8 Configs + 20 Seeds (Fixed)

Uses features.parquet directly (like V10). Fixes:
1. Personalization added at runtime (not importing 02_feature_engineering)
2. Uses features_v11_personalized.parquet for speed
3. Same per-target tuning logic as V10

Expected: ~30 mins, no SIGKILL
"""

import sys, re, json, warnings, logging, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
N_SPLITS = 5
N_TOP_FEATURES = 20
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

LEAKAGE_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}

LEAKAGE_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count'}

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def get_feature_cols(feat):
    return [c for c in feat.columns
            if c not in META_COLS | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def remove_leakage_features(feature_cols, target):
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_Q]
    return feature_cols

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        df = df.merge(subj_stats, on='subject_id', how='left')
        mask_std_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(mask_std_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std'])
        personal_cols.append(f'{col}_zscore')
    return df, personal_cols

def rank_features(feat, feature_cols, target, random_seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feature_cols].fillna(0).values.astype(np.float64)
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 50,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': random_seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
    }
    sn = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    del ds, model, importances, X, y
    gc.collect()
    return ranked

def lgb_cv_predict(feat, selected_cols, target, seeds, spw):
    y = feat[target].values.astype(np.float64)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    sanitized = [sanitize(c) for c in selected_cols]
    for seed_i, seed in enumerate(seeds):
        cfg = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
            'random_state': seed, 'scale_pos_weight': spw,
        }
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            X_tr = feat.iloc[tr_idx][selected_cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va_idx][selected_cols].fillna(0).values.astype(np.float64)
            train_set = lgb.Dataset(X_tr, label=y[tr_idx], feature_name=sanitized, params={'verbose': '-1'})
            val_set = lgb.Dataset(X_va, label=y[va_idx], feature_name=sanitized, reference=train_set, params={'verbose': '-1'})
            m = lgb.train(cfg, train_set, num_boost_round=500, valid_sets=[val_set],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_full[va_idx, seed_i] = m.predict(X_va)
            del train_set, val_set, m, X_tr, X_va
        gc.collect()
    return oof_full

def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    calibrated = pred + shift
    calibrated = np.clip(calibrated, 0.0001, 0.9999)
    return calibrated

def main():
    log.info("=" * 70)
    log.info("V37_fix — Per-Target Hyper + V8 Configs + 20 Seeds (Fixed)")
    log.info("=" * 70)

    # Load features
    feat_path = DATA_PROCESSED / "features.parquet"
    log.info(f"Loading features from {feat_path}")
    feat = pd.read_parquet(feat_path)
    log.info(f"Features loaded: {feat.shape}")

    # Add personalization
    feature_cols = get_feature_cols(feat)
    log.info(f"Base features: {len(feature_cols)}")
    feat, personal_cols = add_personalization(feat, feature_cols)
    log.info(f"After personalization: {feat.shape}, {len(personal_cols)} z-score cols")

    all_feat = get_feature_cols(feat)
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")

    # Per-target tuning
    all_cal = {}
    log.info("\n=== Per-target tuning ===")

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={train_rate[target]:.3f}) ---")
        y = feat[target].values.astype(np.float64)
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        leak_free = remove_leakage_features(all_feat, target)
        log.info(f"  Leak-free features: {len(leak_free)}")

        # Rank features
        ranked = rank_features(feat, leak_free, target)
        top20 = [r[0] for r in ranked[:20]]
        log.info(f"  Top-20 features: {top20[:5]}...")

        # 20-seed ensemble with top-20
        log.info("  Training 20-seed ensemble...")
        oof_full = lgb_cv_predict(feat, top20, target, SEEDS, spw)
        oof_avg = np.clip(oof_full.mean(axis=1), 0.0001, 0.9999)
        del oof_full
        gc.collect()

        # Calibration
        cal = simple_mean_match(oof_avg, train_rate[target])
        cal_loss = log_loss(y, cal, labels=[0, 1])

        log.info(f"  Final: Cal={cal_loss:.4f}")
        all_cal[target] = cal

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V37_fix SUMMARY")
    for t in TARGET_COLS:
        cal_l = log_loss(feat[t], all_cal[t], labels=[0, 1])
        log.info(f"  {t}: Cal={cal_l:.4f}")

    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"\n  Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Delta: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

    # Generate submission
    log.info("\n=== Generating submission ===")
    # For submission, train final models on all data
    all_selected = {}
    for target in TARGET_COLS:
        leak_free = remove_leakage_features(all_feat, target)
        ranked = rank_features(feat, leak_free, target)
        all_selected[target] = [r[0] for r in ranked[:20]]
        log.info(f"  {target}: selected {len(all_selected[target])} features")

    # Load test data
    log.info("Loading test data...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)

    spec2 = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    ld_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ld_mod)

    parquet_dfs = {}
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet", "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }
    sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date
    test_dates = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())

    for name, fname in parquet_names.items():
        df = pd.read_parquet(f'data_raw/ch2025_data_items/{fname}')
        df = ld_mod.build_merge_key(df)
        parquet_dfs[name] = df[df['date'].astype(str).isin(test_dates)]
    labels = pd.read_csv('data_raw/ch2026_metrics_train.csv', parse_dates=['sleep_date', 'lifelog_date'])

    log.info("Running feature engineering on test data...")
    test_feat = feat_eng.create_day_features(parquet_dfs, labels)
    log.info(f"Test features: {test_feat.shape}")

    # Personalize test data
    test_feat = test_feat.merge(
        feat[['subject_id'] + personal_cols], on=['subject_id'] + personal_cols, how='left'
    )
    # Fill any NaN with 0
    for col in personal_cols:
        if col in test_feat.columns:
            test_feat[col] = test_feat[col].fillna(0)

    # Predict
    submission = sample.copy()
    for target in TARGET_COLS:
        log.info(f"  Predicting {target}...")
        sel = all_selected[target]
        # Ensure all cols exist
        available = [c for c in sel if c in test_feat.columns]
        log.info(f"    Using {len(available)}/{len(sel)} features")

        spw = train_rate[target]
        cfg = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
            'scale_pos_weight': spw,
        }
        sn = [sanitize(c) for c in available]
        preds = []
        for seed in SEEDS:
            cfg_s = {**cfg, 'random_state': seed}
            X_test = test_feat[available].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_test, feature_name=sn, params={'verbose': '-1'})
            m = lgb.train(cfg_s, ds, num_boost_round=500)
            preds.append(m.predict(X_test))
            del m, ds
        avg_pred = np.clip(np.mean(preds, axis=0), 0.0001, 0.9999)
        cal_pred = simple_mean_match(avg_pred, train_rate[target])
        submission[target] = cal_pred
        del avg_pred, cal_pred, preds

    # Save
    out_path = MODEL_DIR / "submission_v37_fix.csv"
    submission.to_csv(out_path, index=False)
    log.info(f"Submission saved to {out_path}")

    # Also save to submit_dir
    sub_dir = SUBMIT_DIR / "v37_fix"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "submission.csv").write_csv(submission)
    log.info(f"Submission saved to {sub_dir / 'submission.csv'}")

if __name__ == "__main__":
    main()
