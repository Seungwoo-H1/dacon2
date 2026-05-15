"""
V34 — V8 Base + V10 Personalization + LOSO CV + Calibration Tuning

Strategy:
1. V8의 per-target hyperparameters를 베이스로 사용
2. V10의 personalization(z-score) 추가
3. LOSO CV (GroupKFold 10-fold)로 엄격한 검증
4. V10의 mean-matching calibration 사용
5. 20 seeds ensemble
6. Importance ranking 기반 top-10 feature selection
7. Calibration exploration: avg_shift vs mean-matching vs 둘 다

Reference: V8(0.6537 LB), V10 personalization + mean-matching cal
"""

import sys, re, json, warnings, logging, importlib.util
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

# 20 seeds ensemble
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(SEEDS)
N_SPLITS = 10  # LOSO

# V8 per-target hyperparameters (from meta_20260501_032657.json)
V8_PER_TARGET_CONFIGS = {
    "Q1": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cst": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q2": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cst": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q3": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cst": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
    "S1": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S2": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S3": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S4": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cst": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
}

# Feature leakage
LEAKAGE_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

# Constant and collinear features to drop
CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std','mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std','mWifi_wifi_bssid_count_max',
]


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def mean_match(pred, target_mean):
    """V10 방식 mean-matching calibration."""
    shift = target_mean - pred.mean()
    calibrated = np.clip(pred + shift, 0.0001, 0.9999)
    return calibrated


# ── Load features via 02_feature_engineering pipeline ──
def load_features():
    """Load features using 02_feature_engineering pipeline (same as V33/V10)."""
    spec_ld = importlib.util.spec_from_file_location(
        "01_load_data", Path('src/01_load_data.py'))
    ld = importlib.util.module_from_spec(spec_ld)
    spec_ld.loader.exec_module(ld)

    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv",
                         parse_dates=["sleep_date", "lifelog_date"])

    data_dir = DATA_RAW / "ch2025_data_items"
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet",
        "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }

    parquet_dfs = {}
    train_dates = set(labels['lifelog_date'].dt.date.astype(str).unique())
    for name, fname in parquet_names.items():
        df = pd.read_parquet(data_dir / fname)
        df = ld.build_merge_key(df)
        df = df[df['date'].astype(str).isin(train_dates)]
        parquet_dfs[name] = df

    spec_fe = importlib.util.spec_from_file_location(
        "02_feature_engineering", Path('src/02_feature_engineering.py'))
    fe = importlib.util.module_from_spec(spec_fe)
    spec_fe.loader.exec_module(fe)

    feat = fe.create_day_features(parquet_dfs, labels)
    log.info(f"  Features loaded: {feat.shape}")
    return feat


# ── Feature utilities ──
def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def remove_leakage_cols(feature_cols, target):
    """Remove leakage features based on target type."""
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_Q]
    return feature_cols


def add_personalization(df, feature_cols):
    """V10 방식: per-subject z-score personalization."""
    df = df.copy()
    personal_cols = []

    for col in feature_cols:
        col_filled = df[col].fillna(0)
        subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        df = df.merge(subj_stats, on='subject_id', how='left')

        mask_std_zero = (df[f'{col}_subj_std'] == 0)
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')

    return df, personal_cols


def rank_features(feat, feature_cols, target, seed=42):
    """Importance ranking via quick LightGBM (gain)."""
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos

    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
    }
    sn = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=100)
    imp = model.feature_importance(importance_type="gain")
    return sorted(zip(feature_cols, imp), key=lambda x: -x[1])


# ── LOSO CV with V8 per-target configs ──
def lgb_loso_cv(feat, cols, target, seeds, target_cfg):
    """LOSO CV (GroupKFold 10-fold) with V8 per-target hyperparameters."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))

    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]

    cfg = {
        'num_leaves': target_cfg['nl'], 'max_depth': target_cfg['md'],
        'learning_rate': target_cfg['lr'], 'n_estimators': target_cfg['ne'],
        'subsample': target_cfg['ss'], 'colsample_bytree': target_cfg['cst'],
        'reg_alpha': target_cfg['ra'], 'reg_lambda': target_cfg['rl'],
        'min_child_samples': target_cfg['mc'], 'verbose': -1,
        'force_row_wise': True, 'n_jobs': -1,
    }

    for si, s in enumerate(seeds):
        seed_cfg = {**cfg, 'random_state': s, 'scale_pos_weight': spw}

        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})

            m = lgb.train(seed_cfg, ds, num_boost_round=cfg['n_estimators'],
                          valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_full[va, si] = m.predict(X_va)

    oof_avg = oof_full.mean(axis=1)

    # Per-fold stats
    fold_losses = []
    for tr, va in gkf.split(feat, y, feat['subject_id']):
        fold_oof = np.clip(oof_full[va].mean(axis=1), 0.0001, 0.9999)
        fold_losses.append(log_loss(y[va], fold_oof, labels=[0, 1]))
    fold_avg = np.mean(fold_losses)
    fold_std = np.std(fold_losses)

    return oof_avg, oof_full, log_loss(y, np.clip(oof_avg, 0.0001, 0.9999), labels=[0, 1]), fold_avg, fold_std


# ── Calibration exploration ──
def explore_calibration(oof_preds, target, train_rate):
    """
    Compare three calibration strategies:
    1. mean-matching only (V10)
    2. avg_shift only (V8)
    3. mean-matching + avg_shift (combo)

    Returns dict with cal_oof_loss and cal_method for each.
    """
    results = {}

    # Method 1: mean-matching
    cal_mm = mean_match(oof_preds, train_rate)
    results['mean_match'] = {'cal': cal_mm, 'loss': log_loss(target, cal_mm, labels=[0, 1])}

    # Method 2: avg_shift (V8 style) — apply global shift
    shift = train_rate - oof_preds.mean()
    cal_shift = np.clip(oof_preds + shift, 0.0001, 0.9999)
    results['avg_shift'] = {'cal': cal_shift, 'loss': log_loss(target, cal_shift, labels=[0, 1])}

    # Method 3: mean-matching + avg_shift correction
    cal_combo = mean_match(oof_preds, train_rate)
    combo_shift = train_rate - cal_combo.mean()
    cal_combo = np.clip(cal_combo + combo_shift, 0.0001, 0.9999)
    results['combo'] = {'cal': cal_combo, 'loss': log_loss(target, cal_combo, labels=[0, 1])}

    return results


# ── Main ──
def main():
    log.info("=" * 70)
    log.info("V34 — V8 Base + V10 Personalization + LOSO CV + Calibration Tuning")
    log.info(f"  Seeds: {N_SEEDS}, Splits: {N_SPLITS} (LOSO)")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n=== Step 1: Loading features ===")
    feat = load_features()

    # ── 2. Prepare feature list ──
    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    # Remove constant/collinear
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    log.info(f"  Base features (from pipeline, no const/collinear): {len(raw_base)}")

    # ── 3. Add personalization ──
    log.info("\n=== Step 2: Adding personalization (V10 z-score) ===")
    feat, personal_cols = add_personalization(feat, raw_base)
    log.info(f"  Added {len(personal_cols)} z-score columns")

    # ── 4. Per-target: feature selection + LOSO CV + calibration exploration ──
    log.info(f"\n=== Step 3: Per-target tuning (LOSO {N_SPLITS}-fold, {N_SEEDS} seeds) ===")

    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Target rates: {train_rate}")

    all_oof = {}
    all_cal = {}
    all_best_cal = {}
    all_best_info = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={train_rate[target]:.3f}) ---")

        # Get V8 config for this target
        v8_cfg = V8_PER_TARGET_CONFIGS[target]
        log.info(f"  V8 config: nl={v8_cfg['nl']} md={v8_cfg['md']} lr={v8_cfg['lr']} ne={v8_cfg['ne']} "
                 f"ss={v8_cfg['ss']} cst={v8_cfg['cst']} ra={v8_cfg['ra']} rl={v8_cfg['rl']} mc={v8_cfg['mc']}")

        # Remove leakage
        leak_free = remove_leakage_cols(get_feature_cols(feat), target)
        log.info(f"  Leakage-free features: {len(leak_free)}")

        # Importance ranking → top-10
        ranked = rank_features(feat, leak_free, target, seed=42)
        sel_cols = [c for c, _ in ranked[:10]]
        log.info(f"  Top-10 features: {[c.split('_')[-1] for c in sel_cols][:5]}...")

        # LOSO CV
        oof_avg, oof_full, cv_loss, fold_avg, fold_std = lgb_loso_cv(
            feat, sel_cols, target, SEEDS, v8_cfg
        )
        log.info(f"  LOSO CV loss: {cv_loss:.4f} (fold avg={fold_avg:.4f}, std={fold_std:.4f})")
        log.info(f"  OOF mean: {oof_avg.mean():.4f}, train rate: {train_rate[target]:.4f}")

        # Calibration exploration
        cal_results = explore_calibration(oof_avg, feat[target].values, train_rate[target])
        log.info(f"  Calibration options:")
        for method, info in cal_results.items():
            log.info(f"    {method}: loss={info['loss']:.4f}, mean={info['cal'].mean():.4f}, "
                     f"shift={info['cal'].mean()-train_rate[target]:+.4f}")

        # Pick best calibration method
        best_method = min(cal_results, key=lambda m: cal_results[m]['loss'])
        best_cal = cal_results[best_method]['cal']
        best_cal_loss = cal_results[best_method]['loss']

        all_oof[target] = oof_avg
        all_cal[target] = best_cal
        all_best_cal[target] = best_method
        all_best_info[target] = {
            'v8_config': v8_cfg,
            'n_features': len(sel_cols),
            'features': sel_cols,
            'cv_loss': cv_loss,
            'fold_avg': fold_avg,
            'fold_std': fold_std,
            'cal_method': best_method,
            'cal_loss': best_cal_loss,
            'cal_results': {m: {'loss': cal_results[m]['loss']} for m in cal_results},
        }
        log.info(f"  ✅ Best cal: {best_method} (cal_loss={best_cal_loss:.4f})")

    # ── 5. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V34 OOF SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'CV Loss':<10} {'FoldAvg':<10} {'FoldStd':<10} {'Cal Method':<12} {'Cal Loss':<10}")
    for t in TARGET_COLS:
        bi = all_best_info[t]
        log.info(f"{t:<6} {bi['cv_loss']:<10.4f} {bi['fold_avg']:<10.4f} {bi['fold_std']:<10.4f} "
                 f"{bi['cal_method']:<12} {bi['cal_loss']:<10.4f}")

    avg_oof = np.mean([log_loss(feat[t], np.clip(all_oof[t], 0.0001, 0.9999), labels=[0, 1]) for t in TARGET_COLS])
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"\n  Avg OOF CV loss:  {avg_oof:.4f}")
    log.info(f"  Avg Cal OOF loss: {avg_cal:.4f}")
    log.info(f"  V8 Submission:    0.6537")
    log.info(f"  Δ vs V8:          {0.6537 - avg_cal:+.4f}")

    # ── 6. Generate submission ──
    log.info(f"\n=== Step 4: Training final models + generating submission ===")

    # Load feature engineering
    spec = importlib.util.spec_from_file_location(
        "02_feature_engineering", Path('src/02_feature_engineering.py'))
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)

    spec2 = importlib.util.spec_from_file_location(
        "01_load_data", Path('src/01_load_data.py'))
    ld_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ld_mod)

    # Load test data
    parquet_dfs = {}
    data_dir = PROJECT_ROOT / "data_raw" / "ch2025_data_items"
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet",
        "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }

    sample = pd.read_csv(DATA_RAW / "ch2026_submission_sample.csv")
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date

    test_dates = set(
        sample["sleep_date"].astype(str).tolist()
        + sample["lifelog_date"].astype(str).tolist()
    )

    for name, fname in parquet_names.items():
        df = pd.read_parquet(data_dir / fname)
        df = ld_mod.build_merge_key(df)
        df = df[df["date"].astype(str).isin(test_dates)]
        parquet_dfs[name] = df

    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    log.info(f"  Test features: {test_features.shape}")

    # Add personalization to test
    test_feat_cols = get_feature_cols(test_features)
    test_features, _ = add_personalization(test_features, test_feat_cols)

    predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    # For each target: train on full data → predict test
    for target in TARGET_COLS:
        log.info(f"\n  [{target}] Training final models ({N_SEEDS} seeds)...")
        selected_cols = all_best_info[target]['features']
        v8_cfg = all_best_info[target]['v8_config']
        cal_method = all_best_info[target]['cal_method']

        y_all = feat[target].values
        X_all = feat[selected_cols].fillna(0).values
        test_X = test_features[selected_cols].fillna(0).values
        sn = [sanitize(c) for c in selected_cols]

        spw = ((y_all == 0).sum()) / max((y_all == 1).sum(), 1)

        cfg = {
            'num_leaves': v8_cfg['nl'], 'max_depth': v8_cfg['md'],
            'learning_rate': v8_cfg['lr'], 'n_estimators': v8_cfg['ne'],
            'subsample': v8_cfg['ss'], 'colsample_bytree': v8_cfg['cst'],
            'reg_alpha': v8_cfg['ra'], 'reg_lambda': v8_cfg['rl'],
            'min_child_samples': v8_cfg['mc'],
            'scale_pos_weight': spw, 'verbose': -1,
            'force_row_wise': True, 'n_jobs': -1,
        }

        all_preds = np.zeros(len(test_X))
        for si, seed in enumerate(SEEDS):
            seed_cfg = {**cfg, 'random_state': seed}
            ds = lgb.Dataset(X_all, label=y_all, feature_name=sn, params={'verbose': '-1'})
            model = lgb.train(seed_cfg, ds, num_boost_round=cfg['n_estimators'])
            all_preds += model.predict(test_X)
            if (si + 1) % 5 == 0:
                log.info(f"    [{target}] seed {si+1}/{N_SEEDS} done")

        all_preds /= N_SEEDS

        # Apply best calibration method
        if cal_method == 'mean_match':
            cal_preds = mean_match(all_preds, train_rate[target])
        elif cal_method == 'avg_shift':
            shift = train_rate[target] - all_preds.mean()
            cal_preds = np.clip(all_preds + shift, 0.0001, 0.9999)
        else:  # combo
            cal_preds = mean_match(all_preds, train_rate[target])
            combo_shift = train_rate[target] - cal_preds.mean()
            cal_preds = np.clip(cal_preds + combo_shift, 0.0001, 0.9999)

        predictions[target] = cal_preds
        log.info(f"    {target}: mean={cal_preds.mean():.4f}, min={cal_preds.min():.4f}, "
                 f"max={cal_preds.max():.4f}, rate={train_rate[target]:.3f}, shift={cal_preds.mean()-train_rate[target]:+.4f}")

    # ── Save submission ──
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v34_{timestamp}.csv'
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")

    # ── Save metadata ──
    meta = {
        'version': 'v34',
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'avg_shift_v8': 0.004803341642460823,
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'calibration_methods': {t: all_best_info[t]['cal_method'] for t in TARGET_COLS},
        'per_target': {},
    }

    for target in TARGET_COLS:
        bi = all_best_info[target]
        meta['per_target'][target] = {
            'config': {k: v for k, v in bi['v8_config'].items()},
            'n_features': bi['n_features'],
            'features': bi['features'],
            'cv_loss': float(bi['cv_loss']),
            'cal_method': bi['cal_method'],
            'cal_oof_loss': float(bi['cal_loss']),
            'cal_results': {m: float(v['loss']) for m, v in bi['cal_results'].items()},
            'oof_mean': float(all_oof[target].mean()),
            'cal_mean': float(all_cal[target].mean()),
            'train_rate': float(train_rate[target]),
            'pred_min': float(predictions[target].min()),
            'pred_max': float(predictions[target].max()),
        }
        log.info(f"\n  {target}: cal_method={bi['cal_method']}, "
                 f"CV={bi['cv_loss']:.4f}, Cal={bi['cal_loss']:.4f}, "
                 f"pred_mean={predictions[target].mean():.4f}")

    meta_path = sub_path.parent / f'meta_v34_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Metadata saved: {meta_path}")

    # ── Final summary ──
    log.info(f"\n{'='*70}")
    log.info("V34 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'CV Loss':<10} {'Cal OOF':<10} {'Cal Method':<12} {'Test Mean':<12} {'Train Rate':<10} {'Shift':<10}")
    for target in TARGET_COLS:
        bi = all_best_info[target]
        oof_l = log_loss(feat[target], np.clip(all_oof[target], 0.0001, 0.9999), labels=[0, 1])
        cal_l = log_loss(feat[target], all_cal[target], labels=[0, 1])
        tm = predictions[target].mean()
        tr = train_rate[target]
        log.info(f"{target:<6} {oof_l:<10.4f} {cal_l:<10.4f} {bi['cal_method']:<12} "
                 f"{tm:<12.4f} {tr:<10.3f} {tm-tr:+.4f}")

    avg_oof = np.mean([log_loss(feat[t], np.clip(all_oof[t], 0.0001, 0.9999), labels=[0, 1]) for t in TARGET_COLS])
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"\n  Avg OOF CV:  {avg_oof:.4f}")
    log.info(f"  Avg Cal OOF: {avg_cal:.4f}")
    log.info(f"  V8 Submission: 0.6537")
    log.info(f"  Δ vs V8: {0.6537 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6537 else '❌ Not improved'})")


if __name__ == "__main__":
    main()
