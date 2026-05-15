"""
07_v10_robust.py — V10: Robust pipeline avoiding v9's calibration trap

Key improvements over v9:
1. NO isotonic calibration (v9's fatal mistake — shift=-0.47 to -0.55 collapsed all predictions to ~0)
2. Simple mean-matching + clip only (like v8 which scored 0.65374)
3. Feature leakage fix: wLight/wHr/wPedo nighttime data excluded from S1-S4
4. Personalization: per-subject z-score normalization
5. 20-seed ensemble (LightGBM only, no CatBoost/XGB overhead)
6. Feature selection: top-20 per target via importance ranking
7. Per-target hyperparameter tuning via 5-fold GroupKFold

Strategy:
- If we can't beat 0.65374 on CV, we won't beat it on test.
- The key is: robust feature engineering + regularization + proper calibration.
"""

import sys
import re
import json
import warnings
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────
sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

# ── Hyperparameters ──────────────────────────────────────
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 5
N_TOP_FEATURES = 20


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_feature_cols(feat):
    return [c for c in feat.columns
            if c not in META_COLS | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


# ── Feature leakage fix ──────────────────────────────────
# v9 used nighttime wrist data (wLight/wHr/wPedo) for S1-S4 targets.
# Audit found: nighttime wLight mean=17.3 (dark), daytime wLight mean=263.9.
# Using full-day wrist aggregation leaks sleep-time data into sleep targets.
# FIX: For S targets, only use daytime wrist data or exclude wrist data entirely.

LEAKAGE_FEATURES_S = {
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

LEAKAGE_FEATURES_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
}


def remove_leakage_features(feature_cols, target):
    """Remove features known to leak into S1-S4 targets."""
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_Q]
    return feature_cols


# ── Personalization: z-score per subject ─────────────────
def add_personalization(df, feature_cols):
    """Add per-subject z-score features."""
    df = df.copy()
    personal_cols = []
    
    for col in feature_cols:
        if df[col].isnull().any():
            # Fill nulls with 0 for z-score calculation
            col_filled = df[col].fillna(0)
        else:
            col_filled = df[col]
        
        # Per-subject mean and std
        subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        
        merged = df.merge(subj_stats, on='subject_id', how='left')
        
        # Z-score (handle zero std)
        mask_std_zero = merged[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        
        merged[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null,
            0.0,
            (merged[col] - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        
        personal_cols.append(f'{col}_zscore')
        df = merged
    
    return df, personal_cols


# ── Feature ranking ──────────────────────────────────────
def rank_features(feat, feature_cols, target, random_seed=42):
    """Quick LightGBM scan to rank features by gain importance."""
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
        'scale_pos_weight': spw, 'random_state': random_seed,
        'min_child_samples': 10,
        'force_row_wise': True, 'n_jobs': -1,
    }
    sanitized = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=100)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    return ranked


# ── Hyperparameter configs ───────────────────────────────
# Conservative configs to avoid overfitting on 450 samples
LGB_CONSERVATIVE = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 15, 'max_depth': 4,
    'learning_rate': 0.03, 'n_estimators': 500,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
    'min_child_samples': 10,
    'force_row_wise': True, 'n_jobs': -1,
    'verbose': -1,
}


def lgb_cv_predict(feat, selected_cols, target, seeds, spw):
    """
    GroupKFold: N_SPLITS folds × N_SEEDS seeds → OOF predictions.
    Returns averaged OOF predictions and per-seed/fold losses.
    """
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}
    
    sanitized = [sanitize(c) for c in selected_cols]
    
    for seed_i, seed in enumerate(seeds):
        cfg = {**LGB_CONSERVATIVE, 'random_state': seed}
        
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[train_idx][selected_cols].fillna(0).values
            X_va = feat.iloc[val_idx][selected_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]
            
            train_set = lgb.Dataset(
                X_tr, label=y_tr, feature_name=sanitized,
                params={'verbose': '-1'}
            )
            val_set = lgb.Dataset(
                X_va, label=y_va, feature_name=sanitized,
                reference=train_set, params={'verbose': '-1'}
            )
            
            params = {**cfg, 'scale_pos_weight': spw}
            model = lgb.train(
                params, train_set, num_boost_round=cfg['n_estimators'],
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            
            pred = model.predict(X_va)
            oof_full[val_idx, seed_i] = pred
            
            fold_loss = log_loss(y_va, pred, labels=[0, 1])
            all_fold_losses[fold].append(fold_loss)
    
    oof_avg = oof_full.mean(axis=1)
    fold_avg_losses = [np.mean(all_fold_losses[i]) for i in range(N_SPLITS)]
    cv_loss = log_loss(y, oof_avg, labels=[0, 1])
    cv_std = np.std(fold_avg_losses)
    
    return oof_avg, oof_full, cv_loss, cv_std, fold_avg_losses


# ── Simple calibration: mean-matching only ───────────────
def simple_mean_match(pred, target_rate):
    """
    Align prediction mean to target rate via shift + clip.
    This is what v8 did — and it worked (0.65374).
    NO isotonic regression (that's what killed v9).
    """
    shift = target_rate - pred.mean()
    calibrated = pred + shift
    calibrated = np.clip(calibrated, 0.0001, 0.9999)
    return calibrated


# ── Per-target hyperparameter tuning ─────────────────────
def tune_target(feat, feature_cols, target):
    """
    Try multiple hyperparameter configs per target via 5-fold CV.
    Select best config.
    """
    configs = [
        # Conservative configs
        {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
        {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C3', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
        {'name': 'C6', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    ]
    
    best_config = None
    best_cv = float('inf')
    best_oof = None
    best_selected_cols = None
    
    # Get top features via importance ranking
    ranked = rank_features(feat, feature_cols, target)
    
    for n_feat in [10, 20, 30]:
        selected_cols = [r[0] for r in ranked[:n_feat]]
        
        y = feat[target].values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos
        
        for cfg in configs:
            # Override LGB_CONSERVATIVE with this config
            test_cfg = {**LGB_CONSERVATIVE,
                       'num_leaves': cfg['nl'],
                       'max_depth': cfg['md'],
                       'learning_rate': cfg['lr'],
                       'n_estimators': cfg['ne'],
                       'subsample': cfg['ss'],
                       'colsample_bytree': cfg['cst'],
                       'reg_alpha': cfg['ra'],
                       'reg_lambda': cfg['rl'],
                       'min_child_samples': cfg['mc'],
            }
            
            oof_avg, oof_full, cv_loss, cv_std, fold_losses = lgb_cv_predict(
                feat, selected_cols, target, RANDOM_SEEDS, spw
            )
            
            # We want low CV loss + low std + prediction mean close to train rate
            train_rate = y.mean()
            pred_mean_shift = abs(oof_avg.mean() - train_rate)
            
            # Composite score: CV loss + 0.5 * std + penalty for shift
            score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift
            
            if score < best_cv:
                best_cv = score
                best_config = {**cfg, '_n_feats': n_feat}
                best_oof = oof_avg
                best_selected_cols = selected_cols
        
        if len(ranked) < 10:
            break
    
    return best_config, best_selected_cols, best_oof


# ── Main pipeline ────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("V10 Robust Pipeline — v9 calibration trap avoided")
    log.info("=" * 70)
    
    # ── 1. Load features ───────────────────────────────────
    feat_path = DATA_PROCESSED / "features.parquet"
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded features: {feat.shape}")
    
    feature_cols = get_feature_cols(feat)
    log.info(f"Raw feature cols: {len(feature_cols)}")
    
    # ── 2. Add personalization ─────────────────────────────
    log.info("Adding personalization (per-subject z-score)...")
    feat, personal_cols = add_personalization(feat, feature_cols)
    log.info(f"After personalization: {feat.shape}, {len(personal_cols)} z-score cols added")
    
    # Also merge test dates for later
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")
    
    # ── 3. Feature ranking + per-target tuning ─────────────
    log.info("\n=== Feature ranking + per-target tuning ===")
    
    all_best_configs = {}
    all_best_cols = {}
    all_oof = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} ---")
        log.info(f"  Train rate: {train_rate[target]:.3f}")
        
        # Remove leakage features for S targets
        leak_free_cols = remove_leakage_features(feature_cols + personal_cols, target)
        log.info(f"  Leakage-free feature cols: {len(leak_free_cols)}")
        
        # Rank features
        ranked = rank_features(feat, leak_free_cols, target)
        top10 = [r[0] for r in ranked[:10]]
        log.info(f"  Top 10 features: {top10[:5]}...")
        
        # Per-target tuning
        best_config, best_cols, best_oof = tune_target(feat, leak_free_cols, target)
        
        all_best_configs[target] = best_config
        all_best_cols[target] = best_cols
        all_oof[target] = best_oof
        
        y = feat[target].values
        oof_cv = log_loss(y, best_oof, labels=[0, 1])
        oof_shift = best_oof.mean() - y.mean()
        
        log.info(f"  Best config: {best_config}")
        log.info(f"  OOF CV loss: {oof_cv:.4f}")
        log.info(f"  OOF mean shift: {oof_shift:+.4f}")
        log.info(f"  Selected {len(best_cols)} features")
    
    # ── 4. Summary comparison ──────────────────────────────
    log.info("\n=== OOF CV Scores (pre-calibration) ===")
    log.info(f"{'Target':<6} {'OOF Loss':<12} {'OOF Mean':<12} {'Train Rate':<12} {'Shift':<12}")
    for target in TARGET_COLS:
        y = feat[target].values
        oof_loss = log_loss(y, all_oof[target], labels=[0, 1])
        oof_mean = all_oof[target].mean()
        tr = train_rate[target]
        log.info(f"{target:<6} {oof_loss:<12.4f} {oof_mean:<12.4f} {tr:<12.3f} {oof_mean-tr:+.4f}")
    
    avg_oof = np.mean([
        log_loss(feat[t], all_oof[t], labels=[0, 1]) for t in TARGET_COLS
    ])
    log.info(f"  Average OOF loss: {avg_oof:.4f}")
    
    # ── 5. Calibration: mean-matching only (NO isotonic) ───
    log.info("\n=== Calibration: mean-matching only ===")
    
    calibrated_oof = {}
    for target in TARGET_COLS:
        cal = simple_mean_match(all_oof[target], train_rate[target])
        calibrated_oof[target] = cal
        
        y = feat[target].values
        cal_loss = log_loss(y, cal, labels=[0, 1])
        shift = cal.mean() - y.mean()
        log.info(f"  {target}: cal_loss={cal_loss:.4f}, mean={cal.mean():.4f}, shift={shift:+.4f}")
    
    avg_cal = np.mean([
        log_loss(feat[t], calibrated_oof[t], labels=[0, 1]) for t in TARGET_COLS
    ])
    log.info(f"  Average calibrated OOF loss: {avg_cal:.4f}")
    
    # ── 6. Generate submission ─────────────────────────────
    log.info("\n=== Step 4: Training final models + generating submission ===")
    
    # Load feature engineering
    spec = importlib.util.spec_from_file_location(
        "02_feature_engineering", Path('src/02_feature_engineering.py')
    )
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)
    
    spec2 = importlib.util.spec_from_file_location(
        "01_load_data", Path('src/01_load_data.py')
    )
    ld_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ld_mod)
    
    # Load test data
    parquet_dfs = {}
    data_dir = Path('data_raw/ch2025_data_items')
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet",
        "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet",
        "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet",
        "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet",
        "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet",
        "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet",
        "wPedo": "ch2025_wPedo.parquet",
    }
    
    sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
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
    log.info(f"Test features: {test_features.shape}")
    
    # Add personalization to test features
    test_feat_cols = get_feature_cols(test_features)
    test_features, _ = add_personalization(test_features, test_feat_cols)
    
    predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    
    # For each target: train on full data → predict test
    for target in TARGET_COLS:
        log.info(f"\n  Training final models for {target}...")
        selected_cols = all_best_cols[target]
        
        y_all = feat[target].values
        X_all = feat[selected_cols].fillna(0).values
        test_X = test_features[selected_cols].fillna(0).values
        sanitized = [sanitize(c) for c in selected_cols]
        
        cfg = all_best_configs[target]
        n_pos = max((y_all == 1).sum(), 1)
        n_neg = (y_all == 0).sum()
        spw = n_neg / n_pos
        
        lgb_params = {**LGB_CONSERVATIVE,
                      'num_leaves': cfg['nl'],
                      'max_depth': cfg['md'],
                      'learning_rate': cfg['lr'],
                      'n_estimators': cfg['ne'],
                      'subsample': cfg['ss'],
                      'colsample_bytree': cfg['cst'],
                      'reg_alpha': cfg['ra'],
                      'reg_lambda': cfg['rl'],
                      'min_child_samples': cfg['mc'],
                      'scale_pos_weight': spw,
        }
        
        # 20-seed ensemble on full training data
        all_preds = np.zeros(len(test_X))
        
        for seed_i, seed in enumerate(RANDOM_SEEDS):
            seed_params = {**lgb_params, 'random_state': seed}
            
            ds_all = lgb.Dataset(
                X_all, label=y_all,
                feature_name=sanitized,
                params={'verbose': '-1'},
            )
            model = lgb.train(seed_params, ds_all, num_boost_round=cfg['ne'])
            all_preds += model.predict(test_X)
            
            if (seed_i + 1) % 5 == 0:
                log.info(f"    [{target}] seed {seed_i + 1}/{N_SEEDS} done")
        
        all_preds /= N_SEEDS
        
        # Simple mean-matching calibration (NO isotonic!)
        cal_preds = simple_mean_match(all_preds, train_rate[target])
        predictions[target] = cal_preds
        
        log.info(
            f"    {target}: mean={cal_preds.mean():.4f}, "
            f"min={cal_preds.min():.4f}, "
            f"max={cal_preds.max():.4f}, "
            f"train_rate={train_rate[target]:.3f}, "
            f"shift={cal_preds.mean()-train_rate[target]:+.4f}"
        )
    
    # ── Save submission ────────────────────────────────────
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f'submission_v10_{timestamp}.csv'
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")
    
    # Save metadata
    meta = {
        'version': 'v10',
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'n_top_features': 'tuned per target',
        'calibration': 'simple mean-matching + clip (NO isotonic)',
        'leakage_fix': 'wrist nighttime data removed from S targets',
        'personalization': 'per-subject z-score',
        'per_target': {},
    }
    for target in TARGET_COLS:
        y = feat[target].values
        oof_loss = log_loss(y, all_oof[target], labels=[0, 1])
        cal_loss = log_loss(y, calibrated_oof[target], labels=[0, 1])
        
        meta['per_target'][target] = {
            'config': all_best_configs[target],
            'n_features': len(all_best_cols[target]),
            'oof_cv_loss': float(oof_loss),
            'cal_oof_loss': float(cal_loss),
            'oof_mean': float(all_oof[target].mean()),
            'cal_mean': float(predictions[target].mean()),
            'train_rate': float(train_rate[target]),
            'pred_min': float(predictions[target].min()),
            'pred_max': float(predictions[target].max()),
        }
        
        log.info(
            f"\n  {target}: config={all_best_configs[target]}, "
            f"features={len(all_best_cols[target])}, "
            f"OOF_loss={oof_loss:.4f}, cal_OOF={cal_loss:.4f}, "
            f"pred_mean={predictions[target].mean():.4f}, "
            f"pred_range=[{predictions[target].min():.4f}, {predictions[target].max():.4f}]"
        )
    
    meta_path = sub_path.parent / f'meta_v10_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Metadata saved: {meta_path}")
    
    # ── Final summary ──────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("V10 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'OOF Loss':<12} {'Cal OOF':<12} {'Test Mean':<12} {'Train Rate':<12} {'Shift'}")
    for target in TARGET_COLS:
        oof_loss = log_loss(feat[target], all_oof[target], labels=[0, 1])
        cal_loss = log_loss(feat[target], calibrated_oof[target], labels=[0, 1])
        test_mean = predictions[target].mean()
        tr = train_rate[target]
        log.info(f"{target:<6} {oof_loss:<12.4f} {cal_loss:<12.4f} {test_mean:<12.4f} {tr:<12.3f} {test_mean-tr:+.4f}")
    
    return predictions


if __name__ == "__main__":
    main()
