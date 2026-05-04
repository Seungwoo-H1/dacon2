"""
09_v11_catboost.py — V11 CatBoost experiment

Strategy:
1. Load features_v11.parquet (4860 base features)
2. Rank features per target using LGBM (fast) on BASE features only
3. Select top-N (10, 20, 30) features per target
4. Add personalization (z-score) ONLY to selected features
5. Train CatBoost with GroupKFold (5 splits) × 10 seeds
6. Compare with V10

This avoids the explosion to 19000+ columns that kills CatBoost on 450 samples.
"""

import sys
import re
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import TARGETS, DATA_PROCESSED, MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

V10_SCORES = {
    'Q1': 0.6338, 'Q2': 0.6034, 'Q3': 0.6119,
    'S1': 0.5680, 'S2': 0.6022, 'S3': 0.5835, 'S4': 0.6240,
}
V10_AVG = 0.6038

N_SPLITS = 5
N_SEEDS_CB = 10


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


# ── Leakage features (same as V10) ─────────────────────────

LEAKAGE_FEATURES_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min',
    'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}

LEAKAGE_FEATURES_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
}


def remove_leakage_features(feature_cols, target):
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_Q]
    return feature_cols


# ── Personalization (z-score per subject) ──────────────────

def add_personalization(feat, feature_cols):
    """Add per-subject z-score for given feature columns ONLY."""
    feat = feat.copy()
    personal_cols = []

    for col in feature_cols:
        col_filled = feat[col].fillna(0)
        subj_stats = col_filled.groupby(feat['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        merged = feat.merge(subj_stats, on='subject_id', how='left')

        mask_std_zero = merged[f'{col}_subj_std'] == 0
        mask_null = feat[col].isnull()
        merged[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null, 0.0,
            (merged[col] - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
        feat = merged

    return feat, personal_cols


# ── Feature ranking with LGBM (fast) ──────────────────────

def rank_features_lgbm(feat, feature_cols, target, random_seed=42):
    """Quick LGBM scan to rank features by gain importance."""
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos

    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': random_seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
    }
    safe_names = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=safe_names, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    return ranked


# ── CatBoost CV ────────────────────────────────────────────

def cb_cv_predict(feat, selected_cols, target, seeds, cb_config, spw):
    """GroupKFold: N_SPLITS folds × N_SEEDS seeds → OOF predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}

    sanitized = [sanitize(c) for c in selected_cols]

    for seed_i, seed in enumerate(seeds):
        cfg = {**cb_config, 'random_seed': seed + 1000}

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[train_idx][selected_cols].fillna(0).values
            X_va = feat.iloc[val_idx][selected_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]

            m = cb.CatBoost(cfg)
            m.fit(
                X_tr, y_tr,
                eval_set=cb.Pool(X_va, y_va, feature_names=sanitized),
                verbose=0,
                use_best_model=True,
                early_stopping_rounds=50,
            )

            pred = np.clip(m.predict(X_va), 0.0001, 0.9999)
            oof_full[val_idx, seed_i] = pred

            fold_loss = log_loss(y_va, pred, labels=[0, 1])
            all_fold_losses[fold].append(fold_loss)

    oof_avg = oof_full.mean(axis=1)
    fold_avg_losses = [np.mean(all_fold_losses[i]) for i in range(N_SPLITS)]
    cv_loss = log_loss(y, oof_avg, labels=[0, 1])
    cv_std = np.std(fold_avg_losses)

    return oof_avg, oof_full, cv_loss, cv_std, fold_avg_losses


# ── Calibration ────────────────────────────────────────────

def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    calibrated = pred + shift
    calibrated = np.clip(calibrated, 0.0001, 0.9999)
    return calibrated


# ── Main pipeline ──────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("V11 CatBoost Experiment — Smart feature selection first")
    log.info("=" * 70)

    # ── 1. Load features (base only, no personalization yet) ──
    feat_path = DATA_PROCESSED / "features_v11.parquet"
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded features: {feat.shape}")

    base_feature_cols = get_feature_cols(feat)
    log.info(f"Base feature cols: {len(base_feature_cols)}")

    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"Target rates: {train_rate}")

    # ── 2. CatBoost configs ─────────────────────────────
    cb_config_a = {
        'iterations': 500, 'depth': 6, 'learning_rate': 0.03,
        'bagging_temperature': 0.5, 'l2_leaf_reg': 3,
        'loss_function': 'Logloss', 'eval_metric': 'Logloss',
        'verbose': 0, 'one_hot_max_size': 2, 'random_strength': 1,
    }

    cb_config_b = {
        'iterations': 300, 'depth': 4, 'learning_rate': 0.05,
        'bagging_temperature': 1.0, 'l2_leaf_reg': 5,
        'loss_function': 'Logloss', 'eval_metric': 'Logloss',
        'verbose': 0, 'one_hot_max_size': 2, 'random_strength': 1,
    }

    configs = {'A': cb_config_a, 'B': cb_config_b}
    all_results = {}

    for config_name, cb_config in configs.items():
        log.info(f"\n{'='*50}")
        log.info(f"CatBoost Config {config_name}")
        log.info(f"  {cb_config}")
        log.info(f"{'='*50}")

        cal_oof_scores = {}
        oof_preds = {}
        selected_feature_counts = {}

        for target in TARGET_COLS:
            log.info(f"\n--- {target} ---")

            # Step 1: Rank base features (fast LGBM scan)
            leak_free_base = remove_leakage_features(base_feature_cols, target)
            ranked = rank_features_lgbm(feat, leak_free_base, target)

            best_cv = float('inf')
            best_cols = None
            best_oof = None

            # Step 2: Try different feature counts
            for n_feat in [10, 20, 30, 50]:
                selected_base = [r[0] for r in ranked[:n_feat] if r[1] > 0]
                log.info(f"  Trying n={n_feat}: {len(selected_base)} selected")

                # Step 3: Add personalization ONLY to selected features
                feat_sel, _ = add_personalization(feat, selected_base)
                all_cols = selected_base + [f"{c}_zscore" for c in selected_base]

                # Step 4: Train CatBoost
                y = feat_sel[target].values
                n_pos = max((y == 1).sum(), 1)
                n_neg = (y == 0).sum()
                spw = n_neg / n_pos

                oof_avg, oof_full, cv_loss, cv_std, _ = cb_cv_predict(
                    feat_sel, all_cols, target,
                    list(range(1, N_SEEDS_CB + 1)), cb_config, spw
                )

                train_rate_val = y.mean()
                pred_mean_shift = abs(oof_avg.mean() - train_rate_val)
                score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_cols = all_cols
                    best_oof = oof_avg
                    log.info(f"    → New best: cv={cv_loss:.4f}, score={score:.4f}")

            # Final calibration
            cal = simple_mean_match(best_oof, train_rate[target])
            cal_loss = log_loss(feat[target].values, cal, labels=[0, 1])
            v10_loss = V10_SCORES[target]
            delta = cal_loss - v10_loss

            cal_oof_scores[target] = cal_loss
            oof_preds[target] = cal
            selected_feature_counts[target] = len(best_cols)

            winner = "V11" if delta < 0 else "V10"
            log.info(f"  V10: {v10_loss:.4f} | V11: {cal_loss:.4f} | Δ={delta:+.4f} | Winner: {winner}")
            log.info(f"  Config: {config_name} | Features: {len(best_cols)} ({sum(1 for c in best_cols if '_zscore' in c)} zscore) | CV: {best_cv:.4f}")

        avg_cal = np.mean(list(cal_oof_scores.values()))
        log.info(f"\n  {config_name} AVG cal OOF: {avg_cal:.4f} | V10: {V10_AVG:.4f} | Δ={avg_cal - V10_AVG:+.4f}")
        all_results[config_name] = cal_oof_scores
        all_results[config_name]['AVG'] = avg_cal
        all_results[config_name]['selected_counts'] = selected_feature_counts

    # ── 3. Final comparison table ─────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== V11 CatBoost Results ===")
    log.info(f"{'Target':<6} {'V10':<10} {'CB-A':<10} {'Δ A':<8} {'CB-B':<10} {'Δ B':<8} {'Winner'}")
    log.info("-" * 70)

    for target in TARGET_COLS:
        v10 = V10_SCORES[target]
        a = all_results['A'][target]
        b = all_results['B'][target]
        da = a - v10
        db = b - v10
        if da < 0 or db < 0:
            winner = "V11"
        elif da <= db:
            winner = "A"
        else:
            winner = "B"
        log.info(f"{target:<6} {v10:<10.4f} {a:<10.4f} {da:+.4f} {b:<10.4f} {db:+.4f} {winner}")

    avg_a = all_results['A']['AVG']
    avg_b = all_results['B']['AVG']
    da_avg = avg_a - V10_AVG
    db_avg = avg_b - V10_AVG
    log.info("-" * 70)
    winner = "V11" if da_avg < 0 or db_avg < 0 else ("A" if da_avg <= db_avg else "B")
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg_a:<10.4f} {da_avg:+.4f} {avg_b:<10.4f} {db_avg:+.4f} {winner}")
    log.info(f"\n{'='*70}")
    if avg_a < avg_b:
        log.info(f"  ✅ Best: Config A (AVG cal OOF = {avg_a:.4f})")
    else:
        log.info(f"  ✅ Best: Config B (AVG cal OOF = {avg_b:.4f})")
    if min(avg_a, avg_b) < V10_AVG:
        log.info(f"  🎉 V11 BEATS V10 by {V10_AVG - min(avg_a, avg_b):.4f}!")
    else:
        log.info(f"  V10 still better by {min(avg_a, avg_b) - V10_AVG:.4f}")
    log.info(f"{'='*70}")

    # ── 4. Save results ───────────────────────────────────
    results_path = MODEL_DIR / 'v11_catboost_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    log.info(f"Saved results: {results_path}")

    meta = {
        'version': 'v11_catboost',
        'features_file': str(feat_path),
        'n_base_features': len(base_feature_cols),
        'n_splits': N_SPLITS,
        'n_seeds': N_SEEDS_CB,
        'calibration': 'simple mean-matching + clip',
        'strategy': 'LGBM rank → select top-N → z-score only selected → CatBoost',
        'per_target': {},
    }
    for target in TARGET_COLS:
        meta['per_target'][target] = {
            'v10_cal_oof': V10_SCORES[target],
            'cb_a_cal_oof': all_results['A'][target],
            'cb_b_cal_oof': all_results['B'][target],
            'n_features_a': all_results['A']['selected_counts'][target],
            'n_features_b': all_results['B']['selected_counts'][target],
        }

    meta_path = MODEL_DIR / 'v11_catboost_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info(f"Saved meta: {meta_path}")

    return all_results


if __name__ == "__main__":
    main()
