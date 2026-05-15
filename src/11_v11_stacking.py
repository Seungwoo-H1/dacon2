"""
11_v11_stacking.py — V11 Stacking Ensemble

Level-1: LGBM OOF predictions (GroupKFold × 30 seeds)
Level-2: LogisticRegression meta-learner

Compares stacking vs individual models.
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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

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
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

V10_SCORES = {
    'Q1': 0.6338, 'Q2': 0.6034, 'Q3': 0.6119,
    'S1': 0.5680, 'S2': 0.6022, 'S3': 0.5835, 'S4': 0.6240,
}
V10_AVG = 0.6038

N_SPLITS = 5
N_SEEDS = 30
N_FEATURES = [20, 50, 100]

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


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def remove_leakage_features(feature_cols, target):
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_Q]
    return feature_cols


def add_personalization(feat, feature_cols):
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


SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001,
         16000, 17001, 18000, 19000, 20000, 21001, 22000, 23000, 24000, 25001]

LGB_CONFIGS = [
    {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    {'name': 'C2', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C3', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C4', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 400, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
]


def lgb_oof(feat, selected_cols, target, seeds, cfg, spw):
    """Generate OOF predictions from LGBM ensemble."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    sanitized = [sanitize(c) for c in selected_cols]

    for seed_i, seed in enumerate(seeds):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'force_row_wise': True, 'n_jobs': -1,
            'scale_pos_weight': spw, 'random_state': seed,
        }
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[train_idx][selected_cols].fillna(0).values
            X_va = feat.iloc[val_idx][selected_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]
            train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={'verbose': '-1'})
            val_set = lgb.Dataset(X_va, label=y_va, feature_name=sanitized,
                                  reference=train_set, params={'verbose': '-1'})
            model = lgb.train(params, train_set, num_boost_round=cfg['ne'],
                              valid_sets=[val_set],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(0)])
            oof_full[val_idx, seed_i] = model.predict(X_va)

    return oof_full.mean(axis=1), oof_full


def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


def main():
    log.info("=" * 70)
    log.info("V11 Stacking Ensemble")
    log.info("=" * 70)

    # ── 1. Load features ────────────────────────────────
    feat_path = DATA_PROCESSED / "features_v11.parquet"
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded: {feat.shape}")

    base_feature_cols = get_feature_cols(feat)
    log.info(f"Base features: {len(base_feature_cols)}")
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}

    # ── 2. Find best individual model config per target ──
    # First, find the best single-model OOF predictions (level-1 base)
    best_config_per_target = {}

    for target in TARGET_COLS:
        log.info(f"\n--- Finding best config for {target} ---")
        leak_free_cols = remove_leakage_features(base_feature_cols, target)

        # Quick ranking
        y = feat[target].values
        X = feat[leak_free_cols].fillna(0).values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw_quick = n_neg / n_pos

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
            'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw_quick, 'random_state': 42,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
        }
        safe_names = [sanitize(c) for c in leak_free_cols]
        ds = lgb.Dataset(X, label=y, feature_name=safe_names, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=50)
        importances = model.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak_free_cols, importances), key=lambda x: -x[1])

        best_cv = float('inf')
        best_info = None

        for n_feat in [20, 50, 100]:
            sel = [r[0] for r in ranked[:n_feat] if r[1] > 0]
            if len(sel) < 5:
                continue

            feat_sel, _ = add_personalization(feat, sel)
            all_cols = sel + [f"{c}_zscore" for c in sel]
            y_full = feat_sel[target].values
            n_pos = max((y_full == 1).sum(), 1)
            n_neg = (y_full == 0).sum()
            spw = n_neg / n_pos

            for cfg in LGB_CONFIGS:
                oof_avg, oof_full = lgb_oof(feat_sel, all_cols, target, SEEDS, cfg, spw)
                cv_loss = log_loss(y_full, oof_avg, labels=[0, 1])

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_info = {
                        'config': cfg,
                        'n_feat': n_feat,
                        'selected_cols': all_cols,
                        'oof_full': oof_full,
                        'spw': spw,
                    }

        best_config_per_target[target] = best_info
        log.info(f"  Best for {target}: {best_info['config']['name']} n_feat={best_info['n_feat']} CV={best_cv:.4f}")

    # ── 3. Stacking: L1 OOF + L2 meta-learner ──────────
    log.info(f"\n{'='*50}")
    log.info("Stacking Training")
    log.info(f"{'='*50}")

    stacking_results = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} ---")
        best_info = best_config_per_target[target]
        feat_sel, _ = add_personalization(feat, best_info['selected_cols'])
        all_cols = best_info['selected_cols']
        oof_full = best_info['oof_full']  # (N_samples, N_seeds)

        y = feat_sel[target].values

        # Use top configs as level-1 features
        # Generate OOF from multiple configs
        l1_features = {}
        for cfg in LGB_CONFIGS:
            oof_avg, oof_full_cfg = lgb_oof(feat_sel, all_cols, target, SEEDS, cfg, best_info['spw'])
            cfg_name = cfg['name']
            l1_features[cfg_name] = oof_avg

        # Also add individual seed predictions as features (first few)
        for seed_i in range(min(5, len(SEEDS))):
            l1_features[f'seed_{SEEDS[seed_i]}'] = oof_full[:, seed_i]

        l1_df = pd.DataFrame(l1_features)
        log.info(f"  Level-1 features: {l1_df.shape[1]}")

        # Level-2: Logistic Regression meta-learner
        # Use GroupKFold to get OOF for level-2
        gkf = GroupKFold(n_splits=N_SPLITS)
        l2_oof = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(feat_sel, y, feat_sel['subject_id'])):
            X_tr = l1_df.iloc[train_idx].values
            X_va = l1_df.iloc[val_idx].values
            y_tr, y_va = y[train_idx], y[val_idx]

            meta = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
            meta.fit(X_tr, y_tr)
            l2_oof[val_idx] = meta.predict_proba(X_va)[:, 1]

        # Calibrate
        cal = simple_mean_match(l2_oof, train_rate[target])
        cal_loss = log_loss(y, cal, labels=[0, 1])

        # Compare with best single model
        single_oof = np.mean([l1_features[c] for c in l1_features], axis=0)
        single_cal = simple_mean_match(single_oof, train_rate[target])
        single_cal_loss = log_loss(y, single_cal, labels=[0, 1])

        v10_loss = V10_SCORES[target]
        delta_stacking = cal_loss - v10_loss
        delta_single = single_cal_loss - v10_loss

        log.info(f"  V10: {v10_loss:.4f} | Stacking: {cal_loss:.4f} | Δ={delta_stacking:+.4f}")
        log.info(f"  V10: {v10_loss:.4f} | Single ensemble avg: {single_cal_loss:.4f} | Δ={delta_single:+.4f}")

        stacking_results[target] = {
            'stacking_cal_oof': cal_loss,
            'single_cal_oof': single_cal_loss,
            'v10': v10_loss,
            'n_l1_features': l1_df.shape[1],
        }

    # ── 4. Summary ──────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== V11 Stacking Results ===")
    log.info(f"{'Target':<6} {'V10':<10} {'Single':<10} {'Stack':<10} {'Δ Stk':<8} {'Winner'}")
    log.info("-" * 70)

    avg_stacking = 0
    avg_single = 0
    for target in TARGET_COLS:
        v10 = stacking_results[target]['v10']
        single = stacking_results[target]['single_cal_oof']
        stack = stacking_results[target]['stacking_cal_oof']
        avg_stacking += stack
        avg_single += single

        if stack <= single and stack <= v10:
            winner = "Stack"
        elif single <= v10:
            winner = "Single"
        else:
            winner = "V10"

        log.info(f"{target:<6} {v10:<10.4f} {single:<10.4f} {stack:<10.4f} {stack - v10:+.4f} {winner}")

    n_targets = len(TARGET_COLS)
    avg_stacking /= n_targets
    avg_single /= n_targets
    avg_v10 = V10_AVG

    log.info("-" * 70)
    log.info(f"{'AVG':<6} {avg_v10:<10.4f} {avg_single:<10.4f} {avg_stacking:<10.4f} {avg_stacking - avg_v10:+.4f}")

    if avg_stacking < avg_v10:
        log.info(f"🎉 Stacking BEATS V10 by {avg_v10 - avg_stacking:.4f}!")
    else:
        log.info(f"V10 still better by {avg_stacking - avg_v10:.4f}")

    # ── 5. Save ─────────────────────────────────────────
    results_path = MODEL_DIR / 'v11_stacking_results.json'
    with open(results_path, 'w') as f:
        json.dump(stacking_results, f, indent=2, default=float)
    log.info(f"Saved: {results_path}")

    meta = {
        'version': 'v11_stacking',
        'n_splits': N_SPLITS,
        'n_seeds': N_SEEDS,
        'n_l1_models': len(LGB_CONFIGS) + 5,
        'level_2': 'LogisticRegression(C=0.1)',
        'average_stacking': float(avg_stacking),
        'average_single': float(avg_single),
        'average_v10': float(avg_v10),
        'per_target': stacking_results,
    }
    meta_path = MODEL_DIR / 'v11_stacking_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info(f"Saved meta: {meta_path}")

    return stacking_results


if __name__ == "__main__":
    main()
