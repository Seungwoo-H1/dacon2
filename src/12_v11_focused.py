"""
12_v11_focused.py — V11 with focused, high-signal features only

Strategy:
1. Take V10's proven pipeline as base
2. Add only the TOP 10 most promising new features from the extended set
3. Try n_feat=20, 30, 40 (with/without personalization)
4. GroupKFold × 20 seeds × multiple configs
5. No stacking (overkill for 450 samples)
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
N_SEEDS = 20

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


def add_personalization(feat, feature_cols):
    """Add per-subject z-score for given feature columns ONLY."""
    feat2 = feat.copy()
    for col in feature_cols:
        col_filled = feat2[col].fillna(0)
        subj_stats = col_filled.groupby(feat2['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        merged = feat2.merge(subj_stats, on='subject_id', how='left')
        mask_std_zero = merged[f'{col}_subj_std'] == 0
        mask_null = feat2[col].isnull()
        merged[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null, 0.0,
            (merged[col] - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        feat2 = merged
    return feat2


def lgb_cv_predict(feat, feature_cols, target, seeds, cfg, spw):
    """GroupKFold × seeds → OOF predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}

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
            X_tr = feat.iloc[train_idx][feature_cols].fillna(0).values
            X_va = feat.iloc[val_idx][feature_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr, params={'verbose': '-1'})
            val_set = lgb.Dataset(X_va, label=y_va, feature_name=train_set.feature_name,
                                  reference=train_set, params={'verbose': '-1'})

            model = lgb.train(
                params, train_set, num_boost_round=cfg['ne'],
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(0)],
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


def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


def main():
    log.info("=" * 70)
    log.info("V11 Focused Experiment — Careful feature selection")
    log.info("=" * 70)

    # ── 1. Load base V10 features (smaller, proven) ─────
    feat_path = DATA_PROCESSED / "features.parquet"
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded V10 features: {feat.shape}")

    # ── 2. Load V11 extended features ──────────────────
    feat_v11 = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    log.info(f"Loaded V11 features: {feat_v11.shape}")

    # ── 3. Find shared features + identify V11-only features
    v10_cols = set(feat.columns) - META_COLS - set(TARGET_COLS)
    v11_cols = set(feat_v11.columns) - META_COLS - set(TARGET_COLS)
    shared_cols = v10_cols & v11_cols
    v11_only_cols = v11_cols - v10_cols
    log.info(f"Shared features: {len(shared_cols)}")
    log.info(f"V11-only features: {len(v11_only_cols)}")

    # ── 4. For each target: rank ALL features (shared + V11-only) ──
    log.info("\n=== Ranking ===")
    target_info = {}

    for target in TARGET_COLS:
        # Get feature columns (no leakage)
        all_feature_cols = [c for c in feat_v11.columns
                          if c not in META_COLS | set(TARGET_COLS)
                          and feat_v11[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
        leak_free = remove_leakage_features(all_feature_cols, target)
        log.info(f"  {target}: {len(leak_free)} leak-free features")

        # Quick LGBM ranking on base features (no personalization yet)
        y = feat_v11[target].values
        X = feat_v11[leak_free].fillna(0).values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        # Use small model for fast ranking
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
            'n_estimators': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds = lgb.Dataset(X, label=y, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=30)
        importances = model.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak_free, importances), key=lambda x: -x[1])

        # Separate shared vs V11-only
        shared_ranked = [r for r in ranked if r[0] in shared_cols and r[1] > 0]
        v11_only_ranked = [r for r in ranked if r[0] in v11_only_cols and r[1] > 0]

        target_info[target] = {
            'ranked': ranked,
            'shared_ranked': shared_ranked,
            'v11_only_ranked': v11_only_ranked,
        }
        log.info(f"  Top5: {[r[0] for r in ranked[:5]]}")

    # ── 5. Try different feature compositions ──────────
    log.info(f"\n=== Tuning ===")

    configs = [
        {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
        {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C3', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'name': 'C4', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
        {'name': 'C5', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    ]

    seeds = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
             6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

    lgbm_results = {}

    for target in TARGET_COLS:
        info = target_info[target]
        log.info(f"\n{'='*50}")
        log.info(f"Target: {target}")

        y_all = feat_v11[target].values
        n_pos = max((y_all == 1).sum(), 1)
        n_neg = (y_all == 0).sum()
        spw = n_neg / n_pos

        best_cv = float('inf')
        best_result = None

        # Strategy: fixed set of promising feature counts and compositions
        # 1. Shared only: top 20, 30
        # 2. Shared + top V11-only: top 15+5, top 20+5, top 20+10
        # 3. Shared only + personalization: top 20

        candidates = []

        # Shared only
        for n in [20, 30]:
            sel = [r[0] for r in info['shared_ranked'][:n] if r[1] > 0]
            candidates.append(('shared_n'+str(n), sel, False))

        # Shared + V11-only
        for n_shared, n_new in [(15, 5), (20, 5), (20, 10), (15, 10)]:
            shared = [r[0] for r in info['shared_ranked'][:n_shared] if r[1] > 0]
            new = [r[0] for r in info['v11_only_ranked'][:n_new] if r[1] > 0]
            sel = shared + new
            candidates.append((f"shared{len(shared)}_new{len(new)}", sel, False))

        # Shared + personalization (z-score on selected)
        for n in [20]:
            sel = [r[0] for r in info['shared_ranked'][:n] if r[1] > 0]
            candidates.append(('shared_n'+str(n)+'_zscore', sel, True))

        for cand_name, selected_base, do_zscore in candidates:
            if len(selected_base) < 5:
                continue

            if do_zscore:
                feat_tuned = add_personalization(feat_v11, selected_base)
                z_cols = [f"{c}_zscore" for c in selected_base]
                all_cols = selected_base + z_cols
            else:
                feat_tuned = feat_v11
                all_cols = selected_base

            for cfg in configs:
                oof_avg, oof_full, cv_loss, cv_std, fold_losses = lgb_cv_predict(
                    feat_tuned, all_cols, target, seeds, cfg, spw
                )

                train_rate_val = y_all.mean()
                pred_mean_shift = abs(oof_avg.mean() - train_rate_val)
                score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_result = {
                        'name': cand_name, 'config': cfg,
                        'oof_avg': oof_avg, 'oof_full': oof_full,
                        'cv_loss': cv_loss, 'cv_std': cv_std,
                        'score': score, 'selected_cols': all_cols,
                        'n_features': len(all_cols),
                    }
                    log.info(f"    {cand_name} {cfg['name']}: CV={cv_loss:.4f} std={cv_std:.4f}")

        if best_result is None:
            log.info(f"  No valid config for {target}")
            continue

        log.info(f"  ✅ Best: {best_result['name']} {best_result['config']['name']} "
                 f"n_feat={best_result['n_features']} CV={best_result['cv_loss']:.4f}")

        # Calibrate
        cal = simple_mean_match(best_result['oof_avg'], y_all.mean())
        cal_loss = log_loss(y_all, cal, labels=[0, 1])
        v10_loss = V10_SCORES[target]
        delta = cal_loss - v10_loss

        log.info(f"  V10: {v10_loss:.4f} | V11: {cal_loss:.4f} | Δ={delta:+.4f} | {'✅ V11' if delta < 0 else '❌ V10'}")

        lgbm_results[target] = {
            'cal_oof': cal_loss, 'cv_oof': best_result['cv_loss'],
            'v10': v10_loss, 'delta': delta,
            'config': best_result['config'],
            'n_features': best_result['n_features'],
            'name': best_result['name'],
        }

    # ── 6. Summary ──────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== V11 FINAL RESULTS ===")
    log.info(f"{'Target':<6} {'V10':<10} {'V11':<10} {'Δ':<8} {'Winner'}")
    log.info("-" * 70)

    avg_v11 = 0
    count = 0
    for target in TARGET_COLS:
        v10 = V10_SCORES[target]
        v11 = lgbm_results.get(target, {}).get('cal_oof', None)
        if v11 is not None:
            avg_v11 += v11
            delta = v11 - v10
            winner = "V11" if delta < 0 else "V10"
            log.info(f"{target:<6} {v10:<10.4f} {v11:<10.4f} {delta:+.4f} {winner}")
            count += 1

    if count > 0:
        avg_v11 /= count

    log.info("-" * 70)
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg_v11:<10.4f} {avg_v11 - V10_AVG:+.4f} "
             f"{'🎉 V11!' if avg_v11 < V10_AVG else 'V10'}")

    # ── 7. Save ─────────────────────────────────────────
    results_path = MODEL_DIR / 'v11_focused_results.json'
    with open(results_path, 'w') as f:
        json.dump(lgbm_results, f, indent=2, default=float)
    log.info(f"Saved: {results_path}")

    meta = {
        'version': 'v11_focused',
        'avg_v10': float(V10_AVG),
        'avg_v11': float(avg_v11),
        'beat_v10': bool(avg_v11 < V10_AVG),
        'per_target': {},
    }
    for target in TARGET_COLS:
        if target in lgbm_results:
            meta['per_target'][target] = lgbm_results[target]

    meta_path = MODEL_DIR / 'v11_focused_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
