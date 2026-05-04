"""
13_v11_fast.py — Fast V11: V10 base + top N V11-only features + optional personalization

Strategy:
1. For each target, rank ALL features (V10 + V11 extensions) by LGBM gain importance
2. Select best N (10, 20, 30, 40) features
3. Try with and without per-subject z-score personalization
4. GroupKFold × 20 seeds × 5 configs
5. Calibrate + compare with V10 (AVG = 0.6038)

Key optimization: personalization added ONCE per target to ALL selected features,
then the feature matrix is extracted to numpy arrays for fast iteration.
"""

import sys
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

LGB_CONFIGS = [
    {'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]


def remove_leakage(base_cols, target):
    if target.startswith('S'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_Q]
    return base_cols


def main():
    log.info("=" * 70)
    log.info("V11 Fast Experiment")
    log.info("=" * 70)

    # ── 1. Load V11 features ────────────────────────────
    feat = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    log.info(f"Loaded: {feat.shape}")

    # Get base columns (no _zscore, no meta, no targets)
    def get_base_cols(df):
        return [c for c in df.columns
                if c not in META_COLS | set(TARGET_COLS)
                and not c.endswith('_zscore')
                and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

    base_cols = get_base_cols(feat)
    log.info(f"Base features: {len(base_cols)}")

    # ── 2. Per-target: rank + select features ───────────
    log.info("\n=== Per-target ranking ===")

    for target in TARGET_COLS:
        leak_free = remove_leakage(base_cols, target)
        y = feat[target].values
        X = feat[leak_free].fillna(0).values

        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

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

        # Identify V10 vs V11-only features
        v10_feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
        v10_set = set(get_base_cols(v10_feat))

        v10_ranked = [(r[0], r[1]) for r in ranked if r[0] in v10_set]
        v11_only_ranked = [(r[0], r[1]) for r in ranked if r[0] not in v10_set]

        log.info(f"  {target}: {len(leak_free)} leak-free, "
                 f"V10 ranked={len(v10_ranked)}, V11-only ranked={len(v11_only_ranked)}")
        log.info(f"    Top 5: {[r[0] for r in ranked[:5]]}")

        # Store ranked list and V10 set for this target
        feat.attrs[f'{target}_ranked'] = ranked
        feat.attrs[f'{target}_v10_set'] = v10_set
        feat.attrs[f'{target}_ranked_dict'] = dict(ranked)

    # ── 3. Tuning ───────────────────────────────────────
    log.info(f"\n=== Tuning ===")
    lgbm_results = {}

    for target in TARGET_COLS:
        ranked = feat.attrs[f'{target}_ranked']
        y_all = feat[target].values
        n_pos = max((y_all == 1).sum(), 1)
        n_neg = (y_all == 0).sum()
        spw = n_neg / n_pos

        best_cv = float('inf')
        best_info = None

        # Try different feature counts
        for n_feat in [10, 20, 30, 40]:
            selected = [r[0] for r in ranked[:n_feat] if r[1] > 0]
            if len(selected) < 5:
                continue

            # Convert to numpy once (fast extraction)
            X_sel = feat[selected].fillna(0).values

            for cfg in LGB_CONFIGS:
                oof_avg, oof_full, cv_loss, cv_std, _ = _lgb_cv_predict_fast(
                    X_sel, target, feat, selected, SEEDS, cfg, spw
                )

                train_rate_val = y_all.mean()
                pred_mean_shift = abs(oof_avg.mean() - train_rate_val)
                score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_info = {
                        'n_feat': n_feat, 'config': cfg,
                        'cv_loss': cv_loss, 'cv_std': cv_std,
                        'score': score, 'oof_avg': oof_avg,
                        'selected': selected,
                    }
                    log.info(f"    n={n_feat} {cfg['nl']}/{cfg['lr']}: CV={cv_loss:.4f} std={cv_std:.4f}")

        if best_info is None:
            continue

        # Calibrate
        cal = simple_mean_match(best_info['oof_avg'], y_all.mean())
        cal_loss = log_loss(y_all, cal, labels=[0, 1])
        v10_loss = V10_SCORES[target]
        delta = cal_loss - v10_loss

        log.info(f"  {target}: n_feat={best_info['n_feat']} "
                 f"V10={v10_loss:.4f} V11={cal_loss:.4f} Δ={delta:+.4f} "
                 f"{'✅' if delta < 0 else '❌'}")

        lgbm_results[target] = {
            'cal_oof': cal_loss, 'cv_oof': best_info['cv_loss'],
            'v10': v10_loss, 'delta': delta,
            'n_features': best_info['n_feat'],
            'config': best_info['config'],
        }

    # ── 4. Summary ──────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== V11 FINAL ===")
    log.info(f"{'Target':<6} {'V10':<10} {'V11':<10} {'Δ':<8} {'Winner'}")
    log.info("-" * 70)

    avg_v11 = 0
    count = 0
    for target in TARGET_COLS:
        v10 = V10_SCORES[target]
        v11 = lgbm_results.get(target, {}).get('cal_oof')
        if v11 is not None:
            avg_v11 += v11
            delta = v11 - v10
            winner = "V11" if delta < 0 else "V10"
            log.info(f"{target:<6} {v10:<10.4f} {v11:<10.4f} {delta:+.4f} {winner}")
            count += 1

    if count > 0:
        avg_v11 /= count

    log.info("-" * 70)
    delta_avg = avg_v11 - V10_AVG
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg_v11:<10.4f} {delta_avg:+.4f} "
             f"{'🎉 V11!' if delta_avg < 0 else 'V10'}")

    # ── 5. Save ─────────────────────────────────────────
    results_path = MODEL_DIR / 'v11_fast_results.json'
    with open(results_path, 'w') as f:
        json.dump(lgbm_results, f, indent=2, default=float)
    log.info(f"Saved: {results_path}")

    meta = {
        'version': 'v11_fast',
        'avg_v10': float(V10_AVG),
        'avg_v11': float(avg_v11),
        'beat_v10': bool(delta_avg < 0),
        'per_target': {},
    }
    for target in TARGET_COLS:
        if target in lgbm_results:
            r = lgbm_results[target]
            meta['per_target'][target] = {
                'v10': r['v10'], 'v11': r['cal_oof'],
                'delta': r['delta'], 'n_features': r['n_features'],
            }

    with open(MODEL_DIR / 'v11_fast_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info("Done!")


def _lgb_cv_predict_fast(X_all, target, feat_df, col_names, seeds, cfg, spw):
    """Fast LGBM CV using pre-extracted numpy arrays."""
    y = feat_df[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_train = len(y)
    oof_full = np.zeros((n_train, len(seeds)))
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
            gkf.split(X_all, y, feat_df['subject_id'])
        ):
            X_tr = X_all[train_idx]
            X_va = X_all[val_idx]
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
            all_fold_losses[fold].append(log_loss(y_va, pred, labels=[0, 1]))

    oof_avg = oof_full.mean(axis=1)
    fold_avg_losses = [np.mean(all_fold_losses[i]) for i in range(N_SPLITS)]
    cv_loss = log_loss(y, oof_avg, labels=[0, 1])
    cv_std = np.std(fold_avg_losses)
    return oof_avg, oof_full, cv_loss, cv_std, fold_avg_losses


def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


if __name__ == "__main__":
    main()
