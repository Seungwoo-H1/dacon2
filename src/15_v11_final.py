"""
15_v11_final.py — V11: Extend features + careful selection

Strategy:
1. V11 extended features + ranking on all 4860 features
2. Select top 10, 20, 30, 40 features per target  
3. Try 3 configs × 10 seeds for each
4. Calibrate + compare with V10
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
N_SEEDS = 10

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

# Reduced configs
LGB_CONFIGS = [
    {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]


def remove_leakage(base_cols, target):
    if target.startswith('S'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_Q]
    return base_cols


def get_base_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and not c.endswith('_zscore')
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def main():
    log.info("=" * 70)
    log.info("V11 Final")
    log.info("=" * 70)

    # Load features
    feat = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    subject_ids = feat['subject_id'].values
    log.info(f"Loaded: {feat.shape}")

    base_cols = get_base_cols(feat)
    log.info(f"Base features: {len(base_cols)}")

    # Pre-extract numpy array
    X_all_base = feat[base_cols].fillna(0).values
    log.info(f"X_all_base: {X_all_base.shape}")

    # Per-target ranking + feature extraction
    log.info("\n=== Ranking ===")
    target_info = {}

    for target in TARGET_COLS:
        leak_free = remove_leakage(base_cols, target)
        y = feat[target].values

        col_to_idx = {c: i for i, c in enumerate(base_cols)}
        leak_free_idx = [col_to_idx[c] for c in leak_free]
        X_leak = X_all_base[:, leak_free_idx]

        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        # LGBM rank
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
            'n_estimators': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds = lgb.Dataset(X_leak, label=y, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=30)
        importances = model.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak_free, importances), key=lambda x: -x[1])

        # Extract numpy arrays for each n_feat
        X_arrays = {}
        for n in [10, 20, 30, 40]:
            sel = [r[0] for r in ranked[:n] if r[1] > 0]
            sel_idx = [col_to_idx[c] for c in sel]
            X_arrays[n] = (sel, X_all_base[:, sel_idx])

        gkf = GroupKFold(n_splits=N_SPLITS)
        gkf_splits = list(gkf.split(X_leak, y, subject_ids))

        target_info[target] = {
            'ranked': ranked,
            'X_arrays': X_arrays,
            'y': y, 'spw': spw, 'gkf_splits': gkf_splits,
        }
        log.info(f"  {target}: {len(leak_free)} leak-free, top5={[r[0] for r in ranked[:5]]}")

    # Tuning
    log.info(f"\n=== Tuning ===")
    lgbm_results = {}

    for target in TARGET_COLS:
        info = target_info[target]
        y = info['y']
        spw = info['spw']
        target_rate = y.mean()

        log.info(f"\n  {target}:")
        best_cv = float('inf')
        best_info = None

        for n_feat, (sel_names, X_sel) in info['X_arrays'].items():
            for cfg_idx, cfg in enumerate(LGB_CONFIGS):
                oof_full = _cv(X_sel, y, info['gkf_splits'], SEEDS, cfg, spw)
                oof_avg = oof_full.mean(axis=1)
                cv_loss = log_loss(y, oof_avg, labels=[0, 1])

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_info = {
                        'n_feat': n_feat, 'config': cfg,
                        'oof_avg': oof_avg, 'cv_loss': cv_loss,
                        'sel': sel_names,
                    }
                    log.info(f"    n={n_feat} cfg={cfg_idx}: CV={cv_loss:.4f}")

        if best_info is None:
            continue

        # Calibrate
        cal = np.clip(best_info['oof_avg'] + (target_rate - best_info['oof_avg'].mean()), 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0, 1])
        v10_loss = V10_SCORES[target]
        delta = cal_loss - v10_loss

        log.info(f"    Best: n_feat={best_info['n_feat']} "
                 f"V10={v10_loss:.4f} V11={cal_loss:.4f} Δ={delta:+.4f} "
                 f"{'✅' if delta < 0 else '❌'}")

        lgbm_results[target] = {
            'cal_oof': cal_loss, 'v10': v10_loss, 'delta': delta,
            'n_features': best_info['n_feat'],
        }

    # Summary
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
    winner_avg = "V11" if delta_avg < 0 else "V10"
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg_v11:<10.4f} {delta_avg:+.4f} {winner_avg}")

    if delta_avg < 0:
        log.info(f"🎉 V11 BEATS V10 by {abs(delta_avg):.4f}!")
    else:
        log.info(f"V10 still better by {abs(delta_avg):.4f}")

    # Save
    results_path = MODEL_DIR / 'v11_final_results.json'
    with open(results_path, 'w') as f:
        json.dump(lgbm_results, f, indent=2, default=float)
    log.info(f"Saved: {results_path}")

    meta = {
        'version': 'v11_final',
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

    with open(MODEL_DIR / 'v11_final_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info("Done!")


def _cv(X_all, y, gkf_splits, seeds, cfg, spw):
    """GroupKFold CV with numpy arrays."""
    n = len(y)
    oof_full = np.zeros((n, len(seeds)))

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

        for fold, (train_idx, val_idx) in enumerate(gkf_splits):
            train_set = lgb.Dataset(X_all[train_idx], label=y[train_idx], params={'verbose': '-1'})
            val_set = lgb.Dataset(X_all[val_idx], label=y[val_idx],
                                  feature_name=train_set.feature_name,
                                  reference=train_set, params={'verbose': '-1'})

            model = lgb.train(params, train_set, num_boost_round=cfg['ne'],
                              valid_sets=[val_set],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(0)])
            oof_full[val_idx, seed_i] = model.predict(X_all[val_idx])

    return oof_full


if __name__ == "__main__":
    main()
