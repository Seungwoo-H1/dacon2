"""
10_v11_lgbm_extended.py — V11 LGBM + Stacking with precomputed personalization

Uses features_v11_personalized.parquet (450 × 8670) which already has z-scores.
1. Rank base features via LGBM importance
2. Select top-20 / top-50 base cols + their z-score cols
3. GroupKFold × 6 configs × 30 seeds
4. Calibrate + compare with V10 (AVG = 0.6038)
5. Stacking ensemble
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


def get_base_feature_cols(df):
    """Return base feature names (without _zscore suffix)."""
    result = []
    for c in df.columns:
        if c in META_COLS or c in TARGET_COLS:
            continue
        if c.endswith('_zscore'):
            continue
        if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]:
            result.append(c)
    return result


def remove_leakage_features(base_cols, target):
    if target.startswith('S'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith('Q'):
        return [c for c in base_cols if c not in LEAKAGE_FEATURES_Q]
    return base_cols


def get_zscore_cols(base_cols, feat_df):
    """Get z-score columns that exist in the dataframe for given base cols."""
    all_cols = feat_df.columns.tolist()
    return [f"{c}_zscore" for c in base_cols if f"{c}_zscore" in all_cols]


def rank_features(feat, base_cols, target, random_seed=42):
    """Quick LGBM rank on base features only."""
    y = feat[target].values
    X = feat[base_cols].fillna(0).values
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
    safe_names = [sanitize(c) for c in base_cols]
    ds = lgb.Dataset(X, label=y, feature_name=safe_names, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(base_cols, importances), key=lambda x: -x[1])
    return ranked


LGB_CONFIGS = [
    {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    {'name': 'C2', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C3', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C4', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 400, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    {'name': 'C5', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    {'name': 'C6', 'nl': 10, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.8, 'cst': 0.8, 'ra': 0.5, 'rl': 1.0, 'mc': 10},
]

SEEDS = list(range(42, 42 + N_SEEDS))


def lgb_cv_predict(feat, selected_cols, target, seeds, cfg, spw):
    """GroupKFold × seeds → OOF predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}
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
    log.info("V11 LGBM Extended + Stacking")
    log.info("=" * 70)

    # ── 1. Load pre-computed personalization ────────────
    feat_path = DATA_PROCESSED / "features_v11_personalized.parquet"
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded: {feat.shape}")

    base_cols = get_base_feature_cols(feat)
    log.info(f"Base features: {len(base_cols)}")
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}

    # ── 2. Per-target ranking + personalization ─────────
    log.info("\n=== Per-target analysis ===")
    target_info = {}

    for target in TARGET_COLS:
        leak_free = remove_leakage_features(base_cols, target)
        ranked = rank_features(feat, leak_free, target)
        top30 = [r[0] for r in ranked[:30] if r[1] > 0]
        log.info(f"  {target}: {len(leak_free)} leak-free, top30={len(top30)}")

        # Build feature combos
        zscore_map = {}
        for base_c in top30:
            zc = f"{base_c}_zscore"
            if zc in feat.columns:
                zscore_map[base_c] = zc

        target_info[target] = {
            'leak_free': leak_free,
            'ranked': ranked,
            'zscore_map': zscore_map,
        }

    # ── 3. LGBM tuning ──────────────────────────────────
    log.info("\n=== LGBM Tuning ===")
    lgbm_results = {}

    for ti, target in enumerate(TARGET_COLS):
        info = target_info[target]
        log.info(f"\n[{ti+1}/{len(TARGET_COLS)}] {target}")

        y_all = feat[target].values
        n_pos = max((y_all == 1).sum(), 1)
        n_neg = (y_all == 0).sum()
        spw = n_neg / n_pos

        best_cv = float('inf')
        best_result = None

        for n_feat in [20, 50]:
            selected_base = [r[0] for r in info['ranked'][:n_feat] if r[1] > 0]
            if len(selected_base) < 5:
                continue
            zcols = [info['zscore_map'][c] for c in selected_base if c in info['zscore_map']]
            all_cols = selected_base + zcols

            for cfg in LGB_CONFIGS:
                oof_avg, oof_full, cv_loss, cv_std, fold_losses = lgb_cv_predict(
                    feat, all_cols, target, SEEDS, cfg, spw
                )
                train_rate_val = y_all.mean()
                pred_mean_shift = abs(oof_avg.mean() - train_rate_val)
                score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift

                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_result = {
                        'config': cfg, 'n_feat': n_feat,
                        'oof_avg': oof_avg, 'oof_full': oof_full,
                        'cv_loss': cv_loss, 'cv_std': cv_std,
                        'score': score, 'selected_cols': all_cols,
                    }
                    log.info(f"    {cfg['name']} n={n_feat}: CV={cv_loss:.4f} std={cv_std:.4f}")

        if best_result is None:
            log.info(f"  No valid config")
            continue

        log.info(f"  ✅ Best: {best_result['config']['name']} n_feat={best_result['n_feat']} CV={best_result['cv_loss']:.4f} cols={len(best_result['selected_cols'])}")

        cal = simple_mean_match(best_result['oof_avg'], train_rate[target])
        cal_loss = log_loss(y_all, cal, labels=[0, 1])
        v10_loss = V10_SCORES[target]
        delta = cal_loss - v10_loss

        log.info(f"  V10: {v10_loss:.4f} | V11: {cal_loss:.4f} | Δ={delta:+.4f} | {'✅ V11' if delta < 0 else '❌ V10'}")

        lgbm_results[target] = {
            'cal_oof': cal_loss, 'cv_oof': best_result['cv_loss'],
            'v10': v10_loss, 'delta': delta,
            'config': best_result['config'],
            'n_features': len(best_result['selected_cols']),
            'cal_preds': cal, 'oof_preds': best_result['oof_avg'],
        }

    # ── 4. Stacking ─────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== Stacking ===")

    stacking_results = {}

    for target in TARGET_COLS:
        if target not in lgbm_results:
            continue

        info = target_info[target]
        y_all = feat[target].values

        # Generate OOF from multiple configs as level-1 features
        l1_features = {}
        used_key = set()

        for cfg in LGB_CONFIGS:
            cfg_key = cfg['name']
            for n_feat in [20, 50]:
                sel_base = [r[0] for r in info['ranked'][:n_feat] if r[1] > 0]
                zcols = [info['zscore_map'][c] for c in sel_base if c in info['zscore_map']]
                all_cols = sel_base + zcols
                l1_key = f"{cfg_key}_n{n_feat}"
                if l1_key not in used_key:
                    used_key.add(l1_key)
                    oof_avg, _, _, _, _ = lgb_cv_predict(feat, all_cols, target, SEEDS, cfg, lgbm_results[target]['cv_oof'] / 0.95)  # approximate spw reuse
                    l1_features[l1_key] = oof_avg

        # Level-2: LogisticRegression with GroupKFold OOF
        gkf = GroupKFold(n_splits=N_SPLITS)
        l2_oof = np.zeros(len(y_all))
        l1_df = pd.DataFrame(l1_features)

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(feat, y_all, feat['subject_id'])
        ):
            X_tr = l1_df.iloc[train_idx].values
            X_va = l1_df.iloc[val_idx].values
            y_tr, y_va = y_all[train_idx], y_all[val_idx]

            meta = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
            meta.fit(X_tr, y_tr)
            l2_oof[val_idx] = meta.predict_proba(X_va)[:, 1]

        cal = simple_mean_match(l2_oof, train_rate[target])
        cal_loss = log_loss(y_all, cal, labels=[0, 1])

        # Single ensemble avg
        single_avg = np.mean(list(l1_features.values()), axis=0)
        single_cal = simple_mean_match(single_avg, train_rate[target])
        single_cal_loss = log_loss(y_all, single_cal, labels=[0, 1])

        v10_loss = V10_SCORES[target]
        stacking_results[target] = {
            'stacking_cal_oof': cal_loss,
            'single_cal_oof': single_cal_loss,
            'v10': v10_loss,
            'n_l1_features': l1_df.shape[1],
        }
        log.info(f"  {target}: L1={l1_df.shape[1]} | Stk={cal_loss:.4f} | Single={single_cal_loss:.4f} | V10={v10_loss:.4f}")

    # ── 5. Summary ──────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("=== V11 FINAL RESULTS ===")
    log.info(f"{'Target':<6} {'V10':<10} {'LGBM':<10} {'Stack':<10} {'Δ LGBM':<8} {'Δ Stk':<8} {'Winner'}")
    log.info("-" * 70)

    avg_lgbm = 0
    avg_stacking = 0
    count = 0

    for target in TARGET_COLS:
        v10 = V10_SCORES[target]
        lgbm = lgbm_results.get(target, {}).get('cal_oof', None)
        stack = stacking_results.get(target, {}).get('stacking_cal_oof', None)

        if lgbm is not None: avg_lgbm += lgbm
        if stack is not None: avg_stacking += stack
        count += 1

        if lgbm is not None and stack is not None:
            dl = lgbm - v10
            ds = stack - v10
            if stack <= lgbm and stack <= v10:
                winner = "Stack"
            elif lgbm <= v10:
                winner = "LGBM"
            else:
                winner = "V10"
            log.info(f"{target:<6} {v10:<10.4f} {lgbm:<10.4f} {stack:<10.4f} {dl:+.4f} {ds:+.4f} {winner}")

    if count > 0:
        avg_lgbm /= count
        avg_stacking /= count

    log.info("-" * 70)
    avg_v10 = V10_AVG
    dl_avg = avg_lgbm - avg_v10
    ds_avg = avg_stacking - avg_v10
    log.info(f"{'AVG':<6} {avg_v10:<10.4f} {avg_lgbm:<10.4f} {avg_stacking:<10.4f} {dl_avg:+.4f} {ds_avg:+.4f}")

    best_avg = min(avg_lgbm, avg_stacking)
    if best_avg < avg_v10:
        best_type = "LGBM" if avg_lgbm < avg_stacking else "Stack"
        log.info(f"\n🎉 V11 BEATS V10! ({best_type} avg={best_avg:.4f}, Δ={avg_v10 - best_avg:.4f})")
    else:
        log.info(f"\nV10 still better by {min(avg_v10 - avg_lgbm, avg_v10 - avg_stacking):.4f}")

    # ── 6. Save ─────────────────────────────────────────
    results_path = MODEL_DIR / 'v11_final_results.json'
    with open(results_path, 'w') as f:
        data = {}
        for t in TARGET_COLS:
            if t in lgbm_results:
                r = lgbm_results[t]
                data[f'lgbm_{t}'] = {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                      for k, v in r.items() if k not in ('cal_preds', 'oof_preds', 'config')}
            if t in stacking_results:
                data[f'stack_{t}'] = stacking_results[t]
        json.dump(data, f, indent=2, default=float)
    log.info(f"Saved: {results_path}")

    meta = {
        'version': 'v11_final',
        'features_file': str(DATA_PROCESSED / "features_v11_personalized.parquet"),
        'n_base_features': len(base_cols),
        'n_splits': N_SPLITS,
        'n_seeds': N_SEEDS,
        'n_configs': len(LGB_CONFIGS),
        'avg_v10': float(avg_v10),
        'avg_lgbm': float(avg_lgbm),
        'avg_stacking': float(avg_stacking),
        'beat_v10': bool(best_avg < avg_v10),
        'per_target': {},
    }
    for target in TARGET_COLS:
        if target in lgbm_results:
            meta['per_target'][f'lgbm_{target}'] = {
                'v10': lgbm_results[target]['v10'],
                'v11': lgbm_results[target]['cal_oof'],
                'n_features': lgbm_results[target]['n_features'],
                'config': lgbm_results[target]['config']['name'],
            }
    meta_path = MODEL_DIR / 'v11_final_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
