"""
V55_re2 — Memory-safe Pairwise + Transformed + Expanded Configs

V55 failed with SIGKILL on S2. Fix:
1. Use np.column_stack for interactions instead of full DataFrame copy
2. Rank features only on top-N candidates to reduce dataset size
3. Delete intermediate DataFrames immediately
4. Keep only top-20 ranked features for training
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_EXTRA_DEEP = {'nl': 25, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.6, 'cb': 0.5, 'ra': 0.1, 'rl': 1.0, 'mc': 25}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V53SAFE = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}

CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48, 'safety': CFG_SAFETY,
        'extra_deep': CFG_EXTRA_DEEP, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP, 'v53safe': CFG_V53SAFE}

SEEDS = list(range(1, 51))

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    personal_cols = []
    df = df.copy()
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
    return df, personal_cols

def build_pairwise_matrix(feat, top_features):
    """Return (X_matrix, feature_names) without creating extra DataFrame."""
    names = []
    arrays = []
    for f in top_features[:10]:
        if f in feat.columns:
            vals = feat[f].fillna(0).values
            arrays.append(vals.reshape(-1,1))
            names.append(f)
    n = len(arrays)
    # pairwise products
    for i in range(n):
        for j in range(i+1, n):
            names.append(f'{names[i]}_x_{names[j]}')
            arrays.append(arrays[i] * arrays[j])
    # pairwise ratios
    for i in range(n):
        for j in range(n):
            if i != j:
                names.append(f'{names[i]}_div_{names[j]}')
                arrays.append(arrays[i] / (arrays[j] + 1e-8))
    # squared
    for i in range(min(5, n)):
        names.append(f'{names[i]}_sq')
        arrays.append(arrays[i] ** 2)
    return np.column_stack(arrays), names

def build_transformed_matrix(feat, top_features):
    """Return (X_matrix, feature_names) for log/sqrt/abs transforms."""
    names = []
    arrays = []
    for f in top_features[:15]:
        if f in feat.columns:
            vals = feat[f].fillna(0).values
            vals_abs = np.abs(vals) + 1e-8
            names.append(f'{f}_log')
            arrays.append(np.sign(vals) * np.log1p(vals_abs))
            names.append(f'{f}_sqrt')
            arrays.append(np.sign(vals) * np.sqrt(vals_abs))
            names.append(f'{f}_abs')
            arrays.append(np.abs(vals))
    return np.column_stack(arrays), names

def rank_features_ranking(y, col_values_dict, n_rank=200, seed=42):
    """Rank features by LGBM importance using numpy arrays directly."""
    # Build X from dict of arrays
    feat_names = list(col_values_dict.keys())
    X = np.column_stack([col_values_dict[f] for f in feat_names]).astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_names]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_names, imp), key=lambda x: -x[1])
    del model, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def train_cv_from_matrix(X_train_list, X_val_list, y, seeds, cfg, n_folds=5):
    """Train CV models when features are numpy arrays."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    n_features = X_train_list[0].shape[1]
    sn = [f'f{i}' for i in range(n_features)]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }

    for si, seed in enumerate(seeds):
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(y)), y, subject_ids)):
            X_tr = np.vstack([X_train_list[tr_idx[fold_idx]]])
            X_va = np.vstack([X_val_list[va_idx[fold_idx]]])
            ds = lgb.Dataset(X_tr, label=y[tr_idx], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va_idx], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_idx, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va

    return oof

def isotonic_calibrate(oof_preds, y_true):
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(oof_preds, y_true)
        return iso.predict(oof_preds), True
    except Exception:
        return oof_preds, False

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V55_re2 — Memory-safe Pairwise + Transformed + Expanded Configs")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Loaded: {feat.shape}")

    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape}")

    all_cols = feat_cols_raw + zscore_cols
    train_rates = {t: feat[t].mean() for t in TARGET_COLS}

    # ── 2. Feature ranking on base features ──
    log.info("\n--- 2. Feature ranking ---")
    ranked_lgb = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        y = feat[target].values.astype(np.float64)
        ranked = rank_features_ranking(y, {c: feat[c].fillna(0).values for c in leak_cols}, seed=42)
        ranked_lgb[target] = ranked
        log.info(f"  {target} top-5: {ranked[:5]}")
    gc.collect()

    # ── 3. Experiments ──
    log.info("\n--- 3. Multi-strategy experiments ---")

    all_results = {}

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        ranked = ranked_lgb[target]

        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}) ===")

        best_overall_cal = float('inf')
        best_overall_oof = None
        best_config_str = None

        # Strategy A: Base (no interactions) — use column_dict approach
        log.info(f"    [A] Base...")
        subject_ids = feat['subject_id'].values
        base_dict = {c: feat[c].fillna(0).values for c in ranked[:20]}
        for cfg_name, cfg in CFGS.items():
            for n_feat in [8, 10, 15, 20]:
                sel_dict = {f: base_dict[f] for f in ranked[:n_feat]}
                X = np.column_stack(list(sel_dict.values())).astype(np.float64)
                # Prepare OOF
                oof = np.zeros((len(y), len(SEEDS)))
                spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                sn = [f'f{i}' for i in range(n_feat)]
                cfg_full = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                }
                gkf = GroupKFold(n_splits=5)
                subject_ids = feat['subject_id'].values
                for si, seed in enumerate(SEEDS):
                    cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
                    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(y)), y, subject_ids)):
                        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn, params={'verbose': '-1'})
                        vd = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=sn, reference=ds, params={'verbose': '-1'})
                        m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                                     valid_sets=[vd],
                                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                        oof[va_idx, si] = m.predict(X[va_idx])
                        del ds, vd, m
                del X
                gc.collect()

                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                iso_cal, ok = isotonic_calibrate(oof_avg, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    iso_cal_loss = log_loss(y, oof_avg, labels=[0, 1])
                    iso_cal = oof_avg

                config_str = f"base_{cfg_name}_n{n_feat}"
                if iso_cal_loss < best_overall_cal:
                    best_overall_cal = iso_cal_loss
                    best_overall_oof = iso_cal
                    best_config_str = config_str

        gc.collect()

        # Strategy B: Pairwise interactions — build matrix once
        log.info(f"    [B] Pairwise...")
        top_feats = ranked[:10]
        pair_matrix, pair_names = build_pairwise_matrix(feat, top_feats)
        gc.collect()
        
        # Keep only features with some variance
        variances = pair_matrix.var(axis=0)
        valid_mask = variances > 1e-10
        pair_matrix = pair_matrix[:, valid_mask]
        pair_names = [n for n, v in zip(pair_names, valid_mask) if v]
        
        # Rank pairwise features
        ranked_pair = rank_features_ranking(y, {n: pair_matrix[:, i] for i, n in enumerate(pair_names[:200])}, seed=42)
        gc.collect()

        for cfg_name, cfg in CFGS.items():
            for n_feat in [8, 10, 15, 20]:
                sel_names = ranked_pair[:n_feat]
                sel_dict = {n: pair_matrix[:, [i for i, nm in enumerate(pair_names) if nm == n][0]] for n in sel_names if n in pair_names}
                X = np.column_stack(list(sel_dict.values())).astype(np.float64)
                
                oof = np.zeros((len(y), len(SEEDS)))
                spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                sn = [f'f{i}' for i in range(n_feat)]
                cfg_full = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                }
                gkf = GroupKFold(n_splits=5)
                for si, seed in enumerate(SEEDS):
                    cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
                    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(y)), y, subject_ids)):
                        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn, params={'verbose': '-1'})
                        vd = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=sn, reference=ds, params={'verbose': '-1'})
                        m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                                     valid_sets=[vd],
                                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                        oof[va_idx, si] = m.predict(X[va_idx])
                        del ds, vd, m
                del X
                gc.collect()

                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                iso_cal, ok = isotonic_calibrate(oof_avg, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    iso_cal_loss = log_loss(y, oof_avg, labels=[0, 1])
                    iso_cal = oof_avg

                config_str = f"pair_{cfg_name}_n{n_feat}"
                if iso_cal_loss < best_overall_cal:
                    best_overall_cal = iso_cal_loss
                    best_overall_oof = iso_cal
                    best_config_str = config_str

        del pair_matrix, pair_names, sel_dict
        gc.collect()

        # Strategy C: Transformed features — build matrix once
        log.info(f"    [C] Transforms...")
        trans_matrix, trans_names = build_transformed_matrix(feat, ranked[:15])
        gc.collect()
        
        # Filter zero-variance features
        variances = trans_matrix.var(axis=0)
        valid_mask = variances > 1e-10
        trans_matrix = trans_matrix[:, valid_mask]
        trans_names = [n for n, v in zip(trans_names, valid_mask) if v]
        
        ranked_trans = rank_features_ranking(y, {n: trans_matrix[:, i] for i, n in enumerate(trans_names[:200])}, seed=42)
        gc.collect()

        for cfg_name, cfg in CFGS.items():
            for n_feat in [8, 10, 15, 20]:
                sel_names = ranked_trans[:n_feat]
                sel_dict = {n: trans_matrix[:, [i for i, nm in enumerate(trans_names) if nm == n][0]] for n in sel_names if n in trans_names}
                X = np.column_stack(list(sel_dict.values())).astype(np.float64)
                
                oof = np.zeros((len(y), len(SEEDS)))
                spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                sn = [f'f{i}' for i in range(n_feat)]
                cfg_full = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                }
                gkf = GroupKFold(n_splits=5)
                for si, seed in enumerate(SEEDS):
                    cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
                    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros(len(y)), y, subject_ids)):
                        ds = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn, params={'verbose': '-1'})
                        vd = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=sn, reference=ds, params={'verbose': '-1'})
                        m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                                     valid_sets=[vd],
                                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                        oof[va_idx, si] = m.predict(X[va_idx])
                        del ds, vd, m
                del X
                gc.collect()

                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                iso_cal, ok = isotonic_calibrate(oof_avg, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    iso_cal_loss = log_loss(y, oof_avg, labels=[0, 1])
                    iso_cal = oof_avg

                config_str = f"trans_{cfg_name}_n{n_feat}"
                if iso_cal_loss < best_overall_cal:
                    best_overall_cal = iso_cal_loss
                    best_overall_oof = iso_cal
                    best_config_str = config_str

        del trans_matrix, trans_names, sel_dict
        gc.collect()

        log.info(f"    ✅ Best: {best_config_str} Cal={best_overall_cal:.4f}")

        all_results[target] = {
            'best_method': best_config_str,
            'cal_oof': best_overall_oof,
            'cal_loss': best_overall_cal,
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V55_re2 SUMMARY")
    log.info(f"{'='*70}")

    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['best_method']} Cal={r['cal_loss']:.4f}")

    avg_cal = np.mean([
        log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V55_re2 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V53 Avg Cal: 0.5479")
    log.info(f"  Δ vs V53: {avg_cal - 0.5479:+.4f} ({'✅ IMPROVED' if avg_cal < 0.5479 else '❌ Not improved'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    oof_path = DATA_PROCESSED / "oof_v55_re2.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    meta = {
        'version': 'V55_re2',
        'name': 'Memory-safe Pairwise + Transformed + Expanded Configs',
        'avg_cal_loss': avg_cal,
        'v53_cal_loss': 0.5479,
        'delta_v53': avg_cal - 0.5479,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'best_method': r['best_method'],
            'cal_loss': r['cal_loss'],
        }
    meta_path = DATA_PROCESSED / "v55_re2_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")

    return all_results

if __name__ == "__main__":
    main()
