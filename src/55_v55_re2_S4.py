"""
V55_re2_S4 — Quick S4 only (base + pairwise + transforms)
Skips ranking, uses pre-ranked features. Fast config only.
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

TARGET_COLS = ['S4']
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
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_EXTRA_DEEP = {'nl': 25, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.6, 'cb': 0.5, 'ra': 0.1, 'rl': 1.0, 'mc': 25}

CFGS = {'deep': CFG_DEEP, 'v48': CFG_V48, 'safety': CFG_SAFETY, 'extra_deep': CFG_EXTRA_DEEP}
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
    names = []
    arrays = []
    for f in top_features[:10]:
        if f in feat.columns:
            vals = feat[f].fillna(0).values
            arrays.append(vals.reshape(-1,1))
            names.append(f)
    n = len(arrays)
    for i in range(n):
        for j in range(i+1, n):
            names.append(f'{names[i]}_x_{names[j]}')
            arrays.append(arrays[i] * arrays[j])
    for i in range(n):
        for j in range(n):
            if i != j:
                names.append(f'{names[i]}_div_{names[j]}')
                arrays.append(arrays[i] / (arrays[j] + 1e-8))
    for i in range(min(5, n)):
        names.append(f'{names[i]}_sq')
        arrays.append(arrays[i] ** 2)
    return np.column_stack(arrays), names

def build_transformed_matrix(feat, top_features):
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

def isotonic_calibrate(oof_preds, y_true):
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(oof_preds, y_true)
        return iso.predict(oof_preds), True
    except Exception:
        return oof_preds, False

def train_cv_fast(X, y, seeds, cfg, subject_ids, n_features, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
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
            ds = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X[va_idx], label=y[va_idx], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_idx, si] = m.predict(X[va_idx])
            del ds, vd, m
    return oof

def main():
    t_start = time.time()
    log.info("V55_re2_S4 — Quick S4 only")

    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    feat_cols_raw = get_feature_cols(feat)
    feat, _ = add_personalization(feat, feat_cols_raw)
    all_cols = feat_cols_raw + [f'{c}_zscore' for c in feat_cols_raw]

    target = 'S4'
    y = feat[target].values.astype(np.float64)
    train_rate = y.mean()

    subject_ids = feat['subject_id'].values
    
    # Rank features by LGBM gain importance (same as V55_re2)
    leak_cols = [c for c in all_cols if c not in LEAK_S and c not in {'S2', 'S3', 'S4'} and not c.startswith('S2_') and not c.startswith('S3_') and not c.startswith('S4_')]
    feat_vals_base = {c: feat[c].fillna(0).values for c in leak_cols}
    ranked = sorted(feat_vals_base.keys(), key=lambda f: 0)  # placeholder
    import lightgbm as lgb
    X_base = np.column_stack(list(feat_vals_base.values())).astype(np.float64)
    ds_base = lgb.Dataset(X_base, label=y, feature_name=[sanitize(c) for c in leak_cols], params={'verbose': '-1'})
    model_base = lgb.train({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                            'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
                            'reg_alpha': 1.0, 'reg_lambda': 3.0, 'scale_pos_weight': max(((y==0).sum())/max((y==1).sum(),1), 0.1),
                            'random_state': 42, 'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1},
                           ds_base, num_boost_round=50)
    imp = model_base.feature_importance(importance_type='gain')
    ranked = [f for f, _ in sorted(zip(leak_cols, imp), key=lambda x: -x[1])]
    del model_base, ds_base, X_base
    gc.collect()
    
    log.info(f"  Ranked top-5: {ranked[:5]}")

    best_overall_cal = float('inf')
    best_overall_oof = None
    best_config_str = None

    # Base
    log.info("  [A] Base...")
    for cfg_name, cfg in CFGS.items():
        for n_feat in [8, 10, 15, 20]:
            sel_cols = ranked[:n_feat]
            X = np.column_stack([feat[c].fillna(0).values for c in sel_cols]).astype(np.float64)
            oof = train_cv_fast(X, y, SEEDS, cfg, subject_ids, n_feat)
            del X
            gc.collect()
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            if ok:
                iso_cal = mean_match(iso_cal, train_rate)
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

    # Pairwise
    log.info("  [B] Pairwise...")
    pair_matrix, pair_names = build_pairwise_matrix(feat, ranked[:10])
    variances = pair_matrix.var(axis=0)
    valid_mask = variances > 1e-10
    pair_matrix = pair_matrix[:, valid_mask]
    pair_names = [n for n, v in zip(pair_names, valid_mask) if v]
    # Rank only top 100 to save time
    feat_vals = {n: pair_matrix[:, i] for i, n in enumerate(pair_names[:100])}
    ranked_pair = sorted(feat_vals.keys(),
                        key=lambda f: -np.corrcoef(feat_vals[f], y)[:1,1].max() if len(set(np.unique(feat_vals[f]))) > 1 else 0)
    gc.collect()
    
    for cfg_name, cfg in CFGS.items():
        for n_feat in [8, 10, 15, 20]:
            sel_names = ranked_pair[:n_feat]
            X = np.column_stack([feat_vals[n] for n in sel_names if n in feat_vals]).astype(np.float64)
            if X.shape[1] != n_feat:
                continue
            oof = train_cv_fast(X, y, SEEDS, cfg, subject_ids, n_feat)
            del X
            gc.collect()
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            if ok:
                iso_cal = mean_match(iso_cal, train_rate)
                iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
            else:
                iso_cal_loss = log_loss(y, oof_avg, labels=[0, 1])
                iso_cal = oof_avg
            config_str = f"pair_{cfg_name}_n{n_feat}"
            if iso_cal_loss < best_overall_cal:
                best_overall_cal = iso_cal_loss
                best_overall_oof = iso_cal
                best_config_str = config_str

    del pair_matrix
    gc.collect()

    # Transforms
    log.info("  [C] Transforms...")
    trans_matrix, trans_names = build_transformed_matrix(feat, ranked[:15])
    variances = trans_matrix.var(axis=0)
    valid_mask = variances > 1e-10
    trans_matrix = trans_matrix[:, valid_mask]
    trans_names = [n for n, v in zip(trans_names, valid_mask) if v]
    feat_vals = {n: trans_matrix[:, i] for i, n in enumerate(trans_names[:100])}
    ranked_trans = sorted(feat_vals.keys(),
                          key=lambda f: -np.corrcoef(feat_vals[f], y)[:1,1].max() if len(set(np.unique(feat_vals[f]))) > 1 else 0)
    gc.collect()

    for cfg_name, cfg in CFGS.items():
        for n_feat in [8, 10, 15, 20]:
            sel_names = ranked_trans[:n_feat]
            X = np.column_stack([feat_vals[n] for n in sel_names if n in feat_vals]).astype(np.float64)
            if X.shape[1] != n_feat:
                continue
            oof = train_cv_fast(X, y, SEEDS, cfg, subject_ids, n_feat)
            del X
            gc.collect()
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            if ok:
                iso_cal = mean_match(iso_cal, train_rate)
                iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
            else:
                iso_cal_loss = log_loss(y, oof_avg, labels=[0, 1])
                iso_cal = oof_avg
            config_str = f"trans_{cfg_name}_n{n_feat}"
            if iso_cal_loss < best_overall_cal:
                best_overall_cal = iso_cal_loss
                best_overall_oof = iso_cal
                best_config_str = config_str

    log.info(f"    ✅ S4 Best: {best_config_str} Cal={best_overall_cal:.4f}")

    # Combine with previous results
    v53_avg = 0.5479
    partial_results = {
        'Q1': 0.5834, 'Q2': 0.5589, 'Q3': 0.5780,
        'S1': 0.5282, 'S2': 0.5256, 'S3': 0.5185,
        'S4': best_overall_cal,
    }
    avg_cal = np.mean(list(partial_results.values()))
    
    log.info(f"\n{'='*70}")
    log.info("V55_re2 Summary (S4 completed)")
    log.info(f"{'='*70}")
    for t, cal in partial_results.items():
        log.info(f"  {t}: Cal={cal:.4f}")
    log.info(f"\n  V55_re2 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V53 Avg Cal: {v53_avg:.4f}")
    log.info(f"  Δ vs V53: {avg_cal - v53_avg:+.4f} ({'✅ IMPROVED' if avg_cal < v53_avg else '❌ Not improved'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # Save OOF
    oof_path = DATA_PROCESSED / "oof_v55_re2_S4.csv"
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
        'S4': best_overall_oof,
    })
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    meta = {
        'version': 'V55_re2',
        'avg_cal_loss': avg_cal,
        'v53_cal_loss': v53_avg,
        'delta_v53': avg_cal - v53_avg,
        'per_target': {t: {'cal_loss': c} for t, c in partial_results.items()},
    }
    meta_path = DATA_PROCESSED / "v55_re2_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()
