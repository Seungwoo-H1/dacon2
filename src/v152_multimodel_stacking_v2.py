"""
V152 — Multi-Model Stacking v2: More Seeds + Higher C Sweep + Larger Feature Set

Builds on V151 (OOF 0.61865, meta C=100).
Improvements:
1. Seeds: 5 → 7 per model family (28 students: 2 LGBM + XGB + CB × 7)
2. Meta C sweep: 100, 300, 500, 1000, 3000, 5000
3. Feature count: top 25 → top 30 per target
4. Same architecture, same base features as V151
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_step_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 7  # 7 seeds per model family
META_C_SWEEP = [100.0, 300.0, 500.0, 1000.0, 3000.0, 5000.0]
TOP_K = 30  # top features per target


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def add_target_encoding(df, target):
    global_mean = df[target].mean()
    group_counts = df.groupby('subject_id')[target].transform('count')
    group_sums = df.groupby('subject_id')[target].transform('sum')
    k = 5
    enc = (group_sums + k * global_mean) / (group_counts + k)
    df[f"{target}_enc"] = enc
    return df


def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.05,
              'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


# V151 default configs
LGBM_WIDE = {'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.05, 'n_estimators': 500,
             'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
             'min_child_samples': 5}
LGBM_DEEP = {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 1000,
             'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
             'min_child_samples': 15}

XGB_PARAMS = {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8,
              'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
              'random_state': SEED, 'verbosity': 0}

CB_PARAMS = {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0,
             'random_strength': 1.0, 'random_state': SEED, 'silent': True}


def v152_run(train_df, test_df, feat_cols):
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Add target encoding + group z-scores (same as V151)
    for t in TARGETS:
        train_df = add_target_encoding(train_df, t)
    
    feat_aug = feat_cols.copy()
    candidates = [c for c in feat_cols if any(x in c for x in ['mean', 'std', 'sum'])
                  and not any(x in c for x in ['subject_id', 'sleep_date', 'lifelog_date'])][:15]
    for c in candidates:
        grp_mean = train_df.groupby('subject_id')[c].transform('mean')
        grp_std = train_df.groupby('subject_id')[c].transform('std').fillna(1e-8)
        train_df[f"{c}_zgrp"] = (train_df[c] - grp_mean) / grp_std
        feat_aug.append(f"{c}_zgrp")
    
    target_enc_cols = [f"{t}_enc" for t in TARGETS]
    all_train_cols = feat_aug + target_enc_cols
    all_test_cols = feat_aug
    
    common_cols = [c for c in all_train_cols if c in all_test_cols]
    
    log.info(f"Features: base={len(feat_cols)}, zgrp={len(candidates)}, enc={len(target_enc_cols)}")
    log.info(f"Common (train∩test): {len(common_cols)}")
    log.info(f"Seeds: {N_SEEDS}, Students/target: {N_SEEDS*4}, Top-K: {TOP_K}")
    
    # Run stacking for each target with each META_C
    best_results = {}
    
    for meta_c in META_C_SWEEP:
        log.info(f"\n{'='*70}")
        log.info(f"Running with meta C={meta_c}")
        log.info(f"{'='*70}")
        
        train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
        test_preds = {t: np.zeros((len(test_df), N_SEEDS * 4)) for t in TARGETS}
        
        for t in TARGETS:
            log.info(f"\n--- {t} ---")
            y = train_df[t].values.astype(np.float64)
            feat_cols_clean = remove_leak(all_train_cols, t)
            common_clean = [c for c in feat_cols_clean if c in all_test_cols]
            
            ranked = rank_features(train_df, common_clean, t)
            sel_cols = ranked[:TOP_K]
            
            all_student_oofs = []
            all_test_preds = []
            
            for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
                seed_oofs = []
                seed_tests = []
                
                # === LGBM Wide ===
                spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                lgb_params = {**LGBM_WIDE, 'scale_pos_weight': spw, 'random_state': seed,
                             'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                
                lgb_oof = np.zeros(len(train_df))
                lgb_test = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    sn = [sanitize_col(str(c)) for c in range(X_tr.shape[1])]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(lgb_params, ds, num_boost_round=LGBM_WIDE['n_estimators'])
                    lgb_oof[va_idx] = m.predict(X_va)
                    lgb_test += m.predict(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64))
                lgb_oof = np.clip(lgb_oof, 0.001, 0.999)
                lgb_test /= N_FOLDS
                seed_oofs.append(lgb_oof)
                seed_tests.append(lgb_test)
                
                # === LGBM Deep ===
                lgb_params_d = {**LGBM_DEEP, 'scale_pos_weight': spw, 'random_state': seed,
                               'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                lgb_oof_d = np.zeros(len(train_df))
                lgb_test_d = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    sn = [sanitize_col(str(c)) for c in range(X_tr.shape[1])]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(lgb_params_d, ds, num_boost_round=LGBM_DEEP['n_estimators'])
                    lgb_oof_d[va_idx] = m.predict(X_va)
                    lgb_test_d += m.predict(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64))
                lgb_oof_d = np.clip(lgb_oof_d, 0.001, 0.999)
                lgb_test_d /= N_FOLDS
                seed_oofs.append(lgb_oof_d)
                seed_tests.append(lgb_test_d)
                
                # === XGBoost ===
                xgb_oof = np.zeros(len(train_df))
                xgb_test = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_va)
                    m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=500)
                    xgb_oof[va_idx] = m.predict(dval)
                    xgb_test += m.predict(xgb.DMatrix(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64)))
                xgb_oof = np.clip(xgb_oof, 0.001, 0.999)
                xgb_test /= N_FOLDS
                seed_oofs.append(xgb_oof)
                seed_tests.append(xgb_test)
                
                # === CatBoost ===
                cb_oof = np.zeros(len(train_df))
                cb_test = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    dtrain = cb.Pool(X_tr, label=y_tr)
                    dval = cb.Pool(X_va)
                    m = cb.CatBoostRegressor(**CB_PARAMS, iterations=500)
                    m.fit(dtrain, eval_set=dval, use_best_model=False)
                    cb_oof[va_idx] = m.predict(dval)
                    cb_test += m.predict(cb.Pool(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64)))
                cb_oof = np.clip(cb_oof, 0.001, 0.999)
                cb_test /= N_FOLDS
                seed_oofs.append(cb_oof)
                seed_tests.append(cb_test)
                
                all_student_oofs.extend(seed_oofs)
                all_test_preds.extend(seed_tests)
                
                avg_seed_oof = np.mean([log_loss(y, s) for s in seed_oofs])
                log.info(f"    Seed {si}: avg OOF={avg_seed_oof:.5f}")
            
            stacked = np.column_stack(all_student_oofs)
            meta = LogisticRegression(C=meta_c, max_iter=2000, random_state=SEED)
            meta.fit(stacked, y)
            
            train_oof[t] = meta.predict_proba(stacked)[:, 1]
            ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
            test_stacked = np.column_stack(all_test_preds)
            test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
            log.info(f"    Stacking OOF (C={meta_c}, {len(all_student_oofs)} students, top-{TOP_K}): {ll:.5f}")
        
        avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                          for t in TARGETS])
        log.info(f"\nMeta C={meta_c}: AVG OOF={avg_oof:.5f}")
        best_results[meta_c] = (avg_oof, train_oof.copy(), {k: v.copy() for k, v in test_preds.items()})
    
    # Pick best META_C
    best_c = min(best_results, key=lambda k: best_results[k][0])
    best_oof, best_train_oof, best_test_preds = best_results[best_c]
    
    log.info(f"\n{'='*70}")
    log.info(f"Best META_C: {best_c} → AVG OOF: {best_oof:.5f}")
    log.info(f"V151 AVG OOF: 0.61865, Δ: {best_oof - 0.61865:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = best_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v152_multimodel_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    meta_data = {
        'version': 'V152',
        'name': 'Multi-Model Stacking v2 (7 seeds, top-30, C sweep 100-5000)',
        'avg_oof': round(float(best_oof), 5),
        'best_meta_c': best_c,
        'n_students_per_target': N_SEEDS * 4,
        'top_k': TOP_K,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(best_train_oof[t], 0.001, 0.999))), 5)
                          for t in TARGETS},
        'v151_avg_oof': 0.61865,
        'delta_vs_v151': round(float(best_oof - 0.61865), 5),
        'v150_avg_oof': 0.62040,
        'delta_vs_v150': round(float(best_oof - 0.62040), 5),
        'all_meta_c_results': {str(k): round(float(v[0]), 5) for k, v in best_results.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v152_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return best_oof, meta_data


t_start = time.time()
log.info("=" * 70)
log.info("V152 — Multi-Model Stacking v2 (7 seeds, top-30, C sweep)")
log.info("=" * 70)

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols = get_feature_cols(train_df)
log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

avg_oof, meta = v152_run(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
