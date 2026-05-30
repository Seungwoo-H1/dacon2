"""
V154 C=100 Submit Only — regenerate submission from completed log

V154 was killed mid-C-sweep. C=100 completed with:
  Q1: 0.64932, Q2: 0.61176, Q3: 0.61872, S1: 0.52363,
  S2: 0.58400, S3: 0.55906, S4: 0.61780
  AVG OOF: 0.59490

This script re-runs ONLY C=100 to produce a submission file
for OOF-LB gap checking.
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
SUBMIT.mkdir(exist_ok=True)

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
N_SEEDS = 12
TOP_K = 30
META_C = 100.0

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


t_start = time.time()
log.info("V154 C=100 Submit Only — 48 students, 12 seeds")

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols = get_feature_cols(train_df)
log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

# Build features
group = train_df['subject_id'].values
gkf = GroupKFold(n_splits=N_FOLDS)

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

# Run V154 for C=100 only
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

        # LGBM Wide
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

        # LGBM Deep
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

        # XGBoost
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

        # CatBoost
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
    meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
    meta.fit(stacked, y)

    train_oof[t] = meta.predict_proba(stacked)[:, 1]
    ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
    test_stacked = np.column_stack(all_test_preds)
    test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    log.info(f"    Stacking OOF (C={META_C}, {len(all_student_oofs)} students, top-{TOP_K}): {ll:.5f}")

avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                  for t in TARGETS])
log.info(f"\nMeta C={META_C}: AVG OOF={avg_oof:.5f}")

# Create submission
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame()
sub['subject_id'] = test_df['subject_id'].values
sub['sleep_date'] = test_df['sleep_date'].values
sub['lifelog_date'] = test_df['lifelog_date'].values
for t in TARGETS:
    sub[t] = test_preds[t]

sub_path = SUBMIT / f"submission_v154_c100_submit_only_{ts}.csv"
sub.to_csv(sub_path, index=False)
log.info(f"Saved: {sub_path}")

# Per-target OOF
per_target_oof = {}
for t in TARGETS:
    oof = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    per_target_oof[t] = round(oof, 5)
    log.info(f"  {t} OOF: {oof:.5f}")

meta_data = {
    'version': 'V154_C100_SubmitOnly',
    'name': 'V154 More Seeds, C=100 only (regenerated for OOF-LB check)',
    'avg_oof': round(float(avg_oof), 5),
    'meta_c': META_C,
    'n_students_per_target': N_SEEDS * 4,
    'top_k': TOP_K,
    'per_target_oof': per_target_oof,
    'v152_avg_oof': 0.61207,
    'delta_vs_v152': round(float(avg_oof - 0.61207), 5),
    'v140_avg_oof': 0.64110,
    'delta_vs_v140': round(float(avg_oof - 0.64110), 5),
    'submission_file': str(sub_path),
    'timestamp': ts,
    'total_time_s': round(time.time() - t_start, 0),
}

meta_path = SUBMIT / f'meta_v154_c100_{ts}.json'
with open(meta_path, 'w') as f:
    json.dump(meta_data, f, indent=2)
log.info(f"Saved: {meta_path}")

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
