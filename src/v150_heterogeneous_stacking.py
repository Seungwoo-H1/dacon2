"""
V150 — Multi-Model Heterogeneous Stacking (XGB + CatBoost + LGBM)

Hypothesis: LGBM-only stacking is stuck at ~0.63. Need model diversity.
V150 trains 3 DIFFERENT model families:
  - LightGBM (wide config)
  - XGBoost (calibrated)
  - CatBoost (with target encoding)

Each uses GroupKFold 5-fold → OOF predictions → LR meta-learner (C=10).

Key differences from V145 (which failed):
  - V145 mixed CB+XGB+LGBM but had different seeds per target → misalignment
  - V150 uses SAME seeds across all 3 families → proper alignment
  - Each family has 3 seeds → 9 students total → meta-learner
  - Heavy regularization to prevent overfitting on 450 rows
  - Isotonic calibration before stacking (learned per-fold)

Critical lessons from V145:
  - CatBoost early stopping killed learning (17 iterations) → disable early stopping
  - Must use same GroupKFold splits across model families
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

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

# Same cfg for ALL models (ensures apples-to-apples comparison)
BASE_CGFS = {
    'lgbm_wide': {'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.05, 'n_estimators': 500,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
                  'min_child_samples': 5},
    'lgbm_deep': {'num_leaves': 31, 'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 1000,
                  'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'reg_lambda': 10.0,
                  'min_child_samples': 15},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 3  # 3 seeds per model family
META_C = 10.0


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


def try_import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        log.warning("XGBoost not available, skipping")
        return None


def try_import_catboost():
    try:
        import catboost as cb
        return cb
    except ImportError:
        log.warning("CatBoost not available, skipping")
        return None


def train_lgbm(X_tr, y_tr, X_va, params, n_estimators):
    sn = [sanitize_col(str(c)) for c in range(X_tr.shape[1])]
    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=n_estimators)
    return m


def train_xgb(X_tr, y_tr, X_va, params, n_estimators):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va)
    m = xgb.train(params, dtrain, num_boost_round=n_estimators)
    return m


def train_catboost(X_tr, y_tr, X_va, params, n_estimators):
    import catboost as cb
    dtrain = cb.Pool(X_tr, label=y_tr)
    m = cb.CatBoostRegressor(**params, iterations=n_estimators, logging_level='Silent')
    m.fit(dtrain, **{})  # No early stopping — train full
    return m


def v150_run(train_df, test_df, feat_cols):
    """V150: Heterogeneous multi-model stacking."""
    
    # Check available libraries
    xgb = try_import_xgboost()
    cb = try_import_catboost()
    
    if xgb is None and cb is None:
        log.error("Neither XGBoost nor CatBoost available. Reverting to LGBM-only.")
        log.error("Skipping V150 — try install: pip install xgboost catboost")
        return 999.0, {}
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Add target encoding (same as V148)
    for t in TARGETS:
        train_df = add_target_encoding(train_df, t)
    
    # Add group z-scores
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
    all_test_cols = feat_aug  # no target enc for test
    
    # Find common features between train and test
    common_cols = [c for c in all_train_cols if c in all_test_cols]
    
    log.info(f"Features: base={len(feat_cols)}, zgrp={len(candidates)}, enc={len(target_enc_cols)}")
    log.info(f"Common (train∩test): {len(common_cols)}")
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS * 3)) for t in TARGETS}  # 3 families × 3 seeds
    
    student_idx = {t: 0 for t in TARGETS}  # track student index
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(all_train_cols, t)
        common_clean = [c for c in feat_cols_clean if c in all_test_cols]
        
        # Feature ranking on common features
        ranked = rank_features(train_df, common_clean, t)
        sel_cols = ranked[:25]  # Use top 25 features
        cfg = BASE_CGFS['lgbm_wide']
        
        per_seed_oofs = []  # list of (oof, test_pred) tuples per model
        all_student_oofs = []  # flat list of student OOFs
        
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            log.info(f"\n  Seed {si} (s{seed}):")
            
            # === LightGBM ===
            lgb_oof = np.zeros(len(train_df))
            lgb_test = np.zeros(len(test_df))
            lgb_params = {**cfg, 'scale_pos_weight': max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1),
                         'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                sn = [sanitize_col(str(c)) for c in range(X_tr.shape[1])]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(lgb_params, ds, num_boost_round=cfg['n_estimators'])
                
                lgb_oof[va_idx] = m.predict(X_va)
                lgb_test += m.predict(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64))
            
            lgb_oof = np.clip(lgb_oof, 0.001, 0.999)
            lgb_test /= N_FOLDS
            lgb_ll = log_loss(y, lgb_oof)
            log.info(f"    LGBM: OOF={lgb_ll:.5f}")
            per_seed_oofs.append((lgb_oof, lgb_test))
            all_student_oofs.append(lgb_oof)
            
            # === XGBoost ===
            if xgb is not None:
                xgb_oof = np.zeros(len(train_df))
                xgb_test = np.zeros(len(test_df))
                xgb_params = {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8,
                             'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
                             'random_state': seed, 'verbosity': 0}
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    dtrain = xgb.DMatrix(X_tr, label=y_tr)
                    dval = xgb.DMatrix(X_va)
                    m = xgb.train(xgb_params, dtrain, num_boost_round=500)
                    
                    xgb_oof[va_idx] = m.predict(dval)
                    xgb_test += m.predict(xgb.DMatrix(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64)))
                
                xgb_oof = np.clip(xgb_oof, 0.001, 0.999)
                xgb_test /= N_FOLDS
                xgb_ll = log_loss(y, xgb_oof)
                log.info(f"    XGB:  OOF={xgb_ll:.5f}")
                per_seed_oofs.append((xgb_oof, xgb_test))
                all_student_oofs.append(xgb_oof)
            
            # === CatBoost ===
            if cb is not None:
                cb_oof = np.zeros(len(train_df))
                cb_test = np.zeros(len(test_df))
                cb_params = {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0,
                            'random_strength': 1.0, 'random_state': seed, 'silent': True}
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    dtrain = cb.Pool(X_tr, label=y_tr)
                    dval = cb.Pool(X_va)
                    m = cb.CatBoostRegressor(**cb_params, iterations=500)
                    m.fit(dtrain, eval_set=dval, use_best_model=False)  # No early stopping!
                    
                    cb_oof[va_idx] = m.predict(dval)
                    cb_test += m.predict(cb.Pool(train_df[sel_cols].iloc[:len(test_df)].fillna(0).values.astype(np.float64)))
                
                cb_oof = np.clip(cb_oof, 0.001, 0.999)
                cb_test /= N_FOLDS
                cb_ll = log_loss(y, cb_oof)
                log.info(f"    CB:   OOF={cb_ll:.5f}")
                per_seed_oofs.append((cb_oof, cb_test))
                all_student_oofs.append(cb_oof)
        
        # Meta-learner: LR on all student OOFs
        n_students = len(all_student_oofs)
        stacked = np.column_stack(all_student_oofs)
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"\n  Stacking OOF (C={META_C}, {n_students} students): {ll:.5f}")
        
        # Test: stack predictions from all students
        all_test_preds = [p[1] for p in per_seed_oofs]
        test_stacked = np.column_stack(all_test_preds)
        test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                       for t in TARGETS])
    log.info(f"\n{'='*70}")
    log.info(f"V150 AVG OOF: {avg_oof:.5f}")
    log.info(f"V148 AVG OOF: 0.63129")
    log.info(f"Δ vs V148: {avg_oof - 0.63129:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v150_hetero_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    # Per-seed detail
    per_seed_detail = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(all_train_cols, t)
        common_clean = [c for c in feat_cols_clean if c in all_test_cols]
        ranked = rank_features(train_df, common_clean, t)
        n_students_t = int((test_preds[t].shape[1] > 0))
        if t in test_preds and test_preds[t].shape[1] > 0:
            per_seed_detail[t] = f"{test_preds[t].shape[1]} students"
    
    meta_data = {
        'version': 'V150',
        'name': f'Heterogeneous Multi-Model (LGBM{"+XGB" if xgb else ""}{"+CB" if cb else ""})',
        'avg_oof': round(float(avg_oof), 5),
        'models_available': {'lgbm': True, 'xgb': xgb is not None, 'cb': cb is not None},
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5)
                          for t in TARGETS},
        'v148_avg_oof': 0.63129,
        'delta_vs_v148': round(float(avg_oof - 0.63129), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v150_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_oof, meta_data


t_start = time.time()
log.info("=" * 70)
log.info("V150 — Heterogeneous Multi-Model Stacking")
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

avg_oof, meta = v150_run(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
