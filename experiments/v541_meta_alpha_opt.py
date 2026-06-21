#!/usr/bin/env python3
"""
V541 — ElasticNet Meta with V537 features (re-evaluation with correct metric)

Key insight from V538: gap = student_ll - meta_ll is misleading for comparing meta models.
Lower gap does NOT mean better expected LB. The correct metric is meta_ll_on_train.
Lower meta_ll_on_train → LOWER expected LB → BETTER.

V537 Ridge: meta_ll_train ≈ 0.65251
V538 EN: meta_ll_train ≈ 0.66530 (WORSE)

So Ridge is better. But we should try:
1. Different alpha for ElasticNet/Lasso
2. Different alpha for Ridge
3. Weighted blend of Ridge + ElasticNet predictions
"""
import sys, gc, logging, json, re, time, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import lightgbm as lgb, xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
          'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
          'wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean',
          'wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
          'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean',
          'wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
          'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}
SEED = 42
N_FOLDS = 5
N_SEEDS = 13

# Same as V537
V534_CONFIG = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}
XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X; gc.collect()
    return [r[0] for r in ranked]

def train_one(seed, X_tr, y_tr, X_va, X_test, feat_names, learner, n_est, **mp):
    if learner == 'xgb':
        params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
        ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
        ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
        m = xgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(ds_va), m.predict(ds_te)
    else:
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
                 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feat_names]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(X_va), m.predict(X_test)


def main():
    global train_df, test_df
    t_start = time.time()
    log.info("=" * 70)
    log.info("V541: Meta Model Alpha Optimization (correct metric: meta_ll_on_train)")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # Z-score
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m_val, s_val = np.mean(tv), np.std(tv, ddof=0)
        if s_val < 1e-8: s_val = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m_val) / s_val
        test_df[zc] = (ev - m_val) / s_val

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(train_df)
    gkf = GroupKFold(n_splits=N_FOLDS)

    log.info('Pre-ranking features...')
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # Generate predictions
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}

    for target in TARGETS:
        bc = V534_CONFIG[target]
        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols): sel_cols = feat_names
        n_est = bc['n_est']
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[bc['xgb_cfg']]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[bc['lgbm_cfg']]}
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        log.info(f'\n{target}: n_est={n_est}, n_feat={bc["n_feat"]}')
        xgb_oofs, lgbm_oofs = [], []
        xgb_tests, lgbm_tests = [], []

        for si in range(N_SEEDS):
            seed = SEED + si * 11
            oof_xgb = np.zeros(n_train)
            oof_lgbm = np.zeros(n_train)
            test_xgb = np.zeros(n_test)
            test_lgbm = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                pvxgb, ttxgb = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'xgb', n_est, **xgb_params)
                pvlgbm, ttlgbm = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'lgbm', n_est, **lgbm_params)
                oof_xgb[va_idx] = pvxgb; oof_lgbm[va_idx] = pvlgbm
                test_xgb += ttxgb; test_lgbm += ttlgbm

            oof_xgb = np.clip(oof_xgb, 0.001, 0.999)
            oof_lgbm = np.clip(oof_lgbm, 0.001, 0.999)
            test_xgb /= N_FOLDS; test_lgbm /= N_FOLDS
            xgb_oofs.append(oof_xgb); lgbm_oofs.append(oof_lgbm)
            xgb_tests.append(test_xgb); lgbm_tests.append(test_lgbm)

        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10); inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        wx = inv_xgb / (inv_xgb + inv_lgbm)
        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_oofs, lgbm_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_tests, lgbm_tests)]
        all_seed_oofs[target] = blended_oofs
        all_test_preds[target] = blended_tests
        log.info(f'  {target}: wx={wx:.3f}, xgb_ll={ll_xgb:.5f}, lgbm_ll={ll_lgbm:.5f}')

    # ================================================================
    # Meta model alpha optimization
    # Metric: meta_ll_on_train (lower = better expected LB)
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V541: Meta Model Comparison (metric: meta_ll_on_train)')
    log.info('=' * 70)
    log.info('IMPORTANT: LOWER meta_ll = LOWER expected LB = BETTER')
    log.info('gap = student_ll - meta_ll → LOWER gap = HIGHER meta_ll = WORSE')
    log.info('')

    ridge_alphas = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0]
    en_alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    en_ratios = [0.2, 0.5, 0.8]
    lasso_alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]

    # Test Ridge α sweep per target
    log.info('\n--- Ridge α sweep (per target) ---')
    best_ridge_alpha = {}
    best_ridge_ll = {}
    for target in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[target])
        y_true = train_df[target].values
        n_seeds = oofs_2d.shape[1]
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        
        best_alpha = None
        best_ll = float('inf')
        for alpha in ridge_alphas:
            meta = Ridge(alpha=alpha)
            meta.fit(X_meta, y_true)
            train_pred = meta.predict(X_meta)
            pmin, pmax = train_pred.min(), train_pred.max()
            if pmax - pmin < 1e-10:
                train_proba = np.ones_like(train_pred) * 0.5
            else:
                train_proba = (train_pred - pmin) / (pmax - pmin)
            train_proba = np.clip(train_proba, 0.001, 0.999)
            ll = log_loss(y_true, train_proba)
            if ll < best_ll:
                best_ll = ll
                best_alpha = alpha
        
        best_ridge_alpha[target] = best_alpha
        best_ridge_ll[target] = best_ll
        log.info(f'  {target}: best α={best_alpha}, meta_ll={best_ll:.5f}')

    # Compute overall avg meta_ll
    avg_ridge_ll = sum(best_ridge_ll.values()) / 7
    avg_student_ll = np.mean([np.mean([log_loss(train_df[t].values, np.clip(np.column_stack(all_seed_oofs[t])[:, si], 0.001, 0.999)) for si in range(N_SEEDS)]) for t in TARGETS])
    
    log.info(f'\n  avg_student_ll={avg_student_ll:.5f}')
    log.info(f'  avg_ridge_ll={avg_ridge_ll:.5f}')
    log.info(f'  expected LB (Ridge) = {avg_ridge_ll:.5f}')

    # Test ElasticNet per target
    log.info('\n--- ElasticNet sweep (per target) ---')
    best_en_ll = {}
    for target in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[target])
        y_true = train_df[target].values
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        
        best_ll = float('inf')
        for alpha in en_alphas:
            for l1r in en_ratios:
                meta = ElasticNet(alpha=alpha, l1_ratio=l1r, random_state=SEED, max_iter=5000)
                meta.fit(X_meta, y_true)
                train_pred = meta.predict(X_meta)
                pmin, pmax = train_pred.min(), train_pred.max()
                if pmax - pmin < 1e-10:
                    train_proba = np.ones_like(train_pred) * 0.5
                else:
                    train_proba = (train_pred - pmin) / (pmax - pmin)
                train_proba = np.clip(train_proba, 0.001, 0.999)
                ll = log_loss(y_true, train_proba)
                if ll < best_ll:
                    best_ll = ll
        best_en_ll[target] = best_ll
        log.info(f'  {target}: meta_ll={best_ll:.5f}')

    avg_en_ll = sum(best_en_ll.values()) / 7
    log.info(f'  avg_en_ll={avg_en_ll:.5f}')
    log.info(f'  improvement over Ridge: {avg_ridge_ll - avg_en_ll:+.5f}')

    # Test Lasso per target
    log.info('\n--- Lasso sweep (per target) ---')
    best_lasso_ll = {}
    for target in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[target])
        y_true = train_df[target].values
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        
        best_ll = float('inf')
        for alpha in lasso_alphas:
            meta = Lasso(alpha=alpha, random_state=SEED, max_iter=5000)
            meta.fit(X_meta, y_true)
            train_pred = meta.predict(X_meta)
            pmin, pmax = train_pred.min(), train_pred.max()
            if pmax - pmin < 1e-10:
                train_proba = np.ones_like(train_pred) * 0.5
            else:
                train_proba = (train_pred - pmin) / (pmax - pmin)
            train_proba = np.clip(train_proba, 0.001, 0.999)
            ll = log_loss(y_true, train_proba)
            if ll < best_ll:
                best_ll = ll
        best_lasso_ll[target] = best_ll
        log.info(f'  {target}: meta_ll={best_ll:.5f}')

    avg_lasso_ll = sum(best_lasso_ll.values()) / 7
    log.info(f'  avg_lasso_ll={avg_lasso_ll:.5f}')
    log.info(f'  improvement over Ridge: {avg_ridge_ll - avg_lasso_ll:+.5f}')

    # Summary
    log.info('\n' + '=' * 70)
    log.info('V541 Summary (metric: meta_ll_on_train, LOWER=BETTER)')
    log.info('=' * 70)
    log.info(f'  avg_student_ll: {avg_student_ll:.5f}')
    log.info(f'  Ridge (optimal α): {avg_ridge_ll:.5f}')
    log.info(f'  ElasticNet:        {avg_en_ll:.5f}')
    log.info(f'  Lasso:             {avg_lasso_ll:.5f}')
    log.info(f'  Best: {min([("Ridge", avg_ridge_ll), ("ElasticNet", avg_en_ll), ("Lasso", avg_lasso_ll)], key=lambda x: x[1])[0]} = {min([avg_ridge_ll, avg_en_ll, avg_lasso_ll]):.5f}')

    log.info(f'\nTotal time: {time.time() - t_start:.1f}s')
    return {'ridge_ll': avg_ridge_ll, 'en_ll': avg_en_ll, 'lasso_ll': avg_lasso_ll, 'student_ll': avg_student_ll}


if __name__ == '__main__':
    main()
