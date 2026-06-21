#!/usr/bin/env python3
"""
V534 — V531 replication + Q1 n_feat sweep with Ridge meta

V531 findings: Ridge meta is vastly superior to LogReg meta.
- Best: S4_n15 + Ridge → avg_gap=-0.02536
- Q2_n10_S4_n15 + Ridge → avg_gap=-0.02536
- V531 SIGTERM truncated, so Q2_n10_wide_agg 등 미완료

V532 (S4_n15 + Ridge): avg_gap=0.04085, Q1=+0.181 (BAD!)
→ V531 config vs V532 config difference: V532 used V528_BASE (S4_n_feat=10) + override S4_n15

V534 Hypothesis: V531 results were from a single-pass run that was truncated.
V531 results used the SAME run's oofs for both LogReg and Ridge meta.
But V532 seems to have different results for S4_n15+Ridge (0.04085 vs -0.02536).

Possible reasons:
1. Different feature ranking (rank_features uses single seed LGBM with 50 trees → unstable)
2. V531 results were partial (only 1 config completed before SIGTERM)
3. Different N_SEEDS or other params

V534 approach:
1. Reproduce V531 configs with fresh random state
2. Sweep Q1 n_feat: 3, 5, 8, 10, 15, 20
3. Use S4_n15 for all (since that was V531's best)
4. Compare: mean+std Ridge, ElasticNet meta, GBRT meta
"""
import sys, gc, logging, json, re, time, warnings, random
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEED = 42
N_FOLDS = 5
N_SEEDS = 13

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count','wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count','wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05,
        'n_estimators': 50, 'scale_pos_weight': spw, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
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


def compute_gap_ridge(oofs_arr, y, alpha=0.001):
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = Ridge(alpha=alpha)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y, train_proba)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_arr])
    return avg_student, meta_ll, avg_student - meta_ll


def compute_gap_enet(oofs_arr, y, alpha=0.01, l1=0.5):
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=SEED)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y, train_proba)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_arr])
    return avg_student, meta_ll, avg_student - meta_ll


def compute_gap_gbrt(oofs_arr, y):
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y, train_proba)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_arr])
    return avg_student, meta_ll, avg_student - meta_ll


def compute_gap_logreg(oofs_arr, y):
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = LogisticRegression(C=10.0, max_iter=2000, random_state=SEED)
    meta.fit(X_meta, y)
    train_proba = meta.predict_proba(X_meta)[:, 1]
    meta_ll = log_loss(y, np.clip(train_proba, 0.001, 0.999))
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_arr])
    return avg_student, meta_ll, avg_student - meta_ll


def run_target_config(config, ranked_features, test_feat_cols, train_df, test_df, gkf, N_FOLDS, N_SEEDS, meta_fn):
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}

    for target in TARGETS:
        bc = config[target]
        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names

        n_est = bc['n_est']
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[bc['xgb_cfg']]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[bc['lgbm_cfg']]}

        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        xgb_seed_oofs = []
        lgbm_seed_oofs = []
        xgb_test_preds = []
        lgbm_test_preds = []

        for si in range(N_SEEDS):
            seed = SEED + si * 11
            seed_oof_xgb = np.zeros(n_train)
            seed_oof_lgbm = np.zeros(n_train)
            seed_test_xgb = np.zeros(n_test)
            seed_test_lgbm = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]

                pvxgb, ttxgb = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'xgb', n_est, **xgb_params)
                pvlgbm, ttlgbm = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'lgbm', n_est, **lgbm_params)

                seed_oof_xgb[va_idx] = pvxgb
                seed_oof_lgbm[va_idx] = pvlgbm
                seed_test_xgb += ttxgb
                seed_test_lgbm += ttlgbm

            seed_oof_xgb = np.clip(seed_oof_xgb, 0.001, 0.999)
            seed_oof_lgbm = np.clip(seed_oof_lgbm, 0.001, 0.999)
            seed_test_xgb /= N_FOLDS
            seed_test_lgbm /= N_FOLDS

            xgb_seed_oofs.append(seed_oof_xgb)
            lgbm_seed_oofs.append(seed_oof_lgbm)
            xgb_test_preds.append(seed_test_xgb)
            lgbm_test_preds.append(seed_test_lgbm)

        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_seed_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_seed_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10)
        inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        wx = inv_xgb / (inv_xgb + inv_lgbm)

        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_test_preds, lgbm_test_preds)]

        all_seed_oofs[target] = blended_oofs
        all_test_preds[target] = blended_tests

    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    total_gap = 0
    target_gaps = {}
    for t in TARGETS:
        _, _, gap = meta_fn(all_seed_oofs[t], train_df[t].values)
        target_gaps[t] = gap
        total_gap += gap
    avg_gap = total_gap / 7
    vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])

    return {
        'key': config.get('_name', 'unknown'),
        'avg_gap': avg_gap,
        'target_gaps': target_gaps,
        'vs308': vs308,
        'test_preds': all_test_preds,
    }


# V528 model configs (same as V531)
XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    'q_medium':  {'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 2.0, 'reg_lambda': 4.0, 'min_child_weight': 5},
    's_wide':    {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V534 — V531 Replication + Q1 n_feat Sweep + Meta Learner Comparison")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # Z-score
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m, s = np.mean(tv), np.std(tv, ddof=0)
        if s < 1e-8: s = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m) / s
        test_df[zc] = (ev - m) / s

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    gkf = GroupKFold(n_splits=N_FOLDS)

    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # V531 base: Q2_n10_S4_n15 (best V531 config)
    BASE_V531 = {
        'Q1':  {'n_feat': 5,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
        'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
        'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
        'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
        'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
        'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
        'S4':  {'n_feat': 15, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
    }

    all_results = []

    # === Phase 1: Q1 n_feat sweep with V531 config (S4_n15, Ridge) ===
    log.info("\n=== PHASE 1: Q1 n_feat sweep (V531 config, S4_n15, Ridge meta) ===")
    q1_nfeats = [3, 5, 8, 10, 15, 20]
    for nf in q1_nfeats:
        config = {**BASE_V531, 'Q1': {'n_feat': nf, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': f'Q1_n{nf}_S4n15'}
        log.info(f"\nConfig: {config['_name']} Q1_n_feat={nf}")
        for t in TARGETS:
            log.info(f"  {t}: n_feat={config[t]['n_feat']}")

        try:
            r = run_target_config(config, ranked_features, test_feat_cols,
                                  train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                  lambda oofs, y: compute_gap_ridge(oofs, y))
            r['meta_type'] = 'Ridge'
            all_results.append(r)
            log.info(f"  Ridge: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7, Q1={r['target_gaps']['Q1']:.5f}")
        except Exception as e:
            log.info(f"  ERROR: {e}")

    # === Phase 2: Best Q1 n_feat + 3 meta learners ===
    best_q1_nf = min(q1_nfeats, key=lambda nf: [r for r in all_results if f'Q1_n{nf}' in r['key']][0]['avg_gap']) if all_results else 5
    log.info(f"\n=== PHASE 2: Q1_n{best_q1_nf} + multiple meta learners ===")
    config2 = {**BASE_V531, 'Q1': {'n_feat': best_q1_nf, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': f'Q1_n{best_q1_nf}_S4n15_multi_meta'}

    for meta_name, meta_fn in [('Ridge', lambda o, y: compute_gap_ridge(o, y)),
                                ('Ridge_a01', lambda o, y: compute_gap_ridge(o, y, alpha=0.01)),
                                ('Ridge_a001', lambda o, y: compute_gap_ridge(o, y, alpha=0.0001)),
                                ('ElasticNet', lambda o, y: compute_gap_enet(o, y)),
                                ('GBRT', lambda o, y: compute_gap_gbrt(o, y)),
                                ('LogReg', lambda o, y: compute_gap_logreg(o, y))]:
        try:
            r = run_target_config(config2, ranked_features, test_feat_cols,
                                  train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                  meta_fn)
            r['meta_type'] = meta_name
            all_results.append(r)
            log.info(f"  {meta_name}: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7")
            for t in TARGETS:
                log.info(f"    {t}: {r['target_gaps'][t]:+.5f}")
        except Exception as e:
            log.info(f"  {meta_name} ERROR: {e}")

    # === Phase 3: Q2 n_feat sweep with best Q1 ===
    log.info(f"\n=== PHASE 3: Q2 n_feat sweep (Q1_n{best_q1_nf}, S4_n15, Ridge) ===")
    q2_nfeats = [8, 10, 12, 14]
    for nf in q2_nfeats:
        config = {**BASE_V531, 'Q1': {'n_feat': best_q1_nf, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600},
                  'Q2': {'n_feat': nf, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800},
                  '_name': f'Q1_n{best_q1_nf}_Q2_n{nf}_S4n15'}
        log.info(f"\nConfig: {config['_name']} Q2_n_feat={nf}")
        try:
            r = run_target_config(config, ranked_features, test_feat_cols,
                                  train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                  lambda oofs, y: compute_gap_ridge(oofs, y))
            r['meta_type'] = 'Ridge'
            all_results.append(r)
            log.info(f"  avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7")
        except Exception as e:
            log.info(f"  ERROR: {e}")

    # === Phase 4: S4 n_feat sweep with best Q1, Q2 ===
    best_q2_nf = min(q2_nfeats, key=lambda nf: [r for r in all_results if f'Q2_n{nf}' in r['key']][0]['avg_gap']) if any(f'Q2_n' in r['key'] for r in all_results) else 10
    log.info(f"\n=== PHASE 4: S4 n_feat sweep (Q1_n{best_q1_nf}, Q2_n{best_q2_nf}, Ridge) ===")
    s4_nfeats = [10, 12, 15, 20]
    for nf in s4_nfeats:
        config = {**BASE_V531, 'Q1': {'n_feat': best_q1_nf, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600},
                  'Q2': {'n_feat': best_q2_nf, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800},
                  'S4': {'n_feat': nf, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300},
                  '_name': f'Q1_n{best_q1_nf}_Q2_n{best_q2_nf}_S4_n{nf}'}
        log.info(f"\nConfig: {config['_name']} S4_n_feat={nf}")
        try:
            r = run_target_config(config, ranked_features, test_feat_cols,
                                  train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                  lambda oofs, y: compute_gap_ridge(oofs, y))
            r['meta_type'] = 'Ridge'
            all_results.append(r)
            log.info(f"  avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7")
        except Exception as e:
            log.info(f"  ERROR: {e}")

    # === Summary ===
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")

    for r in sorted(all_results, key=lambda x: x['avg_gap']):
        marker = ""
        if r['avg_gap'] < -0.03: marker = " 🎯🎯🎯🎯"
        elif r['avg_gap'] < -0.02: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < -0.01: marker = " 🎯🎯"
        elif r['avg_gap'] < 0.0: marker = " 🎯"
        elif r['avg_gap'] < 0.02: marker = " 👍"
        log.info(f"  {r['key']} [{r['meta_type']}]: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7{marker}")
        for t in TARGETS:
            v308_g = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}[t]
            vs = "✅" if r['target_gaps'][t] < v308_g else "❌"
            log.info(f"    {t}: {r['target_gaps'][t]:+.5f} V308={v308_g:.3f} {vs}")

    best = min(all_results, key=lambda x: x['avg_gap'])
    log.info(f"\n🏆 BEST: {best['key']} [{best['meta_type']}] with avg_gap={best['avg_gap']:.5f}")

    # Save result
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        'version': 'V534',
        'name': 'V531 Replication + Q1 n_feat Sweep + Meta Learner Comparison',
        'results': [{k: v for k, v in r.items() if k != 'test_preds'} for r in all_results],
        'best_key': best['key'],
        'best_meta': best['meta_type'],
        'best_gap': float(best['avg_gap']),
        'best_target_gaps': best['target_gaps'],
        'vs308': best['vs308'],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v534_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()