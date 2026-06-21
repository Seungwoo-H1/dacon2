#!/usr/bin/env python3
"""
V533 — Q1/Q2 Gap Fix + Meta Learner Exploration

V532 findings:
- Q1 gap=+0.181 (V308 0.113보다 훨씬 나쁨!)
- Q2 gap=+0.080 (V308 0.079보다 약간 나쁨)
- S2 gap=-0.067 (과도한 underfitting/calibration)
- Other targets good

Hypotheses:
1. Q1 needs more features / different config (V528 n_feat=5가 너무 적을 수 있음)
2. Q2 needs deeper model / more features
3. Target-specific meta: some targets need LogReg, some need Ridge
4. Ridge alpha tuning per target
5. Q1/Q2 separate hyperparameter search
"""
import sys, gc, logging, json, re, time, warnings
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
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, 'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50, 'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
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
        params = {**mp, 'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feat_names]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(X_va), m.predict(X_test)


def compute_gap(target_oofs, y, meta_type='ridge', alpha=0.001, enet_alpha=0.01, enet_l1=0.5):
    oofs_2d = np.column_stack(target_oofs)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    
    if meta_type == 'logreg':
        meta = LogisticRegression(C=10.0, max_iter=2000, random_state=SEED)
        meta.fit(X_meta, y)
        train_pred = meta.predict_proba(X_meta)[:, 1]
    elif meta_type == 'ridge':
        meta = Ridge(alpha=alpha)
        meta.fit(X_meta, y)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_pred = np.clip(train_proba, 0.001, 0.999)
    elif meta_type == 'enet':
        meta = ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1, max_iter=5000, random_state=SEED)
        meta.fit(X_meta, y)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_pred = np.clip(train_proba, 0.001, 0.999)
    elif meta_type == 'gbrt':
        meta = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED)
        meta.fit(X_meta, y)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_pred = np.clip(train_proba, 0.001, 0.999)
    else:
        raise ValueError(f"Unknown meta_type: {meta_type}")
    
    train_pred = np.clip(train_pred, 0.001, 0.999)
    meta_ll = log_loss(y, train_pred)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in target_oofs])
    return avg_student, meta_ll, avg_student - meta_ll


# V528 base config
V528_BASE = {
    'Q1':  {'n_feat': 5,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    'q_medium':  {'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 2.0, 'reg_lambda': 4.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
    'q_wide':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 2},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}


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
    
    # Compute gaps
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


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V533 — Q1/Q2 Gap Fix + Meta Learner Exploration")
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
    
    # === CONFIGURATIONS ===
    configs = {}
    
    # Config 1: V532 baseline (Q1_n5 Q2_n10 S4_n10 + Ridge) - to reproduce V532 result
    configs['V532_baseline'] = {**V528_BASE, '_name': 'V532_baseline'}
    
    # Config 2: Q1_n_feat=10 (increase from 5)
    configs['Q1_n10'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': 'Q1_n10'}
    
    # Config 3: Q1_n_feat=15
    configs['Q1_n15'] = {**V528_BASE, 'Q1': {'n_feat': 15, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': 'Q1_n15'}
    
    # Config 4: Q1_n_feat=20
    configs['Q1_n20'] = {**V528_BASE, 'Q1': {'n_feat': 20, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': 'Q1_n20'}
    
    # Config 5: Q1 with wider xgb config (q_wide)
    configs['Q1_q_wide'] = {**V528_BASE, 'Q1': {'n_feat': 5, 'xgb_cfg': 'q_wide', 'lgbm_cfg': 'wide', 'n_est': 600}, '_name': 'Q1_q_wide'}
    
    # Config 6: Q2 n_feat=20 (increase from 10)
    configs['Q2_n20'] = {**V528_BASE, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, '_name': 'Q2_n20'}
    
    # Config 7: Q2 n_feat=30
    configs['Q2_n30'] = {**V528_BASE, 'Q2': {'n_feat': 30, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, '_name': 'Q2_n30'}
    
    # Config 8: Q1_n10 + Q2_n20
    configs['Q1_n10_Q2_n20'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, '_name': 'Q1_n10_Q2_n20'}
    
    # Config 9: Q1_n10 + Q2_n20 + S4_n15
    configs['Q1_n10_Q2_n20_S4_n15'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, 'S4': {'n_feat': 15, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, '_name': 'Q1_n10_Q2_n20_S4_n15'}
    
    # Config 10: V528 config with LogReg meta (reproduce V528)
    configs['V528_LogReg'] = {**V528_BASE, '_name': 'V528_LogReg'}
    
    # Config 11: V528 with S4_n15 + LogReg meta
    configs['V528_S4n15_LogReg'] = {**V528_BASE, 'S4': {'n_feat': 15, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, '_name': 'V528_S4n15_LogReg'}
    
    # Config 12: Q1_n10_Q2_n20_S4_n15 + LogReg
    configs['Q1_n10_Q2_n20_S4_n15_LogReg'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, 'S4': {'n_feat': 15, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, '_name': 'Q1_n10_Q2_n20_S4_n15_LogReg'}
    
    # Config 13: V532 baseline with ElasticNet meta
    configs['V532_baseline_ENet'] = {**V528_BASE, '_name': 'V532_baseline_ENet'}
    
    # Config 14: V532 baseline with GBRT meta
    configs['V532_baseline_GBRT'] = {**V528_BASE, '_name': 'V532_baseline_GBRT'}
    
    # Config 15: Q1_n10_Q2_n20_S4_n15 + ElasticNet
    configs['Q1_n10_Q2_n20_S4_n15_ENet'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, 'S4': {'n_feat': 15, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, '_name': 'Q1_n10_Q2_n20_S4_n15_ENet'}
    
    # Config 16: S4_n15 + Q1_n10 + Q2_n20 (Ridge, alpha=0.01)
    configs['Q1_n10_Q2_n20_S4_n15_Ridge_a01'] = {**V528_BASE, 'Q1': {'n_feat': 10, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'Q2': {'n_feat': 20, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, 'S4': {'n_feat': 15, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, '_name': 'Q1_n10_Q2_n20_S4_n15_Ridge_a01'}
    
    # Config 17: V532 baseline + Ridge alpha=0.1 (stronger regularization)
    configs['V532_baseline_Ridge_a01'] = {**V528_BASE, '_name': 'V532_baseline_Ridge_a01'}
    
    all_results = []
    
    for name, config in configs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Config: {config['_name']}")
        log.info(f"{'='*60}")
        
        for t in TARGETS:
            log.info(f"  {t}: n_feat={config[t]['n_feat']} xgb={config[t]['xgb_cfg']} lgbm={config[t]['lgbm_cfg']} n_est={config[t]['n_est']}")
        
        try:
            # Ridge meta
            result_ridge = run_target_config(config, ranked_features, test_feat_cols,
                                            train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                            lambda oofs, y: compute_gap(oofs, y, 'ridge'))
            
            # LogReg meta
            result_logreg = run_target_config(config, ranked_features, test_feat_cols,
                                             train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                             lambda oofs, y: compute_gap(oofs, y, 'logreg'))
            
            # ElasticNet meta
            result_enet = run_target_config(config, ranked_features, test_feat_cols,
                                           train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                           lambda oofs, y: compute_gap(oofs, y, 'enet'))
            
            # GBRT meta
            result_gbrt = run_target_config(config, ranked_features, test_feat_cols,
                                           train_df, test_df, gkf, N_FOLDS, N_SEEDS,
                                           lambda oofs, y: compute_gap(oofs, y, 'gbrt'))
            
            result_ridge['meta_type'] = 'Ridge'
            result_logreg['meta_type'] = 'LogReg'
            result_enet['meta_type'] = 'ElasticNet'
            result_gbrt['meta_type'] = 'GBRT'
            
            log.info(f"  Ridge:  avg_gap={result_ridge['avg_gap']:.5f}, vs308={result_ridge['vs308']}/7")
            log.info(f"  LogReg: avg_gap={result_logreg['avg_gap']:.5f}, vs308={result_logreg['vs308']}/7")
            log.info(f"  ENet:   avg_gap={result_enet['avg_gap']:.5f}, vs308={result_enet['vs308']}/7")
            log.info(f"  GBRT:   avg_gap={result_gbrt['avg_gap']:.5f}, vs308={result_gbrt['vs308']}/7")
            
            all_results.extend([result_ridge, result_logreg, result_enet, result_gbrt])
        except Exception as e:
            log.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    
    for r in sorted(all_results, key=lambda x: x['avg_gap']):
        marker = ""
        if r['avg_gap'] < -0.02: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < -0.01: marker = " 🎯🎯"
        elif r['avg_gap'] < 0.0: marker = " 🎯"
        log.info(f"  {r['key']} [{r['meta_type']}]: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7{marker}")
    
    best = min(all_results, key=lambda x: x['avg_gap'])
    log.info(f"\n🏆 BEST: {best['key']} [{best['meta_type']}] with avg_gap={best['avg_gap']:.5f}")
    
    for t in TARGETS:
        v308_gaps_t = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}[t]
        vs = "✅" if best['target_gaps'][t] < v308_gaps_t else "❌"
        log.info(f"  {t}: {best['target_gaps'][t]:+.5f} V308={v308_gaps_t:.3f} {vs}")
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best result JSON
    result = {
        'version': 'V533',
        'name': 'Q1/Q2 Gap Fix + Meta Learner Exploration',
        'results': [{k: v for k, v in r.items() if k != 'test_preds'} for r in all_results],
        'best_key': best['key'],
        'best_meta': best['meta_type'],
        'best_gap': float(best['avg_gap']),
        'vs308': best['vs308'],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v533_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()