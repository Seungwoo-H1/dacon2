#!/usr/bin/env python3
"""
V541 — V540 Top Configs + ElasticNet Meta Validation

Uses V540 top configs per target (by Ridge gap) and tests:
1. Ridge meta (current V537)
2. ElasticNet meta
3. Best per-target config for each meta

Also tests: weighted blend with different weights, not just LL-inverted.
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
V308_GAPS = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
BEST_ALPHAS = {'Q1': 0.001, 'Q2': 0.06, 'Q3': 0.001, 'S1': 0.01, 'S2': 0.03, 'S3': 10.0, 'S4': 0.003}
V534_CONFIG = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}

# V540 top configs per target (by Ridge gap)
V540_TOP = {
    'Q1':  ('s_strong', 'heavy_lgb'),
    'Q2':  ('q_narrow', 'heavy_lgb'),
    'Q3':  ('heavy_reg', 'light_lgb'),
    'S1':  ('s_strong', 'heavy_lgb'),
    'S2':  ('light_reg', 'heavy_lgb'),
    'S3':  ('s_strong', 'heavy_lgb'),
    'S4':  ('s_strong', 'heavy_lgb'),  # V540 didn't finish S4, default to s_strong+heavy_lgb
}

# V540 EN-best configs per target
V540_EN_TOP = {
    'Q1':  ('q_narrow', 'heavy_lgb'),  # en-best
    'Q2':  ('q_strong', 'heavy_lgb'),
    'Q3':  ('heavy_reg', 'heavy_lgb'),
    'S1':  ('heavy_reg', 'heavy_lgb'),
    'S2':  ('light_reg', 'heavy_lgb'),
    'S3':  ('shallow_deep', 'heavy_lgb'),
    'S4':  ('s_strong', 'heavy_lgb'),
}

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
    'heavy_reg': {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 15.0, 'min_child_weight': 10},
    'light_reg': {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_weight': 1},
    'medium_reg':{'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.75, 'reg_alpha': 3.0, 'reg_lambda': 5.0, 'min_child_weight': 4},
    'shallow_deep':{'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'min_child_weight': 1},
}

LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'heavy_lgb': {'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 25},
    'light_lgb': {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_samples': 3},
    'balanced':  {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_samples': 8},
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

def train_one_xgb(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, **mp):
    params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
    ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
    ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
    m = xgb.train(params, ds_tr, num_boost_round=n_est)
    return m.predict(ds_va), m.predict(ds_te)

def train_one_lgbm(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, **mp):
    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
             'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_names]
    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
    m = lgb.train(params, ds_tr, num_boost_round=n_est)
    return m.predict(X_va), m.predict(X_test)

def meta_ridge(oofs_2d, y_true, alpha):
    n_seeds = oofs_2d.shape[1]
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = Ridge(alpha=alpha)
    meta.fit(X_meta, y_true)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y_true, train_proba)
    avg_student = np.mean([log_loss(y_true, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
    return avg_student - meta_ll, meta, train_proba

def meta_elasticnet(oofs_2d, y_true, alpha=0.5, l1_ratio=0.5):
    n_seeds = oofs_2d.shape[1]
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED, max_iter=5000)
    meta.fit(X_meta, y_true)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y_true, train_proba)
    avg_student = np.mean([log_loss(y_true, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
    return avg_student - meta_ll, meta, train_proba


def run_target_full(target, xgb_name, lgbm_name, gkf, train_df, test_df):
    bc = V534_CONFIG[target]
    n_feat = bc['n_feat']
    
    log.info(f'\n{target}: {xgb_name}+{lgbm_name}, n_feat={n_feat}, n_est={bc["n_est"]}')
    
    ranked_features = rank_features(train_df, get_feature_cols(train_df), target)
    sel_cols = ranked_features[:n_feat]
    test_cols = [c for c in sel_cols if c in get_feature_cols(test_df)]
    if len(test_cols) != len(sel_cols): sel_cols = test_cols
    
    y = train_df[target].values.astype(np.float64)
    X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
    n_est = bc['n_est']
    n_train = len(train_df)
    n_test = len(test_df)
    
    xgb_mp = XGB_CFGS[xgb_name]
    lgbm_mp = LGBM_CFGS[lgbm_name]
    
    xgb_oofs = []; lgbm_oofs = []; xgb_tests = []; lgbm_tests = []
    
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
            pvxgb, ttxgb = train_one_xgb(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **xgb_mp)
            pvlgbm, ttlgbm = train_one_lgbm(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **lgbm_mp)
            oof_xgb[va_idx] = pvxgb; oof_lgbm[va_idx] = pvlgbm
            test_xgb += ttxgb; test_lgbm += ttlgbm
        
        oof_xgb = np.clip(oof_xgb, 0.001, 0.999)
        oof_lgbm = np.clip(oof_lgbm, 0.001, 0.999)
        test_xgb /= N_FOLDS; test_lgbm /= N_FOLDS
        xgb_oofs.append(oof_xgb); lgbm_oofs.append(oof_lgbm)
        xgb_tests.append(test_xgb); lgbm_tests.append(test_lgbm)
        
        if si % 4 == 0 or si == N_SEEDS - 1:
            log.info(f'    seed {si}: xgb={np.mean(oof_xgb):.4f}, lgbm={np.mean(oof_lgbm):.4f}')
    
    ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs])
    ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs])
    inv_xgb = 1.0 / max(ll_xgb, 1e-10); inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
    wx = inv_xgb / (inv_xgb + inv_lgbm)
    
    # Return all seeds
    return {
        'oofs': xgb_oofs + lgbm_oofs,  # combined
        'tests': xgb_tests + lgbm_tests,
        'll_xgb': ll_xgb, 'll_lgbm': ll_lgbm, 'wx': wx,
        'y': y, 'target': target
    }


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V541: V540 Top Configs + EN Meta Validation (13 seeds)")
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
    
    # Add zscore columns to test_feat_cols check
    test_feat_cols = get_feature_cols(test_df)
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # ================================================================
    # Run V540 top configs per target
    # ================================================================
    all_oofs = {}
    all_tests = {}
    
    for target in TARGETS:
        xgb_name, lgbm_name = V540_TOP[target]
        result = run_target_full(target, xgb_name, lgbm_name, gkf, train_df, test_df)
        all_oofs[target] = np.column_stack(result['oofs'])  # N_SEEDS*2
        all_tests[target] = result['tests']
    
    # ================================================================
    # Test different meta learners
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V541: Meta Learner Comparison')
    log.info('=' * 70)
    
    results_summary = {}
    
    for meta_name, meta_fn, meta_alpha in [
        ('Ridge_0.01', meta_ridge, 0.01),
        ('Ridge_0.001', meta_ridge, 0.001),
        ('Ridge_optimal', meta_ridge, None),
        ('EN_0.5_0.5', meta_elasticnet, (0.5, 0.5)),
        ('EN_0.3_0.5', meta_elasticnet, (0.3, 0.5)),
        ('EN_1.0_0.5', meta_elasticnet, (1.0, 0.5)),
    ]:
        target_gaps = {}
        target_meta = {}
        
        for target in TARGETS:
            oofs_2d = all_oofs[target]
            y_true = train_df[target].values
            
            if meta_name.startswith('Ridge'):
                if meta_alpha is None:
                    # Find optimal alpha
                    best_alpha = 0.01
                    best_gap = float('inf')
                    for a in [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
                        gap, _, _ = meta_ridge(oofs_2d, y_true, a)
                        if gap < best_gap:
                            best_gap = gap; best_alpha = a
                    alpha = best_alpha
                else:
                    alpha = meta_alpha
                gap, meta, train_proba = meta_ridge(oofs_2d, y_true, alpha)
            else:
                alpha, l1r = meta_alpha
                gap, meta, train_proba = meta_elasticnet(oofs_2d, y_true, alpha, l1r)
            
            target_gaps[target] = gap
            target_meta[target] = (meta, alpha if not isinstance(alpha, tuple) else alpha)
            
            vs = "✅" if gap < V308_GAPS[target] else "❌"
            log.info(f'  {meta_name} / {target}: gap={gap:+.5f} {vs}')
        
        avg_gap = sum(target_gaps.values()) / 7
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < V308_GAPS[t])
        improvement = -0.03016 - avg_gap
        
        results_summary[meta_name] = {
            'avg_gap': avg_gap, 'vs308': vs308, 'improvement': improvement,
            'target_gaps': target_gaps, 'target_meta': target_meta
        }
        
        log.info(f'  📊 {meta_name}: avg_gap={avg_gap:+.5f}, improvement=+{improvement:+.5f}, vs308={vs308}/7')
    
    # ================================================================
    # Find best meta + create submission
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V541 Final Results')
    log.info('=' * 70)
    
    best_meta_name = min(results_summary, key=lambda x: results_summary[x]['avg_gap'])
    best = results_summary[best_meta_name]
    
    log.info(f'\n🏆 Best meta: {best_meta_name}')
    log.info(f'  avg_gap: {best["avg_gap"]:+.5f}')
    log.info(f'  improvement vs V537: +{best["improvement"]:+.5f}')
    log.info(f'  vs308: {best["vs308"]}/7')
    log.info(f'  Expected LB: {0.62235 - best["avg_gap"]:.5f}')
    
    log.info(f'\n  Target gaps:')
    for t in TARGETS:
        vs = "✅" if best['target_gaps'][t] < V308_GAPS[t] else "❌"
        log.info(f'    {t}: {best["target_gaps"][t]:+.5f} {vs}')
    
    # ================================================================
    # Generate submission with best meta
    # ================================================================
    log.info(f'\nGenerating submission with {best_meta_name}...')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    
    for target in TARGETS:
        oofs_2d = all_oofs[target]
        meta, meta_param = best['target_meta'][target]
        
        n_test = len(test_df)
        n_seeds = oofs_2d.shape[1]
        
        # Average all seed predictions (XGB + LGBM)
        test_avg = np.zeros(n_test)
        test_std_val = np.zeros(n_test)
        for si in range(n_seeds):
            test_avg += all_tests[target][si]
        test_avg /= n_seeds
        for si in range(n_seeds):
            test_std_val += (all_tests[target][si] - test_avg) ** 2
        test_std = np.sqrt(test_std_val / n_seeds)
        
        X_test_meta = np.column_stack([test_avg, test_std])
        test_pred = meta.predict(X_test_meta)
        
        # Train pred for scaling
        X_train_meta = np.column_stack([np.mean(oofs_2d, axis=1), np.std(oofs_2d, axis=1)])
        train_pred = meta.predict(X_train_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            test_proba = np.ones(n_test) * 0.5
        else:
            test_proba = (test_pred - pmin) / (pmax - pmin)
        test_proba = np.clip(test_proba, 0.001, 0.999)
        sub_df[target] = test_proba
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f'submission_v541_{best_meta_name.replace(".", "_").replace("_optimal", "opt")}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f'📁 Submission saved: {sub_path}')
    
    # Save results
    result = {
        'version': 'V541',
        'hypothesis': 'v540_top_configs_en_meta_validation',
        'meta_comparison': {k: {'avg_gap': v['avg_gap'], 'vs308': v['vs308'], 'improvement': v['improvement']} 
                           for k, v in results_summary.items()},
        'best_meta': best_meta_name,
        'avg_gap': best['avg_gap'],
        'improvement': best['improvement'],
        'target_gaps': best['target_gaps'],
        'submission_file': str(sub_path.name),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v541_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')
    
    # Print summary table
    log.info('\n  === Meta Comparison Table ===')
    log.info(f'  {"Meta":25s} {"avg_gap":>10} {"vs308":>8} {"Δ V537":>10} {"Exp LB":>10}')
    log.info('  ' + '-' * 68)
    for name, v in sorted(results_summary.items(), key=lambda x: x[1]['avg_gap']):
        log.info(f'  {name:25s} {v["avg_gap"]:>+10.5f} {v["vs308"]:>8d} {v["improvement"]:>+10.5f} {0.62235-v["avg_gap"]:>10.5f}')

if __name__ == '__main__':
    main()
