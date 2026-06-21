#!/usr/bin/env python3
"""
V542 — V537 Fine Alpha + V540 Top Configs Hybrid

Hypothesis: V540 top configs (s_strong+heavy_lgb) beat V534 base configs,
but V537 fine per-target Ridge alphas should still apply to the Ridge meta.

Test:
1. V540 top configs per target + per-target Ridge alpha sweep
2. Find optimal α per target with V540 configs
3. Compare V537 alphas vs V542 alphas
"""
import sys, gc, logging, json, re, time, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
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

# V540 top configs
V540_TOP = {
    'Q1':  ('s_strong', 'heavy_lgb', 600),
    'Q2':  ('q_narrow', 'heavy_lgb', 800),
    'Q3':  ('heavy_reg', 'light_lgb', 500),
    'S1':  ('s_strong', 'heavy_lgb', 500),
    'S2':  ('light_reg', 'heavy_lgb', 500),
    'S3':  ('s_strong', 'heavy_lgb', 1000),
    'S4':  ('s_strong', 'heavy_lgb', 300),
}

# V537 best alphas (for reference)
V537_ALPHAS = {'Q1': 0.001, 'Q2': 0.06, 'Q3': 0.001, 'S1': 0.01, 'S2': 0.03, 'S3': 10.0, 'S4': 0.003}

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
    'heavy_reg': {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 15.0, 'min_child_weight': 10},
    'light_reg': {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_weight': 1},
}

LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'heavy_lgb': {'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 25},
    'light_lgb': {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_samples': 3},
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


def run_target_with_configs(target, xgb_cfg, lgbm_cfg, n_est, n_feat, gkf, train_df, test_df):
    ranked = rank_features(train_df, get_feature_cols(train_df), target)
    sel_cols = ranked[:n_feat]
    test_cols = [c for c in sel_cols if c in get_feature_cols(test_df)]
    if len(test_cols) != len(sel_cols): sel_cols = test_cols
    
    y = train_df[target].values.astype(np.float64)
    X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    
    xgb_mp = XGB_CFGS[xgb_cfg]
    lgbm_mp = LGBM_CFGS[lgbm_cfg]
    
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
    
    # Return all 2*N_SEEDS predictions
    all_oofs = xgb_oofs + lgbm_oofs
    all_tests = xgb_tests + lgbm_tests
    return all_oofs, all_tests, y, sel_cols


def find_optimal_ridge_alpha(oofs_2d, y_true):
    best_alpha = 0.01
    best_gap = float('inf')
    for a in [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        meta = Ridge(alpha=a)
        avg_p = np.mean(oofs_2d, axis=1)
        std_p = np.std(oofs_2d, axis=1)
        meta.fit(np.column_stack([avg_p, std_p]), y_true)
        tp = meta.predict(np.column_stack([avg_p, std_p]))
        pmin, pmax = tp.min(), tp.max()
        if pmax - pmin < 1e-10:
            proba = np.ones(len(tp)) * 0.5
        else:
            proba = np.clip((tp - pmin) / (pmax - pmin), 0.001, 0.999)
        meta_ll = log_loss(y_true, proba)
        n_seeds = oofs_2d.shape[1]
        avg_student = np.mean([log_loss(y_true, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
        gap = avg_student - meta_ll
        if gap < best_gap:
            best_gap = gap
            best_alpha = a
    return best_alpha, best_gap


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V542: V537 Fine Alpha + V540 Top Configs (Per-Target Alpha Sweep)")
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
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Run all targets with V540 top configs
    all_data = {}
    for target in TARGETS:
        xgb_cfg, lgbm_cfg, n_est = V540_TOP[target]
        bc = {'Q1': 3, 'Q2': 10, 'Q3': 7, 'S1': 3, 'S2': 7, 'S3': 23, 'S4': 20}
        n_feat = bc[target]
        
        log.info(f'\n{target}: {xgb_cfg}+{lgbm_cfg}, n_feat={n_feat}, n_est={n_est}')
        oofs, tests, y, sel_cols = run_target_with_configs(target, xgb_cfg, lgbm_cfg, n_est, n_feat, gkf, train_df, test_df)
        all_data[target] = {'oofs': oofs, 'tests': tests, 'y': y, 'sel_cols': sel_cols}
    
    # ================================================================
    # Per-target optimal Ridge alpha sweep
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V542: Per-Target Optimal Ridge Alpha')
    log.info('=' * 70)
    
    v542_alphas = {}
    v542_gaps = {}
    v542_models = {}
    
    for target in TARGETS:
        oofs_2d = np.column_stack(all_data[target]['oofs'])
        y = all_data[target]['y']
        
        optimal_alpha, gap = find_optimal_ridge_alpha(oofs_2d, y)
        v542_alphas[target] = optimal_alpha
        v542_gaps[target] = gap
        v542_models[target] = Ridge(alpha=optimal_alpha)
        
        v537_alpha = V537_ALPHAS[target]
        # Compare with V537 alpha
        meta_v537 = Ridge(alpha=v537_alpha)
        avg_p = np.mean(oofs_2d, axis=1)
        std_p = np.std(oofs_2d, axis=1)
        meta_v537.fit(np.column_stack([avg_p, std_p]), y)
        tp_v537 = meta_v537.predict(np.column_stack([avg_p, std_p]))
        pmin, pmax = tp_v537.min(), tp_v537.max()
        if pmax - pmin < 1e-10:
            proba_v537 = np.ones(len(tp_v537)) * 0.5
        else:
            proba_v537 = np.clip((tp_v537 - pmin) / (pmax - pmin), 0.001, 0.999)
        meta_ll_v537 = log_loss(y, proba_v537)
        n_seeds = oofs_2d.shape[1]
        avg_student = np.mean([log_loss(y, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
        gap_v537 = avg_student - meta_ll_v537
        
        vs = "✅" if gap < V308_GAPS[target] else "❌"
        better = "🎯" if gap < gap_v537 else ""
        log.info(f'  {target}: V542α={optimal_alpha} (gap={gap:+.5f}) vs V537α={v537_alpha} (gap={gap_v537:+.5f}) {better} {vs}')
    
    avg_gap_v542 = sum(v542_gaps.values()) / 7
    avg_gap_v537 = sum(
        find_optimal_ridge_alpha(np.column_stack(all_data[t]['oofs']), all_data[t]['y'])[1] 
        for t in TARGETS
    )  # This is actually the same — need to compute with V537 alphas
    
    # Actually compute V537 avg gap with V540 configs
    v537_gaps_v540_configs = {}
    for target in TARGETS:
        oofs_2d = np.column_stack(all_data[target]['oofs'])
        y = all_data[target]['y']
        meta = Ridge(alpha=V537_ALPHAS[target])
        avg_p = np.mean(oofs_2d, axis=1)
        std_p = np.std(oofs_2d, axis=1)
        meta.fit(np.column_stack([avg_p, std_p]), y)
        tp = meta.predict(np.column_stack([avg_p, std_p]))
        pmin, pmax = tp.min(), tp.max()
        if pmax - pmin < 1e-10:
            proba = np.ones(len(tp)) * 0.5
        else:
            proba = np.clip((tp - pmin) / (pmax - pmin), 0.001, 0.999)
        meta_ll = log_loss(y, proba)
        n_seeds = oofs_2d.shape[1]
        avg_student = np.mean([log_loss(y, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
        v537_gaps_v540_configs[target] = avg_student - meta_ll
    
    avg_gap_v537_on_v540 = sum(v537_gaps_v540_configs.values()) / 7
    improvement = avg_gap_v537_on_v540 - avg_gap_v542  # positive = V542 better
    
    log.info(f'\n  V537 alphas on V540 configs: avg_gap={avg_gap_v537_on_v540:+.5f}')
    log.info(f'  V542 alphas on V540 configs: avg_gap={avg_gap_v542:+.5f}')
    log.info(f'  Improvement (V542 over V537): +{improvement:+.5f}')
    log.info(f'  V542 over V537 (OOF): +{(-0.03016) - avg_gap_v542:+.5f}')
    
    # ================================================================
    # Generate submission
    # ================================================================
    log.info(f'\nGenerating submission with V542 optimal alphas...')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    
    for target in TARGETS:
        oofs = all_data[target]['oofs']
        tests = all_data[target]['tests']
        n_seeds = len(oofs)
        y = all_data[target]['y']
        
        n_test = len(test_df)
        n_oofs = len(oofs)
        oofs_2d = np.column_stack(oofs)
        
        test_avg = np.zeros(n_test)
        for si in range(n_seeds):
            test_avg += tests[si]
        test_avg /= n_seeds
        
        test_var = np.zeros(n_test)
        for si in range(n_seeds):
            test_var += (tests[si] - test_avg) ** 2
        test_std = np.sqrt(test_var / n_seeds)
        
        X_test_meta = np.column_stack([test_avg, test_std])
        X_train_meta = np.column_stack([np.mean(oofs_2d, axis=1), np.std(oofs_2d, axis=1)])
        
        # Fit Ridge meta on training data
        meta = Ridge(alpha=v542_alphas[target])
        meta.fit(X_train_meta, y)
        
        test_pred = meta.predict(X_test_meta)
        train_pred = meta.predict(X_train_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            test_proba = np.ones(n_test) * 0.5
        else:
            test_proba = (test_pred - pmin) / (pmax - pmin)
        test_proba = np.clip(test_proba, 0.001, 0.999)
        sub_df[target] = test_proba
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f'submission_v542_per_target_alpha_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f'📁 Submission: {sub_path}')
    
    # ================================================================
    # Compare V541 (fixed α) vs V542 (per-target α)
    # ================================================================
    log.info('\n  === V541 (fixed α=0.01) vs V542 (per-target α) ===')
    log.info(f'  {"Version":20s} {"avg_gap":>10} {"vs308":>8} {"Exp LB":>10}')
    log.info('  ' + '-' * 52)
    
    # V541 Ridge_0.01 avg_gap was -0.04889
    v541_avg_gap = -0.04889
    log.info(f'  {"V541 Ridge_0.01":20s} {v541_avg_gap:>+10.5f} {7:>8d} {0.62235-v541_avg_gap:>10.5f}')
    log.info(f'  {"V542 per-target α":20s} {avg_gap_v542:>+10.5f} {sum(1 for t in TARGETS if v542_gaps[t] < V308_GAPS[t]):>8d} {0.62235-avg_gap_v542:>10.5f}')
    
    # Save
    result = {
        'version': 'V542',
        'hypothesis': 'per_target_ridge_alpha_v540_configs',
        'v542_alphas': v542_alphas,
        'v542_gaps': {k: float(v) for k, v in v542_gaps.items()},
        'v537_gaps_on_v540_configs': {k: float(v) for k, v in v537_gaps_v540_configs.items()},
        'avg_gap_v542': float(avg_gap_v542),
        'avg_gap_v537_on_v540': float(avg_gap_v537_on_v540),
        'improvement': float(improvement),
        'submission_file': str(sub_path.name),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v542_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')

if __name__ == '__main__':
    main()
