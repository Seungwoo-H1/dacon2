#!/usr/bin/env python3
"""
V545 — V537 Base Configs + V541 Meta (Ridge_optimal)

Hypothesis: V537 configs (Q1_n3_q_narrow, Q2_n10_q_deep, etc.) with V541's
V540 configs + Ridge_optimal alpha might give different results than
V540 configs alone. Test if V534_base configs with V541 meta still competitive.

Actually: V541 used V540 top configs. This test uses V534 original configs
with the same V541 meta approach to see which base configs are better.
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

# V534 original configs (from MEMORY.md)
V534_CONFIGS = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}

# V540 top configs
V540_TOP = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'heavy_lgb', 'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'heavy_lgb', 'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'heavy_reg', 'lgbm_cfg': 'light_lgb', 'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'heavy_lgb', 'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 'light_reg', 'lgbm_cfg': 'heavy_lgb', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 's_strong',  'lgbm_cfg': 'heavy_lgb', 'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 's_strong',  'lgbm_cfg': 'heavy_lgb', 'n_est': 300},
}

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
    'balanced':  {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_samples': 8},
}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def rank_features(df, feat_cols, target, seed=SEED):
    y = df[target].values.astype(np.float64)
    X = df[feat_cols].fillna(0).values.astype(np.float64)
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


def run_with_configs(configs_dict, gkf, train_df, test_df, v542_alphas):
    """Run all targets with given configs and return results."""
    all_data = {}
    
    for target in TARGETS:
        bc = configs_dict[target]
        xgb_cfg, lgbm_cfg, n_est, n_feat = bc['xgb_cfg'], bc['lgbm_cfg'], bc['n_est'], bc['n_feat']
        
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
        
        all_data[target] = {
            'oofs': xgb_oofs + lgbm_oofs,
            'tests': xgb_tests + lgbm_tests,
            'y': y,
        }
        log.info(f'  {target}: done ({xgb_cfg}+{lgbm_cfg})')
    
    # Evaluate with Ridge_optimal (V542 alphas)
    target_gaps = {}
    for target in TARGETS:
        oofs_2d = np.column_stack(all_data[target]['oofs'])
        y = all_data[target]['y']
        alpha = v542_alphas[target]
        
        meta = Ridge(alpha=alpha)
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
        target_gaps[target] = avg_student - meta_ll
    
    avg_gap = sum(target_gaps.values()) / 7
    return all_data, target_gaps, avg_gap


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V545: V534 Base Configs vs V540 Configs with V542 Optimal Alphas")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
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
    
    # V542 optimal alphas
    v542_alphas = {'Q1': 0.0001, 'Q2': 10.0, 'Q3': 0.0001, 'S1': 0.0001, 'S2': 0.03, 'S3': 10.0, 'S4': 0.0001}
    
    # ================================================================
    # Test V534 base configs with V542 alphas
    # ================================================================
    log.info('\n--- V534 Base Configs + V542 Alphas ---')
    v534_cfgs = {}
    for target in TARGETS:
        bc = V534_CONFIGS[target]
        v534_cfgs[target] = {'n_feat': bc['n_feat'], 'xgb_cfg': bc['xgb_cfg'], 'lgbm_cfg': bc['lgbm_cfg'], 'n_est': bc['n_est']}
    
    v534_data, v534_gaps, v534_avg_gap = run_with_configs(v534_cfgs, gkf, train_df, test_df, v542_alphas)
    
    log.info(f'\n  V534 Base + V542 Alphas: avg_gap={v534_avg_gap:+.5f}')
    for t in TARGETS:
        vs = "✅" if v534_gaps[t] < V308_GAPS[t] else "❌"
        log.info(f'    {t}: {v534_gaps[t]:+.5f} {vs}')
    
    # ================================================================
    # Test V540 configs with V542 alphas (already known from V541/V542)
    # ================================================================
    log.info('\n--- V540 Configs + V542 Alphas ---')
    v540_cfgs = {}
    for target in TARGETS:
        bc = V540_TOP[target]
        v540_cfgs[target] = {'n_feat': bc['n_feat'], 'xgb_cfg': bc['xgb_cfg'], 'lgbm_cfg': bc['lgbm_cfg'], 'n_est': bc['n_est']}
    
    v540_data, v540_gaps, v540_avg_gap = run_with_configs(v540_cfgs, gkf, train_df, test_df, v542_alphas)
    
    log.info(f'\n  V540 Configs + V542 Alphas: avg_gap={v540_avg_gap:+.5f}')
    for t in TARGETS:
        vs = "✅" if v540_gaps[t] < V308_GAPS[t] else "❌"
        log.info(f'    {t}: {v540_gaps[t]:+.5f} {vs}')
    
    # ================================================================
    # Generate submission with best configs
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V545 Final Results')
    log.info('=' * 70)
    
    # Compare
    log.info(f'  {"Config":20s} {"avg_gap":>10} {"vs308":>8} {"Δ V537":>10} {"Exp LB":>10}')
    log.info('  ' + '-' * 64)
    
    v537_gap = -0.03016
    v534_vs308 = sum(1 for t in TARGETS if v534_gaps[t] < V308_GAPS[t])
    v540_vs308 = sum(1 for t in TARGETS if v540_gaps[t] < V308_GAPS[t])
    
    log.info(f'  {"V537 baseline":20s} {v537_gap:>+10.5f} {7:>8d} {0:>10.5f} {0.62235-v537_gap:>10.5f}')
    log.info(f'  {"V534 Base+V542α":20s} {v534_avg_gap:>+10.5f} {v534_vs308:>8d} {(-0.03016)-v534_avg_gap:>+10.5f} {0.62235-v534_avg_gap:>10.5f}')
    log.info(f'  {"V540 Cfgs+V542α":20s} {v540_avg_gap:>+10.5f} {v540_vs308:>8d} {(-0.03016)-v540_avg_gap:>+10.5f} {0.62235-v540_avg_gap:>10.5f}')
    
    best_avg_gap = min(v534_avg_gap, v540_avg_gap)
    if v540_avg_gap <= v534_avg_gap:
        best_configs = v540_cfgs
        best_data = v540_data
        best_name = 'V540_Configs'
    else:
        best_configs = v534_cfgs
        best_data = v534_data
        best_name = 'V534_Base'
    
    log.info(f'\n🏆 Best: {best_name} (avg_gap={best_avg_gap:+.5f})')
    
    # Create submission
    log.info(f'\nGenerating submission with {best_name}...')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    
    for target in TARGETS:
        oofs = best_data[target]['oofs']
        tests = best_data[target]['tests']
        y = best_data[target]['y']
        n_seeds = len(oofs)
        
        n_test = len(test_df)
        oofs_2d = np.column_stack(oofs)
        
        # Average test predictions
        test_avg = np.zeros(n_test)
        for si in range(n_seeds):
            test_avg += tests[si]
        test_avg /= n_seeds
        
        test_var = np.zeros(n_test)
        for si in range(n_seeds):
            test_var += (tests[si] - test_avg) ** 2
        test_std = np.sqrt(test_var / n_seeds)
        
        # Fit Ridge with V542 alpha
        alpha = v542_alphas[target]
        meta = Ridge(alpha=alpha)
        X_train_meta = np.column_stack([np.mean(oofs_2d, axis=1), np.std(oofs_2d, axis=1)])
        meta.fit(X_train_meta, y)
        
        X_test_meta = np.column_stack([test_avg, test_std])
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
    sub_path = SUBMIT / f'submission_v545_{best_name.lower()}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f'📁 Submission: {sub_path}')
    
    # Save results
    result = {
        'version': 'V545',
        'hypothesis': 'v534_base_vs_v540_configs_with_v542_alphas',
        'v534_base_gap': float(v534_avg_gap),
        'v540_configs_gap': float(v540_avg_gap),
        'best': best_name,
        'best_gap': float(best_avg_gap),
        'v534_target_gaps': {k: float(v) for k, v in v534_gaps.items()},
        'v540_target_gaps': {k: float(v) for k, v in v540_gaps.items()},
        'submission_file': str(sub_path.name),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v545_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')

if __name__ == '__main__':
    main()
