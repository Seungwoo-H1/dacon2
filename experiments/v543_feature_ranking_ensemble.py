#!/usr/bin/env python3
"""
V543 — Ensemble of Feature Rankings + V540 Configs

Hypothesis: Feature ranking changes across seeds. Ensemble multiple
rankings → more stable features → better gap.

Method:
1. Rank features with 3 different seeds
2. For each target, use features from each ranking
3. Test: single ranking (best), ensemble ranking (union/intersection),
   and per-rank-model stacking
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

# V540 configs + optimal alphas from V542
V540_TOP = {
    'Q1':  ('s_strong', 'heavy_lgb', 600, 3, 0.0001),
    'Q2':  ('q_narrow', 'heavy_lgb', 800, 10, 10.0),
    'Q3':  ('heavy_reg', 'light_lgb', 500, 7, 0.0001),
    'S1':  ('s_strong', 'heavy_lgb', 500, 3, 0.0001),
    'S2':  ('light_reg', 'heavy_lgb', 500, 7, 0.03),
    'S3':  ('s_strong', 'heavy_lgb', 1000, 23, 10.0),
    'S4':  ('s_strong', 'heavy_lgb', 300, 20, 0.0001),
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

def rank_features_multi(df, feat_cols, target, seeds):
    """Rank features with multiple seeds and return averaged rank."""
    all_ranks = {}
    for seed in seeds:
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
        for col, imp_val in zip(feat_cols, imp):
            all_ranks.setdefault(col, []).append(imp_val)
        del m, ds, X
    avg_imp = {col: np.mean(ims) for col, ims in all_ranks.items()}
    ranked = sorted(avg_imp.keys(), key=lambda c: -avg_imp[c])
    return ranked

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


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V543: Feature Ranking Ensemble + V540 Configs")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
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
    all_feat_cols = get_feature_cols(train_df)
    
    # Rank features with 3 different seeds
    ranking_seeds = [42, 123, 456]
    log.info(f'Ranking features with seeds {ranking_seeds}...')
    
    feature_rankings = {}
    for target in TARGETS:
        log.info(f'  {target}...')
        feature_rankings[target] = rank_features_multi(train_df, all_feat_cols, target, ranking_seeds)
    
    # ================================================================
    # Test: single seed vs ensemble ranking
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V543: Single Seed vs Ensemble Ranking')
    log.info('=' * 70)
    
    results = {}
    
    for n_feat_option in [3, 5, 7, 10, 15, 20]:
        log.info(f'\n--- n_feat={n_feat_option} ---')
        for ranking_name, use_ensemble in [('single_seed', False), ('ensemble', True)]:
            target_gaps = {}
            target_configs = {}
            
            for target in TARGETS:
                bc = V540_TOP[target]
                xgb_cfg, lgbm_cfg, n_est, default_n_feat, alpha = bc
                
                # Use n_feat_option for targets where default matches, otherwise use min
                if target in ['Q1', 'S1']:
                    nf = min(n_feat_option, 5)
                elif target in ['Q2']:
                    nf = min(n_feat_option, 14)
                elif target in ['Q3']:
                    nf = min(n_feat_option, 10)
                elif target in ['S2']:
                    nf = min(n_feat_option, 10)
                elif target in ['S3']:
                    nf = min(n_feat_option, 30)
                elif target in ['S4']:
                    nf = min(n_feat_option, 25)
                else:
                    nf = n_feat_option
                
                if use_ensemble:
                    ranked = feature_rankings[target]
                else:
                    # Use default seed (42) ranking
                    y = train_df[target].values.astype(np.float64)
                    X = train_df[all_feat_cols].fillna(0).values.astype(np.float64)
                    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
                              'scale_pos_weight': spw, 'random_state': 42, 'force_row_wise': True, 'n_jobs': 1}
                    sn = [sanitize_col(c) for c in all_feat_cols]
                    ds = lgb.Dataset(X, label=y, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=50)
                    imp = m.feature_importance(importance_type='gain')
                    ranked = sorted(zip(all_feat_cols, imp), key=lambda x: -x[1])
                    ranked = [r[0] for r in ranked]
                
                sel_cols = ranked[:nf]
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
                
                oofs_2d = np.column_stack(xgb_oofs + lgbm_oofs)
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
            vs308 = sum(1 for t in TARGETS if target_gaps[t] < V308_GAPS[t])
            key = f'nfeat={n_feat_option}_{ranking_name}'
            results[key] = {'avg_gap': avg_gap, 'vs308': vs308, 'target_gaps': target_gaps}
            log.info(f'  {key}: avg_gap={avg_gap:+.5f}, vs308={vs308}/7')
    
    # ================================================================
    # Find best config and generate submission
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V543 Final Results')
    log.info('=' * 70)
    
    best_key = min(results, key=lambda k: results[k]['avg_gap'])
    best = results[best_key]
    
    log.info(f'\n🏆 Best: {best_key}')
    log.info(f'  avg_gap: {best["avg_gap"]:+.5f}')
    log.info(f'  improvement vs V537: +{(-0.03016) - best["avg_gap"]:+.5f}')
    log.info(f'  vs308: {best["vs308"]}/7')
    log.info(f'  Expected LB: {0.62235 - best["avg_gap"]:.5f}')
    
    log.info(f'\n  Target gaps:')
    for t in TARGETS:
        vs = "✅" if best['target_gaps'][t] < V308_GAPS[t] else "❌"
        log.info(f'    {t}: {best["target_gaps"][t]:+.5f} {vs}')
    
    # Save results
    result = {
        'version': 'V543',
        'hypothesis': 'feature_ranking_ensemble_v540_configs',
        'all_results': {k: {'avg_gap': v['avg_gap'], 'vs308': v['vs308']} for k, v in results.items()},
        'best': best_key,
        'avg_gap': best['avg_gap'],
        'improvement': (-0.03016) - best['avg_gap'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v543_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')
    
    # Summary table
    log.info('\n  === Results Summary ===')
    log.info(f'  {"Config":35s} {"avg_gap":>10} {"vs308":>8} {"Δ V537":>10}')
    log.info('  ' + '-' * 67)
    for k, v in sorted(results.items(), key=lambda x: x[1]['avg_gap']):
        log.info(f'  {k:35s} {v["avg_gap"]:>+10.5f} {v["vs308"]:>8d} {(-0.03016)-v["avg_gap"]:>+10.5f}')

if __name__ == '__main__':
    main()
