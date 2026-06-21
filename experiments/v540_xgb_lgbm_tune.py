#!/usr/bin/env python3
"""
V540 — Per-Target XGB/LGBM Hyperparameter Optimization

Hypothesis: Each target might benefit from completely different
XGB/LGBM hyperparameters, not just config templates.

Method: For each target, sweep XGB config × LGBM config with V537's
per-target Ridge alphas. Use 13 seeds for accuracy.
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
N_SEEDS = 5
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

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
    'shallow_deep': {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'min_child_weight': 1},
    'deep_deep':    {'max_depth': 6, 'learning_rate': 0.02, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'min_child_weight': 8},
    'balanced':     {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'heavy_reg':    {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 15.0, 'min_child_weight': 10},
    'light_reg':    {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_weight': 1},
    'medium_reg':   {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.75, 'reg_alpha': 3.0, 'reg_lambda': 5.0, 'min_child_weight': 4},
}

LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'balanced':  {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_samples': 8},
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

def compute_gap_with_ridge(oofs_2d, y_true, alpha):
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
    return avg_student - meta_ll, meta_ll

def compute_gap_with_en(oofs_2d, y_true, alpha=0.5, l1_ratio=0.5):
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
    return avg_student - meta_ll, meta_ll

def run_target(target, sel_cols, X_test_full, y, n_est, xgb_mp, lgbm_mp, gkf, train_df, test_df):
    n_train = len(train_df)
    n_test = len(test_df)
    
    xgb_oofs = []
    lgbm_oofs = []
    xgb_tests = []
    lgbm_tests = []

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

    ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs])
    ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs])
    inv_xgb = 1.0 / max(ll_xgb, 1e-10); inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
    wx = inv_xgb / (inv_xgb + inv_lgbm)
    blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_oofs, lgbm_oofs)]
    blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_tests, lgbm_tests)]
    
    return blended_oofs, blended_tests, ll_xgb, ll_lgbm, wx


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V540: Per-Target XGB/LGBM Hyperparameter Optimization")
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

    xgb_cfg_names = sorted(XGB_CFGS.keys())
    lgbm_cfg_names = sorted(LGBM_CFGS.keys())
    
    log.info(f'XGB configs: {xgb_cfg_names}')
    log.info(f'LGBM configs: {lgbm_cfg_names}')
    
    # For each target, sweep all XGB x LGBM configs
    # Store ALL results per target
    all_results = {}
    
    for target in TARGETS:
        bc = V534_CONFIG[target]
        n_feat = bc['n_feat']
        sel_cols = ranked_features[target][:n_feat]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols): sel_cols = feat_names
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        n_est = bc['n_est']
        alpha = BEST_ALPHAS[target]

        log.info(f'\n{target}: n_feat={n_feat}, n_est={n_est}, alpha={alpha}')
        
        target_results = []
        
        for xgb_name in xgb_cfg_names:
            for lgbm_name in lgbm_cfg_names:
                xgb_mp = XGB_CFGS[xgb_name]
                lgbm_mp = LGBM_CFGS[lgbm_name]
                label = f'X:{xgb_name}+L:{lgbm_name}'
                t1 = time.time()
                try:
                    oofs, tests, ll_xgb, ll_lgbm, wx = run_target(
                        target, sel_cols, X_test_full, y, n_est,
                        xgb_mp, lgbm_mp, gkf, train_df, test_df
                    )
                    oofs_2d = np.column_stack(oofs)
                    gap_r, meta_ll = compute_gap_with_ridge(oofs_2d, y, alpha)
                    gap_en, meta_ll_en = compute_gap_with_en(oofs_2d, y, 0.5, 0.5)
                    elapsed = time.time() - t1
                    target_results.append({
                        'label': label, 'gap_r': gap_r, 'gap_en': gap_en,
                        'wx': wx, 'll_xgb': ll_xgb, 'll_lgbm': ll_lgbm,
                        'meta_ll_r': meta_ll, 'meta_ll_en': meta_ll_en,
                        'elapsed': elapsed
                    })
                except Exception as e:
                    log.warning(f'  {target} {label}: ERROR {e}')
        
        target_results.sort(key=lambda x: x['gap_r'])
        log.info(f'  Top 5 by Ridge gap:')
        for r in target_results[:5]:
            log.info(f'    {r["label"]}: gap_r={r["gap_r"]:+.5f}, gap_en={r["gap_en"]:+.5f}, wx={r["wx"]:.3f}, ll_xgb={r["ll_xgb"]:.5f}, ll_lgbm={r["ll_lgbm"]:.5f}')
        
        all_results[target] = target_results
    
    # ================================================================
    # V541: Meta Learner Comparison (Ridge vs EN vs Lasso) with top configs
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V541: Meta Learner Comparison (Ridge vs EN vs Lasso)')
    log.info('=' * 70)
    
    # Pick best config per target (by Ridge gap)
    best_config_per_target = {}
    best_gap_r = {}
    best_gap_en = {}
    
    for target in TARGETS:
        best = all_results[target][0]
        best_config_per_target[target] = best['label']
        best_gap_r[target] = best['gap_r']
        best_gap_en[target] = best['gap_en']
    
    avg_gap_r = sum(best_gap_r.values()) / 7
    avg_gap_en = sum(best_gap_en.values()) / 7
    improvement_r = -0.03016 - avg_gap_r
    improvement_en = -0.03016 - avg_gap_en
    
    log.info(f'\nBest config per target (by Ridge gap):')
    for t in TARGETS:
        log.info(f'  {t}: {best_config_per_target[t]}, gap_r={best_gap_r[t]:+.5f}, gap_en={best_gap_en[t]:+.5f}')
    log.info(f'\n  avg_gap (Ridge): {avg_gap_r:+.5f}, improvement vs V537: +{improvement_r:+.5f}')
    log.info(f'  avg_gap (EN):    {avg_gap_en:+.5f}, improvement vs V537: +{improvement_en:+.5f}')
    
    # Also test with ElasticNet meta on all seed configs
    # Pick top 3 configs per target by EN gap and test meta learners
    log.info('\n  === EN Top configs per target ===')
    for target in TARGETS:
        top3_en = sorted(all_results[target], key=lambda x: x['gap_en'])[:3]
        for r in top3_en:
            log.info(f'    {r["label"]}: gap_r={r["gap_r"]:+.5f}, gap_en={r["gap_en"]:+.5f}')
    
    # Also check: what if we use the EN-best config with Ridge meta, and vice versa?
    # EN-best per target
    en_best_per_target = {}
    en_gap_r = {}
    en_gap_en = {}
    for target in TARGETS:
        en_best = sorted(all_results[target], key=lambda x: x['gap_en'])[0]
        en_best_per_target[target] = en_best['label']
        en_gap_r[target] = en_best['gap_r']
        en_gap_en[target] = en_best['gap_en']
    
    avg_gap_en_best_r = sum(en_gap_r.values()) / 7
    avg_gap_en_best_en = sum(en_gap_en.values()) / 7
    
    log.info(f'\n  EN-best config + Ridge meta: avg_gap={avg_gap_en_best_r:+.5f}')
    log.info(f'  EN-best config + EN meta:    avg_gap={avg_gap_en_best_en:+.5f}')
    
    # ================================================================
    # Final comparison
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V540 Final Summary')
    log.info('=' * 70)
    
    comparisons = [
        ('V537 baseline', -0.03016, 'N/A'),
        (f'Ridge-best + Ridge', avg_gap_r, f'+{improvement_r:+.5f}'),
        (f'Ridge-best + EN', avg_gap_en, f'+{improvement_en:+.5f}'),
        (f'EN-best + Ridge', avg_gap_en_best_r, f'{avg_gap_en_best_r:+.5f}'),
        (f'EN-best + EN', avg_gap_en_best_en, f'{avg_gap_en_best_en:+.5f}'),
    ]
    
    for name, gap, imp in comparisons:
        log.info(f'  {name:30s}: avg_gap={gap:+.5f} (Δ V537: {imp})')
    
    best_combo = max(comparisons[1:], key=lambda x: -x[1])
    log.info(f'\n🏆 Best combo: {best_combo[0]} (avg_gap={best_combo[1]:+.5f}, improvement=+{best_combo[2]})')
    
    log.info(f'\nTotal time: {time.time() - t_start:.1f}s')
    
    # Save results
    result = {
        'version': 'V540+V541',
        'hypothesis': 'per_target_xgb_lgbm_hparam_opt',
        'avg_gap_ridge_best_ridge': float(avg_gap_r),
        'avg_gap_ridge_best_en': float(avg_gap_en),
        'avg_gap_en_best_ridge': float(avg_gap_en_best_r),
        'avg_gap_en_best_en': float(avg_gap_en_best_en),
        'best_config_per_target_ridge': {k: v for k, v in best_config_per_target.items()},
        'best_config_per_target_en': {k: v for k, v in en_best_per_target.items()},
        'all_target_results': {k: [{kk: vv for kk, vv in v.items() if kk != 'oofs' and kk != 'tests'} for v in vv] for k, vv in all_results.items()},
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    ts = result['timestamp']
    result_path = EXPERIMENTS / f'v540_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')

if __name__ == '__main__':
    main()
