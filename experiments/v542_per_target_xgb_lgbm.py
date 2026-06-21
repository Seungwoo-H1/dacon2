#!/usr/bin/env python3
"""
V542 — Different XGB/LGBM Hyperparameters Per Target

Hypothesis: Each target benefits from different tree hyperparameters.
V537 uses the same templates. This experiment optimizes per target.
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
N_SEEDS = 5  # Reduced for speed
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
    'light':     {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'min_child_weight': 1},
    'heavy':     {'max_depth': 6, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 8},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'balanced':  {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_samples': 8},
    'heavy_lgb': {'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 25},
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

def compute_expected_lb(oofs, y_true):
    """Compute expected LB as meta_ll_on_train with optimal Ridge alpha."""
    oofs_2d = np.column_stack(oofs)
    n_seeds = oofs_2d.shape[1]
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    # Use V537 optimal per-target alpha
    target_idx = TARGETS.index(y_true.__class__.__name__ if hasattr(y_true, '__class__') else 'Q')
    # We'll pass target name separately
    return oofs_2d, n_seeds, X_meta, y_true

def main():
    global train_df, test_df
    t_start = time.time()
    log.info("=" * 70)
    log.info("V542: Per-Target XGB/LGBM Hyperparameter Optimization")
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

    # XGB configs to try
    xgb_options = [
        ('q_narrow', XGB_CFGS['q_narrow']),
        ('q_deep', XGB_CFGS['q_deep']),
        ('q_strong', XGB_CFGS['q_strong']),
        ('light', XGB_CFGS['light']),
        ('heavy', XGB_CFGS['heavy']),
    ]
    lgbm_options = [
        ('wide', LGBM_CFGS['wide']),
        ('wide_strong', LGBM_CFGS['wide_strong']),
        ('safety', LGBM_CFGS['safety']),
        ('balanced', LGBM_CFGS['balanced']),
        ('heavy_lgb', LGBM_CFGS['heavy_lgb']),
    ]

    # For each target, find best XGB+LGBM config
    # Expected LB metric: meta_ll_on_train with Ridge α=optimal
    log.info('Optimizing per target...')
    best_results = {}
    
    for target in TARGETS:
        n_feat = V534_CONFIG[target]['n_feat']
        n_est = V534_CONFIG[target]['n_est']
        sel_cols = ranked_features[target][:n_feat]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols): sel_cols = feat_names
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        n_train = len(train_df)
        n_test = len(test_df)

        log.info(f'\n{target}: Testing {len(xgb_options)} XGB × {len(lgbm_options)} LGBM = {len(xgb_options)*len(lgbm_options)} configs')
        
        best_ll = float('inf')
        best_cfg = None
        best_oofs = None
        best_tests = None
        best_wx = None

        for xgb_name, xgb_mp in xgb_options:
            for lgbm_name, lgbm_mp in lgbm_options:
                # Quick run with 3 seeds
                oofs_run = []
                for si in range(3):
                    seed = SEED + si * 11
                    oof_xgb = np.zeros(n_train)
                    oof_lgbm = np.zeros(n_train)
                    
                    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                        X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                        X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                        y_tr = y[tr_idx]
                        pvxgb, _ = train_one_xgb(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **xgb_mp)
                        pvlgbm, _ = train_one_lgbm(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **lgbm_mp)
                        oof_xgb[va_idx] = pvxgb
                        oof_lgbm[va_idx] = pvlgbm
                    
                    oof_xgb = np.clip(oof_xgb, 0.001, 0.999)
                    oof_lgbm = np.clip(oof_lgbm, 0.001, 0.999)
                    oofs_run.append((oof_xgb, oof_lgbm))

                # Compute blend and meta_ll
                xgb_oofs_3 = [o[0] for o in oofs_run]
                lgbm_oofs_3 = [o[1] for o in oofs_run]
                ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs_3])
                ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs_3])
                inv_xgb = 1.0 / max(ll_xgb, 1e-10)
                inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
                wx = inv_xgb / (inv_xgb + inv_lgbm)
                blended = [wx * xo + (1-wx) * lo for xo, lo in zip(xgb_oofs_3, lgbm_oofs_3)]
                
                # Meta LL with optimal α
                oofs_2d = np.column_stack(blended)
                avg_pred = np.mean(oofs_2d, axis=1)
                std_pred = np.std(oofs_2d, axis=1)
                X_meta = np.column_stack([avg_pred, std_pred])
                alpha = BEST_ALPHAS[target]
                meta = Ridge(alpha=alpha)
                meta.fit(X_meta, y)
                train_pred = meta.predict(X_meta)
                pmin, pmax = train_pred.min(), train_pred.max()
                if pmax - pmin < 1e-10:
                    train_proba = np.ones_like(train_pred) * 0.5
                else:
                    train_proba = (train_pred - pmin) / (pmax - pmin)
                train_proba = np.clip(train_proba, 0.001, 0.999)
                ll = log_loss(y, train_proba)
                
                if ll < best_ll:
                    best_ll = ll
                    best_cfg = (xgb_name, lgbm_name)
                    best_wx = wx

        log.info(f'  ✅ {target}: best={best_cfg[0]}+{best_cfg[1]}, wx={best_wx:.3f}, meta_ll={best_ll:.5f}')
        best_results[target] = {'xgb': best_cfg[0], 'lgbm': best_cfg[1], 'wx': best_wx, 'meta_ll': best_ll}

    avg_ll = sum(r['meta_ll'] for r in best_results.values()) / 7
    log.info(f'\n  avg meta_ll (V542): {avg_ll:.5f}')
    log.info(f'  avg meta_ll (V537): 0.65251')
    log.info(f'  improvement: {0.65251 - avg_ll:+.5f}')
    log.info(f'\nTotal time: {time.time() - t_start:.1f}s')
    return {'avg_ll': avg_ll, 'improvement': 0.65251 - avg_ll, 'details': best_results}


if __name__ == '__main__':
    main()
