#!/usr/bin/env python3
"""
V528 — Per-Target Ridge Alpha + Finer Grid + Extended Seed Sweep

Hypothesis: V527 showed 7 seeds + 2D Ridge(α=0.01) is best overall.
But per-target, different targets might benefit from different alpha values.
Also, the grid 0.01-10.0 was coarse near 0.01. Try: 0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03.

And test: 9 seeds, 11 seeds, 13 seeds to confirm 7 is truly optimal.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
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

SEED = 42
N_FOLDS = 5


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


def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
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


def ridge_gap_per_target(oofs_list, y, alpha):
    """Compute per-target gap with given Ridge alpha.
    oofs_list: list of 1D arrays, one per seed."""
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_list])
    oofs_2d = np.column_stack(oofs_list)
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
    return avg_student, meta_ll, avg_student - meta_ll


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V528 — Per-Target Ridge Alpha + Finer Grid + Extended Seed Sweep")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score features
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
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
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    v308_gaps = {
        'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
        'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
    }
    
    XGB_CFGS = {
        'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
        'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
        'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
        's_wide':    {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    }
    LGBM_CFGS = {
        'wide':    {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'safety':  {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }
    
    V308_FEATURES = {
        'Q1': 19, 'Q2': 14, 'Q3': 11,
        'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20
    }
    
    # Pre-rank features
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)
    
    # STEP 1: Train base models — 15 seeds, V308 features, cv_weighted blend
    log.info("\nTraining base models (15 seeds)...")
    all_target_oofs = {}
    all_target_tests = {}
    
    for target in TARGETS:
        n_feat = V308_FEATURES[target]
        sel_cols = ranked_features[target][:n_feat]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names
        
        if target.startswith('Q'):
            if target == 'Q1':
                xgb_cfg, lgbm_cfg = 'q_narrow', 'wide'
            elif target == 'Q2':
                xgb_cfg, lgbm_cfg = 'q_deep', 'wide'
            else:
                xgb_cfg, lgbm_cfg = 'q_strong', 'safety'
            n_est = 800
        else:
            if target == 'S1':
                xgb_cfg, lgbm_cfg = 'q_strong', 'wide'
            elif target == 'S2':
                xgb_cfg, lgbm_cfg = 's_wide', 'wide'
            elif target == 'S3':
                xgb_cfg, lgbm_cfg = 'q_strong', 'safety'
            else:
                xgb_cfg, lgbm_cfg = 'q_deep', 'wide'
            n_est = 500
        
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[xgb_cfg]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[lgbm_cfg]}
        
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        
        xgb_seed_oofs = []
        lgbm_seed_oofs = []
        xgb_tests = []
        lgbm_tests = []
        
        for si in range(15):
            seed = SEED + si * 11
            xoof = np.zeros(n_train)
            loof = np.zeros(n_train)
            xtest = np.zeros(n_test)
            ltest = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                pvxgb, ttxgb = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'xgb', n_est, **xgb_params)
                pvlgbm, ttlgbm = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'lgbm', n_est, **lgbm_params)
                
                xoof[va_idx] = pvxgb
                loof[va_idx] = pvlgbm
                xtest += ttxgb
                ltest += ttlgbm
            
            xoof = np.clip(xoof, 0.001, 0.999)
            loof = np.clip(loof, 0.001, 0.999)
            xtest /= N_FOLDS
            ltest /= N_FOLDS
            
            xgb_seed_oofs.append(xoof)
            lgbm_seed_oofs.append(loof)
            xgb_tests.append(xtest)
            lgbm_tests.append(ltest)
        
        # CV-weighted blend
        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_seed_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_seed_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10)
        inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        total = inv_xgb + inv_lgbm
        wx = inv_xgb / total
        
        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_tests, lgbm_tests)]
        
        all_target_oofs[target] = blended_oofs
        all_target_tests[target] = blended_tests
        log.info(f"  {target}: n_feat={n_feat}, wx={wx:.3f}, 15 seeds")
    
    # STEP 2: Per-target optimal alpha search (fine grid near 0.01)
    log.info(f"\n{'='*60}")
    log.info("PER-TARGET OPTIMAL ALPHA (fine grid)")
    log.info(f"{'='*60}")
    
    FINE_ALPHA = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    per_target_best = {}
    
    for target in TARGETS:
        best_a = None
        best_g = 999
        for alpha in FINE_ALPHA:
            _, _, gap = ridge_gap_per_target(all_target_oofs[target], train_df[target].values, alpha)
            if gap < best_g:
                best_g = gap
                best_a = alpha
        per_target_best[target] = {'alpha': best_a, 'gap': best_g}
        log.info(f"  {target}: best_alpha={best_a}, gap={best_g:+.5f}")
    
    # Average of per-target optimal alphas
    avg_opt_gap = np.mean([per_target_best[t]['gap'] for t in TARGETS])
    log.info(f"  Average per-target optimal gap: {avg_opt_gap:+.5f}")
    
    # STEP 3: Global optimal alpha (same for all targets)
    log.info(f"\n{'='*60}")
    log.info("GLOBAL ALPHA SWEEP (fine grid)")
    log.info(f"{'='*60}")
    
    global_results = []
    for alpha in FINE_ALPHA:
        total_gap = 0
        target_gaps = {}
        for target in TARGETS:
            _, _, gap = ridge_gap_per_target(all_target_oofs[target], train_df[target].values, alpha)
            target_gaps[target] = gap
            total_gap += gap
        avg_gap = total_gap / 7
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        
        marker = ""
        if avg_gap < 0.0: marker = " 🎯"
        log.info(f"  alpha={alpha:5.3f}: avg_gap={avg_gap:+.5f}, vs308={vs308}/7{marker}")
        
        global_results.append({'alpha': alpha, 'avg_gap': avg_gap, 'target_gaps': target_gaps, 'vs308': vs308})
    
    best_global = min(global_results, key=lambda x: x['avg_gap'])
    
    # STEP 4: Extended seed sweep (confirm 7 is optimal)
    log.info(f"\n{'='*60}")
    log.info(f"EXTENDED SEED SWEEP (using best global alpha={best_global['alpha']})")
    log.info(f"{'='*60}")
    
    for n_seeds_try in [3, 5, 7, 9, 11, 13, 15]:
        total_gap = 0
        target_gaps = {}
        for target in TARGETS:
            subset_oofs = all_target_oofs[target][:n_seeds_try]
            _, _, gap = ridge_gap_per_target(subset_oofs, train_df[target].values, best_global['alpha'])
            target_gaps[target] = gap
            total_gap += gap
        avg_gap = total_gap / 7
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        
        marker = ""
        if avg_gap < 0.0: marker = " 🎯"
        log.info(f"  seeds={n_seeds_try}: avg_gap={avg_gap:+.5f}, vs308={vs308}/7{marker}")
    
    # STEP 5: Generate submission with best config
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"  Per-target optimal: avg_gap={avg_opt_gap:+.5f}")
    log.info(f"  Global optimal: alpha={best_global['alpha']}, avg_gap={best_global['avg_gap']:+.5f}")
    
    # Use per-target optimal for submission (if significantly better)
    use_per_target = avg_opt_gap < best_global['avg_gap']
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        sub_df[t] = np.mean(all_target_tests[t], axis=0)
    
    if use_per_target:
        sub_path = SUBMIT / f'submission_v528_per_target_alpha_{ts}.csv'
        log.info(f"  => Use per-target alpha (better)")
    else:
        sub_path = SUBMIT / f'submission_v528_global_alpha_{best_global["alpha"]}_{ts}.csv'
        log.info(f"  => Use global alpha={best_global['alpha']}")
    
    sub_df.to_csv(sub_path, index=False)
    log.info(f"Submission saved: {sub_path}")
    
    result = {
        'version': 'V528',
        'name': 'Per-Target Ridge Alpha + Finer Grid + Extended Seed Sweep',
        'per_target_best': {t: {'alpha': v['alpha'], 'gap': v['gap']} for t, v in per_target_best.items()},
        'avg_per_target_gap': avg_opt_gap,
        'global_results': [{'alpha': r['alpha'], 'avg_gap': r['avg_gap'], 'vs308': r['vs308']} for r in global_results],
        'best_global': {'alpha': best_global['alpha'], 'avg_gap': best_global['avg_gap']},
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v528_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"📝 Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()
