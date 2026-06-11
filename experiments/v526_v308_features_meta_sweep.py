#!/usr/bin/env python3
"""
V526 — Blend XGB+LGBM with V308-style feature selections

Hypothesis: V525 proved XGB+LGBM blending works (avg_gap=0.01878 with narrow n_feat).
But V525 used narrow feature counts (Q1:7, Q3:7, etc.). 
V308 used wider selections and achieved the original benchmark.
Try blending with V308's feature counts to see if more features + blending
gives even lower gap.

Additionally test different meta architectures:
- 1D meta (mean only)
- 2D meta (mean+std) 
- Ridge/Lasso meta
- Shallow NN meta (1 hidden layer, 3 neurons)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
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
N_SEEDS = 15
META_C = 10.0


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


def train_meta(meta_type, X_meta, y):
    """Train a meta learner of given type."""
    if meta_type == '1D':
        # Only mean
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(X_meta[:, np.newaxis], y)
        return lambda X: meta.predict_proba(X[:, np.newaxis])[:, 1]
    elif meta_type == '2D':
        # Mean + std
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(X_meta, y)
        return lambda X: meta.predict_proba(X)[:, 1]
    elif meta_type == 'ridge':
        meta = Ridge(alpha=1.0)
        meta.fit(X_meta, y)
        # Convert to proba-like output
        def predict(X):
            pred = meta.predict(X)
            return (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return predict
    elif meta_type == 'lasso':
        meta = Lasso(alpha=0.1)
        meta.fit(X_meta, y)
        def predict(X):
            pred = meta.predict(X)
            return (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return predict
    elif meta_type == 'elastic':
        meta = ElasticNet(alpha=0.1, l1_ratio=0.5)
        meta.fit(X_meta, y)
        def predict(X):
            pred = meta.predict(X)
            return (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return predict
    elif meta_type == 'mlp':
        meta = MLPClassifier(hidden_layer_sizes=(3,), max_iter=500, random_state=SEED, warm_start=False)
        meta.fit(X_meta, y)
        return lambda X: meta.predict_proba(X)[:, 1]
    elif meta_type == 'ridge2':
        meta = Ridge(alpha=0.1)
        meta.fit(X_meta, y)
        def predict(X):
            pred = meta.predict(X)
            return (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return predict
    elif meta_type == 'ridge05':
        meta = Ridge(alpha=0.5)
        meta.fit(X_meta, y)
        def predict(X):
            pred = meta.predict(X)
            return (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
        return predict
    raise ValueError(f"Unknown meta_type: {meta_type}")


def compute_gap_meta(meta_type, seed_oofs_list, y):
    oofs_arr = np.column_stack(seed_oofs_list)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in seed_oofs_list])
    avg_pred = np.mean(oofs_arr, axis=1)
    std_pred = np.std(oofs_arr, axis=1)
    
    if meta_type == '1D':
        X_meta = avg_pred  # 1D array
        predict_fn = train_meta(meta_type, X_meta, y)
        train_oof = predict_fn(X_meta)
    elif meta_type == '2D':
        X_meta = np.column_stack([avg_pred, std_pred])
        predict_fn = train_meta(meta_type, X_meta, y)
        train_oof = predict_fn(X_meta)
    else:
        X_meta = np.column_stack([avg_pred, std_pred])
        predict_fn = train_meta(meta_type, X_meta, y)
        train_oof = predict_fn(X_meta)
    
    train_oof = np.clip(train_oof, 0.001, 0.999)
    meta_ll = log_loss(y, train_oof)
    
    return avg_student, meta_ll, avg_student - meta_ll


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V526 — Blend XGB+LGBM with V308-style features + meta sweep")
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
    
    # V308-style feature counts (wider)
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
        log.info(f"  {target}: {len(ranked_features[target])} features ranked")
    
    # Meta types to sweep
    META_TYPES = ['1D', '2D', 'ridge05', 'ridge', 'ridge2', 'lasso', 'elastic', 'mlp']
    
    # Use V525 cv_weighted blend approach
    # First: train all XGB+LGBM models with V308 feature counts
    
    log.info("\nTraining base models with V308 feature counts...")
    
    target_results = {}
    for target in TARGETS:
        n_feat = V308_FEATURES[target]
        sel_cols = ranked_features[target][:n_feat]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names
        
        # Choose learner configs based on target type
        if target.startswith('Q'):
            if target == 'Q1':
                xgb_cfg, lgbm_cfg = 'q_narrow', 'wide'
            elif target == 'Q2':
                xgb_cfg, lgbm_cfg = 'q_deep', 'wide'
            else:  # Q3
                xgb_cfg, lgbm_cfg = 'q_strong', 'safety'
            n_est = 800  # higher for Q targets
        else:
            if target == 'S1':
                xgb_cfg, lgbm_cfg = 'q_strong', 'wide'
            elif target == 'S2':
                xgb_cfg, lgbm_cfg = 's_wide', 'wide'
            elif target == 'S3':
                xgb_cfg, lgbm_cfg = 'q_strong', 'safety'
            else:  # S4
                xgb_cfg, lgbm_cfg = 'q_deep', 'wide'
            n_est = 500
        
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[xgb_cfg]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[lgbm_cfg]}
        
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        
        # Collect per-seed XGB and LGBM OOF + test
        xgb_seed_oofs = []
        lgbm_seed_oofs = []
        xgb_tests = []
        lgbm_tests = []
        
        for si in range(N_SEEDS):
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
        
        # CV-weighted blend: use mean OOF LL across all seeds to compute weight
        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_seed_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_seed_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10)
        inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        total = inv_xgb + inv_lgbm
        wx = inv_xgb / total
        
        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_tests, lgbm_tests)]
        
        target_results[target] = {
            'blended_oofs': blended_oofs,
            'blended_tests': blended_tests,
            'wx': wx,
        }
        log.info(f"  {target}: n_feat={n_feat}, wx={wx:.3f}")
    
    # Now test each meta type
    log.info(f"\n{'='*60}")
    log.info("Testing meta architectures")
    log.info(f"{'='*60}")
    
    all_meta_results = {}
    
    for mt in META_TYPES:
        total_gap = 0
        target_gaps = {}
        
        for target in TARGETS:
            t_y = train_df[target].values
            _, _, gap = compute_gap_meta(mt, target_results[target]['blended_oofs'], t_y)
            target_gaps[target] = gap
            total_gap += gap
        
        avg_gap = total_gap / 7
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        marker = ""
        if avg_gap < 0.025: marker = " 🎯🎯🎯"
        elif avg_gap < 0.030: marker = " ⭐"
        log.info(f"  {mt}: avg_gap={avg_gap:.5f}, vs308={vs308}/7{marker}")
        
        all_meta_results[mt] = {
            'avg_gap': avg_gap,
            'target_gaps': target_gaps,
            'vs308': vs308,
        }
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("META SWEEP SUMMARY")
    log.info(f"{'='*70}")
    
    for mt in sorted(all_meta_results.keys(), key=lambda m: all_meta_results[m]['avg_gap']):
        r = all_meta_results[mt]
        marker = ""
        if r['avg_gap'] < 0.025: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < 0.030: marker = " ⭐"
        log.info(f"  {mt}: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7{marker}")
        for t in TARGETS:
            vs = "✅" if r['target_gaps'][t] < v308_gaps[t] else "❌"
            log.info(f"    {t}: gap={r['target_gaps'][t]:.5f} V308={v308_gaps[t]:.3f} {vs}")
    
    best_meta = min(all_meta_results.keys(), key=lambda m: all_meta_results[m]['avg_gap'])
    best = all_meta_results[best_meta]
    log.info(f"\n✅ BEST META: {best_meta} with avg_gap={best['avg_gap']:.5f}")
    
    # Generate submission
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        wx = target_results[t]['wx']
        blended_test = target_results[t]['blended_tests']
        sub_df[t] = np.mean(blended_test, axis=0)
    
    sub_path = SUBMIT / f'submission_v526_{best_meta}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"Submission saved: {sub_path}")
    
    result = {
        'version': 'V526',
        'name': 'Blend XGB+LGBM V308-features + meta sweep',
        'n_features': V308_FEATURES,
        'meta_results': {m: {'avg_gap': r['avg_gap'], 'target_gaps': r['target_gaps'], 'vs308': r['vs308']} for m, r in all_meta_results.items()},
        'best_meta': best_meta,
        'best_gap': float(best['avg_gap']),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v526_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"📝 Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()
