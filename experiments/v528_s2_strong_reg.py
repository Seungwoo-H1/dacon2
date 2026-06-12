#!/usr/bin/env python3
"""
V528 — S2 strong regularization + per-target n_est sweep

V527 findings:
- S2 n_feat=7 is optimal (more features = worse due to overfitting)
- S2 n_est=500 was best (n_est=300→0.03744, n_est=500→0.03744, similar)
- S1_n3 + S4_n15 + S2_n7_n500 = avg_gap=0.01422

New approaches for V528:
1. S2 stronger regularization (higher reg_alpha, reg_lambda)
2. S2 different learner config (deeper trees, more aggressive)
3. Q1 n_feat=3 (even smaller) + S4 n_feat=10
4. Reduce Q2 gap (0.0276 is the 2nd largest after S2)
5. S3 n_feat=30 (increase from 23)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
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

def compute_gap(seed_oofs_list, y):
    oofs_arr = np.column_stack(seed_oofs_list)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in seed_oofs_list])
    avg_pred = np.mean(oofs_arr, axis=1)
    std_pred = np.std(oofs_arr, axis=1)
    meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
    meta.fit(np.column_stack([avg_pred, std_pred]), y)
    train_oof = meta.predict_proba(np.column_stack([avg_pred, std_pred]))[:, 1]
    meta_ll = log_loss(y, np.clip(train_oof, 0.001, 0.999))
    return avg_student, meta_ll, avg_student - meta_ll

def run_experiment(config_name, config, ranked_features, test_feat_cols,
                   train_df, test_df, gkf, N_FOLDS, N_SEEDS):
    n_train = len(train_df)
    n_test = len(test_df)
    
    XGB_CFGS = {
        'q_narrow':   {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
        'q_deep':     {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
        'q_strong':   {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
        's_wide':     {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
        's_strong':   {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
        'q_narrow_str': {'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    }
    LGBM_CFGS = {
        'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
        'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }
    
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}
    
    for ti, target in enumerate(TARGETS):
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
        
        blended_oofs = [0.5*x + 0.5*y for x, y in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [0.5*x + 0.5*y for x, y in zip(xgb_test_preds, lgbm_test_preds)]
        
        all_seed_oofs[target] = blended_oofs
        all_test_preds[target] = blended_tests
        
        _, _, gap = compute_gap(blended_oofs, y)
        log.info(f"  {target}: gap={gap:.5f}")
    
    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    total_gap = 0
    target_gaps = {}
    for t in TARGETS:
        t_y = train_df[t].values
        _, _, gap = compute_gap(all_seed_oofs[t], t_y)
        target_gaps[t] = gap
        total_gap += gap
    avg_gap = total_gap / 7
    
    vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
    avg_oof = sum(compute_gap(all_seed_oofs[t], train_df[t].values.astype(np.float64))[1] for t in TARGETS) / 7
    
    return {
        'key': config_name,
        'avg_gap': avg_gap,
        'target_gaps': target_gaps,
        'vs308': vs308,
        'avg_oof': avg_oof,
        'test_preds': all_test_preds,
    }

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V528 — S2 strong reg + Q1/S3/Q2 tuning + per-target n_est")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
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
    
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)
    
    # Best baseline from V527
    BASELINE = {
        'Q1':  {'n_feat': 5,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',   'n_est': 600},
        'Q2':  {'n_feat': 14, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',   'n_est': 800},
        'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety', 'n_est': 500},
        'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',   'n_est': 500},
        'S2':  {'n_feat': 7,  'xgb_cfg': 's_wide',    'lgbm_cfg': 'wide',   'n_est': 500},
        'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety', 'n_est': 1000},
        'S4':  {'n_feat': 15, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',   'n_est': 300},
    }
    
    configs = {
        # Stronger S2 regularization
        'S2_strong': {**BASELINE, 'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500}},
        # Stronger Q1 + S2
        'Q1_str_S2_strong': {**BASELINE, 'Q1': {'n_feat': 5, 'xgb_cfg': 'q_narrow_str', 'lgbm_cfg': 'wide_strong', 'n_est': 600}, 'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500}},
        # Q1 n_feat=3 + S4 n_feat=10
        'Q1_n3_S4_n10': {**BASELINE, 'Q1': {'n_feat': 3, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'S4': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}},
        # Q1_n3 + S4_n10 + S2_strong
        'Q1_n3_S4_n10_S2_strong': {**BASELINE, 'Q1': {'n_feat': 3, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'S4': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, 'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500}},
        # Q2 n_feat=10 (smaller)
        'Q2_n10': {**BASELINE, 'Q2': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}},
        # Q2_n10 + S4_n10 + S2_strong
        'Q2_n10_S4_n10_S2_strong': {**BASELINE, 'Q2': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800}, 'S4': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, 'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500}},
        # S3 n_feat=30 (bigger)
        'S3_n30': {**BASELINE, 'S3': {'n_feat': 30, 'xgb_cfg': 'q_strong', 'lgbm_cfg': 'safety', 'n_est': 1000}},
        # Q1_n3 + S3_n30 + S4_n10 + S2_strong
        'Q1_n3_S3_n30_S4_n10_S2_strong': {**BASELINE, 'Q1': {'n_feat': 3, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600}, 'S3': {'n_feat': 30, 'xgb_cfg': 'q_strong', 'lgbm_cfg': 'safety', 'n_est': 1000}, 'S4': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300}, 'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500}},
        # Best combo so far: all improvements
        'BEST_COMBO': {
            'Q1': {'n_feat': 3, 'xgb_cfg': 'q_narrow', 'lgbm_cfg': 'wide', 'n_est': 600},
            'Q2': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 800},
            'Q3': {'n_feat': 7, 'xgb_cfg': 'q_strong', 'lgbm_cfg': 'safety', 'n_est': 500},
            'S1': {'n_feat': 3, 'xgb_cfg': 'q_strong', 'lgbm_cfg': 'wide', 'n_est': 500},
            'S2': {'n_feat': 7, 'xgb_cfg': 's_strong', 'lgbm_cfg': 'wide_strong', 'n_est': 500},
            'S3': {'n_feat': 30, 'xgb_cfg': 'q_strong', 'lgbm_cfg': 'safety', 'n_est': 1000},
            'S4': {'n_feat': 10, 'xgb_cfg': 'q_deep', 'lgbm_cfg': 'wide', 'n_est': 300},
        },
    }
    
    all_results = []
    
    for name, config in configs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Config: {name}")
        log.info(f"{'='*60}")
        
        for t in TARGETS:
            log.info(f"  {t}: n_feat={config[t]['n_feat']} n_est={config[t]['n_est']}")
        
        try:
            result = run_experiment(name, config, ranked_features, test_feat_cols,
                                   train_df, test_df,
                                   GroupKFold(n_splits=N_FOLDS), N_FOLDS, N_SEEDS)
            all_results.append(result)
            log.info(f"  => avg_gap={result['avg_gap']:.5f}, vs308={result['vs308']}/7, avg_oof={result['avg_oof']:.5f}")
        except Exception as e:
            log.info(f"  => ERROR: {e}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    
    for r in sorted(all_results, key=lambda x: x['avg_gap']):
        marker = ""
        if r['avg_gap'] < 0.014: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < 0.016: marker = " ⭐⭐⭐"
        log.info(f"  {r['key']}: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7, oof={r['avg_oof']:.5f}{marker}")
    
    best = min(all_results, key=lambda x: x['avg_gap'])
    log.info(f"\n✅ BEST: {best['key']} with avg_gap={best['avg_gap']:.5f}")
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        sub_df[t] = np.mean(best['test_preds'][t], axis=0)
    sub_path = SUBMIT / f'submission_v528_{best["key"]}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"Submission saved: {sub_path}")
    
    result = {
        'version': 'V528',
        'name': 'S2 strong reg + per-target tuning',
        'results': [{k: (v if not isinstance(v, np.ndarray) else str(v)) for k, v in r.items() if k != 'test_preds'} for r in all_results],
        'best_key': best['key'],
        'best_gap': float(best['avg_gap']),
        'best_oof': float(best['avg_oof']),
        'best_vs308': best['vs308'],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v528_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()
