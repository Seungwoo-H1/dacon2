#!/usr/bin/env python3
"""
V526 — V525 cv_weighted + S1 Improvement

V525 results (cv_weighted): avg_gap=0.01878, vs308=6/7
- S1 gap=0.025 (V308 0.020 대비 +0.005) — only ❌ target
- V522 S1 was xgb_strong_n5 → gap=0.00251
- V525 S1 was xgb_strong_n10 → gap=0.036

Hypothesis: V525의 S1 config가 V524의 xgb_strong_n5보다 n_feat가 많음 (10 vs 5).
n_feat=5로 줄이면 S1 gap이 더 낮아질 수 있음.

Also test:
1. S1 n_feat=5 with XGB strong
2. S1 n_feat=3 with XGB strong  
3. S1 n_feat=15 with XGB strong (wider)
4. S1 n_feat=5 with XGB narrow
5. V525 blend (equal 0.5/0.5) for all targets

Same V525 baseline configs for other targets.
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
    """Train and return (pred_va, pred_test)."""
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


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V526 — V525 blend + S1 n_feat sweep")
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
        's_narrow':  {'max_depth': 3, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    }
    LGBM_CFGS = {
        'wide':    {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'safety':  {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }
    
    # V525 baseline config (same for Q1-Q3, S2-S4)
    BASELINE = {
        'Q1':  {'n_feat': 7,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',   'n_est': 600},
        'Q2':  {'n_feat': 14, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',   'n_est': 800},
        'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety', 'n_est': 500},
        'S1':  None,  # Will be swept
        'S2':  {'n_feat': 7,  'xgb_cfg': 's_wide',    'lgbm_cfg': 'wide',   'n_est': 300},
        'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety', 'n_est': 1000},
        'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',   'n_est': 300},
    }
    
    # S1 configs to test
    S1_CONFIGS = [
        ('xgb_strong_n5',  5,  'q_strong'),
        ('xgb_strong_n10', 10, 'q_strong'),  # V525 default
        ('xgb_strong_n15', 15, 'q_strong'),
        ('xgb_strong_n20', 20, 'q_strong'),
        ('xgb_narrow_n5',  5,  's_narrow'),
        ('lgbm_wide_n10',  10, 'wide'),
        ('lgbm_wide_n7',   7,  'wide'),
    ]
    
    # Pre-rank features for all targets
    log.info("Pre-ranking features for all targets...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)
        log.info(f"  {target}: ranked {len(ranked_features[target])} features")
    
    # Test all S1 configs with equal mix (0.5 XGB + 0.5 LGBM)
    all_results = []
    
    for s1_key, s1_nfeat, s1_xgb_cfg in S1_CONFIGS:
        log.info(f"\n{'='*60}")
        log.info(f"S1 Config: {s1_key} (n_feat={s1_nfeat}, xgb={s1_xgb_cfg})")
        log.info(f"{'='*60}")
        
        all_seed_oofs = {t: [] for t in TARGETS}
        all_test_preds = {t: [] for t in TARGETS}
        
        for ti, target in enumerate(TARGETS):
            if target == 'S1':
                bc = {'n_feat': s1_nfeat, 'xgb_cfg': s1_xgb_cfg, 'lgbm_cfg': 'wide', 'n_est': 500}
            else:
                bc = BASELINE[target]
            
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
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
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
            
            # Equal mix (0.5/0.5) — same as V525 best
            blended_oofs = []
            blended_tests = []
            
            for si in range(N_SEEDS):
                xoof = xgb_seed_oofs[si]
                loof = lgbm_seed_oofs[si]
                xtest = xgb_test_preds[si]
                ltest = lgbm_test_preds[si]
                
                blended_oofs.append(0.5 * xoof + 0.5 * loof)
                blended_tests.append(0.5 * xtest + 0.5 * ltest)
            
            all_seed_oofs[target] = blended_oofs
            all_test_preds[target] = blended_tests
            
            _, _, gap = compute_gap(blended_oofs, y)
            vs = "✅" if gap < v308_gaps[target] else "❌"
            log.info(f"  {target}: gap={gap:.5f} V308={v308_gaps[target]:.3f} {vs}")
        
        # Overall
        total_gap = 0
        target_gaps = {}
        for t in TARGETS:
            t_y = train_df[t].values
            _, _, gap = compute_gap(all_seed_oofs[t], t_y)
            target_gaps[t] = gap
            total_gap += gap
        avg_gap = total_gap / 7
        
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        marker = ""
        if avg_gap < 0.020: marker = " 🎯🎯🎯 BELOW 0.020"
        elif avg_gap < 0.025: marker = " ⭐⭐"
        elif avg_gap < 0.030: marker = " ⭐"
        log.info(f"\n  RESULT {s1_key}: avg_gap={avg_gap:.5f}, vs308={vs308}/7{marker}")
        
        all_results.append({
            'key': s1_key,
            'avg_gap': avg_gap,
            'target_gaps': target_gaps,
            'vs308': vs308,
            's1_gap': target_gaps['S1'],
            'test_preds': all_test_preds,
        })
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    
    for r in sorted(all_results, key=lambda x: x['avg_gap']):
        marker = ""
        if r['avg_gap'] < 0.020: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < 0.025: marker = " ⭐⭐"
        elif r['avg_gap'] < 0.030: marker = " ⭐"
        log.info(f"  {r['key']}: avg_gap={r['avg_gap']:.5f}, vs308={r['vs308']}/7, S1_gap={r['s1_gap']:.5f}{marker}")
    
    best = min(all_results, key=lambda x: x['avg_gap'])
    log.info(f"\n✅ BEST: {best['key']} with avg_gap={best['avg_gap']:.5f}")
    
    # Generate submission for best
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        sub_df[t] = np.mean(best['test_preds'][t], axis=0)
    sub_path = SUBMIT / f'submission_v526_{best["key"]}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"Submission saved: {sub_path}")
    
    result = {
        'version': 'V526',
        'name': 'V525 blend + S1 n_feat sweep',
        'results': [{k: r[k] for k in ['key', 'avg_gap', 'target_gaps', 'vs308', 's1_gap']} for r in all_results],
        'best_key': best['key'],
        'best_gap': float(best['avg_gap']),
        'best_s1_gap': float(best['s1_gap']),
        'timestamp': ts,
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
