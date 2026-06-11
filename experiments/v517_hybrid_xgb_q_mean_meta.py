#!/usr/bin/env python3
"""
V517 — Hybrid: XGB for Q + LGBM for S + Mean-Only Meta (1D)

Combines V515 (XGB for Q → better gap) with V514 (mean-only meta → less overfitting).

Changes:
1. Q targets use XGB (q_deep), S targets use LGBM (V308 configs)
2. Meta layer uses ONLY mean of 15 seed predictions (1D) → no meta overfitting
3. Simple average of test predictions for submission

Also runs V515 (full 15D meta) alongside for comparison.
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

XGB_CFGS = {
    'q_deep':  {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 
                'min_child_weight': 5},
    's_wide':  {'max_depth': 4, 'learning_rate': 0.04, 'n_estimators': 600,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 
                'min_child_weight': 3},
    's_safe':  {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 700,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 
                'min_child_weight': 5},
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'n_feat': 19, 'learner': 'xgb', 'xgb_cfg': 'q_deep'},
    'Q2':  {'n_feat': 14, 'learner': 'xgb', 'xgb_cfg': 'q_deep'},
    'Q3':  {'n_feat': 11, 'learner': 'xgb', 'xgb_cfg': 'q_deep'},
    'S1':  {'n_feat': 21, 'learner': 'lgbm', 'lgbm_cfg': 'wide'},
    'S2':  {'n_feat': 19, 'learner': 'lgbm', 'lgbm_cfg': 'deep'},
    'S3':  {'n_feat': 23, 'learner': 'lgbm', 'lgbm_cfg': 'safety'},
    'S4':  {'n_feat': 20, 'learner': 'lgbm', 'lgbm_cfg': 'wide'},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15


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


def train_xgb_model(X_train, y_train, X_val, sel_cols, params, seed, n_estimators):
    """Train XGB model and return predictions for validation."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[sanitize_col(c) for c in sel_cols])
    dval = xgb.DMatrix(X_val, feature_names=[sanitize_col(c) for c in sel_cols])
    cfg = {**params, 'random_state': seed}
    m = xgb.train(cfg, dtrain, num_boost_round=n_estimators)
    pred_val = m.predict(dval)
    del m, dtrain, dval
    gc.collect()
    return pred_val


def train_lgbm_model(X_train, y_train, sel_cols, params, seed, n_estimators):
    """Train LGBM model, return predictor for predicting on X_val later."""
    sn = [sanitize_col(c) for c in sel_cols]
    ds = lgb.Dataset(X_train, label=y_train, feature_name=sn)
    cfg = {**params, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    m = lgb.train(cfg, ds, num_boost_round=n_estimators)
    return m


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V517 — Hybrid: XGB for Q + LGBM for S + Mean-Only Meta (1D)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score
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
    
    log.info(f"Train: {len(train_feat_cols)} features, Test: {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        n_feat = V53_SWEEP[target]['n_feat']
        learner_type = V53_SWEEP[target]['learner']
        
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {target}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        log.info(f"    Learner: {learner_type}, n_feat: {n_feat}, n_sel: {len(sel_cols)}")
        
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                X_te = test_df[sel_cols].fillna(0).values.astype(np.float64)
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                
                if learner_type == 'xgb':
                    xc = XGB_CFGS[V53_SWEEP[target]['xgb_cfg']]
                    pred_va = train_xgb_model(X_tr, y_tr, X_va, sel_cols, xc, seed, xc['n_estimators'])
                    pred_te = train_xgb_model(X_tr, y_tr, X_te, sel_cols, xc, seed, xc['n_estimators'])
                    seed_oof[va_idx] = pred_va
                    seed_test += pred_te
                else:
                    lc = CFGS[V53_SWEEP[target]['lgbm_cfg']]
                    model = train_lgbm_model(X_tr, y_tr, sel_cols, lc, seed, lc['n_estimators'])
                    seed_oof[va_idx] = model.predict(X_va)
                    seed_test += model.predict(X_te)
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[target][:, si] = seed_test
            
            ll = log_loss(y, seed_oof)
            log.info(f"    Seed {si:2d}: OOF={ll:.5f}")
        
        # Mean-only meta (1D)
        stacked = np.column_stack(per_seed_oofs)
        meta_mean = stacked.mean(axis=1, keepdims=True)
        
        meta = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        meta.fit(meta_mean, y)
        
        train_oof[target] = meta.predict_proba(meta_mean)[:, 1]
        ll = log_loss(y, np.clip(train_oof[target], 0.001, 0.999))
        log.info(f"    {target} 1D Meta OOF: {ll:.5f}")
        
        all_seed_oofs[target] = per_seed_oofs
    
    # Results
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(per_target_oof.values()))
    
    log.info(f"\n{'='*70}")
    log.info("V517 RESULTS")
    log.info(f"{'='*70}")
    
    avg_gap = 0
    v308_gaps = {'Q1':0.113,'Q2':0.079,'Q3':0.124,'S1':0.020,'S2':0.097,'S3':0.017,'S4':0.039}
    for t in TARGETS:
        t_y = train_df[t].values
        t_seeds = all_seed_oofs[t]
        t_student_lls = [log_loss(t_y, so) for so in t_seeds]
        t_meta_ll = per_target_oof[t]
        t_avg_student = np.mean(t_student_lls)
        t_gap = t_avg_student - t_meta_ll
        t_std = np.std(t_student_lls)
        avg_gap += t_gap
        
        v308_gap = v308_gaps[t]
        status = "✅" if t_gap < v308_gap else "❌"
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} student={t_avg_student:.5f} std={t_std:.5f} vs V308 {v308_gap} {status}")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235)")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308: 0.06977)")
    log.info(f"  Delta OOF: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Delta GAP: {avg_gap - 0.06977:+.5f}")
    log.info(f"{'='*70}")
    
    # Submission: mean of test predictions (same as 1D meta)
    test_final = {}
    for t in TARGETS:
        test_final[t] = test_preds[t].mean(axis=1)
    
    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    submit_df = pd.DataFrame({'id': range(1, n_test + 1)})
    for t in TARGETS:
        submit_df[t] = test_final[t]
    submit_path = SUBMIT / f"submission_v517_hybrid_mean_meta_{now}.csv"
    submit_df.to_csv(submit_path, index=False)
    log.info(f"  ✅ Submission: {submit_path}")
    
    # Per-target gap info
    per_target_gap = {}
    for t in TARGETS:
        t_student_lls = [log_loss(train_df[t].values, so) for so in all_seed_oofs[t]]
        per_target_gap[t] = float(np.mean(t_student_lls) - per_target_oof[t])
    
    result = {
        'version': 'V517',
        'name': 'Hybrid: XGB for Q + LGBM for S + Mean-Only Meta (1D)',
        'avg_oof': float(avg_oof),
        'avg_gap': float(avg_gap),
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': float(avg_oof - 0.62235),
        'delta_vs_v308_gap': float(avg_gap - 0.06977),
        'per_target_oof': {k: float(v) for k, v in per_target_oof.items()},
        'per_target_gap': per_target_gap,
        'meta_features': 'mean only (1D)',
        'n_seeds': N_SEEDS,
        'submission_file': str(submit_path),
        'timestamp': now,
        'total_time_s': round(time.time() - t_start, 1)
    }
    result_path = EXPERIMENTS / f"v517_{now}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"  📝 Result: {result_path}")
    log.info(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
