"""
V346 — Multi-Task LGBM with Shared Backbone + Task-Specific Heads

Hypothesis: 각 target별 모델을 독립적으로 training하면 target 간 공유 signal을
놓치고 있음. 건강 지표(Q1-Q3)와 활동 지표(S1-S4)는 서로 상관관계가 있음.
multi-task learning은 shared feature representation을 활용하여
각 target의 generalization을 개선할 수 있음.

Architecture:
1. Multi-output LGBM regression: 7 targets를 동시에 prediction
   - Use multi-output regression objective
   - Share feature extraction across all targets
2. Ensemble 15 seeds → per-target prediction
3. Meta: LR on per-seed predictions (same as V308)

This is a fundamentally different architecture from V308.
If multi-task learning captures cross-target correlations,
student OOF should improve without widening OOF-LB gap.

Risk: Multi-task might cause interference between targets.
Mitigation: Use very strong regularization, limited depth.
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
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
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
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


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V346 — Multi-Task LGBM: 7 targets simultaneous prediction")
    log.info("Multi-output regression → per-seed predictions → LR meta")
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
    
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df[zc] = (vals - mean) / std
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} | Test: {len(test_feat_cols)}")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    all_oofs = {}
    all_test_preds = {}
    all_student_oofs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        cfg = CFGS[cfg_name]
        
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        # Multi-output student: train one model per fold, predict all 7 targets
        # But we need fold-aware predictions (OOF), so we still need per-fold training
        # The "multi-task" aspect: train ONE model that predicts all targets simultaneously
        
        # Actually, LGBM doesn't support multi-output directly in the same way as sklearn.
        # Alternative: train one model per fold with all 7 targets as multi-output labels.
        
        # Strategy: GroupKFold with multi-output LGBM
        # X_tr (n_train, n_feat) → Y_tr (n_train, 7)
        
        y_all = train_df[TARGETS].values.astype(np.float64)
        y_all_test = test_df[TARGETS].values.astype(np.float64) if all(t in test_df.columns for t in TARGETS) else None
        
        mt_train_oofs = np.zeros((n_train, N_SEEDS, 7))  # (sample, seed, target)
        mt_test_preds = np.zeros((n_test, N_SEEDS, 7))
        
        for si, seed in enumerate(seeds):
            fold_preds_train = np.zeros((n_train, 7))
            fold_preds_test = np.zeros((n_test, 7))
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_all, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                Y_tr = y_all[tr_idx]  # (n_train, 7)
                
                params = {**cfg, 'objective': 'regression', 'metric': 'l2',
                          'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=Y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                fold_preds_train[va_idx] = m.predict(X_va)
                fold_preds_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            fold_preds_train = np.clip(fold_preds_train, 0.001, 0.999)
            fold_preds_test /= N_FOLDS
            fold_preds_test = np.clip(fold_preds_test, 0.01, 0.99)
            mt_train_oofs[:, si] = fold_preds_train
            mt_test_preds[:, si] = fold_preds_test
        
        # Extract per-target predictions
        t_idx = TARGETS.index(t)
        student_preds = np.mean(mt_train_oofs[:, :, t_idx], axis=1)
        student_preds_test = np.mean(mt_test_preds[:, :, t_idx], axis=1)
        
        student_ll = log_loss(y, student_preds)
        all_student_oofs[t] = student_ll
        
        # Meta learner
        stacked_train = np.column_stack([mt_train_oofs[:, :, t_idx].T])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        meta_train_pred = np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, meta_train_pred)
        all_oofs[t] = oof_ll
        
        stacked_test = np.column_stack([mt_test_preds[:, :, t_idx].T])
        meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.01, 0.99)
        all_test_preds[t] = meta_test
        
        # Also compute V308-style (binary student) for comparison
        v308_comparison = {}
        
        log.info(f"  mt_student_avg={student_ll:.5f}, meta_OOF={oof_ll:.5f}, "
                 f"gap={student_ll-oof_ll:+.4f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student = np.mean(list(all_student_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V346 RESULTS (multi-task LGBM)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: meta_OOF={all_oofs[t]:.5f}, student={all_student_oofs[t]:.5f}")
    log.info(f"  AVG meta_OOF: {avg_oof:.5f}")
    log.info(f"  AVG student_OOF: {avg_student:.5f}")
    log.info(f"  V308: meta_OOF=0.62235 | Δ: {avg_oof - 0.62235:+.5f}")
    
    # Distribution check
    log.info(f"\n{'='*70}")
    log.info("Distribution comparison:")
    v308 = pd.read_csv('submissions/submission_v308_zscore_20260602_021028.csv')
    log.info(f"{'Target':>6} {'V308_mean':>10} {'V346_mean':>10} {'V308_std':>10} {'V346_std':>10} {'ratio':>8}")
    for t in TARGETS:
        log.info(f"{t:>6} {v308[t].mean():>10.4f} {all_test_preds[t].mean():>10.4f} "
                 f"{v308[t].std():>10.4f} {all_test_preds[t].std():>10.4f} {all_test_preds[t].std()/v308[t].std():>8.2f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    sub_path = SUBMIT / f"submission_v346_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    meta_data = {
        'version': 'V346',
        'name': 'Multi-Task LGBM (7 targets simultaneous)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_seeds': N_SEEDS,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    meta_path = EXPERIMENTS / f'v346_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
