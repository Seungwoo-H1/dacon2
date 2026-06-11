#!/usr/bin/env python3
"""
V518 — Full: XGB for Q + LGBM for S, Q1 n_feat=7 (V517's best config)

Based on V517 results:
- Q1: XGB narrow, n_feat=7 → gap 0.041 ✅ (vs V308 0.113)
- Q2: XGB deep, n_feat=14 → gap 0.061 ✅ 
- Q3: XGB deep, n_feat=11 → gap 0.076 ✅
- S1-S4: LGBM (V53 configs)

This is the best gap combo: avg_gap 0.048 (V308 0.070 → -31% gap!)

Key targets:
- S1 is the only target with gap 0.022 vs V308 0.020 (slightly worse)
- Q1 is now the best-performing Q target (was worst before)
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
    'q_narrow': {'max_depth': 4, 'learning_rate': 0.04, 'n_estimators': 600,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 
                'min_child_weight': 3, 'random_state': 42},
    'q_deep':  {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 
                'min_child_weight': 5, 'random_state': 42},
}

LGBM_CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

# V518 config: Q1 n_feat=7, rest from V515
TARGET_CFGS = {
    'Q1':  {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 'q_narrow'},
    'Q2':  {'n_feat': 14, 'learner': 'xgb', 'cfg_name': 'q_deep'},
    'Q3':  {'n_feat': 11, 'learner': 'xgb', 'cfg_name': 'q_deep'},
    'S1':  {'n_feat': 21, 'learner': 'lgbm', 'cfg_name': 'wide'},
    'S2':  {'n_feat': 19, 'learner': 'lgbm', 'cfg_name': 'deep'},
    'S3':  {'n_feat': 23, 'learner': 'lgbm', 'cfg_name': 'safety'},
    'S4':  {'n_feat': 20, 'learner': 'lgbm', 'cfg_name': 'wide'},
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
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V518 — XGB for Q (Q1 n_feat=7) + LGBM for S")
    log.info("V517 best config in production")
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
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features | Test: {len(test_feat_cols)}")
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    all_seed_test = {t: [] for t in TARGETS}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        
        cfg = TARGET_CFGS[target]
        n_feat = cfg['n_feat']
        learner = cfg['learner']
        cfg_name = cfg['cfg_name']
        
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        log.info(f"    Learner: {learner}, n_feat: {n_feat}, n_sel: {len(sel_cols)}, cfg: {cfg_name}")
        
        per_seed_oofs = []
        per_seed_test_preds = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                X_test = test_df[sel_cols].fillna(0).values.astype(np.float64)
                
                if learner == 'xgb':
                    xgb_cfg = XGB_CFGS[cfg_name]
                    params = {
                        'objective': 'binary:logistic', 'eval_metric': 'logloss',
                        'max_depth': xgb_cfg['max_depth'], 'learning_rate': xgb_cfg['learning_rate'],
                        'n_estimators': xgb_cfg['n_estimators'], 'subsample': xgb_cfg['subsample'],
                        'colsample_bytree': xgb_cfg['colsample_bytree'],
                        'reg_alpha': xgb_cfg['reg_alpha'], 'reg_lambda': xgb_cfg['reg_lambda'],
                        'min_child_weight': xgb_cfg['min_child_weight'],
                        'random_state': seed, 'n_jobs': 1, 'verbosity': 0,
                    }
                    ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=sel_cols)
                    m = xgb.train(params, ds, num_boost_round=xgb_cfg['n_estimators'])
                    pred_va = m.predict(xgb.DMatrix(X_va, feature_names=sel_cols))
                    pred_test = m.predict(xgb.DMatrix(X_test, feature_names=sel_cols))
                    n_est = xgb_cfg['n_estimators']
                else:
                    lgbm_cfg = LGBM_CFGS[cfg_name]
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**lgbm_cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=lgbm_cfg['n_estimators'])
                    pred_va = m.predict(X_va)
                    pred_test = m.predict(X_test)
                    n_est = lgbm_cfg['n_estimators']
                
                seed_oof[va_idx] = pred_va
                seed_test += pred_test
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test_preds.append(seed_test)
            
            if si < 3 or si % 5 == 0:
                ll = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={ll:.5f}")
        
        t_student_lls = [log_loss(y, so) for so in per_seed_oofs]
        t_avg_student = np.mean(t_student_lls)
        t_std_student = np.std(t_student_lls)
        log.info(f"    Student avg OOF: {t_avg_student:.5f}, std: {t_std_student:.5f}")
        
        # Meta: 15 seed predictions → LR
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof[target] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[target], 0.001, 0.999))
        log.info(f"    {target} Stacking OOF (C={META_C}): {ll:.5f}")
        
        stacked_test = np.column_stack(per_seed_test_preds)
        test_preds[target] = meta.predict_proba(stacked_test)[:, 1]
        
        all_seed_oofs[target] = per_seed_oofs
        all_seed_test[target] = per_seed_test_preds
    
    # Results
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(per_target_oof.values()))
    
    v308_gaps = {
        'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
        'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
    }
    
    log.info(f"\n{'='*70}")
    log.info("V518 RESULTS")
    log.info(f"{'='*70}")
    
    avg_gap = 0
    per_target_gap = {}
    for t in TARGETS:
        t_y = train_df[t].values
        t_seeds = all_seed_oofs[t]
        t_student_lls = [log_loss(t_y, so) for so in t_seeds]
        t_meta_ll = per_target_oof[t]
        t_avg_student = np.mean(t_student_lls)
        t_gap = t_avg_student - t_meta_ll
        t_seed_std = np.std(t_student_lls)
        avg_gap += t_gap
        per_target_gap[t] = t_gap
        v308_gap = v308_gaps[t]
        status = "✅" if t_gap < v308_gap else "❌"
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} student={t_avg_student:.5f} std={t_seed_std:.5f} vs V308 {v308_gap:.3f} {status}")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308 actual: 0.06977, Δ: {avg_gap-0.06977:+.5f})")
    
    if avg_gap < 0.025:
        log.info("  🎯 GAP TARGET HIT!")
    else:
        log.info(f"  ⚠️ Gap target not hit (need < 0.025)")
    
    # Generate submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'sleep_date': test_df['sleep_date'].values,
        'lifelog_date': test_df['lifelog_date'].values,
    })
    for t in TARGETS:
        sub_df[t] = test_preds[t]
    
    sub_path = SUBMIT / f'submission_v518_xgb_q_lgbm_s_{timestamp}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n  ✅ Submission: {sub_path}")
    
    result = {
        'version': 'V518',
        'name': 'XGB for Q (Q1 n_feat=7) + LGBM for S',
        'avg_oof': float(avg_oof),
        'avg_gap': float(avg_gap),
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': float(avg_oof - 0.62235),
        'delta_vs_v308_gap': float(avg_gap - 0.06977),
        'per_target_oof': {k: float(v) for k, v in per_target_oof.items()},
        'per_target_gap': {k: float(v) for k, v in per_target_gap.items()},
        'per_target_seed_std': {t: float(np.std([log_loss(train_df[t].values, so) for so in all_seed_oofs[t]])) for t in TARGETS},
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v518_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"  📝 Result saved: {result_path}")
    log.info(f"\n  Total time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    main()
