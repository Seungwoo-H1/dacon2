#!/usr/bin/env python3
"""
V519 — Aggressive reduction: Q1 n_feat=7 + Q3 n_feat=7 + S1 n_feat=10

V518 results: avg_gap 0.04837, best target: Q1 gap 0.041
Remaining issues:
- Q3 gap 0.076 (student 0.695, meta 0.619) — biggest single contributor
- S1 gap 0.022 (student 0.599, meta 0.577) — barely above V308 threshold

Hypothesis: Q3 with fewer features (currently n_feat=11, V53 v48 config)
could have lower student variance and gap. Also try S1 with fewer features.
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
                'min_child_weight': 3},
    'q_deep':  {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 800,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 
                'min_child_weight': 5},
    'q_strong': {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 
                'min_child_weight': 5},
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

# V519: Q3 narrow features, S1 narrow features, Q1=7 stays
V519_CONFIGS = {
    # Test multiple Q3 variants
    'Q3_5':   {'n_feat': 5,  'learner': 'xgb', 'cfg_name': 'q_strong'},
    'Q3_7':   {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 'q_strong'},
    'Q3_9':   {'n_feat': 9,  'learner': 'xgb', 'cfg_name': 'q_deep'},
    # S1 variants
    'S1_10':  {'n_feat': 10, 'learner': 'lgbm', 'cfg_name': 'wide'},
    'S1_15':  {'n_feat': 15, 'learner': 'lgbm', 'cfg_name': 'wide'},
    # Fixed from V518
    'Q1':     {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 'q_narrow'},
    'Q2':     {'n_feat': 14, 'learner': 'xgb', 'cfg_name': 'q_deep'},
    'S2':     {'n_feat': 19, 'learner': 'lgbm', 'cfg_name': 'deep'},
    'S3':     {'n_feat': 23, 'learner': 'lgbm', 'cfg_name': 'safety'},
    'S4':     {'n_feat': 20, 'learner': 'lgbm', 'cfg_name': 'wide'},
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
    log.info("V519 — Aggressive reduction: Q3 n_feat=5/7/9, S1 n_feat=10/15")
    log.info("Hypothesis: fewer features for Q3 and S1 reduces gap")
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
    
    # Store results for each config
    config_results = {}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target}")
        
        if target == 'Q3':
            configs_to_test = ['Q3_5', 'Q3_7', 'Q3_9']
        elif target == 'S1':
            configs_to_test = ['S1_10', 'S1_15']
        else:
            configs_to_test = [target]
        
        for config_name in configs_to_test:
            n_feat = V519_CONFIGS[config_name]['n_feat']
            learner = V519_CONFIGS[config_name]['learner']
            cfg_name = V519_CONFIGS[config_name]['cfg_name']
            
            log.info(f"\n  Config: {config_name}, n_feat: {n_feat}, learner: {learner}")
            
            feat_cols_clean = remove_leak(train_feat_cols, target)
            ranked = rank_features(train_df, feat_cols_clean, target)
            sel_cols = ranked[:n_feat]
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            if len(sel_cols_test) != len(sel_cols):
                sel_cols = sel_cols_test
            
            per_seed_oofs = []
            per_seed_test_preds = []
            for si in range(N_SEEDS):
                seed = SEED + si * 11
                seed_oof = np.zeros(n_train)
                seed_test = np.zeros(n_test)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, train_df[target].values, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = train_df[target].values[tr_idx]
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
                    
                    seed_oof[va_idx] = pred_va
                    seed_test += pred_test
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                per_seed_test_preds.append(seed_test)
            
            seed_key = f"{target}_{config_name}" if target in ('Q3', 'S1') else config_name
            config_results[seed_key] = {
                'oofs': per_seed_oofs,
                'test_preds': per_seed_test_preds,
                'n_feat': n_feat,
                'learner': learner,
                'cfg': cfg_name,
            }
    
    # Now build full configs and compare
    v308_gaps = {
        'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
        'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
    }
    
    # Fixed targets: Q1, Q2, Q3 (use each variant), S1 (use each variant), S2, S3, S4
    # Build combinations: Q3 variant × S1 variant
    q3_variants = ['Q3_5', 'Q3_7', 'Q3_9']
    s1_variants = ['S1_10', 'S1_15']
    
    log.info(f"\n{'='*70}")
    log.info("BUILDING FULL CONFIGS AND COMPARING")
    log.info(f"{'='*70}")
    
    best_avg_gap = 999
    best_config_name = None
    best_result = None
    
    for q3_v in q3_variants:
        for s1_v in s1_variants:
            config_label = f"Q3={q3_v} S1={s1_v}"
            log.info(f"\n--- {config_label} ---")
            
            avg_gap = 0
            per_target_info = {}
            
            for target in TARGETS:
                if target == 'Q3':
                    seed_key = f"Q3_{q3_v}"
                elif target == 'S1':
                    seed_key = f"S1_{s1_v}"
                else:
                    seed_key = target
                
                y = train_df[target].values
                seeds = config_results[seed_key]['oofs']
                student_lls = [log_loss(y, so) for so in seeds]
                avg_student = np.mean(student_lls)
                seed_std = np.std(student_lls)
                
                stacked = np.column_stack(seeds)
                meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
                meta.fit(stacked, y)
                oof_pred = meta.predict_proba(stacked)[:, 1]
                meta_ll = log_loss(y, np.clip(oof_pred, 0.001, 0.999))
                
                gap = avg_student - meta_ll
                v308_gap = v308_gaps[target]
                status = "✅" if gap < v308_gap else "❌"
                
                per_target_info[target] = {
                    'meta_ll': float(meta_ll),
                    'gap': float(gap),
                    'avg_student': float(avg_student),
                    'seed_std': float(seed_std),
                }
                
                avg_gap += gap
                log.info(f"  {target}: meta={meta_ll:.5f} gap={gap:.5f} student={avg_student:.5f} std={seed_std:.5f} vs V308 {v308_gap:.3f} {status}")
            
            avg_gap /= len(TARGETS)
            log.info(f"  AVG GAP: {avg_gap:.5f}")
            
            if avg_gap < best_avg_gap:
                best_avg_gap = avg_gap
                best_config_name = config_label
                best_result = (q3_v, s1_v, per_target_info)
    
    log.info(f"\n{'='*70}")
    log.info(f"BEST CONFIG: {best_config_name} (avg_gap={best_avg_gap:.5f})")
    log.info(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'v519_{timestamp}.json'
    result = {
        'config_results': {k: {'n_feat': v['n_feat'], 'learner': v['learner'], 'cfg': v['cfg']} 
                          for k, v in config_results.items()},
        'comparisons': {},
        'best': {'name': best_config_name, 'avg_gap': float(best_avg_gap), 'targets': best_result[2]},
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"  📝 Result saved: {result_path}")
    log.info(f"\n  Total time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    main()
