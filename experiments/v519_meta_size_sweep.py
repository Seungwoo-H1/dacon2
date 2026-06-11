#!/usr/bin/env python3
"""
V519 — V518 Best Config + 2D Meta + Submission

V518 found:
- XGB for Q (Q1 n_feat=7, Q2/Q3 n_feat=14/11), LGBM for S
- 15D meta: gap 0.048, OOF 0.623
- 2D meta: gap 0.037, OOF 0.634

V519: Run V518 config with 2D meta (mean+std) as primary and submit.
Also test 4D meta (mean+std+min+max) for comparison.

Key hypothesis: 2D/4D meta eliminates meta overfitting → gap drops to ~0.035.
If gap is truly ~0.035 with 2D meta and OOF is 0.634, LB might still beat V308
because gap is the dominant factor.

V308: OOF 0.622, gap 0.070 → LB 0.639
V519 2D: OOF 0.634, gap 0.037 → if gap halves, LB should be lower.
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

TARGET_CFGS = {
    'Q1':  {'n_feat': 7,  'learner': 'xgb', 'xgb_cfg': 'q_narrow'},
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


def train_model(X_tr, y_tr, X_va, X_test, sel_cols, learner, cfg, seed, n_est):
    if learner == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': cfg['max_depth'], 'learning_rate': cfg['learning_rate'],
            'n_estimators': cfg['n_estimators'], 'subsample': cfg['subsample'],
            'colsample_bytree': cfg['colsample_bytree'],
            'reg_alpha': cfg['reg_alpha'], 'reg_lambda': cfg['reg_lambda'],
            'min_child_weight': cfg['min_child_weight'],
            'random_state': seed, 'n_jobs': 1, 'verbosity': 0,
        }
        ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=[sanitize_col(c) for c in sel_cols])
        m = xgb.train(params, ds, num_boost_round=cfg['n_estimators'])
        pred_va = m.predict(xgb.DMatrix(X_va, feature_names=[sanitize_col(c) for c in sel_cols]))
        pred_test = m.predict(xgb.DMatrix(X_test, feature_names=[sanitize_col(c) for c in sel_cols]))
    else:
        params = {**cfg, 'random_state': seed,
                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in sel_cols]
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds, num_boost_round=n_est)
        pred_va = m.predict(X_va)
        pred_test = m.predict(X_test)
    return pred_va, pred_test


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V519 — V518 Config + Multi-Meta-Size Comparison")
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
    log.info(f"Features: {len(train_feat_cols)} train, {len(test_feat_cols)} test")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    all_seed_oofs = {t: [] for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        tc = TARGET_CFGS[target]
        n_feat = tc['n_feat']
        learner = tc['learner']
        
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        log.info(f"    Learner: {learner}, n_feat: {n_feat}, n_sel: {len(sel_cols)}")
        
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
                
                if learner == 'xgb':
                    cfg = XGB_CFGS[tc['xgb_cfg']]
                    pred_va, pred_te = train_model(X_tr, y_tr, X_va, X_te, sel_cols,
                                                    'xgb', cfg, seed, cfg['n_estimators'])
                else:
                    cfg = LGBM_CFGS[tc['lgbm_cfg']]
                    pred_va, pred_te = train_model(X_tr, y_tr, X_va, X_te, sel_cols,
                                                    'lgbm', cfg, seed, cfg['n_estimators'])
                
                seed_oof[va_idx] = pred_va
                seed_test += pred_te
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[target][:, si] = seed_test
            ll = log_loss(y, seed_oof)
            log.info(f"    Seed {si:2d}: OOF={ll:.5f}")
        
        all_seed_oofs[target] = per_seed_oofs
    
    # Build stacked predictions and test for different meta sizes
    stacked = {t: np.column_stack(all_seed_oofs[t]) for t in TARGETS}
    
    meta_sizes = {
        '1D': lambda s: s.mean(axis=1, keepdims=True),
        '2D': lambda s: np.column_stack([s.mean(axis=1), s.std(axis=1)]),
        '4D': lambda s: np.column_stack([s.mean(axis=1), s.std(axis=1), s.min(axis=1), s.max(axis=1)]),
    }
    
    meta_c_vals = [5.0, 10.0, 20.0]
    
    log.info(f"\n{'='*70}")
    log.info("META COMPARISON")
    log.info(f"{'='*70}")
    
    v308_gaps = {'Q1':0.113,'Q2':0.079,'Q3':0.124,'S1':0.020,'S2':0.097,'S3':0.017,'S4':0.039}
    
    results = {}
    for ms_name, ms_fn in meta_sizes.items():
        log.info(f"\n--- {ms_name} Meta ---")
        per_target_oof = {}
        per_target_gap = {}
        for t in TARGETS:
            s = stacked[t]
            meta_feats = ms_fn(s)
            
            for mc in meta_c_vals:
                meta = LogisticRegression(C=mc, max_iter=1000, random_state=SEED)
                meta.fit(meta_feats, train_df[t].values)
                oof_pred = meta.predict_proba(meta_feats)[:, 1]
                ll = log_loss(train_df[t].values, np.clip(oof_pred, 0.001, 0.999))
                
                seed_lls = [log_loss(train_df[t].values, so) for so in all_seed_oofs[t]]
                avg_student = np.mean(seed_lls)
                gap = avg_student - ll
                per_target_oof[f'{ms_name}_C{mc}'] = ll
                per_target_gap[f'{ms_name}_C{mc}'] = gap
        
        # Compute avg OOF and gap per meta size
        avg_oof = np.mean([per_target_oof[f'{ms_name}_C{mc}'] for mc in meta_c_vals])
        avg_gap = np.mean([per_target_gap[f'{ms_name}_C{mc}'] for mc in meta_c_vals])
        
        # Also compute per target
        log.info(f"  Avg OOF ({ms_name}): {avg_oof:.5f}")
        log.info(f"  Avg GAP ({ms_name}): {avg_gap:.5f} (V308: 0.06977)")
        
        for mc in meta_c_vals:
            avg_oof_mc = np.mean([per_target_oof[f'{ms_name}_C{mc}']])
            avg_gap_mc = np.mean([per_target_gap[f'{ms_name}_C{mc}']])
            log.info(f"  C={mc}: AVG OOF={avg_oof_mc:.5f} AVG GAP={avg_gap_mc:.5f}")
            
            # Per-target for C=10
            if mc == 10.0:
                for t in TARGETS:
                    gap_val = per_target_gap[f'{ms_name}_C{mc}']
                    v308_gap = v308_gaps[t]
                    status = "✅" if gap_val < v308_gap else "❌"
        
        results[ms_name] = {
            'per_target_oof': per_target_oof,
            'per_target_gap': per_target_gap,
        }
    
    # Find best config
    best_ms = None
    best_avg_gap = float('inf')
    best_avg_oof = float('inf')
    for ms_name, res in results.items():
        for mc in meta_c_vals:
            avg_oof = np.mean([res['per_target_oof'][f'{ms_name}_C{mc}']])
            avg_gap = np.mean([res['per_target_gap'][f'{ms_name}_C{mc}']])
            if avg_gap < best_avg_gap or (avg_gap == best_avg_gap and avg_oof < best_avg_oof):
                best_ms = ms_name
                best_mc = mc
                best_avg_gap = avg_gap
                best_avg_oof = avg_oof
    
    log.info(f"\n{'='*70}")
    log.info(f"BEST: {best_ms} meta, C={best_mc}, avg_gap={best_avg_gap:.5f}, avg_oof={best_avg_oof:.5f}")
    log.info(f"{'='*70}")
    
    # Generate submission with best config
    now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    test_final = {}
    for t in TARGETS:
        test_final[t] = test_preds[t].mean(axis=1)
    
    submit_df = pd.DataFrame({'id': range(1, n_test + 1)})
    for t in TARGETS:
        submit_df[t] = test_final[t]
    submit_path = SUBMIT / f"submission_v519_best_meta_{best_ms}_C{best_mc}_{now}.csv"
    submit_df.to_csv(submit_path, index=False)
    log.info(f"  ✅ Submission: {submit_path}")
    
    # Save result
    result = {
        'version': 'V519',
        'name': f'V518 Config + {best_ms} Meta, C={best_mc}',
        'best_meta_size': best_ms,
        'best_meta_c': best_mc,
        'avg_oof': float(best_avg_oof),
        'avg_gap': float(best_avg_gap),
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': float(best_avg_oof - 0.62235),
        'delta_vs_v308_gap': float(best_avg_gap - 0.06977),
        'all_results': {},
        'n_seeds': N_SEEDS,
        'submission_file': str(submit_path),
        'timestamp': now,
        'total_time_s': round(time.time() - t_start, 1)
    }
    for ms_name, res in results.items():
        result['all_results'][ms_name] = {}
        for mc in meta_c_vals:
            avg_oof = np.mean([res['per_target_oof'][f'{ms_name}_C{mc}']])
            avg_gap = np.mean([res['per_target_gap'][f'{ms_name}_C{mc}']])
            result['all_results'][ms_name][f'C{mc}'] = {'avg_oof': float(avg_oof), 'avg_gap': float(avg_gap)}
    
    result_path = EXPERIMENTS / f"v519_{now}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"  📝 Result: {result_path}")
    log.info(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
