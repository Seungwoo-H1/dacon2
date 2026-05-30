"""
V159 — Non-linear Meta-learner (LightGBM instead of LR)

Hypothesis: V146's LR meta-learner (C=10) is linear and may miss 
non-linear interactions between the 5 student models. A shallow GBM 
meta-learner can capture these interactions.

Key insight: V146's student predictions are 5-dimensional. 
LR can only learn linear combinations. A GBM meta-learner can learn:
- Feature interactions between student predictions
- Non-linear thresholds (e.g., student1 > 0.7 AND student2 > 0.6)
- Complex aggregation patterns

Risk: Low — same 5 student seeds, same feature selection, only meta changes
Expected: OOF improvement 0.001-0.003

Why this time: V146's meta C=10 already improved over V140's C=0.1 by 0.009.
There's still room for non-linear meta to improve on linear LR.
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
N_SEEDS = 5
META_C_LR = 10.0


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
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def run_experiment(train_df, test_df, feat_cols, meta_type='gbm', meta_params=None):
    """
    V146 stacking with configurable meta-learner.
    meta_type: 'lr' or 'gbm'
    meta_params: dict for meta-learner config
    Returns: (train_oof, test_meta_preds, v146_oof)
    """
    global t_start
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_seed = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    test_meta = {t: np.zeros(n_test) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        cfg = CFGS[cfg_name]
        
        log.info(f"    Selected {n_feat} features")
        
        per_seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_seed[t][:, si] = seed_test
            
            log.info(f"    Seed {si} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        # Meta learner
        stacked = np.column_stack(per_seed_oofs)
        
        if meta_type == 'lr':
            meta = LogisticRegression(C=META_C_LR, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
        elif meta_type == 'gbm':
            # Shallow GBM meta-learner
            if meta_params is None:
                meta_params = {
                    'num_leaves': 8,
                    'max_depth': 2,
                    'learning_rate': 0.1,
                    'n_estimators': 50,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 1.0,
                    'min_child_samples': 5,
                    'verbose': -1,
                    'random_state': SEED,
                    'n_jobs': 1,
                    'force_row_wise': True,
                }
            
            sn_meta = [f'seed_{i}' for i in range(N_SEEDS)]
            ds_meta = lgb.Dataset(stacked, label=y, feature_name=sn_meta)
            meta = lgb.train(meta_params, ds_meta, num_boost_round=meta_params['n_estimators'])
        
        if meta_type == 'lr':
            train_oof[t] = meta.predict_proba(stacked)[:, 1]
        else:
            train_oof[t] = np.clip(meta.predict(stacked), 0.001, 0.999)
        
        if meta_type == 'lr':
            test_stacked = np.column_stack([test_seed[t][:, i] for i in range(N_SEEDS)])
            test_meta[t] = meta.predict_proba(test_stacked)[:, 1]
        else:
            # GBM meta: predict_proba returns [[neg_prob, pos_prob]]
            test_stacked = np.column_stack([test_seed[t][:, i] for i in range(N_SEEDS)])
            test_proba = meta.predict(test_stacked)
            # lgb predict on proba dataset returns log-odds or probability depending on metric
            # Since we used binary objective, predict() returns probability
            test_meta[t] = np.clip(test_proba, 0.001, 0.999)
    
    return train_oof, test_meta


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V159 — Non-linear Meta-learner (LightGBM)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Base features: {len(feat_cols)}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    # Run with both meta types for comparison
    for meta_type in ['lr', 'gbm']:
        tag = 'LR' if meta_type == 'lr' else 'GBM'
        log.info(f"\n{'='*70}")
        log.info(f"Meta-learner: {tag}")
        log.info(f"{'='*70}")
        
        train_oof, test_meta = run_experiment(train_df, test_df, feat_cols, meta_type=meta_type)
        
        # Compute OOF
        v146_oof = {}
        for t in TARGETS:
            v146_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        avg_oof = np.mean(list(v146_oof.values()))
        
        log.info(f"\n{tag} Results:")
        for t in TARGETS:
            log.info(f"  {t}: {v146_oof[t]:.5f}")
        log.info(f"  AVG OOF: {avg_oof:.5f}")
        log.info(f"  Δ vs V146 (LR C=10, 0.63169): {avg_oof - 0.63169:+.5f}")
        
        # Save GBM results as submission
        if meta_type == 'gbm':
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub = pd.DataFrame()
            sub['subject_id'] = test_df['subject_id'].values
            sub['sleep_date'] = test_df['sleep_date'].values
            sub['lifelog_date'] = test_df['lifelog_date'].values
            for t in TARGETS:
                sub[t] = test_meta[t]
            
            sub_path = SUBMIT / f"submission_v159_nonlinear_meta_{ts}.csv"
            sub.to_csv(sub_path, index=False)
            log.info(f"Saved: {sub_path}")
            
            meta_data = {
                'version': 'V159',
                'name': 'Non-linear Meta-learner (LightGBM)',
                'meta_type': 'gbm',
                'avg_oof': round(float(avg_oof), 5),
                'v146_avg_oof': 0.63169,
                'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
                'per_target_oof': {t: round(float(v146_oof[t]), 5) for t in TARGETS},
                'meta_params': {
                    'num_leaves': 8,
                    'max_depth': 2,
                    'learning_rate': 0.1,
                    'n_estimators': 50,
                },
                'submission_file': str(sub_path),
                'timestamp': ts,
                'total_time_s': round(time.time() - t_start, 0),
            }
            
            meta_path = SUBMIT / f'meta_v159_{ts}.json'
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            log.info(f"Saved: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
