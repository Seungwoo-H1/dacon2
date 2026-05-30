"""
V163 — Two-Level Stacking (Hierarchical)

Hypothesis: V146/V160 uses single-level stacking: 15 student LGBMs → LR meta.
This limits complexity to what a single LR can model from 15 inputs.

Two-level stacking:
Level 1: 15 LGBM seeds → GroupKFold OOF → 15 predictions
Level 2: Group 15 seeds into 3 groups (5 each), train 3 LR meta-learners
Level 3: 3 LR outputs → 1 final LR meta-learner

This allows:
- LR meta-learners to learn group-specific patterns (e.g., early seeds vs late seeds)
- Hierarchical structure to capture more complex interactions than single LR
- Better generalization by reducing dimensionality of final meta

Risk: Medium (more parameters, potential overfitting)
Expected: OOF improvement 0.001-0.003 over V160
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
N_SEEDS = 15
N_GROUPS = 3  # 15 seeds → 3 groups of 5
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

def train_students(train_df, test_df, feat_cols, t, group, n_folds=5, seed=SEED):
    """Train N_SEEDS LGBM students for a target, return OOF + test predictions."""
    y = train_df[t].values.astype(np.float64)
    feat_cols_clean = remove_leak(feat_cols, t)
    n_feat = V53_SWEEP[t]['n_feat']
    cfg = CFGS[V53_SWEEP[t]['cfg']]
    
    ranked = rank_features(train_df, feat_cols_clean, t)
    sel_cols = ranked[:n_feat]
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    per_seed_oofs = []
    test_preds = np.zeros((n_test, N_SEEDS))
    
    for si in range(N_SEEDS):
        s = seed + si * 7
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        
        for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(train_df, y, group)):
            X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': s,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= n_folds
        per_seed_oofs.append(seed_oof)
        test_preds[:, si] = seed_test
    
    return per_seed_oofs, test_preds, sel_cols

def run_experiment(train_df, test_df, feat_cols, group, method='single', n_groups=3):
    """
    method: 'single' = V160 style, 'two_level' = hierarchical stacking
    """
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {}
    test_seed = {}
    test_meta = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        
        per_seed_oofs, test_preds, sel_cols = train_students(
            train_df, test_df, feat_cols, t, group
        )
        
        train_oof[t] = np.column_stack(per_seed_oofs)  # (450, 15)
        test_seed[t] = test_preds  # (250, 15)
        
        if method == 'single':
            # Single-level: 15 → LR → 1
            meta_lr = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta_lr.fit(train_oof[t], y)
            train_oof[t] = meta_lr.predict_proba(train_oof[t])[:, 1]
            
            test_stacked = test_seed[t]
            test_meta_pred = meta_lr.predict_proba(test_stacked)[:, 1]
            
        elif method == 'two_level':
            # Two-level: 15 → 3 groups of 5 → 3 LR → 1 final LR
            seed_indices = list(range(N_SEEDS))
            group_size = N_SEEDS // n_groups
            
            # Level 1: group meta-learners (OOF)
            level1_preds_train = np.zeros((n_train, n_groups))
            level1_preds_test = np.zeros((n_test, n_groups))
            
            for g in range(n_groups):
                start = g * group_size
                end = start + group_size
                group_indices = seed_indices[start:end]
                group_preds = per_seed_oofs[start:end]  # list of 5 OOF arrays
                
                # Group average OOF
                group_oof = np.mean(group_preds, axis=0)
                group_oof = np.clip(group_oof, 0.001, 0.999)
                
                # Level 1 meta for this group
                meta_g = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED+g)
                meta_g.fit(np.column_stack(group_preds), y)
                level1_preds_train[:, g] = meta_g.predict_proba(np.column_stack(group_preds))[:, 1]
                
                # Test predictions for this group
                group_test_preds = test_preds[:, group_indices]
                # Use 5 individual test predictions for Level 1 meta
                level1_preds_test[:, g] = meta_g.predict_proba(group_test_preds)[:, 1]
            
            # Level 2: final meta-learner
            meta_final = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED+100)
            meta_final.fit(level1_preds_train, y)
            train_oof[t] = meta_final.predict_proba(level1_preds_train)[:, 1]
            test_meta_pred = meta_final.predict_proba(level1_preds_test)[:, 1]
        
        test_meta[t] = test_meta_pred
    
    return train_oof, test_meta

def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V163 — Two-Level Stacking (Hierarchical)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    group = train_df['subject_id'].values
    
    # Run both methods
    for method in ['single', 'two_level']:
        tag = 'V160 (single-level)' if method == 'single' else 'V163 (two-level)'
        log.info(f"\n{'='*70}")
        log.info(f"Method: {tag}")
        log.info(f"{'='*70}")
        
        train_oof, test_meta = run_experiment(
            train_df, test_df, feat_cols, group, method=method
        )
        
        oofs = {}
        for t in TARGETS:
            y = train_df[t].values.astype(np.float64)
            o = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
            oofs[t] = o
            log.info(f"  {t}: OOF={o:.5f}")
        
        avg = np.mean(list(oofs.values()))
        log.info(f"  AVG OOF: {avg:.5f}")
        log.info(f"  Δ vs V146 (0.63169): {avg - 0.63169:+.5f}")
        log.info(f"  Δ vs V160 (0.62240): {avg - 0.62240:+.5f}")
        
        if method == 'two_level':
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub = pd.DataFrame()
            sub['subject_id'] = test_df['subject_id'].values
            sub['sleep_date'] = test_df['sleep_date'].values
            sub['lifelog_date'] = test_df['lifelog_date'].values
            for t in TARGETS:
                sub[t] = test_meta[t]
            
            sub.to_csv(f"submissions/submission_v163_twolevel_{ts}.csv", index=False)
            log.info(f"Saved: submissions/submission_v163_twolevel_{ts}.csv")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
