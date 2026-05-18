"""
V142 — Target-Specific Drift Correction + Stacking

Hypothesis: V141 failed because it applied global drift correction,
but drift is TARGET-SPECIFIC. The features that cause train/test
mismatch depend on the TARGET, not globally.

Approach:
  1. Per-target adversarial validation: train/test split by target value
  2. Per-target drift feature identification
  3. Per-target fold-level drift correction (importance weighting)
  4. Stacking with corrected features

Key insight:
  V140 succeeds because OOF≈LB — stable generalization.
  V141's drift features overlap with prediction-important features,
  so removing them hurts. Instead, we correct at the fold level
  where train distribution matters for validation.
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
N_SEEDS = 5
META_C = 3.0


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


def rank_features_stability_weighted(train_df, feat_cols, target, stability, seed=SEED):
    """
    Rank features using stability-weighted importance.
    Features with high fold-importance CV are down-weighted.
    """
    y = train_df[target].values.astype(np.float64)
    fc_leaked = remove_leak(feat_cols, target)
    stab = stability[target]
    fc_idx = {c: i for i, c in enumerate(fc_leaked)}
    
    # Quick importance for ranking (lighter than full ranking)
    X = train_df[fc_leaked].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    ds = lgb.Dataset(X, label=y)
    m = lgb.train(params, ds, num_boost_round=50)
    raw_imp = m.feature_importance(importance_type='gain')
    
    # Apply stability weight
    adjusted_imp = np.zeros(len(fc_leaked))
    for i, col in enumerate(fc_leaked):
        cv = stab['cv_imp'][i]
        # Stability bonus: features with consistent importance across folds get higher score
        stability_weight = 1.0 / (1.0 + cv * 0.5)  # milder penalty than V141
        adjusted_imp[i] = raw_imp[i] * stability_weight
    
    ranked = sorted(zip(fc_leaked, adjusted_imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]


def train_stacking_v142(train_df, test_df, feat_cols, stability, n_seeds=5, meta_C=3.0, use_sample_weights=False):
    """
    V142 stacking pipeline:
    - stability-weighted feature selection (milder than V141)
    - optional sample weighting based on fold-level drift
    - 5 seeds + LR meta-learner (C=3.0)
    """
    t_start = time.time()
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), n_seeds)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n  --- {t} ---")
        y = train_df[t].values.astype(np.float64)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # Stability-weighted feature selection
        ranked = rank_features_stability_weighted(train_df, feat_cols, t, stability)
        sel_cols = ranked[:n_feat]
        cfg = CFGS[cfg_name]
        
        # Compute per-fold sample weights (fold-level train/test proximity)
        # For each fold's validation set, compute how similar its features are to other folds
        # If a fold's validation distribution differs from the rest, down-weight it
        fold_drift_weights = np.ones(len(train_df))
        
        if use_sample_weights:
            X_all = train_df[sel_cols].fillna(0).values.astype(np.float64)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                # Compute fold-level mean for each feature
                fold_mean = X_all[va_idx].mean(axis=0)
                train_mean = X_all[tr_idx].mean(axis=0)
                # Distance from fold mean to overall mean
                overall_mean = X_all.mean(axis=0)
                feature_drift = np.abs(fold_mean - overall_mean) / (np.std(X_all, axis=0) + 1e-10)
                fold_drift_score = feature_drift.mean()
                # Higher drift → lower weight
                fold_drift_weights[va_idx] = 1.0 / (1.0 + fold_drift_score * 0.5)
            
            # Normalize
            fold_drift_weights = fold_drift_weights / fold_drift_weights.mean()
        
        # Level 0: N_SEEDS models
        per_seed_oofs = []
        for si in range(n_seeds):
            seed = SEED + si * 7
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                # Apply fold drift weights to training data
                w_tr = fold_drift_weights[tr_idx] if use_sample_weights else None
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                
                if w_tr is not None:
                    ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=sn)
                else:
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            log.info(f"    Seed {si} train OOF: {log_loss(y, seed_oof):.5f}")
        
        # Level 1: Stack → LR meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=meta_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"    Stacking OOF: {ll:.5f}")
        
        test_stacked = np.column_stack([test_preds[t][:, i] for i in range(n_seeds)])
        test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) 
                       for t in TARGETS])
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v142_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    
    meta_data = {
        'version': 'V142',
        'name': 'Stability-Weighted Stacking (5 seeds + stability feat sel + fold drift weights)',
        'avg_oof': round(float(avg_oof), 5),
        'meta_C': meta_C,
        'n_seeds': n_seeds,
        'use_sample_weights': use_sample_weights,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) 
                          for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v142_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f}")
    log.info(f"  Saved: {sub_path}")
    
    return avg_oof, meta_data, stability


if __name__ == '__main__':
    t_start = time.time()
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
    
    # Compute stability
    log.info("\nComputing feature stability...")
    group = train_df['subject_id'].values
    stability = {}
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        fc_leaked = remove_leak(feat_cols, t)
        gkf = GroupKFold(n_splits=N_FOLDS)
        
        imps = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[fc_leaked].iloc[tr_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            cfg = CFGS[cfg_name]
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': SEED,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            ds = lgb.Dataset(X_tr, label=y_tr)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            imp = m.feature_importance(importance_type='gain')
            imps.append(imp)
        
        imps = np.array(imps)
        mean_imp = imps.mean(axis=0)
        std_imp = imps.std(axis=0)
        cv_imp = std_imp / (mean_imp + 1e-10)
        stability[t] = {'feat_cols': fc_leaked, 'mean_imp': mean_imp, 'cv_imp': cv_imp}
    
    log.info("Stability computed.")
    
    # Experiment A: without sample weights
    log.info("\n" + "=" * 70)
    log.info("V142-A: stability-weighted feat sel, NO sample weights")
    log.info("=" * 70)
    oofA, metaA, stab = train_stacking_v142(train_df, test_df, feat_cols, stability,
                                             n_seeds=N_SEEDS, meta_C=META_C, use_sample_weights=False)
    
    # Experiment B: with fold drift sample weights
    log.info("\n" + "=" * 70)
    log.info("V142-B: stability-weighted feat sel, WITH fold drift weights")
    log.info("=" * 70)
    oofB, metaB, _ = train_stacking_v142(train_df, test_df, feat_cols, stab,
                                          n_seeds=N_SEEDS, meta_C=META_C, use_sample_weights=True)
    
    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"V140 baseline:    OOF=0.64110")
    log.info(f"V141 drift-aware: OOF=0.63678")
    log.info(f"V142-A (no weights):  OOF={oofA:.5f}  Δ={oofA - 0.64110:+.5f}")
    log.info(f"V142-B (drift wt):    OOF={oofB:.5f}  Δ={oofB - 0.64110:+.5f}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
