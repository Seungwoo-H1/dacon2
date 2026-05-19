"""
V138 — Proper Leave-One-Fold-Out Stacking + Cross-Target Features

Problem with V254 Approach C (Stacking):
  Meta-learner was trained on OOF predictions → overfitting
  Result: Q1=0.4955 but overall avg=0.629 (worse than V127's 0.537)

Fix: Proper CV stacking (LOO-folds for level-0 → meta-learner trains on 
     out-of-fold preds from level-0, never sees the same data twice)

Also test: Cross-target raw features with proper CV

Architecture:
┌─────────────────────────────────────────────────────┐
│ Approach 1: 3-fold CV Stacking                       │
│   Level 0: 3 models per target (different seeds)     │
│   Level 0 OOF: GroupKFold 5-fold                     │
│   Level 1: LogisticRegression on out-of-fold preds   │
│   → Meta-learner never trains on data it evaluated   │
│                                                      │
│ Approach 2: Cross-Target Raw Features + Stacking      │
│   Features = base features + 6 raw target columns     │
│   Level 0: 3 models per target (different seeds)      │
│   Level 1: Proper CV stacking                         │
│                                                      │
│ Approach 3: LOO Target Meta (proper CV version)       │
│   Features = base features + 6 OTHER TARGET OOF preds │
│   Level 0: Base OOF via GroupKFold 5-fold             │
│   Level 1: Train meta model with proper CV            │
└─────────────────────────────────────────────────────┘
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy
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

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        subj_mean = grp[f'{col}_subj_mean']
        subj_std = grp[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        zc = f'{col}_zscore'
        df[zc] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(zc)
        gc.collect()
    return df, personal_cols

def rank_features(feat, feat_cols, target, seed=SEED):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


# ================================================================
# Level-0: Train 3 models per target with GroupKFold
# Returns: oof_preds (dict[t] → ndarray), level0_models per fold
# ================================================================

def train_level0(feat, feat_cols, targets, n_seeds=3, cfg_name='deep', n_trees=500,
                 n_feat=20):
    """
    Train N_SEEDS models per target via GroupKFold.
    Returns oof_preds dict[target] = ndarray(n_samples)
    """
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    oof_preds = {t: np.zeros(len(feat)) for t in targets}
    
    for t in targets:
        y = feat[t].values.astype(np.float64)
        ranked = rank_features(feat, feat_cols, t)
        sel_cols = ranked[:n_feat]
        
        cfg = CFGS[cfg_name]
        
        for si, seed in enumerate(range(SEED, SEED + n_seeds)):
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr, y_va = y[tr_idx], y[va_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=n_trees)
                
                oof_preds[t][va_idx] += m.predict(X_va)
        
        # Average across seeds
        oof_preds[t] /= n_seeds
        oof_preds[t] = np.clip(oof_preds[t], 0.001, 0.999)
    
    return oof_preds


# ================================================================
# Approach 1: Proper CV Stacking
# ================================================================

def approach_proper_stacking(feat):
    """
    Proper leave-one-fold-out stacking.
    
    Level 0: 3 models per target (GroupKFold 5-fold), average predictions
    Level 1: For each sample, collect its out-of-fold level-0 predictions
             → train LogisticRegression on these
             → predict with the meta-learner
    """
    log.info("  [Proper Stacking] Level 0: 3 models × 7 targets (GroupKFold 5-fold)")
    
    all_feat_cols = get_feature_cols(feat)
    oof_level0 = train_level0(feat, all_feat_cols, TARGETS, n_seeds=3,
                               cfg_name='deep', n_trees=1000, n_feat=20)
    
    # Check level 0 OOF
    level0_oof = {}
    for t in TARGETS:
        ll = log_loss(feat[t].values, oof_level0[t])
        level0_oof[t] = ll
        log.info(f"    {t} Level-0 OOF: {ll:.5f}")
    
    log.info("  [Proper Stacking] Level 1: LogisticRegression on OOF preds")
    
    # Level 1: Stack OOF predictions per target
    oof_stacked = {}
    for t in TARGETS:
        # Stack level-0 predictions from 3 seeds
        # Need per-seed OOF, not averaged
        all_feat_cols_leaked = remove_leak(all_feat_cols, t)
        y = feat[t].values.astype(np.float64)
        group = feat['subject_id'].values
        gkf = GroupKFold(n_splits=N_FOLDS)
        
        # Get per-seed OOF
        seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + 3)):
            seed_oof = np.zeros(len(feat))
            ranked = rank_features(feat, all_feat_cols_leaked, t)
            sel_cols = ranked[:20]
            cfg = CFGS[V53_SWEEP[t]['cfg']]
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=CFGS[V53_SWEEP[t]['cfg']]['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_oofs.append(seed_oof)
        
        # Stack: (n_samples, 3)
        stacked = np.column_stack(seed_oofs)
        
        # Train meta-learner on OOF predictions
        meta = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        # Predict with meta-learner
        oof_stacked[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(oof_stacked[t], 0.001, 0.999))
        log.info(f"    {t}: Stacking OOF={ll:.5f}, meta weights={meta.coef_[0].round(3)}")
    
    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_stacked[t], 0.001, 0.999)) 
                       for t in TARGETS])
    log.info(f"  [Proper Stacking] AVG OOF: {avg_oof:.5f}")
    return oof_stacked, avg_oof, level0_oof


# ================================================================
# Approach 2: Cross-Target Raw Features
# ================================================================

def approach_cross_target_raw(feat):
    """
    Add other 6 targets as raw features, then proper CV stacking.
    """
    log.info("  [Cross-Target Raw] Features + 6 raw targets")
    
    base_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    level0_oof = {}
    
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        other_targets = [ot for ot in TARGETS if ot != t]
        extended_cols = base_feat_cols + other_targets
        
        ranked = rank_features(feat, extended_cols, t)
        n_feat = 25  # Allow extra features for cross-target
        sel_cols = ranked[:n_feat]
        
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        # Level 0: 3 models
        seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + 3)):
            seed_oof = np.zeros(len(feat))
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr, y_va = y[tr_idx], y[va_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_oofs.append(seed_oof)
        
        # Level 1: Stack
        stacked = np.column_stack(seed_oofs)
        meta = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        oof_preds[t] = meta.predict_proba(stacked)[:, 1]
        
        ll = log_loss(y, np.clip(oof_preds[t], 0.001, 0.999))
        level0_oof[t] = ll
        log.info(f"    {t}: Cross-Target OOF={ll:.5f} (feat={len(sel_cols)}, selected={n_feat})")
    
    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) 
                       for t in TARGETS])
    log.info(f"  [Cross-Target Raw] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof, level0_oof


# ================================================================
# Approach 3: LOO Target Meta (base OOF + meta model)
# ================================================================

def approach_loo_target_meta(feat):
    """
    Train base OOF for each target. Then use base OOF as meta features
    for the target-specific model, with proper CV.
    
    For sample i, its meta features are the OOF predictions of other targets
    from models that DID NOT see sample i.
    """
    log.info("  [LOO Target Meta] Base OOF + meta model")
    
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Step 1: Get base OOF for each target
    base_oof = {}
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        all_feat_cols_leaked = remove_leak(all_feat_cols, t)
        ranked = rank_features(feat, all_feat_cols_leaked, t)
        sel_cols = ranked[:V53_SWEEP[t]['n_feat']]
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        fold_oof = np.zeros(len(feat))
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            fold_oof[va_idx] = m.predict(X_va)
        
        base_oof[t] = np.clip(fold_oof, 0.001, 0.999)
        ll = log_loss(y, fold_oof)
        log.info(f"    Base OOF {t}: {ll:.5f}")
    
    # Step 2: Train meta models
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    meta_oof = {}
    
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        other_targets = [ot for ot in TARGETS if ot != t]
        
        # Build meta features per fold
        meta_features_oof = np.zeros((len(feat), len(other_targets)))
        meta_features_base = feat[remove_leak(all_feat_cols, t)].fillna(0)
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
            # For validation samples, use base OOF from models that didn't see them
            for j, ot in enumerate(other_targets):
                meta_features_oof[va_idx, j] = base_oof[ot][va_idx]
        
        # Stack: base features + meta features
        X_all = np.column_stack([meta_features_base.values.astype(np.float64), 
                                  meta_features_oof])
        
        # Split
        X_tr = X_all[tr_idx]
        X_va = X_all[va_idx]
        
        # Train per-fold meta model
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        spw = max(((y[tr_idx] == 0).sum()) / max((y[tr_idx] == 1).sum(), 1), 0.1)
        params = {**cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(f"feat_{i}") for i in range(X_all.shape[1])]
        ds = lgb.Dataset(X_tr, label=y[tr_idx], feature_name=sn)
        m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
        
        oof_preds[t][va_idx] = m.predict(X_va)
    
    oof_preds = {t: np.clip(oof_preds[t], 0.001, 0.999) for t in TARGETS}
    
    for t in TARGETS:
        ll = log_loss(feat[t].values, oof_preds[t])
        meta_oof[t] = ll
        log.info(f"    {t}: Meta OOF={ll:.5f}")
    
    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) 
                       for t in TARGETS])
    log.info(f"  [LOO Target Meta] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof, meta_oof


# ================================================================
# Approach 4: V127 Baseline (for reference)
# ================================================================

def v127_baseline(feat):
    log.info("  [V127 Baseline] Per-target independent models")
    
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        all_feat_cols_leaked = remove_leak(all_feat_cols, t)
        ranked = rank_features(feat, all_feat_cols_leaked, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked[:n_feat]
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        fold_lls = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            oof_preds[t][va_idx] = m.predict(X_va)
            fold_lls.append(log_loss(y[va_idx], np.clip(oof_preds[t][va_idx], 0.001, 0.999)))
        
        oof_preds[t] = np.clip(oof_preds[t], 0.001, 0.999)
        avg_ll = np.mean(fold_lls)
        log.info(f"    {t}: OOF={avg_ll:.5f} (n_feat={n_feat})")
    
    avg_oof = np.mean([log_loss(feat[t].values, oof_preds[t]) for t in TARGETS])
    log.info(f"  [V127 Baseline] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


# ================================================================
# MAIN
# ================================================================

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V138 — Proper CV Stacking + Cross-Target Experiments")
    log.info("=" * 70)
    
    # Load data
    feat = pd.read_parquet(DATA / "features.parquet")
    for df in [feat]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    log.info(f"Data: {feat.shape}, Features: {len(get_feature_cols(feat))}")
    log.info(f"Target means: {[f'{t}: {feat[t].mean():.3f}' for t in TARGETS]}")
    
    results = {}
    
    # 1) V127 Baseline
    log.info("\n" + "─" * 70)
    log.info("Approach 1: V127 Baseline")
    log.info("─" * 70)
    oof, avg = v127_baseline(feat)
    results['V127_Baseline'] = {'avg_oof': avg, 'per_target_oof': 
        {t: log_loss(feat[t].values, oof[t]) for t in TARGETS}}
    
    # 2) Proper Stacking (Level-0: 3 seeds, Level-1: LR meta)
    log.info("\n" + "─" * 70)
    log.info("Approach 2: Proper CV Stacking")
    log.info("─" * 70)
    oof, avg, l0 = approach_proper_stacking(feat)
    results['Proper_Stacking'] = {'avg_oof': avg, 'per_target_oof': 
        {t: log_loss(feat[t].values, oof[t]) for t in TARGETS},
        'level0_oof': l0}
    
    # 3) Cross-Target Raw + Stacking
    log.info("\n" + "─" * 70)
    log.info("Approach 3: Cross-Target Raw Features + Stacking")
    log.info("─" * 70)
    oof, avg, l0 = approach_cross_target_raw(feat)
    results['CrossTarget_Raw_Stacking'] = {'avg_oof': avg, 'per_target_oof': 
        {t: log_loss(feat[t].values, oof[t]) for t in TARGETS},
        'level0_oof': l0}
    
    # 4) LOO Target Meta
    log.info("\n" + "─" * 70)
    log.info("Approach 4: LOO Target Meta")
    log.info("─" * 70)
    oof, avg, meta_oof = approach_loo_target_meta(feat)
    results['LOO_Target_Meta'] = {'avg_oof': avg, 'per_target_oof': 
        {t: log_loss(feat[t].values, oof[t]) for t in TARGETS},
        'meta_oof': meta_oof}
    
    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    baseline = results['V127_Baseline']['avg_oof']
    log.info(f"{'Approach':<40} {'AVG OOF':>10} {'Δ vs V127':>12}")
    log.info(f"{'─' * 65}")
    
    for name, data in results.items():
        avg = data['avg_oof']
        delta = avg - baseline
        marker = " ✅" if delta < -0.01 else (" ⚠️" if delta > 0 else "")
        log.info(f"{name:<40} {avg:>10.5f} {delta:>+12.5f}{marker}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {k: v for k, v in results.items()}
    meta['total_time_s'] = round(time.time() - t_start, 0)
    meta['timestamp'] = ts
    
    meta_path = SUBMIT / f'meta_v138_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Saved: {meta_path}")
    
    return results


if __name__ == '__main__':
    main()
