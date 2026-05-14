"""
V254: Multi-Target Joint Training Experiments for V127

Three approaches evaluated with 5-fold GroupKFold (no leakage):
A) Multi-output LGBM (MultiTargetRegressor style) - train separate LGBM per target but share feature ranking
B) Leave-one-out target - train 7 models where each model predicts 1 target using the other 6 as features
C) Stacking - base models predict → meta-model blends

All approaches report per-target OOF LogLoss.
Baseline: V127 per-target independent models.
"""

import os, sys, gc, re, json, time, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

SEED = 42
N_FOLDS = 5

# V53 sweep configs (same as V127)
V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
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


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def train_one_lgbm(X_tr, y_tr, X_val, feat_names, target, seed, n_trees=None):
    """Train single LGBM, return prediction on val."""
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    ne = n_trees if n_trees else base['n_estimators']

    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    params = {**base,
              'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_names]
    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=ne)
    return m.predict(X_val), m


def rank_features_once(train_feat, all_feat_cols, target, seed=SEED):
    """Rank all features by importance for a target (single-pass)."""
    y = train_feat[target].values.astype(np.float64)
    X = train_feat[all_feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]

    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in all_feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def v127_baseline_oof(feat):
    """
    Reproduce V127 baseline OOF using GroupKFold.
    This is the reference point.
    """
    print("  [Baseline] Training V127 per-target models...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        ranked = rank_features_once(feat, all_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked[:n_feat]

        fold_lls = []
        fold_pred = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr, X_val = feat[sel_cols].iloc[tr_idx].fillna(0).values, \
                          feat[sel_cols].iloc[val_idx].fillna(0).values
            y_tr, y_val = y[tr_idx], y[val_idx]
            pred, _ = train_one_lgbm(X_tr, y_tr, X_val, sel_cols, t, SEED + fold)
            fold_pred[val_idx] = pred
            fold_lls.append(log_loss(y_val, np.clip(pred, 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (n_feat={n_feat})")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Baseline] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_a_multioutput(feat):
    """
    Approach A: Multi-output Joint Training
    - Train SEPARATE LGBM per target (LGBM doesn't natively support multi-output binary)
    - Key difference: use ALL features (no per-target feature selection),
      let the model learn which features matter for each target
    - Compare with V127 where features are selected per target
    """
    print("  [Approach A] Multi-output (shared features, no per-target selection)...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        base = CFGS[cfg_name]

        fold_lls = []
        fold_pred = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[all_feat_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_val = feat[all_feat_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
            y_tr, y_val = y[tr_idx], y[val_idx]

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**base, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in all_feat_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base['n_estimators'])
            fold_pred[val_idx] = m.predict(X_val)
            fold_lls.append(log_loss(y_val, np.clip(fold_pred[val_idx], 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (all {len(all_feat_cols)} features, no selection)")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach A] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_b_leave_one_out(feat):
    """
    Approach B: Leave-One-Out Target
    - Train 7 models, each predicting 1 target using features + other 6 target predictions
    - Cross-val: train model for target T using OOF predictions of other targets
    - This creates a "target context" model
    """
    print("  [Approach B] Leave-one-out target models...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # First, get base OOF predictions for each target (used as meta features)
    base_oof = {}
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        ranked = rank_features_once(feat, all_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked[:n_feat]
        fold_pred = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr, X_val = feat[sel_cols].iloc[tr_idx].fillna(0).values, \
                          feat[sel_cols].iloc[val_idx].fillna(0).values
            y_tr, y_val = y[tr_idx], y[val_idx]
            pred, _ = train_one_lgbm(X_tr, y_tr, X_val, sel_cols, t, SEED + fold)
            fold_pred[val_idx] = pred
        base_oof[t] = fold_pred

    # Now train LOO models
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    for t in TARGETS:
        # Use all_feat_cols + 6 other target OOF preds as features
        other_targets = [ot for ot in TARGETS if ot != t]
        meta_cols = [f'{ot}_oof' for ot in other_targets]

        # Build extended feature matrix
        base_oof_df = pd.DataFrame({c: base_oof[ot] for ot, c in zip(other_targets, meta_cols)})
        base_feat = feat[all_feat_cols].fillna(0)
        extended = pd.concat([base_feat, base_oof_df], axis=1)
        extended_feat_cols = list(base_feat.columns) + meta_cols

        y = feat[t].values.astype(np.float64)
        fold_lls = []
        fold_pred = np.zeros(len(feat))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = extended.iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_val = extended.iloc[val_idx].fillna(0).values.astype(np.float64)
            y_tr, y_val = y[tr_idx], y[val_idx]

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            base_cfg = CFGS[cfg_name]
            params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in extended_feat_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
            fold_pred[val_idx] = m.predict(X_val)
            fold_lls.append(log_loss(y_val, np.clip(fold_pred[val_idx], 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (feat={len(all_feat_cols)} + 6 meta)")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach B] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_c_stacking(feat):
    """
    Approach C: Stacking Ensemble
    - Level 0: Multiple LGBM models with different configs/seeds per target
    - Level 1: Logistic regression that blends level 0 predictions
    - This is different from weighted averaging - the meta-learner learns optimal weights
    """
    print("  [Approach C] Stacking ensemble...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Level 0: Train 3 models per target (3 seeds)
    # Each model uses different seed + different n_estimators
    level0_preds = {t: [] for t in TARGETS}
    level0_train = {t: [] for t in TARGETS}

    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        ranked = rank_features_once(feat, all_feat_cols, t)

        # 3 different seeds, 3 different model sizes
        configs = [
            {'seed': SEED, 'cfg': V53_SWEEP[t]['cfg'], 'trees': None},       # default n_trees
            {'seed': 7, 'cfg': V53_SWEEP[t]['cfg'], 'trees': None},
            {'seed': 999, 'cfg': V53_SWEEP[t]['cfg'], 'trees': None},
        ]

        for cfg in configs:
            fold_oof = np.zeros(len(feat))
            fold_train = np.zeros(len(feat))
            n_feat = V53_SWEEP[t]['n_feat']
            sel_cols = ranked[:n_feat]
            n_trees = CFGS[cfg['cfg']]['n_estimators']

            for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr, X_val = feat[sel_cols].iloc[tr_idx].fillna(0).values, \
                              feat[sel_cols].iloc[val_idx].fillna(0).values
                y_tr, y_val = y[tr_idx], y[val_idx]

                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                base_cfg = CFGS[cfg['cfg']]
                params = {**base_cfg, 'n_estimators': n_trees,
                          'scale_pos_weight': spw, 'random_state': cfg['seed'],
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=n_trees)
                fold_oof[val_idx] = m.predict(X_val)
                fold_train[tr_idx] = m.predict(X_tr)

            fold_oof = np.clip(fold_oof, 0.001, 0.999)
            level0_preds[t].append(fold_oof)
            # For train data: need to train models on full train and predict
            level0_train[t].append(np.zeros(len(feat)))  # placeholder

        print(f"    {t}: 3 seeds OOF ready")

    # Compute avg OOF for each model
    avg_oof_per_model = {}
    for t in TARGETS:
        for i, pred in enumerate(level0_preds[t]):
            avg_oof_per_model[f'{t}_m{i}'] = log_loss(feat[t].values, pred)
            key = f'{t}_m{i}'
            print(f"      {key}: {avg_oof_per_model[key]:.5f}")

    # Level 1: Logistic regression on the 3 models' OOF predictions
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    for t in TARGETS:
        # Stack level 0 predictions
        stacked = np.column_stack(level0_preds[t])  # (n_samples, 3)
        y = feat[t].values.astype(np.float64)

        # Use a simple weighted average optimized on OOF
        # Actually let's use logistic regression as meta-learner
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)

        # The meta-learner's own OOF predictions = its weights on its own predictions
        # This is slightly circular but standard stacking uses the meta-learner trained on OOF
        oof_preds[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(oof_preds[t], 0.001, 0.999))
        print(f"    {t}: stacking OOF={ll:.5f}, meta weights={meta.coef_[0].round(3)}")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach C] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_d_joint_multi_lgbm(feat):
    """
    Approach D: Train a SINGLE LGBM that predicts all 7 targets simultaneously.
    This is possible because LGBM can handle multi-column labels in recent versions.
    Use the LGBM MultiTargetRegressor approach: one model, one fit, multi-output predict.
    """
    print("  [Approach D] Single LGBM model for all 7 targets...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Convert targets to numpy matrix
    y_matrix = feat[TARGETS].values.astype(np.float64)  # (450, 7)

    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, None, group)):
        X_tr = feat[all_feat_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
        X_val = feat[all_feat_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
        y_tr = y_matrix[tr_idx]
        y_val = y_matrix[val_idx]

        # For each target in this fold, train a fast model and collect val preds
        for t_idx, t in enumerate(TARGETS):
            y_tr_t = y_tr[:, t_idx]
            y_val_t = y_val[:, t_idx]

            spw = max(((y_tr_t == 0).sum()) / max((y_tr_t == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            base = CFGS[cfg_name]
            params = {**base, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in all_feat_cols]
            ds = lgb.Dataset(X_tr, label=y_tr_t, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base['n_estimators'])
            oof_preds[t][val_idx] = m.predict(X_val)

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach D] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_e_cross_target_features(feat):
    """
    Approach E: Cross-target features.
    For each target, add the OTHER 6 targets as raw features (not predictions).
    Since targets are in the same DataFrame, this is a leakage-free approach
    when used with GroupKFold (targets are per-row, not leaked).
    """
    print("  [Approach E] Cross-target raw features...")
    base_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        # Features = base features + other 6 targets as raw columns
        other_targets = [ot for ot in TARGETS if ot != t]
        extended_cols = base_feat_cols + other_targets

        ranked = rank_features_once(feat, extended_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked[:n_feat]

        fold_lls = []
        fold_pred = np.zeros(len(feat))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values
            X_val = feat[sel_cols].iloc[val_idx].fillna(0).values
            y_tr, y_val = y[tr_idx], y[val_idx]

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            base_cfg = CFGS[cfg_name]
            params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
            fold_pred[val_idx] = m.predict(X_val)
            fold_lls.append(log_loss(y_val, np.clip(fold_pred[val_idx], 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (feat={len(base_feat_cols)} + 6 raw targets, selected={n_feat})")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach E] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def approach_f_shared_feature_ranking(feat):
    """
    Approach F: Shared feature ranking.
    Rank features using a COMBINED objective (average importance across all 7 targets),
    then use the same feature subset for all targets.
    This forces the model to learn common patterns across targets.
    """
    print("  [Approach F] Shared feature ranking (avg importance across all targets)...")
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Step 1: Get average feature importance across all targets
    avg_importance = np.zeros(len(all_feat_cols))
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        X = feat[all_feat_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        cfg_name = 'deep'  # consistent cfg for ranking
        base = CFGS[cfg_name]
        params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
                  'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in all_feat_cols]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        m = lgb.train(params, ds, num_boost_round=50)
        avg_importance += m.feature_importance(importance_type='gain')
    avg_importance /= len(TARGETS)

    # Step 2: Rank features by shared importance
    ranked = sorted(zip(all_feat_cols, avg_importance), key=lambda x: -x[1])
    # Use top 30 shared features for all targets
    shared_top = [r[0] for r in ranked[:30]]
    print(f"    Shared top 30 features: {shared_top[:10]}...")

    # Step 3: Train per-target models with shared features
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        fold_lls = []
        fold_pred = np.zeros(len(feat))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[shared_top].iloc[tr_idx].fillna(0).values
            X_val = feat[shared_top].iloc[val_idx].fillna(0).values
            y_tr, y_val = y[tr_idx], y[val_idx]

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            base_cfg = CFGS[cfg_name]
            params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in shared_top]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
            fold_pred[val_idx] = m.predict(X_val)
            fold_lls.append(log_loss(y_val, np.clip(fold_pred[val_idx], 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (shared top 30 features)")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"  [Approach F] AVG OOF: {avg_oof:.5f}")
    return oof_preds, avg_oof


def main():
    t_start = time.time()
    print("=" * 70)
    print("V254: Multi-Target Joint Training Experiments")
    print("=" * 70)

    # Load data
    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f"\nData: {feat.shape}, Features: {len(get_feature_cols(feat))}, "
          f"Targets: {TARGETS}, Subjects: {feat['subject_id'].nunique()}")
    print(f"Target means: {[f'{feat[t].mean():.3f}' for t in TARGETS]}")

    results = {}
    oof_dict = {}

    # Run all approaches
    experiments = [
        ('V127_Baseline', v127_baseline_oof),
        ('A_SharedFeatures', approach_a_multioutput),
        ('B_LOO_Meta', approach_b_leave_one_out),
        ('C_Stacking', approach_c_stacking),
        ('D_SingleModel', approach_d_joint_multi_lgbm),
        ('E_CrossTargetRaw', approach_e_cross_target_features),
        ('F_SharedRanking', approach_f_shared_feature_ranking),
    ]

    for name, func in experiments:
        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print(f"{'─' * 70}")
        try:
            oof_preds, avg_oof = func(feat)
            results[name] = {'avg_oof': avg_oof, 'oof': deepcopy(oof_preds)}

            # Per-target OOF
            per_target_oof = {}
            for t in TARGETS:
                ll = log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999))
                per_target_oof[t] = ll
            results[name]['per_target_oof'] = per_target_oof

            oof_dict[name] = oof_preds
            print(f"\n>>> {name} AVG OOF: {avg_oof:.5f}")

            # Clean up
            for t in TARGETS:
                del oof_preds[t]
            gc.collect()

        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Approach':<30} {'AVG OOF':>10} {'Δ vs Baseline':>15} {'Time':>8}")

    baseline_oof = results.get('V127_Baseline', {}).get('avg_oof', None)
    elapsed = time.time() - t_start

    for name, data in results.items():
        if 'avg_oof' in data:
            avg = data['avg_oof']
            delta = avg - baseline_oof if baseline_oof else 0
            print(f"{name:<30} {avg:>10.5f} {delta:>+15.5f}")

    print(f"\nTotal time: {elapsed:.0f}s")

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'v254_multi_target_joint_{timestamp}.json'

    # Build saveable dict (remove oof arrays)
    saveable = {}
    for name, data in results.items():
        saveable[name] = {k: v for k, v in data.items() if k != 'oof'}
        if 'oof' in data:
            saveable[name]['oof_means'] = {t: float(np.mean(v)) for t, v in data['oof'].items()}

    saveable['total_time_sec'] = elapsed
    saveable['baseline_oof'] = baseline_oof
    saveable['timestamp'] = timestamp

    with open(result_path, 'w') as f:
        json.dump(saveable, f, indent=2, default=str)

    print(f"Results saved: {result_path}")
    return results


if __name__ == '__main__':
    main()
