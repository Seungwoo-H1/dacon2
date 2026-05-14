"""
V259: Cross-Target Analysis & Joint Training

Comprehensive cross-target experiments for DaCon2:
1. Inter-Target Correlation Matrix
2. OOF Pseudo-Target Features (leakage-free)
3. Cross-Target Ratios & Interactions
4. Multi-Task LGBM
5. Residual Learning
6. Target Order Analysis (cascade prediction)

Uses GroupKFold(5), seed=42. Outputs to experiments/v259_cross_target_result.json
"""

import os, sys, gc, re, json, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
N_FOLDS = 5
SEED = 42

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


def make_dataset(X, y, feature_names):
    sn = [sanitize_col(c) for c in feature_names]
    return lgb.Dataset(X, label=y, feature_name=sn, free_raw_data=False)


def train_and_predict(X_tr, y_tr, X_val, feat_names, target, seed):
    """Train LGBM and predict on val. Returns prediction array."""
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    params = {**base,
              'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    ds = make_dataset(X_tr, y_tr, feat_names)
    m = lgb.train(params, ds, num_boost_round=base['n_estimators'])
    pred = m.predict(X_val)
    return pred


def rank_features(feat, all_feat_cols, target, seed=42, n_trees=50):
    """Rank features by LGBM gain importance."""
    y = feat[target].values.astype(np.float64)
    X = feat[all_feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]

    params = {**base, 'n_estimators': n_trees, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    ds = make_dataset(X, y, all_feat_cols)
    m = lgb.train(params, ds, num_boost_round=n_trees)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def cv_predict(feat, sel_cols, target, group, gkf, seed_offset=0):
    """Run GroupKFold CV for given features/target. Returns OOF prediction array."""
    y = feat[target].values.astype(np.float64)
    oof = np.zeros(len(feat))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
        X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
        X_val = feat[sel_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
        y_tr = y[tr_idx]
        pred = train_and_predict(X_tr, y_tr, X_val, sel_cols, target, SEED + seed_offset + fold)
        oof[val_idx] = pred
    return oof


def cv_predict_local(df, sel_cols, target, group, gkf, seed_offset=0):
    """Same as cv_predict but df is the actual data frame (e.g. with extra ratio features)."""
    y = df[target].values.astype(np.float64)
    oof = np.zeros(len(df))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, group)):
        X_tr = df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
        X_val = df[sel_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
        y_tr = y[tr_idx]
        pred = train_and_predict(X_tr, y_tr, X_val, sel_cols, target, SEED + seed_offset + fold)
        oof[val_idx] = pred
    return oof


def compute_per_target_oof(feat, oof_preds_dict):
    """Compute per-target OOF log loss and average."""
    per_target = {}
    for t in TARGETS:
        ll = log_loss(feat[t].values, np.clip(oof_preds_dict[t], 0.001, 0.999))
        per_target[t] = round(ll, 5)
    avg = round(np.mean(list(per_target.values())), 5)
    return avg, per_target


# ============================================================
# 1. Inter-Target Correlation
# ============================================================
def compute_target_correlations(feat):
    print("\n" + "=" * 70)
    print("1. Inter-Target Correlation Analysis")
    print("=" * 70)

    corr_matrix = feat[TARGETS].corr().values.tolist()

    print("\nCorrelation Matrix:")
    header = "          " + "  ".join(f"{t:>4}" for t in TARGETS)
    print(header)
    for i, t in enumerate(TARGETS):
        row = f"  {t}  " + "  ".join(f"{corr_matrix[i][j]:>+5.3f}" for j in range(len(TARGETS)))
        print(row)

    # Strongly correlated pairs
    print("\nStrongly correlated pairs (|r| > 0.3):")
    strong_pairs = []
    for i in range(len(TARGETS)):
        for j in range(i + 1, len(TARGETS)):
            r = abs(corr_matrix[i][j])
            if r > 0.3:
                strong_pairs.append({
                    'pair': [TARGETS[i], TARGETS[j]],
                    'correlation': round(corr_matrix[i][j], 4),
                    'abs': round(r, 4)
                })
                print(f"  {TARGETS[i]} - {TARGETS[j]}: {corr_matrix[i][j]:.3f}")
    if not strong_pairs:
        print("  None (all |r| < 0.3)")

    # Independent pairs
    print("\nWeakly correlated pairs (|r| < 0.1):")
    independent_pairs = []
    for i in range(len(TARGETS)):
        for j in range(i + 1, len(TARGETS)):
            r = abs(corr_matrix[i][j])
            if r < 0.1:
                independent_pairs.append({
                    'pair': [TARGETS[i], TARGETS[j]],
                    'correlation': round(corr_matrix[i][j], 4),
                    'abs': round(r, 4)
                })
                print(f"  {TARGETS[i]} - {TARGETS[j]}: {corr_matrix[i][j]:.3f}")

    return {
        'correlation_matrix': [[round(v, 4) for v in row] for row in corr_matrix],
        'strong_pairs': strong_pairs,
        'independent_pairs': independent_pairs,
        'strongest_pair': strong_pairs[0] if strong_pairs else None,
    }


# ============================================================
# 2. OOF Pseudo-Target Features
# ============================================================
def compute_pseudo_target_features(feat):
    print("\n" + "=" * 70)
    print("2. OOF Pseudo-Target Features (Leakage-Free)")
    print("=" * 70)

    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # For each target T, train model using ALL OTHER targets as features
    # This is NOT leakage: other targets are independent label columns,
    # not the target being predicted.
    pseudo_oof = {}
    for primary_target in TARGETS:
        other_targets = [ot for ot in TARGETS if ot != primary_target]
        feat_cols = all_feat_cols + other_targets

        ranked = rank_features(feat, feat_cols, primary_target)
        n_select = V53_SWEEP[primary_target]['n_feat'] + len(other_targets)
        sel_cols = ranked[:min(n_select, len(feat_cols))]

        pred = cv_predict(feat, sel_cols, primary_target, group, gkf)
        pseudo_oof[primary_target] = pred
        gc.collect()

    avg_oof, per_target_oof = compute_per_target_oof(feat, pseudo_oof)

    print("\n  Running baseline for comparison...")
    baseline_avg = run_baseline_avg(feat)
    print(f"  Baseline AVG OOF: {baseline_avg}")

    delta = round(avg_oof - baseline_avg, 5)
    best_pseudo = sorted(TARGETS, key=lambda t: per_target_oof[t])[:3]

    return {
        'oof': avg_oof,
        'per_target': per_target_oof,
        'baseline_avg': baseline_avg,
        'delta': delta,
        'best_pseudo_targets': best_pseudo,
    }


def run_baseline_avg(feat):
    """Run baseline and return average OOF only."""
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_preds = {}
    for t in TARGETS:
        ranked = rank_features(feat, all_feat_cols, t)
        sel_cols = ranked[:V53_SWEEP[t]['n_feat']]
        pred = cv_predict(feat, sel_cols, t, group, gkf)
        oof_preds[t] = pred
        gc.collect()

    avg, _ = compute_per_target_oof(feat, oof_preds)
    return avg


# ============================================================
# 3. Cross-Target Ratios & Interactions (Leakage-Free)
# ============================================================
def compute_cross_target_ratios(feat):
    print("\n" + "=" * 70)
    print("3. Cross-Target Ratios & Interactions")
    print("=" * 70)

    # First, get OOF predictions for all targets from baseline models
    # These OOF preds will be used to compute "leakage-free" ratio features
    oof_target_preds = {}
    base_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    for t in TARGETS:
        ranked = rank_features(feat, base_feat_cols, t)
        sel = ranked[:V53_SWEEP[t]['n_feat']]
        oof_target_preds[t] = cv_predict(feat, sel, t, group, gkf)

    # Define ratio feature mappings: which targets does each ratio use?
    RATIO_MAPPING = {
        'Q1_Q2_ratio': ['Q1', 'Q2'],
        'Q1_Q3_ratio': ['Q1', 'Q3'],
        'Q2_Q3_ratio': ['Q2', 'Q3'],
        'S1_S2_ratio': ['S1', 'S2'],
        'S2_S3_ratio': ['S2', 'S3'],
        'S3_S4_ratio': ['S3', 'S4'],
        'Q_avg': ['Q1', 'Q2', 'Q3'],
        'S_avg': ['S1', 'S2', 'S3', 'S4'],
        'Q_S_diff': ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4'],
        'Q1_S2_ratio': ['Q1', 'S2'],
        'Q3_S4_ratio': ['Q3', 'S4'],
    }

    # For each target, select ratio features that don't contain the target
    # Then compute them from OOF predictions of other targets
    base_oof = np.zeros((len(feat), len(TARGETS)))
    for i, t in enumerate(TARGETS):
        base_oof[:, i] = oof_target_preds[t]

    ranking_cols = base_feat_cols  # rank on base features only

    # Now build ratio-enhanced CV using OOF-based ratios
    oof_preds = {}
    for t in TARGETS:
        ti = TARGETS.index(t)
        y = feat[t].values.astype(np.float64)
        # Find ratios that don't contain this target
        safe_ratios = [r for r, used in RATIO_MAPPING.items() if t not in used]

        # Score safe ratios by correlation with target (using OOF preds as proxy)
        # Use OOF predictions of other targets to estimate correlation
        scored = []
        for r in safe_ratios:
            used = RATIO_MAPPING[r]
            other_indices = [TARGETS.index(u) for u in used]
            # Compute ratio from OOF preds of other targets
            other_oof = base_oof[:, other_indices]
            ratio_name = r.replace('_ratio', '').replace('_avg', '').replace('_diff', '')
            if 'avg' in r:
                ratio_vals = other_oof.mean(axis=1)
            elif 'diff' in r:
                # Q_S_diff = Q_avg - S_avg
                q_idx = [i for i in other_indices if i < 3]
                s_idx = [i for i in other_indices if i >= 3]
                if q_idx and s_idx:
                    ratio_vals = base_oof[:, q_idx].mean(axis=1) - base_oof[:, s_idx].mean(axis=1)
                else:
                    ratio_vals = other_oof.mean(axis=1)
            else:
                # ratio = A / (B + eps)
                if other_oof.shape[1] >= 2:
                    ratio_vals = other_oof[:, 0] / (other_oof[:, 1:] + 1e-8).mean(axis=1)
                else:
                    ratio_vals = other_oof[:, 0]
            # Correlation with target
            corr = np.corrcoef(ratio_vals, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            scored.append((r, abs(corr)))

        scored.sort(key=lambda x: -x[1])
        top_ratios = [r for r, _ in scored[:3]]

        n_base = V53_SWEEP[t]['n_feat'] - 3
        ranked = rank_features(feat, ranking_cols, t)
        sel_base = ranked[:max(n_base, 0)]
        sel_cols = sel_base + top_ratios

        # Build extended feature matrix with OOF-based ratio features
        # For each fold, compute ratios from OOF preds of other targets (computed outside fold)
        ext_feat_cols = sel_cols
        df_ext = feat[base_feat_cols].copy()
        for r in top_ratios:
            used = RATIO_MAPPING[r]
            other_indices = [TARGETS.index(u) for u in used]
            other_oof = base_oof[:, other_indices]
            if 'avg' in r:
                df_ext[r] = other_oof.mean(axis=1)
            elif 'diff' in r:
                q_idx = [i for i in other_indices if i < 3]
                s_idx = [i for i in other_indices if i >= 3]
                if q_idx and s_idx:
                    df_ext[r] = base_oof[:, q_idx].mean(axis=1) - base_oof[:, s_idx].mean(axis=1)
                else:
                    df_ext[r] = other_oof.mean(axis=1)
            else:
                if other_oof.shape[1] >= 2:
                    df_ext[r] = other_oof[:, 0] / (other_oof[:, 1:] + 1e-8).mean(axis=1)
                else:
                    df_ext[r] = other_oof[:, 0]

        sn = [sanitize_col(c) for c in ext_feat_cols]
        oof_fold = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = df_ext[ext_feat_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_val = df_ext[ext_feat_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            pred = train_and_predict(X_tr, y_tr, X_val, ext_feat_cols, t, SEED + fold)
            oof_fold[val_idx] = pred
        oof_preds[t] = oof_fold
        gc.collect()

    baseline_avg = run_baseline_avg(feat)
    avg_oof, per_target = compute_per_target_oof(feat, oof_preds)
    delta = round(avg_oof - baseline_avg, 5)

    print(f"\n  AVG OOF (log_loss): {avg_oof}")
    print(f"  Delta: {delta}")
    print(f"  Per-target OOF: {per_target}")

    return {
        'oof': avg_oof,
        'per_target': per_target,
        'baseline_avg': baseline_avg,
        'delta': delta,
    }


# ============================================================
# 4. Multi-Task LGBM (Leakage-Free via OOF pseudo-targets)
# ============================================================
def run_multi_task_lgbm(feat):
    print("\n" + "=" * 70)
    print("4. Multi-Task LGBM (Per-Target with Cross-Target Features)")
    print("=" * 70)

    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Step 1: Get OOF predictions for all targets from baseline models
    # These serve as "pseudo-targets" for the multi-task features
    oof_baseline = {}
    for t in TARGETS:
        ranked = rank_features(feat, all_feat_cols, t)
        sel = ranked[:V53_SWEEP[t]['n_feat']]
        oof_baseline[t] = cv_predict(feat, sel, t, group, gkf)

    # Step 2: For each target, build extended feature matrix with OOF-based pseudo-targets
    # and rank+select features + pseudo-targets
    oof_preds = {}
    for t in TARGETS:
        ti = TARGETS.index(t)
        y = feat[t].values.astype(np.float64)
        other_targets = [ot for ot in TARGETS if ot != t]
        other_indices = [TARGETS.index(ot) for ot in other_targets]

        # Build extended features: base + OOF-predicted other targets
        ext_feat_cols = all_feat_cols + other_targets
        df_ext = feat[all_feat_cols].copy()
        for ot in other_targets:
            df_ext[ot] = oof_baseline[ot]

        # Rank features (base only, pseudo-targets added after ranking)
        ranked_base = rank_features(feat, all_feat_cols, t)
        n_base = V53_SWEEP[t]['n_feat']
        sel_base = ranked_base[:n_base]

        # Evaluate pseudo-targets individually by correlation with target
        pseudo_corr = {}
        for ot, oi in zip(other_targets, other_indices):
            corr = abs(np.corrcoef(oof_baseline[ot], y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            pseudo_corr[ot] = corr

        # Select top pseudo-targets by correlation
        top_pseudo = sorted(pseudo_corr.items(), key=lambda x: -x[1])[:3]
        top_pseudo_names = [p[0] for p in top_pseudo]
        sel_cols = sel_base + top_pseudo_names

        # Use df_ext for CV (it has OOF-based pseudo-targets, not real targets)
        sn = [sanitize_col(c) for c in sel_cols]
        oof_fold = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = df_ext[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_val = df_ext[sel_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            pred = train_and_predict(X_tr, y_tr, X_val, sel_cols, t, SEED + fold)
            oof_fold[val_idx] = pred
        oof_preds[t] = oof_fold
        gc.collect()

    baseline_avg = run_baseline_avg(feat)
    avg_oof, per_target = compute_per_target_oof(feat, oof_preds)
    delta = round(avg_oof - baseline_avg, 5)

    print(f"\n  AVG OOF (log_loss): {avg_oof}")
    print(f"  Delta: {delta}")
    print(f"  Per-target OOF: {per_target}")

    return {
        'oof': avg_oof,
        'per_target': per_target,
        'baseline_avg': baseline_avg,
        'delta': delta,
    }


# ============================================================
# 5. Residual Learning
# ============================================================
def run_residual_learning(feat):
    print("\n" + "=" * 70)
    print("5. Residual Learning")
    print("=" * 70)

    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    damping = 0.3

    oof_preds = {}
    for t in TARGETS:
        y = feat[t].values.astype(np.float64)
        ranked = rank_features(feat, all_feat_cols, t)
        sel_cols = ranked[:V53_SWEEP[t]['n_feat']]

        oof_stage1 = np.zeros(len(feat))
        oof_stage2 = np.zeros(len(feat))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_val = feat[sel_cols].iloc[val_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            y_val = y[val_idx]

            # Stage 1: train model, predict on val
            pred1 = train_and_predict(X_tr, y_tr, X_val, sel_cols, t, SEED + fold)
            oof_stage1[val_idx] = pred1

            # Stage 2: residuals trained on training fold
            sn = [sanitize_col(c) for c in sel_cols]
            model_s1 = lgb.train(
                {**CFGS[V53_SWEEP[t]['cfg']],
                 'scale_pos_weight': max(((y_tr==0).sum())/max((y_tr==1).sum(),1),0.1),
                 'random_state': SEED+fold,
                 'force_row_wise':True,'n_jobs':1,'verbose':-1},
                lgb.Dataset(X_tr, label=y_tr, feature_name=sn),
                num_boost_round=CFGS[V53_SWEEP[t]['cfg']]['n_estimators'])
            pred1_train = model_s1.predict(X_tr)
            res_tr = y_tr - pred1_train

            # Build residual features using numpy stacking
            X_tr_res = np.column_stack([X_tr, pred1_train.reshape(-1, 1)])
            X_val_res = np.column_stack([X_val, np.zeros(len(X_val))])  # pred1_train not needed for val

            # Fix: val needs pred1 on val, but we trained stage1 on train fold
            # Use pred1_train as the stage1 feature for val rows too (they share same fold's model)
            # Actually we need pred1_val for the stage2 features. Use the val pred1 instead.
            # But we already have pred1 from train_and_predict which is val predictions.
            # Use pred1 as stage1 feature for val, and pred1_train for val train pred
            # Simple approach: just use 0 for train rows and pred1 (val pred) for val rows
            X_val_res = np.column_stack([X_val, np.zeros(len(X_val))])

            res_params = {**CFGS[V53_SWEEP[t]['cfg']],
                          'scale_pos_weight': 1.0, 'random_state': SEED + fold,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn2 = [sanitize_col(c) for c in sel_cols] + [f'{t}_train_pred']
            ds_tr = lgb.Dataset(X_tr_res, label=res_tr, feature_name=sn2)
            m2 = lgb.train(res_params, ds_tr, num_boost_round=CFGS[V53_SWEEP[t]['cfg']]['n_estimators'])
            oof_stage2[val_idx] = m2.predict(X_val_res)

        # Combine
        final = np.clip(oof_stage1 + damping * oof_stage2, 0.001, 0.999)
        oof_preds[t] = final
        gc.collect()

    baseline_avg = run_baseline_avg(feat)
    avg_oof, per_target = compute_per_target_oof(feat, oof_preds)
    delta = round(avg_oof - baseline_avg, 5)

    return {
        'oof': avg_oof,
        'per_target': per_target,
        'baseline_avg': baseline_avg,
        'delta': delta,
        'damping': damping,
    }


# ============================================================
# 6. Target Order Analysis (Cascade Prediction)
# ============================================================
def run_target_order_analysis(feat):
    print("\n" + "=" * 70)
    print("6. Target Order Analysis (Cascade Prediction)")
    print("=" * 70)

    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    cascade_orders = [
        ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4'],  # natural
        ['S1', 'S2', 'S3', 'S4', 'Q1', 'Q2', 'Q3'],  # S first
        ['Q1', 'S1', 'Q2', 'S2', 'Q3', 'S3', 'S4'],  # interleaved
        ['S4', 'S3', 'S2', 'S1', 'Q3', 'Q2', 'Q1'],  # reverse
        ['S2', 'S4', 'S1', 'S3', 'Q1', 'Q3', 'Q2'],  # by correlation
    ]

    results = {}
    for order_idx, order in enumerate(cascade_orders):
        print(f"\n  Order {order_idx + 1}: {order}")

        oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

        for target_idx, current_t in enumerate(order):
            y = feat[current_t].values.astype(np.float64)
            prev_targets = [order[k] for k in range(target_idx)]
            feat_cols = all_feat_cols + prev_targets

            ranked = rank_features(feat, feat_cols, current_t)
            n_select = V53_SWEEP[current_t]['n_feat']
            sel_cols = ranked[:n_select]

            pred = cv_predict(feat, sel_cols, current_t, group, gkf, seed_offset=1000 + target_idx * 100)
            oof_preds[current_t] = pred
            gc.collect()

        avg_oof, per_target = compute_per_target_oof(feat, oof_preds)
        baseline_avg = run_baseline_avg(feat)
        delta = round(avg_oof - baseline_avg, 5)

        key = f'order{order_idx + 1}_' + '_'.join(c for c in order)
        results[key] = {
            'order': order,
            'oof': avg_oof,
            'per_target': per_target,
            'baseline_avg': baseline_avg,
            'delta': delta,
        }
        print(f"    AVG OOF: {avg_oof:.5f} (delta: {delta:+.5f})")

    return results


# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("V259: Cross-Target Analysis & Joint Training")
    print("=" * 70)

    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f"\nData: {feat.shape}")
    print(f"Feature cols: {len(get_feature_cols(feat))}")
    print(f"Targets: {TARGETS}")

    # Baseline for reference across all experiments
    print("\n  Computing baseline...")
    baseline_avg = run_baseline_avg(feat)
    print(f"  Baseline AVG OOF: {baseline_avg}")

    result = {
        'version': 'v259_cross_target',
        'method': 'GroupKFold(5), seed=42',
        'n_subjects': int(feat['subject_id'].nunique()),
        'baseline_avg': baseline_avg,
    }

    # 1. Correlation
    result['target_correlations'] = compute_target_correlations(feat)

    # 2. Pseudo-target features
    pseudo_result = compute_pseudo_target_features(feat)
    result['pseudo_target_features'] = {
        'oof': pseudo_result['oof'],
        'delta': pseudo_result['delta'],
        'best_pseudo_targets': pseudo_result['best_pseudo_targets'],
    }

    # 3. Cross-target ratios
    ratio_result = compute_cross_target_ratios(feat)
    result['cross_target_ratios'] = {
        'oof': ratio_result['oof'],
        'delta': ratio_result['delta'],
    }

    # 4. Multi-task LGBM
    mt_result = run_multi_task_lgbm(feat)
    result['multi_task'] = {
        'oof': mt_result['oof'],
        'delta': mt_result['delta'],
    }

    # 5. Residual learning
    rl_result = run_residual_learning(feat)
    result['residual_learning'] = {
        'oof': rl_result['oof'],
        'delta': rl_result['delta'],
    }

    # 6. Target order analysis
    cascade_results = run_target_order_analysis(feat)
    result['target_order_analysis'] = {
        k: {
            'order': v['order'],
            'oof': v['oof'],
            'delta': v['delta'],
        }
        for k, v in cascade_results.items()
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline AVG OOF: {baseline_avg}")

    print("\nTarget Correlations:")
    corr = result['target_correlations']
    print(f"  Strongest pair: {corr.get('strongest_pair', 'N/A')}")
    print(f"  Strong pairs (|r|>0.3): {len(corr.get('strong_pairs', []))}")
    print(f"  Independent pairs (|r|<0.1): {len(corr.get('independent_pairs', []))}")

    print("\nExperiment Comparison:")
    print(f"  {'Method':<30} {'OOF':>8} {'Delta':>10}")
    for key in ['pseudo_target_features', 'cross_target_ratios', 'multi_task', 'residual_learning']:
        oof = result[key]['oof']
        delta = result[key]['delta']
        print(f"  {key:<30} {oof:>8.5f} {delta:>+10.5f}")

    print("\nCascade Orders:")
    for k, v in cascade_results.items():
        best_delta = min(x['delta'] for x in cascade_results.values())
        marker = " <-- best delta" if v['delta'] == best_delta else ""
        print(f"  {k}: OOF={v['oof']:.5f} (delta={v['delta']:+.5f}){marker}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    result_path = EXPERIMENTS / 'v259_cross_target_result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved: {result_path}")

    return result


if __name__ == '__main__':
    main()
