"""
V61 — Multi-Seed Stacking Ensemble (V58 base + bug-free multi-seed + S4 dedicated + C-tuned)

Improvements over V58:
  1. Bug-free multi-seed: each fold trains N_SEEDS models per model type, averages OOF correctly
  2. S4 dedicated: wider feature set (n_feat=25), single LGBM (no stacking — proven optimal in V53/V58)
  3. C-tuned meta-learner: grid search C=0.1, 0.3, 0.5, 1.0, 2.0, 5.0 on OOF
  4. Per-target model selection with proper comparison to V53 swept

Architecture:
  Level 0: LGBM(N seeds) + CatBoost(N seeds) + XGBoost(N seeds) → averaged OOF per model
  Level 1: LogisticRegression(tuned C) → stacked predictions
  S4: Single LGBM (no stacking, wider features)
"""

import sys, gc, logging, json, re, time, warnings, itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS = ROOT / "experiments"
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

# V53 swept configs — S4 gets wider feature set
V53_CONFIGS = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 5},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety', 'n_feat': 21},
    'S4': {'cfg': 'wide', 'n_feat': 25},  # dedicated wider set for S4
}

# S4 dedicated config
S4_CFG = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

N_SEEDS = 5  # seeds per model per fold
C_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def add_personalization(df, feature_cols):
    """Add subject-level zscore features (batch agg, no fragmentation)."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols


def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ─── Single model trainers ──────────────────────────────────────────────────


def train_lgbm(X_train, y_train, feat_names, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    return model


def train_catboost(X_train, y_train, feat_names, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'num_boost_round': cfg['ne'],
        'learning_rate': cfg['lr'], 'depth': cfg['md'] + 1,
        'subsample': cfg['ss'], 'colsample_bylevel': cfg['cb'],
        'l2_leaf_reg': cfg['rl'],
        'random_seed': seed, 'thread_count': 1, 'verbose': 0,
        'scale_pos_weight': spw, 'max_ctr_complexity': 1,
        'boosting_type': 'Ordered',
    }
    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=None)
    return model


def train_xgboost(X_train, y_train, feat_names, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'],
        'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_weight': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'tree_method': 'hist',
        'verbosity': 0, 'n_jobs': 1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=None, verbose=False)
    return model


# ─── Feature ranking ─────────────────────────────────────────────────────────


def rank_features_importance(train, feat_cols, target, cfgs, n_seeds=5):
    y = train[target].values.astype(np.float64)
    X = train[feat_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in feat_cols]

    imp_sum = np.zeros(len(feat_cols))

    for seed in range(1, n_seeds + 1):
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'min_child_samples': 15, 'random_state': seed,
            'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
        }
        ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=50)
        imp_sum += model.feature_importance(importance_type='gain')
        del model, ds

    imp_avg = imp_sum / n_seeds
    ranked = sorted(zip(feat_cols, imp_avg), key=lambda x: -x[1])
    del X
    gc.collect()
    return [r[0] for r in ranked]


# ─── C-tuned meta-learner ───────────────────────────────────────────────────


def tune_meta_c(oof_stack, y, C_values):
    """Grid search best C for LogisticRegression meta-learner on OOF."""
    best_c = 1.0
    best_loss = float('inf')

    for c_val in C_values:
        meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=1000, random_state=42)
        meta.fit(oof_stack, y)
        preds = np.clip(meta.predict_proba(oof_stack)[:, 1], 1e-15, 1 - 1e-15)
        loss = logloss(y, preds)
        if loss < best_loss:
            best_loss = loss
            best_c = c_val

    return best_c, best_loss


# ─── Multi-seed OOF training (bug-free) ─────────────────────────────────────


def train_multi_seed_oof(train, targets_feat_cols, target, cfg, n_splits=3, seed_base=42):
    """
    Bug-free multi-seed OOF:
    - Each fold: train N_SEEDS models per type, average predictions on validation set
    - Accumulate across folds correctly (each fold contributes independently)
    - Stack the 3 averaged model OOF predictions
    - C-tune meta-learner
    """
    feat_cols = targets_feat_cols[target]
    y = train[target].values.astype(np.float64)
    groups = train['subject_id'].values
    X = train[feat_cols].fillna(0).values.astype(np.float64)

    gkf = GroupKFold(n_splits=n_splits)

    n_models = 3  # lgb, cat, xgb

    # For each fold, compute OOF predictions for each model (averaged across seeds)
    # Then stack across folds
    oof_per_fold = []
    losses_per_model = {0: [], 1: [], 2: []}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        sn = [sanitize(c) for c in feat_cols]

        # Average predictions across seeds for each model
        pred_lgb = np.zeros(len(val_idx))
        pred_cat = np.zeros(len(val_idx))
        pred_xgb = np.zeros(len(val_idx))

        for s in range(N_SEEDS):
            seed = seed_base + fold * N_SEEDS + s

            # LightGBM
            m = train_lgbm(X_tr, y_tr, sn, cfg, seed)
            pred_lgb += np.clip(m.predict(X_val), 0.0001, 0.9999)
            del m

            # CatBoost
            m = train_catboost(X_tr, y_tr, sn, cfg, seed + 100)
            pred_cat += np.clip(m.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
            del m

            # XGBoost
            m = train_xgboost(X_tr, y_tr, sn, cfg, seed + 200)
            pred_xgb += np.clip(m.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
            del m

        # Average across seeds
        pred_lgb /= N_SEEDS
        pred_cat /= N_SEEDS
        pred_xgb /= N_SEEDS

        # Record fold losses
        losses_per_model[0].append(logloss(y_val, pred_lgb))
        losses_per_model[1].append(logloss(y_val, pred_cat))
        losses_per_model[2].append(logloss(y_val, pred_xgb))

        # Store this fold's OOF predictions
        oof_per_fold.append({
            'val_idx': val_idx,
            'lgb': pred_lgb,
            'cat': pred_cat,
            'xgb': pred_xgb,
        })

        gc.collect()

    # Build full OOF arrays from per-fold predictions
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for fold_oof in oof_per_fold:
        idx = fold_oof['val_idx']
        oof_lgb[idx] = fold_oof['lgb']
        oof_cat[idx] = fold_oof['cat']
        oof_xgb[idx] = fold_oof['xgb']

    # Stack
    oof_stack = np.column_stack([oof_lgb, oof_cat, oof_xgb])

    # C-tune meta-learner
    best_c, best_meta_loss = tune_meta_c(oof_stack, y, C_VALUES)

    # Train final meta-learner with best C
    meta = LogisticRegression(C=best_c, solver='lbfgs', max_iter=1000, random_state=seed_base)
    meta.fit(oof_stack, y)

    # Evaluate stacked
    oof_stacked = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
    stacked_loss = logloss(y, oof_stacked)

    avg_losses = {i: np.mean(v) for i, v in losses_per_model.items()}

    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_stack': oof_stacked,
        'meta_model': meta,
        'best_c': best_c,
        'best_meta_loss': best_meta_loss,
        'stacked_loss': stacked_loss,
        'avg_losses': avg_losses,
    }


# ─── Multi-seed full training → test ─────────────────────────────────────────


def train_multi_seed_full(train_feat, test_feat, feat_cols, y_train, cfg, seed_base=42):
    """Train each model with N_SEEDS seeds on full data, average predictions."""
    X_train = train_feat[feat_cols].fillna(0).values.astype(np.float64)
    X_test = test_feat[feat_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in feat_cols]

    preds_per_model = {}

    for model_name, trainer in [('lgb', train_lgbm), ('cat', train_catboost), ('xgb', train_xgboost)]:
        seed_preds = []
        for s in range(N_SEEDS):
            seed = seed_base + s
            m = trainer(X_train, y_train, sn, cfg, seed)
            if model_name == 'cat':
                pred = m.predict_proba(X_test)[:, 1]
            else:
                pred = m.predict(X_test)
            seed_preds.append(np.clip(pred, 0.0001, 0.9999))
            del m

        preds_per_model[model_name] = np.clip(np.mean(seed_preds, axis=0), 0.0001, 0.9999)
        gc.collect()

    return preds_per_model


# ─── S4 Single LGBM (dedicated, no stacking) ────────────────────────────────


def train_s4_single(train_feat, test_feat, feat_cols, y_train, cfg, seed_base=42):
    """S4: Single LGBM with wider features, averaged across seeds."""
    X_train = train_feat[feat_cols].fillna(0).values.astype(np.float64)
    X_test = test_feat[feat_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in feat_cols]

    seed_preds = []
    for s in range(N_SEEDS):
        m = train_lgbm(X_train, y_train, sn, cfg, seed_base + s)
        pred = np.clip(m.predict(X_test), 0.0001, 0.9999)
        seed_preds.append(pred)
        del m

    return np.clip(np.mean(seed_preds, axis=0), 0.0001, 0.9999)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("V61 — Multi-Seed Stacking (bug-free + C-tuned + S4 dedicated)")
    log.info("=" * 80)

    # Load data
    log.info("Loading data...")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")

    train_cols_order = list(train.columns)
    test = test[train_cols_order]

    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    # Personalization
    feat_cols = get_feature_cols(train)
    train, zscore_cols = add_personalization(train, feat_cols)
    test, _ = add_personalization(test, feat_cols)

    all_cols = feat_cols + zscore_cols
    log.info(f"  Features: {len(feat_cols)} base + {len(zscore_cols)} zscore = {len(all_cols)} total")

    # Feature ranking
    log.info("Ranking features...")
    targets_feat_cols = {}

    for target in TARGETS:
        config = V53_CONFIGS[target]
        cfg_name = config['cfg']
        n_feat = config['n_feat']
        base_cfg = CFGS[cfg_name]

        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(train, leak_cols, target, CFGS, n_seeds=5)
        sel_cols = ranked[:n_feat]
        targets_feat_cols[target] = sel_cols

        log.info(f"  {target}: cfg={cfg_name}, n_feat={n_feat}, top5={sel_cols[:5]}")

    # ── OOF evaluation ──
    log.info("\n" + "=" * 80)
    log.info("Phase 1: OOF evaluation (Multi-seed, GroupKFold n_splits=3)")
    log.info("=" * 80)

    oof_results = {}
    cv_scores = {}

    for target in TARGETS:
        log.info(f"\n  --- {target} ---")

        if target == 'S4':
            base_cfg = S4_CFG
        else:
            cfg_name = V53_CONFIGS[target]['cfg']
            base_cfg = CFGS[cfg_name]

        result = train_multi_seed_oof(train, targets_feat_cols, target, base_cfg, n_splits=3, seed_base=42)
        oof_results[target] = result

        cv_scores[target] = {
            'lgb': result['avg_losses'][0],
            'cat': result['avg_losses'][1],
            'xgb': result['avg_losses'][2],
            'stack': result['stacked_loss'],
            'best_c': result['best_c'],
            'meta_c_loss': result['best_meta_loss'],
        }

        log.info(f"  Best C: {result['best_c']} (meta loss={result['best_meta_loss']:.4f})")
        log.info(f"  Fold losses — LGBM: {result['avg_losses'][0]:.4f}, "
                f"CatBoost: {result['avg_losses'][1]:.4f}, "
                f"XGBoost: {result['avg_losses'][2]:.4f}")
        log.info(f"  Stacked (OOF): {result['stacked_loss']:.4f}")

    # ── Comparison with V53 swept and V58 ──
    log.info("\n" + "=" * 80)
    log.info("Phase 2: Comparison")
    log.info("=" * 80)

    v53_swept = {'Q1': 0.7591, 'Q2': 0.6929, 'Q3': 0.6893, 'S1': 0.6029, 'S2': 0.6621, 'S3': 0.7144, 'S4': 0.6438}
    v58_results = {
        'Q1': 0.6469, 'Q2': 0.6310, 'Q3': 0.6337, 'S1': 0.5653,
        'S2': 0.6249, 'S3': 0.6223, 'S4': 0.6532
    }

    for target in TARGETS:
        s = cv_scores[target]
        v53_loss = v53_swept[target]
        v58_loss = v58_results[target]
        delta_v53 = v53_loss - s['stack']
        delta_v58 = v58_loss - s['stack']
        log.info(f"  {target}: V53={v53_loss:.4f} | V58={v58_loss:.4f} | V61={s['stack']:.4f} | "
                f"Δ_vs_V53=+{delta_v53:.4f} | Δ_vs_V58={delta_v58:+.4f} | best_C={s['best_c']}")

    # Best model per target
    log.info("\n  Per-target selection:")
    best_per_target = {}
    for target in TARGETS:
        s = cv_scores[target]
        if target == 'S4':
            # S4: single LGBM (proven better in V53/V58)
            best_per_target[target] = 'lgb_single'
            log.info(f"    {target}: lgb_single (S4 dedicated, no stacking)")
            continue
        best_single = min(s['lgb'], s['cat'], s['xgb'])
        if s['stack'] < best_single and s['stack'] < v53_swept[target]:
            best_per_target[target] = 'stack'
            log.info(f"    {target}: stack (OOF={s['stack']:.4f})")
        else:
            # Use best single model
            best_model = min(['lgb', 'cat', 'xgb'], key=lambda m: s[m])
            best_per_target[target] = best_model
            log.info(f"    {target}: {best_model} (OOF={s[best_model]:.4f})")

    # ── Generate test predictions ──
    log.info("\n" + "=" * 80)
    log.info("Phase 3: Test predictions")
    log.info("=" * 80)

    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    predictions = {}

    for target in TARGETS:
        log.info(f"\n  --- {target} ---")

        if target == 'S4':
            base_cfg = S4_CFG
            feat_cols = targets_feat_cols[target]
            y_train = train[target].values.astype(np.float64)
            predictions[target] = train_s4_single(train, test, feat_cols, y_train, base_cfg, seed_base=42)
            log.info(f"  → Using S4 single LGBM ({N_SEEDS} seeds)")
        else:
            cfg_name = V53_CONFIGS[target]['cfg']
            base_cfg = CFGS[cfg_name]
            feat_cols = targets_feat_cols[target]
            y_train = train[target].values.astype(np.float64)

            preds = train_multi_seed_full(train, test, feat_cols, y_train, base_cfg, seed_base=42)
            log.info(f"  LGBM mean={preds['lgb'].mean():.4f}, CatBoost mean={preds['cat'].mean():.4f}, "
                     f"XGBoost mean={preds['xgb'].mean():.4f}")

            strategy = best_per_target[target]
            if strategy == 'stack':
                oof_stack_full = np.column_stack([preds['lgb'], preds['cat'], preds['xgb']])
                predictions[target] = np.clip(
                    oof_results[target]['meta_model'].predict_proba(oof_stack_full)[:, 1],
                    0.0001, 0.9999
                )
                log.info(f"  → Using stacked ensemble (C={oof_results[target]['best_c']})")
            else:
                predictions[target] = preds[strategy]
                log.info(f"  → Using best single: {strategy}")

    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v61_multiseed_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    log.info(f"\n{'='*80}")
    log.info(f"✅ Submission saved: {sub_path}")
    log.info(f"Rows: {len(sub)}")
    for t in TARGETS:
        log.info(f"  {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")
    log.info(f"{'='*80}")

    # Save meta
    meta = {
        'version': 'V61_multiseed',
        'name': 'Multi-Seed Stacking (bug-free + C-tuned + S4 dedicated)',
        'timestamp': datetime.now().isoformat(),
        'submission_file': str(sub_path),
        'n_seeds_per_model': N_SEEDS,
        'C_values_tested': C_VALUES,
        'n_splits': 3,
        'n_features': {t: V53_CONFIGS[t]['n_feat'] for t in TARGETS},
        's4_dedicated': True,
        'v53_swept_cv': v53_swept,
        'v58_results': v58_results,
        'cv_results': cv_scores,
        'best_per_target': best_per_target,
    }
    meta_path = SUBMIT / f'meta_v61_multiseed_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")

    # Save OOF
    oof_df = pd.DataFrame(index=train.index)
    for target in TARGETS:
        oof_df[f'{target}_lgb'] = oof_results[target]['oof_lgb']
        oof_df[f'{target}_cat'] = oof_results[target]['oof_cat']
        oof_df[f'{target}_xgb'] = oof_results[target]['oof_xgb']
        oof_df[f'{target}_stack'] = oof_results[target]['oof_stack']
    oof_path = EXPERIMENTS / f'oof_v61.csv'
    oof_df.to_parquet(oof_path)
    log.info(f"  OOF saved: {oof_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
