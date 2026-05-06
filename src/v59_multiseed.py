"""
V59 — Multi-Seed Stacking Ensemble with S4 Specialization

Improvements over V58:
  1. Multi-seed ensemble per model (5 seeds each) before stacking → more stable OOF
  2. S4 gets dedicated treatment: wider feature set + deeper model
  3. Stacking meta-learner tuned: C=0.5 for stronger regularization
  4. V58 comparison baked in

Architecture:
  Level 0: LGBM(5 seeds) + CatBoost(5 seeds) + XGBoost(5 seeds) → averaged OOF
  Level 1: LogisticRegression(C=0.5) → final predictions
"""

import sys, gc, logging, json, re, time, warnings
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
    'S4': {'cfg': 'wide', 'n_feat': 25},  # wider feat set for S4
}

# S4 gets its own config: wider, shallower (similar to what worked for S4 in V53)
S4_CONFIG = {'cfg': 'wide', 'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

N_SEEDS = 5  # seeds per model per fold


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


# ─── Multi-seed OOF training ────────────────────────────────────────────────


def train_multi_seed_oof(train, targets_feat_cols, target, cfg, n_splits=3, seed_base=42):
    """
    Train LGBM + CatBoost + XGBoost with multiple seeds each, using GroupKFold.
    Average predictions across seeds for each model.
    Then stack the 3 model OOF predictions.
    """
    feat_cols = targets_feat_cols[target]
    y = train[target].values.astype(np.float64)
    groups = train['subject_id'].values
    X = train[feat_cols].fillna(0).values.astype(np.float64)
    
    gkf = GroupKFold(n_splits=n_splits)
    
    n_models = 3  # lgb, cat, xgb
    
    # Accumulate OOF predictions across folds and seeds
    oof_accum = np.zeros((len(y), n_models))
    fold_losses_per_model = {i: [] for i in range(n_models)}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        sn = [sanitize(c) for c in feat_cols]
        
        # Train multiple seeds per model
        for seed_offset in range(N_SEEDS):
            seed = seed_base + fold * N_SEEDS + seed_offset
            
            # LightGBM
            m_lgb = train_lgbm(X_tr, y_tr, sn, cfg, seed)
            preds_lgb = np.clip(m_lgb.predict(X_val), 0.0001, 0.9999)
            oof_accum[val_idx, 0] += preds_lgb / N_SEEDS  # distribute equally
            fold_losses_per_model[0].append(logloss(y_val, np.clip(oof_accum[val_idx, 0], 0.0001, 0.9999)))
            del m_lgb
            
            # CatBoost
            m_cat = train_catboost(X_tr, y_tr, sn, cfg, seed + 100)
            preds_cat = np.clip(m_cat.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
            oof_accum[val_idx, 1] += preds_cat / N_SEEDS
            fold_losses_per_model[1].append(logloss(y_val, np.clip(oof_accum[val_idx, 1], 0.0001, 0.9999)))
            del m_cat
            
            # XGBoost
            m_xgb = train_xgboost(X_tr, y_tr, sn, cfg, seed + 200)
            preds_xgb = np.clip(m_xgb.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
            oof_accum[val_idx, 2] += preds_xgb / N_SEEDS
            fold_losses_per_model[2].append(logloss(y_val, np.clip(oof_accum[val_idx, 2], 0.0001, 0.9999)))
            del m_xgb
        
        gc.collect()
    
    # Average OOF predictions
    oof_lgb = oof_accum[:, 0]
    oof_cat = oof_accum[:, 1]
    oof_xgb = oof_accum[:, 2]
    
    # Build stacking features
    oof_stack = np.column_stack([oof_lgb, oof_cat, oof_xgb])
    
    # Train meta-learner with tuned C
    meta = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, random_state=seed_base)
    meta.fit(oof_stack, y)
    
    # Evaluate stacked
    oof_stacked = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
    stacked_loss = logloss(y, oof_stacked)
    
    avg_losses = {i: np.mean(v) for i, v in fold_losses_per_model.items()}
    
    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_stack': oof_stacked,
        'meta_model': meta,
        'avg_losses': avg_losses,
        'stacked_loss': stacked_loss,
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


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("V59 — Multi-Seed Stacking Ensemble (LGBM×5 + CatBoost×5 + XGBoost×5)")
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
        
        # S4 uses its own wider config
        if target == 'S4':
            base_cfg = S4_CONFIG
        else:
            config = V53_CONFIGS[target]
            cfg_name = config['cfg']
            base_cfg = CFGS[cfg_name]
        
        result = train_multi_seed_oof(train, targets_feat_cols, target, base_cfg, n_splits=3, seed_base=42)
        oof_results[target] = result
        
        cv_scores[target] = {
            'lgb': result['avg_losses'][0],
            'cat': result['avg_losses'][1],
            'xgb': result['avg_losses'][2],
            'stack': result['stacked_loss'],
        }
        
        log.info(f"  Fold losses — LGBM: {result['avg_losses'][0]:.4f}, "
                f"CatBoost: {result['avg_losses'][1]:.4f}, "
                f"XGBoost: {result['avg_losses'][2]:.4f}")
        log.info(f"  Stacked (OOF): {result['stacked_loss']:.4f}")
    
    # ── Comparison with V53 and V58 ──
    log.info("\n" + "=" * 80)
    log.info("Phase 2: Comparison with V53 swept (0.6806) and V58 (0.6276)")
    log.info("=" * 80)
    
    v53_swept = {'Q1': 0.7591, 'Q2': 0.6929, 'Q3': 0.6893, 'S1': 0.6029, 'S2': 0.6621, 'S3': 0.7144, 'S4': 0.6438}
    v58_avg = 0.6276
    
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
        log.info(f"  {target}: V53={v53_loss:.4f} | V58={v58_loss:.4f} | V59={s['stack']:.4f} | "
                f"Δ_vs_V53=+{delta_v53:.4f} | Δ_vs_V58={delta_v58:+.4f}")
    
    # Best model per target
    log.info("\n  Per-target selection:")
    best_per_target = {}
    for target in TARGETS:
        s = cv_scores[target]
        if s['stack'] < min(s['lgb'], s['cat'], s['xgb']):
            best_per_target[target] = 'stack'
        else:
            best_per_target[target] = 'single'
        log.info(f"    {target}: stack={s['stack']:.4f} vs best_single={min(s['lgb'],s['cat'],s['xgb']):.4f}")
    
    # ── Generate test predictions ──
    log.info("\n" + "=" * 80)
    log.info("Phase 3: Test predictions")
    log.info("=" * 80)
    
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    predictions = {}
    
    for target in TARGETS:
        log.info(f"\n  --- {target} ---")
        
        if target == 'S4':
            base_cfg = S4_CONFIG
        else:
            config = V53_CONFIGS[target]
            cfg_name = config['cfg']
            base_cfg = CFGS[cfg_name]
        
        feat_cols = targets_feat_cols[target]
        y_train = train[target].values.astype(np.float64)
        
        # Multi-seed full training
        preds = train_multi_seed_full(train, test, feat_cols, y_train, base_cfg, seed_base=42)
        
        log.info(f"  LGBM mean={preds['lgb'].mean():.4f}, CatBoost mean={preds['cat'].mean():.4f}, "
                f"XGBoost mean={preds['xgb'].mean():.4f}")
        
        # Stack
        oof_stack_full = np.column_stack([preds['lgb'], preds['cat'], preds['xgb']])
        stacked_pred = np.clip(oof_results[target]['meta_model'].predict_proba(oof_stack_full)[:, 1], 0.0001, 0.9999)
        
        # Choose best strategy
        if best_per_target[target] == 'stack':
            predictions[target] = stacked_pred
            log.info(f"  → Using stacked ensemble")
        else:
            # Pick best single model
            best_model = min(['lgb', 'cat', 'xgb'], key=lambda m: preds[m].mean())
            predictions[target] = preds[best_model]
            log.info(f"  → Using best single: {best_model}")
    
    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v59_multiseed_{ts}.csv"
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
        'version': 'V59_multiseed',
        'name': 'Multi-Seed Stacking Ensemble (5 seeds × 3 models)',
        'timestamp': datetime.now().isoformat(),
        'submission_file': str(sub_path),
        'n_seeds_per_model': N_SEEDS,
        'n_splits': 3,
        'n_features': {t: V53_CONFIGS[t]['n_feat'] for t in TARGETS},
        's4_special': True,
        'v53_swept_cv': v53_swept,
        'v58_avg': v58_avg,
        'cv_results': cv_scores,
        'best_per_target': best_per_target,
    }
    meta_path = SUBMIT / f'meta_v59_multiseed_{ts}.json'
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
    oof_path = EXPERIMENTS / f'oof_v59.csv'
    oof_df.to_parquet(oof_path)
    log.info(f"  OOF saved: {oof_path}")
    
    # Save test preds
    test_preds_df = pd.DataFrame(index=test.index)
    for target in TARGETS:
        for m in ['lgb', 'cat', 'xgb']:
            test_preds_df[f'{target}_{m}'] = preds_per_model.get(target, {}).get(m, np.zeros(250))
    test_preds_path = EXPERIMENTS / f'test_preds_v59.csv'
    test_preds_df.to_parquet(test_preds_path)
    log.info(f"  Test preds saved: {test_preds_path}")


if __name__ == "__main__":
    main()
