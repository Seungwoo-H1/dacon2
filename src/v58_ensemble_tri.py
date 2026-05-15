"""
V58 — Three-Model Stacking Ensemble (LGBM + CatBoost + XGBoost)

Key idea: Combine the three tree-based models using OOF predictions as
meta-features for a simple logistic regression meta-learner. This captures
complementary patterns each model picks up.

Architecture:
  Level 0: LGBM + CatBoost + XGBoost (OOF)
  Level 1: LogisticRegression (calibrated) → final predictions

Features: Same as V53 (base 141 + zscore 141 = 282, target-specific selection)

Improvements over V53:
  1. Multi-model diversity (LGBM/CatBoost/XGBoost have different inductive biases)
  2. OOF stacking prevents meta-learner overfitting
  3. Calibrated meta-learner (Platt scaling)
  4. Per-target feature selection via importance ranking
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

# V53 swept configs (n_feat optimized)
V53_CONFIGS = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 5},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety', 'n_feat': 21},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}


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


# ─── Model trainers ──────────────────────────────────────────────────────────


def train_lgbm(X_train, y_train, feat_names, cfg, seed):
    """Train LightGBM classifier."""
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
    """Train CatBoost classifier."""
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'num_boost_round': cfg['ne'],
        'learning_rate': cfg['lr'], 'depth': cfg['md'] + 1,  # CatBoost depth = md+1 approx
        'subsample': cfg['ss'], 'colsample_bylevel': cfg['cb'],
        'l2_leaf_reg': cfg['rl'],  # CatBoost uses l2_leaf_reg instead of reg_lambda
        'random_seed': seed, 'thread_count': 1, 'verbose': 0,
        'scale_pos_weight': spw, 'max_ctr_complexity': 1,
        'boosting_type': 'Ordered',
    }
    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=None)
    return model


def train_xgboost(X_train, y_train, feat_names, cfg, seed):
    """Train XGBoost classifier."""
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
    """Rank features by LGBM gain importance, averaged over seeds."""
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
    
    # Average importance
    imp_avg = imp_sum / n_seeds
    # Avoid dividing by zero — use gain directly
    ranked = sorted(zip(feat_cols, imp_avg), key=lambda x: -x[1])
    del X
    gc.collect()
    return [r[0] for r in ranked]


# ─── Three-model OOF training ────────────────────────────────────────────────


def train_three_model_oof(train, targets_feat_cols, target, cfg, n_splits=3, seed_base=42):
    """
    Train LGBM + CatBoost + XGBoost using GroupKFold (n_splits=3).
    Returns OOF predictions for each model + stacked ensemble.
    """
    feat_cols = targets_feat_cols[target]
    y = train[target].values.astype(np.float64)
    groups = train['subject_id'].values
    X = train[feat_cols].fillna(0).values.astype(np.float64)
    
    gkf = GroupKFold(n_splits=n_splits)
    
    # OOF predictions for each model
    oof_lgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    
    fold_losses = {'lgb': [], 'cat': [], 'xgb': []}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        sn = [sanitize(c) for c in feat_cols]
        
        # LightGBM
        m_lgb = train_lgbm(X_tr, y_tr, sn, cfg, seed_base + fold)
        preds_lgb = np.clip(m_lgb.predict(X_val), 0.0001, 0.9999)
        oof_lgb[val_idx] = preds_lgb
        fold_losses['lgb'].append(logloss(y_val, preds_lgb))
        del m_lgb
        gc.collect()
        
        # CatBoost
        m_cat = train_catboost(X_tr, y_tr, sn, cfg, seed_base + fold + 100)
        preds_cat = np.clip(m_cat.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
        oof_cat[val_idx] = preds_cat
        fold_losses['cat'].append(logloss(y_val, preds_cat))
        del m_cat
        gc.collect()
        
        # XGBoost
        m_xgb = train_xgboost(X_tr, y_tr, sn, cfg, seed_base + fold + 200)
        preds_xgb = np.clip(m_xgb.predict_proba(X_val)[:, 1], 0.0001, 0.9999)
        oof_xgb[val_idx] = preds_xgb
        fold_losses['xgb'].append(logloss(y_val, preds_xgb))
        del m_xgb
        gc.collect()
    
    # Build stacking features from OOF
    oof_stack = np.column_stack([oof_lgb, oof_cat, oof_xgb])
    
    # Train meta-learner (logistic regression with calibration)
    meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=seed_base)
    meta.fit(oof_stack, y)
    
    # Evaluate stacked model on OOF
    oof_stacked = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
    stacked_loss = logloss(y, oof_stacked)
    
    avg_losses = {k: np.mean(v) for k, v in fold_losses.items()}
    
    return {
        'oof_lgb': oof_lgb,
        'oof_cat': oof_cat,
        'oof_xgb': oof_xgb,
        'oof_stack': oof_stacked,
        'meta_model': meta,
        'fold_losses': fold_losses,
        'avg_losses': avg_losses,
        'stacked_loss': stacked_loss,
    }


# ─── Three-model full training → test prediction ─────────────────────────────


def train_three_model_full(train_feat, test_feat, feat_cols, y_train, cfg, seed_base=42):
    """Train LGBM + CatBoost + XGBoost on ALL train data, predict test."""
    X_train = train_feat[feat_cols].fillna(0).values.astype(np.float64)
    X_test = test_feat[feat_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in feat_cols]
    
    # Train each model on full data with different seeds
    preds_per_seed = {}
    for i, (name, trainer) in enumerate([
        ('lgb', lambda X, y: train_lgbm(X, y, sn, cfg, seed_base)),
        ('cat', lambda X, y: train_catboost(X, y, sn, cfg, seed_base + 100)),
        ('xgb', lambda X, y: train_xgboost(X, y, sn, cfg, seed_base + 200)),
    ]):
        m = trainer(X_train, y_train)
        if name == 'cat':
            pred_test = m.predict_proba(X_test)[:, 1]
        else:
            pred_test = m.predict(X_test)
        preds_per_seed[name] = np.clip(pred_test, 0.0001, 0.9999)
        del m
        gc.collect()
    
    return preds_per_seed


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("V58 — Three-Model Stacking Ensemble (LGBM + CatBoost + XGBoost)")
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
    
    # Feature ranking (same as V53 swept)
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
    
    # ── OOF evaluation with GroupKFold (n_splits=3) ──
    log.info("\n" + "=" * 80)
    log.info("Phase 1: OOF evaluation (GroupKFold n_splits=3)")
    log.info("=" * 80)
    
    oof_results = {}
    cv_scores = {}
    
    for target in TARGETS:
        log.info(f"\n  --- {target} ---")
        config = V53_CONFIGS[target]
        cfg_name = config['cfg']
        base_cfg = CFGS[cfg_name]
        
        result = train_three_model_oof(train, targets_feat_cols, target, base_cfg, n_splits=3, seed_base=42)
        oof_results[target] = result
        
        avg_fold = np.mean(list(result['avg_losses'].values()))
        cv_scores[target] = {
            'lgb': result['avg_losses']['lgb'],
            'cat': result['avg_losses']['cat'],
            'xgb': result['avg_losses']['xgb'],
            'stack': result['stacked_loss'],
            'avg_models': avg_fold,
        }
        
        log.info(f"  Fold losses — LGBM: {result['avg_losses']['lgb']:.4f}, "
                f"CatBoost: {result['avg_losses']['cat']:.4f}, "
                f"XGBoost: {result['avg_losses']['xgb']:.4f}")
        log.info(f"  Stacked (OOF): {result['stacked_loss']:.4f}")
    
    # ── A/B comparison with V53 swept ──
    log.info("\n" + "=" * 80)
    log.info("Phase 2: Comparison with V53 swept (CV avg 0.6806)")
    log.info("=" * 80)
    
    v53_swept = {'Q1': 0.7591, 'Q2': 0.6929, 'Q3': 0.6893, 'S1': 0.6029, 'S2': 0.6621, 'S3': 0.7144, 'S4': 0.6438}
    
    for target in TARGETS:
        s = cv_scores[target]
        v53_loss = v53_swept[target]
        stack_improve = v53_loss - s['stack']
        log.info(f"  {target}: V53={v53_loss:.4f} | LGBM={s['lgb']:.4f} | Cat={s['cat']:.4f} | XGB={s['xgb']:.4f} | Stack={s['stack']:.4f} | Δ={stack_improve:+.4f}")
    
    # Determine best model per target
    log.info("\n  Per-target best model:")
    best_model_per_target = {}
    for target in TARGETS:
        s = cv_scores[target]
        # Compare stacked vs V53 swept
        if s['stack'] < v53_swept[target]:
            best = 'stack'
        elif s['lgb'] < s['cat'] and s['lgb'] < s['xgb']:
            best = 'lgb'
        elif s['cat'] < s['xgb']:
            best = 'cat'
        else:
            best = 'xgb'
        best_model_per_target[target] = best
        log.info(f"    {target}: {best} (OOF={s[best]:.4f})")
    
    # ── Phase 3: Generate test predictions ──
    log.info("\n" + "=" * 80)
    log.info("Phase 3: Test predictions")
    log.info("=" * 80)
    
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    predictions = {}
    test_preds_per_model = {target: {} for target in TARGETS}
    
    for target in TARGETS:
        log.info(f"\n  --- {target} ---")
        config = V53_CONFIGS[target]
        cfg_name = config['cfg']
        base_cfg = CFGS[cfg_name]
        feat_cols = targets_feat_cols[target]
        y_train = train[target].values.astype(np.float64)
        
        # Train all three models on full data
        preds = train_three_model_full(train, test, feat_cols, y_train, base_cfg, seed_base=42)
        
        test_preds_per_model[target] = preds
        log.info(f"  LGBM mean={preds['lgb'].mean():.4f}, CatBoost mean={preds['cat'].mean():.4f}, XGBoost mean={preds['xgb'].mean():.4f}")
        
        # Use stacked model if it improved, otherwise best single model
        meta_model = oof_results[target]['meta_model']
        oof_stack = np.column_stack([preds['lgb'], preds['cat'], preds['xgb']])
        stacked_pred = np.clip(meta_model.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
        
        # Fallback: if stacked not better, use best single model
        if cv_scores[target]['stack'] >= v53_swept[target]:
            best = best_model_per_target[target]
            predictions[target] = preds[best]
            log.info(f"  → Using {best} (V53 swept better or equal)")
        else:
            predictions[target] = stacked_pred
            log.info(f"  → Using stacked ensemble")
    
    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v58_ensemble_{ts}.csv"
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
        'version': 'V58_ensemble',
        'name': 'Three-Model Stacking Ensemble (LGBM + CatBoost + XGBoost)',
        'timestamp': datetime.now().isoformat(),
        'submission_file': str(sub_path),
        'n_splits': 3,
        'n_features_v53_swept': {t: V53_CONFIGS[t]['n_feat'] for t in TARGETS},
        'v53_swept_cv': v53_swept,
        'cv_results': cv_scores,
        'best_model_per_target': best_model_per_target,
        'features': {t: targets_feat_cols[t] for t in TARGETS},
    }
    meta_path = SUBMIT / f'meta_v58_ensemble_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")
    
    # Save OOF predictions for debugging
    oof_df = pd.DataFrame(index=train.index)
    for target in TARGETS:
        oof_df[f'{target}_lgb'] = oof_results[target]['oof_lgb']
        oof_df[f'{target}_cat'] = oof_results[target]['oof_cat']
        oof_df[f'{target}_xgb'] = oof_results[target]['oof_xgb']
        oof_df[f'{target}_stack'] = oof_results[target]['oof_stack']
    oof_path = EXPERIMENTS / f'oof_v58.csv'
    oof_df.to_parquet(oof_path)
    log.info(f"  OOF saved: {oof_path}")
    
    # Save test predictions per model for analysis
    test_preds_df = pd.DataFrame(index=test.index)
    for target in TARGETS:
        for model_name in ['lgb', 'cat', 'xgb']:
            test_preds_df[f'{target}_{model_name}'] = test_preds_per_model[target][model_name]
        # Best prediction
        if cv_scores[target]['stack'] < v53_swept[target]:
            test_preds_df[f'{target}_best'] = oof_results[target]['meta_model'].predict_proba(
                np.column_stack([test_preds_per_model[target][m] for m in ['lgb', 'cat', 'xgb']])
            )[:, 1]
        else:
            test_preds_df[f'{target}_best'] = test_preds_per_model[target][best_model_per_target[target]]
    test_preds_path = EXPERIMENTS / f'test_preds_v58.csv'
    test_preds_df.to_parquet(test_preds_path)
    log.info(f"  Test preds saved: {test_preds_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
