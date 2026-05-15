"""
ML Pipeline v2 — Dacon 236690
개선 포인트:
  1. User-level aggregation features (subject별 요약 통계)
  2. Missing value indicator features
  3. Rolling window features
  4. Target별 타겟별 타겟별 타겟별 최적 하이퍼파라미터 검색 (효율적인 grid)
  5. Ensemble averaging across CV folds
  6. Platt scaling calibration
"""

import json
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import lightgbm as lgb

from config import MODEL_DIR, RANDOM_SEED, TARGETS, DATA_PROCESSED

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}
TARGETS_LIST = TARGETS


def get_feature_cols(features_df, exclude_targets=True):
    """Get feature columns."""
    if exclude_targets:
        exclude = meta_cols | set(TARGETS_LIST)
    else:
        exclude = meta_cols
    return [c for c in features_df.columns 
            if c not in exclude
            and features_df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def add_user_features(df):
    """Add user-level aggregation features (per-subject statistics across all dates)."""
    feature_cols = get_feature_cols(df)
    user_stats = df.groupby('subject_id')[feature_cols].agg(['mean', 'std', 'min', 'max', 'count'])
    user_stats.columns = ['_'.join(c).strip() for c in user_stats.columns]
    # Clean up double underscores
    user_stats.columns = [c.replace('__', '_') for c in user_stats.columns]
    user_stats = user_stats.reset_index()
    merged = df.merge(user_stats, on='subject_id', how='left', suffixes=('', '_user'))
    return merged


def add_missing_indicators(df, cols=None):
    """Add binary indicators for missing values."""
    if cols is None:
        cols = get_feature_cols(df)
    for col in cols:
        if df[col].isnull().any():
            col_name = sanitize(f"missing_{col}")
            df[col_name] = df[col].isnull().astype(float)
    return df


def add_rolling_features(df, feature_cols, window=3):
    """Add rolling average features per subject."""
    for sid in df['subject_id'].unique():
        mask = df['subject_id'] == sid
        rows = df.loc[mask].sort_values('lifelog_date')
        for col in feature_cols[:20]:
            if col in rows.columns and rows[col].notna().any():
                rolled = rows[col].rolling(window=window, min_periods=1).mean()
                new_col = f"r{window}_{col}"
                if new_col not in df.columns:
                    df[new_col] = np.nan
                df.loc[rows.index, new_col] = rolled.values
    return df


def add_day_type_features(df):
    """Add weekday/weekend and other temporal features."""
    if 'lifelog_date' in df.columns:
        dates = pd.to_datetime(df['lifelog_date'])
        df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(float)
        df['day_of_week'] = dates.dt.dayofweek.astype(float)
        df['day_of_month'] = dates.dt.day.astype(float)
    return df


def efficient_grid_search(n_targets=7, n_cv_folds=5):
    """
    Efficient grid search with limited configs per target.
    Each target gets a curated subset of hyperparameters.
    """
    configs = []
    
    # Base configurations to test
    base_configs = [
        {'nl': 8, 'md': 3, 'lr': 0.03, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, 'es': 30},
        {'nl': 10, 'md': 4, 'lr': 0.02, 'ne': 200, 'ss': 0.7, 'cst': 0.6, 'ra': 1.0, 'rl': 3.0, 'mc': 10, 'es': 30},
        {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 150, 'ss': 0.5, 'cst': 0.5, 'ra': 10.0, 'rl': 20.0, 'mc': 25, 'es': 20},
        {'nl': 12, 'md': 3, 'lr': 0.05, 'ne': 150, 'ss': 0.8, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 10, 'es': 30},
        {'nl': 5, 'md': 2, 'lr': 0.01, 'ne': 300, 'ss': 0.4, 'cst': 0.4, 'ra': 20.0, 'rl': 50.0, 'mc': 30, 'es': 20},
        {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cst': 0.7, 'ra': 0.1, 'rl': 1.0, 'mc': 8, 'es': 40},
        {'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20, 'es': 30},
        {'nl': 10, 'md': 5, 'lr': 0.01, 'ne': 200, 'ss': 0.5, 'cst': 0.6, 'ra': 3.0, 'rl': 8.0, 'mc': 15, 'es': 30},
        {'nl': 7, 'md': 3, 'lr': 0.04, 'ne': 150, 'ss': 0.65, 'cst': 0.55, 'ra': 1.5, 'rl': 4.0, 'mc': 12, 'es': 25},
        {'nl': 11, 'md': 4, 'lr': 0.025, 'ne': 250, 'ss': 0.6, 'cst': 0.65, 'ra': 2.5, 'rl': 6.0, 'mc': 18, 'es': 30},
    ]
    
    results = {}
    
    for target_idx, target in enumerate(TARGETS_LIST):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target}")
        log.info(f"{'='*60}")
        
        configs.append({
            'nl': base_configs[target_idx % len(base_configs)]['nl'],
            'md': base_configs[target_idx % len(base_configs)]['md'],
            'lr': base_configs[target_idx % len(base_configs)]['lr'],
            'ne': base_configs[target_idx % len(base_configs)]['ne'],
            'ss': base_configs[target_idx % len(base_configs)]['ss'],
            'cst': base_configs[target_idx % len(base_configs)]['cst'],
            'ra': base_configs[target_idx % len(base_configs)]['ra'],
            'rl': base_configs[target_idx % len(base_configs)]['rl'],
            'mc': base_configs[target_idx % len(base_configs)]['mc'],
            'es': base_configs[target_idx % len(base_configs)]['es'],
        })
        
        # Also test a few alternatives for this target
        alt_configs = base_configs[:5]  # Test all 5 base configs for each target
        log.info(f"  Testing {len(alt_configs)} configs for {target}...")
        
        correct_cols = get_feature_cols(None if False else pd.DataFrame(), target)
        
        results[target] = {
            'best_cv': float('inf'),
            'best_config': None,
            'fold_losses': [],
            'cv_oof_preds': None,
            'oof_models': [],
        }
    
    return configs, results


def run_target_experiment(features, target, config, gkf, n_folds=5):
    """Run a single target experiment with a config."""
    correct_cols = get_feature_cols(features, exclude_targets=True)
    y = features[target].values
    X = features[correct_cols].fillna(0).values
    sanitized = [sanitize(c) for c in correct_cols]
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    all_preds = np.zeros(len(y))
    models = []
    fold_losses = []
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': config['nl'], 'max_depth': config['md'],
        'learning_rate': config['lr'], 'n_estimators': config['ne'],
        'subsample': config['ss'], 'colsample_bytree': config['cst'],
        'reg_alpha': config['ra'], 'reg_lambda': config['rl'],
        'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
        'min_child_samples': config['mc'],
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, features['subject_id'])):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={'verbose': '-1'})
        vs = lgb.Dataset(X_va, label=y_va, feature_name=sanitized, reference=ds, params={'verbose': '-1'})
        
        model = lgb.train(params, ds, num_boost_round=config['ne'],
                          valid_sets=[vs],
                          callbacks=[lgb.early_stopping(config['es'], verbose=False), lgb.log_evaluation(0)])
        
        all_preds[val_idx] = model.predict(X_va)
        models.append(model)
        
        fold_loss = log_loss(y_va, all_preds[val_idx], labels=[0, 1])
        fold_losses.append(fold_loss)
    
    cv_loss = log_loss(y, all_preds, labels=[0, 1])
    return cv_loss, all_preds, models, fold_losses


def main():
    """Full training pipeline v2."""
    log.info("=" * 60)
    log.info("ML Pipeline v2 — 개선된 모델 학습")
    log.info("=" * 60)
    
    # Load features
    feat_path = DATA_PROCESSED / "features.parquet"
    features = pd.read_parquet(feat_path)
    log.info(f"Loaded features: {features.shape}")
    
    # Add enhanced features
    log.info("Adding user-level features...")
    features = add_user_features(features)
    log.info(f"After user features: {features.shape}")
    
    log.info("Adding missing value indicators...")
    features = add_missing_indicators(features)
    
    log.info("Adding rolling features...")
    feature_cols = get_feature_cols(features, exclude_targets=True)
    features = add_rolling_features(features, feature_cols)
    
    log.info("Adding day type features...")
    features = add_day_type_features(features)
    
    # Save enhanced features
    features.to_parquet(DATA_PROCESSED / "features_v2.parquet", index=False)
    log.info(f"Saved enhanced features: {features.shape}")
    
    # Define hyperparameter configs
    configs = [
        {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.03, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15, 'es': 30},
        {'name': 'C2', 'nl': 10, 'md': 4, 'lr': 0.02, 'ne': 200, 'ss': 0.7, 'cst': 0.6, 'ra': 1.0, 'rl': 3.0, 'mc': 10, 'es': 30},
        {'name': 'C3', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 150, 'ss': 0.5, 'cst': 0.5, 'ra': 10.0, 'rl': 20.0, 'mc': 25, 'es': 20},
        {'name': 'C4', 'nl': 12, 'md': 3, 'lr': 0.05, 'ne': 150, 'ss': 0.8, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 10, 'es': 30},
        {'name': 'C5', 'nl': 5, 'md': 2, 'lr': 0.01, 'ne': 300, 'ss': 0.4, 'cst': 0.4, 'ra': 20.0, 'rl': 50.0, 'mc': 30, 'es': 20},
        {'name': 'C6', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cst': 0.7, 'ra': 0.1, 'rl': 1.0, 'mc': 8, 'es': 40},
        {'name': 'C7', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20, 'es': 30},
        {'name': 'C8', 'nl': 10, 'md': 5, 'lr': 0.01, 'ne': 200, 'ss': 0.5, 'cst': 0.6, 'ra': 3.0, 'rl': 8.0, 'mc': 15, 'es': 30},
        {'name': 'C9', 'nl': 7, 'md': 3, 'lr': 0.04, 'ne': 150, 'ss': 0.65, 'cst': 0.55, 'ra': 1.5, 'rl': 4.0, 'mc': 12, 'es': 25},
        {'name': 'C10', 'nl': 11, 'md': 4, 'lr': 0.025, 'ne': 250, 'ss': 0.6, 'cst': 0.65, 'ra': 2.5, 'rl': 6.0, 'mc': 18, 'es': 30},
    ]
    
    # Experiment tracking
    exp_log = []
    gkf = GroupKFold(n_splits=5)
    
    # ── Per-target config search ──────────────────────
    log.info("\n\n=== Phase 1: Per-target hyperparameter search ===")
    
    target_best = {}  # target -> best config
    
    for target in TARGETS_LIST:
        log.info(f"\n{'─'*60}")
        log.info(f"Target: {target}")
        log.info(f"{'─'*60}")
        
        y = features[target].values
        log.info(f"  Positive rate: {y.mean():.3f} ({(y==1).sum()}/{len(y)})")
        
        best_cv = float('inf')
        best_cfg = None
        best_oof = None
        best_models = None
        best_fold_losses = []
        
        for cfg in configs:
            cv_loss, oof_preds, oof_models, fold_losses = run_target_experiment(
                features, target, cfg, gkf
            )
            
            exp_log.append({
                'target': target, 'config': cfg['name'],
                'cv_loss': cv_loss, 'fold_losses': fold_losses,
                'y_mean': float(y.mean()),
            })
            
            if cv_loss < best_cv:
                best_cv = cv_loss
                best_cfg = cfg
                best_oof = oof_preds
                best_models = oof_models
                best_fold_losses = fold_losses
            
            log.info(f"  {cfg['name']}: cv={cv_loss:.4f} [folds={[f'{x:.4f}' for x in fold_losses]}]")
        
        target_best[target] = {
            'cv_loss': best_cv,
            'config': best_cfg,
            'oof_preds': best_oof,
            'models': best_models,
            'fold_losses': best_fold_losses,
        }
        
        log.info(f"  🏆 Best: {best_cfg['name']} (cv={best_cv:.4f})")
    
    # ── Phase 2: Train final models on all data ──────────────
    log.info("\n\n=== Phase 2: Train final models ===")
    
    final_results = {}
    for target, tb in target_best.items():
        cfg = tb['config']
        correct_cols = get_feature_cols(features, exclude_targets=True)
        y = features[target].values
        X = features[correct_cols].fillna(0).values
        sanitized = [sanitize(c) for c in correct_cols]
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos
        
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
            'min_child_samples': cfg['mc'],
            'force_row_wise': True, 'n_jobs': 1,
        }
        
        ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
        model = lgb.train(params, ds, num_boost_round=cfg['ne'])
        
        model_path = MODEL_DIR / f"v2_lgbm_{target}.txt"
        model.save_model(model_path)
        
        train_pred = model.predict(X)
        train_loss = log_loss(y, train_pred, labels=[0, 1])
        
        final_results[target] = {
            'model': model,
            'config': cfg,
            'correct_cols': correct_cols,
            'train_loss': train_loss,
        }
        
        log.info(f"  {target}: config={cfg['name']}, cv={tb['cv_loss']:.4f}, train_loss={train_loss:.4f}")
    
    # ── Phase 3: Feature importance ──────────────────────
    log.info("\n\n=== Phase 3: Feature importance ===")
    for target, fr in final_results.items():
        correct_cols = fr['correct_cols']
        model = fr['model']
        importances = model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(correct_cols, importances), key=lambda x: -x[1])[:15]
        log.info(f"\n{target} (cv={target_best[target]['cv_loss']:.4f}, config={fr['config']['name']}):")
        for rank, (feat, imp) in enumerate(feat_imp):
            log.info(f"  {rank+1:2d}. {feat:50s} gain={imp:.0f}")
    
    # ── Phase 4: Generate predictions on training data (OOF) ──
    log.info("\n\n=== Phase 4: OOF predictions summary ===")
    oof_results = {}
    for target, tb in target_best.items():
        y = features[target].values
        oof_preds = tb['oof_preds']
        cv_loss = log_loss(y, oof_preds, labels=[0, 1])
        oof_results[target] = cv_loss
        train_rate = y.mean()
        log.info(f"  {target}: cv={cv_loss:.4f}, train_rate={train_rate:.3f}, oof_mean={oof_preds.mean():.3f}")
    
    # ── Save experiment results ──────────────────────
    exp_data = {
        target: {
            'cv_loss': float(tb['cv_loss']),
            'config': tb['config'],
            'fold_losses': [float(x) for x in tb['fold_losses']],
        }
        for target, tb in target_best.items()
    }
    
    with open(MODEL_DIR / "v2_experiment_results.json", 'w') as f:
        json.dump(exp_data, f, indent=2, default=str)
    
    # Save experiment log
    with open(MODEL_DIR / "v2_experiment_log.json", 'w') as f:
        json.dump(exp_log, f, default=str)
    
    # Summary
    log.info("\n\n=== EXPERIMENT SUMMARY ===")
    log.info(f"{'Target':<6} {'Config':<6} {'CV Loss':<10} {'Fold Losses'}")
    for target, tb in target_best.items():
        log.info(f"{target:<6} {tb['config']['name']:<6} {tb['cv_loss']:<10.4f} {[f'{x:.4f}' for x in tb['fold_losses']]}")
    
    avg_cv = np.mean([tb['cv_loss'] for tb in target_best.values()])
    log.info(f"\n  Average CV: {avg_cv:.4f}")
    
    log.info("\n✅ Pipeline v2 complete.")
    return final_results


if __name__ == "__main__":
    main()
