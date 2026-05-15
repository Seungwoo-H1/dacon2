"""
32_v30_xgb_gpu_ensemble.py — V30: XGBoost GPU Ensemble

기대 효과: -0.01 ~ -0.03 log-loss vs V25 (LGBM only)
- XGBoost GPU (`hist` + `gpu_predictor`) 병렬 ensemble
- V25의 features.parquet를 기반으로 실행
"""

import sys
import logging
import warnings
from pathlib import Path
import time
import glob

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import DATA_PROCESSED, TARGETS, SUBMIT_DIR, PROJECT_ROOT


def get_target_cols():
    """Get actual target column names from train data."""
    train_csv = DATA_PROCESSED.parent / "ch2026_metrics_train.csv"
    if train_csv.exists():
        train = pd.read_csv(train_csv)
    else:
        # fallback: check raw data
        raw_train = PROJECT_ROOT / "data_raw" / "ch2026_metrics_train.csv"
        train = pd.read_csv(raw_train)
    
    targets = []
    for t in TARGETS:
        # Check both actual_X and X format
        col = f'actual_{t}'
        if col in train.columns:
            targets.append(col)
        elif t in train.columns:
            targets.append(t)
        else:
            log.warning(f"Target {t} not found in train data")
    
    return targets


def train_xgb_for_target(X_train, y_train, X_val, y_val, val_ids, target_col, seed):
    """
    Train single XGBoost GPU model.
    Returns predictions aligned with val_ids.
    """
    try:
        import xgboost as xgb
    except ImportError:
        log.error("XGBoost not installed. Run: pip3 install xgboost --break-system-packages")
        return None

    try:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'predictor': 'gpu_predictor',
            'device': 'cuda:0',
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': seed,
            'verbosity': 0,
        }

        model = xgb.XGBClassifier(**params)

        X_train_c = X_train.astype(np.float32)
        X_val_c = X_val.astype(np.float32)

        model.fit(
            X_train_c, y_train.values,
            eval_set=[(X_val_c, y_val.values)],
            verbose=False,
        )

        preds = model.predict_proba(X_val_c)[:, 1]

        result = pd.DataFrame({
            'id': val_ids,
            'pred': preds,
        })
        return result

    except Exception as e:
        log.warning(f"XGB GPU training failed for {target_col} (seed={seed}): {e}")
        return None


def main():
    log.info("=" * 70)
    log.info("V30 XGBoost GPU Ensemble")
    log.info("=" * 70)

    start_time = time.time()

    # --- 1. Load features ---
    log.info("\n[1/5] Loading features...")
    
    # Try V25 features first, then fall back to extended
    feat_files = glob.glob(str(DATA_PROCESSED / 'features_v25*.parquet'))
    if feat_files:
        feat_path = feat_files[0]
        log.info(f"  V25 features: {feat_path}")
    else:
        feat_path = DATA_PROCESSED / 'features_extended.parquet'
        log.info(f"  Using extended features: {feat_path}")
    
    feat = pd.read_parquet(feat_path)
    log.info(f"  Train shape: {feat.shape}")
    log.info(f"  Columns: {list(feat.columns)[:30]}...")

    # --- 2. Add id column (subject_id + lifelog_date) ---
    log.info("\n[2/5] Adding id column...")
    if 'id' not in feat.columns:
        if 'lifelog_date' in feat.columns and 'subject_id' in feat.columns:
            feat['id'] = feat['subject_id'] + '_' + feat['lifelog_date'].astype(str)
        else:
            # fallback: just use row index
            feat['id'] = feat.index.astype(str)
    log.info(f"  ID column created: {feat['id'].iloc[:3].tolist()}...")

    # --- 3. Find target columns ---
    target_cols = get_target_cols()
    log.info(f"\n[3/5] Targets: {target_cols}")

    # --- 4. Determine fold structure ---
    log.info("\n[4/5] Detecting fold structure...")
    
    # Check if features have fold columns
    fold_cols = [c for c in feat.columns if c.startswith('fold_')]
    if fold_cols:
        n_folds = max(int(c.split('_')[-1]) for c in fold_cols) + 1
        log.info(f"  Found {n_folds} folds: {fold_cols}")
    else:
        # Create 5-fold CV split
        log.info("  No fold columns found. Creating 5-fold CV split.")
        from sklearn.model_selection import GroupKFold
        for i in range(5):
            feat[f'fold_{i}'] = -1
        fold_cols = [f'fold_{i}' for i in range(5)]
        n_folds = 5

    # --- 5. Train XGBoost GPU for each target ---
    log.info("\n[5/5] Training XGBoost GPU models...")
    
    # Feature columns (exclude meta, fold, target)
    exclude_cols = ['id', 'subject_id'] + target_cols + fold_cols
    # Also exclude subject-level meta columns
    for col in list(feat.columns):
        if col.startswith('subject_') or col.startswith('meta_'):
            if col not in exclude_cols:
                exclude_cols.append(col)

    feature_cols = [c for c in feat.columns if c not in exclude_cols]
    
    # Remove datetime and non-numeric columns (XGBoost requires float)
    drop_cols = []
    for c in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(feat[c]):
            drop_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(feat[c]):
            drop_cols.append(c)
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    
    # Fill NaN with 0
    X = feat[feature_cols].fillna(0)
    
    log.info(f"  Feature count: {len(feature_cols)}")
    log.info(f"  Dropped non-numeric/datetime: {drop_cols[:10]}...")

    all_results = {}
    
    for target_col in target_cols:
        log.info(f"\n  → Training for {target_col}...")
        
        y = feat[target_col]
        
        # Drop rows with NaN in target
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        ids_valid = feat.loc[valid_mask, 'id'].values
        
        # Create 5-fold CV
        group_kf = GroupKFold(n_splits=5)
        val_preds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(group_kf.split(X_valid, y_valid, ids_valid)):
            X_train, X_val = X_valid.iloc[train_idx], X_valid.iloc[val_idx]
            y_train, y_val = y_valid.iloc[train_idx], y_valid.iloc[val_idx]
            val_ids = ids_valid[val_idx]
            
            # Train multiple seeds and average
            seed_preds = []
            for seed in [42, 123, 456, 789, 1024]:
                preds = train_xgb_for_target(X_train, y_train, X_val, y_val, val_ids, target_col, seed)
                if preds is not None:
                    seed_preds.append(preds['pred'].values)
            
            if seed_preds:
                avg_pred = np.mean(seed_preds, axis=0)
                fold_result = pd.DataFrame({
                    'id': val_ids,
                    'pred': avg_pred,
                })
                val_preds.append(fold_result)
                log.info(f"    Fold {fold_idx}: {len(val_ids)} samples, mean_pred={avg_pred.mean():.4f}")
        
        if val_preds:
            val_df = pd.concat(val_preds, ignore_index=True)
            all_results[target_col] = val_df
            
            # Save OOF
            oof_path = DATA_PROCESSED / f'xgb_oof_{target_col.replace("actual_", "")}.csv'
            val_df.to_csv(oof_path, index=False)
            log.info(f"  ✓ Saved OOF: {oof_path}")
    
    # --- 5. Evaluate & Compare ---
    log.info("\n[5/5] Evaluation...")
    
    for target_col in target_cols:
        if target_col not in all_results:
            continue
        
        oof = all_results[target_col]
        actual = feat.merge(oof, on='id', how='inner')[target_col].values
        
        # Quick log_loss
        from sklearn.metrics import log_loss
        
        # Align lengths
        min_len = min(len(actual), len(oof['pred']))
        actual = actual[:min_len]
        preds = oof['pred'].values[:min_len]
        
        try:
            ll = log_loss(actual, preds)
            log.info(f"  {target_col}: log_loss={ll:.4f}")
        except Exception as e:
            log.warning(f"  Evaluation failed for {target_col}: {e}")

    # --- 6. Generate Submission ---
    log.info("\n--- Generating Submission ---")
    
    # For test set, train on all data and predict
    test_feat_path = DATA_PROCESSED / 'test_features.parquet'
    if test_feat_path.exists():
        test = pd.read_parquet(test_feat_path)
        # Add id column if missing
        if 'id' not in test.columns:
            if 'subject_id' in test.columns:
                test['id'] = test['subject_id']
            else:
                test['id'] = test.index.astype(str)
        test_ids = test['id'].values
        
        for target_col in target_cols:
            if target_col not in all_results:
                continue
            
            log.info(f"  → Predicting test for {target_col}...")
            test_X = test[feature_cols].fillna(0)
            
            # Train on full training data
            X_all = feat[feature_cols]
            y_all = feat[target_col].dropna()
            
            X_all_valid = X_all[y_all.index]
            
            seed_preds = []
            for seed in [42, 123, 456, 789, 1024]:
                try:
                    import xgboost as xgb
                    params = {
                        'objective': 'binary:logistic',
                        'tree_method': 'hist',
                        'predictor': 'gpu_predictor',
                        'device': 'cuda:0',
                        'max_depth': 6,
                        'learning_rate': 0.03,
                        'n_estimators': 500,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.5,
                        'reg_lambda': 2.0,
                        'min_child_weight': 3,
                        'gamma': 0.1,
                        'random_state': seed,
                        'verbosity': 0,
                    }
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_all_valid.astype(np.float32), y_all.values)
                    test_pred = model.predict_proba(test_X.astype(np.float32))[:, 1]
                    seed_preds.append(test_pred)
                    log.info(f"    Seed {seed}: mean={test_pred.mean():.4f}")
                except Exception as e:
                    log.warning(f"    Seed {seed} failed: {e}")
            
            if seed_preds:
                avg_pred = np.mean(seed_preds, axis=0)
                all_results[target_col] = pd.DataFrame({
                    'id': test_ids,
                    'pred': avg_pred,
                })
        
        # Save submission
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        submission = pd.DataFrame({'id': test_ids})
        for target_col in target_cols:
            if target_col in all_results and target_col.startswith('actual_'):
                short_name = target_col.replace('actual_', '')
                submission[f'actual_{short_name}'] = all_results[target_col]['pred'].values
        
        sub_path = SUBMIT_DIR / f'submission_v30_xgb_{timestamp}.csv'
        submission.to_csv(sub_path, index=False)
        log.info(f"\n✅ Saved submission: {sub_path}")
        log.info(f"   Shape: {submission.shape}")
    
    elapsed = time.time() - start_time
    log.info(f"\n⏱  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
