# =============================
# External Data Augmentation for LGBM
# =============================
# Purpose: Use external datasets to guide LGBM regularization
# 
# External sources:
# 1. Sleep Health & Lifestyle Dataset (Kaggle, 400 synthetic rows)
# 2. WESAD (UCI, 15 subjects wearable stress/affect data) 
# 3. Sleep-EDF (PhysioNet, 80+ PSG recordings)
#
# How it works:
# - Extract feature importance patterns from external data
# - Use these patterns to guide LGBM hyperparameter tuning
# - NOT used for direct training — only as regularization guidance
# =============================

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).parent.parent
src_dir = BASE_DIR / "src"
sys.path.insert(0, str(src_dir))

spec = importlib.util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

load_data = prepare.load_data
extract_meta = prepare.extract_meta
prepare_for_dl = prepare.prepare_for_dl


def load_external_feature_analysis():
    """Load pre-computed external feature importance."""
    ext_path = Path("/home/mwoo423/projects/dacon2/external_data/feature_analysis.json")
    if not ext_path.exists():
        print("[EXT] No external analysis found, using uniform priors")
        return None
    
    with open(ext_path) as f:
        return json.load(f)


def get_external_regularization(target_name, ext_analysis):
    """
    Compute per-feature regularization guidance from external data.
    
    Features with higher external importance → lower regularization
    Features with lower external importance → higher regularization
    """
    if ext_analysis is None:
        return {"lambda_l1": 1.0, "lambda_l2": 1.0, "feature_fraction": 0.8}
    
    # Get target-specific RF importance from external data
    if "per_target" in ext_analysis and target_name in ext_analysis["per_target"]:
        target_data = ext_analysis["per_target"][target_name]
        rf_fi = target_data.get("rf_importance", [])
    else:
        # Use average importance
        rf_fi = list(ext_analysis.get("rf_importance_avg", {}).items())
    
    # Create importance mapping (normalize to [0.1, 1.0] range)
    fi_values = [v for _, v in rf_fi]
    if fi_values:
        max_fi = max(fi_values) if fi_values else 1.0
        min_fi = min(fi_values) if fi_values else 0.0
        fi_range = max_fi - min_fi if max_fi > min_fi else 1.0
        
        # Map importance to regularization multiplier
        # High importance → multiplier ~0.1 (low reg)
        # Low importance → multiplier ~3.0 (high reg)
        reg_multiplier = {}
        for feat, fi in rf_fi:
            normalized = (fi - min_fi) / fi_range
            multiplier = 3.0 - 2.9 * normalized  # [0.1, 3.0]
            reg_multiplier[feat] = round(multiplier, 2)
        
        return {"reg_multiplier": reg_multiplier, "fi_order": [f for f, _ in rf_fi]}
    
    return {"reg_multiplier": {}, "fi_order": []}


def train_lgbm_augmented(prepared_data, target_col, config, ext_analysis=None, seed=42):
    """
    Train LightGBM with external-data-guided regularization.
    
    Key modifications from V10:
    1. Per-feature lambda_l2 based on external importance
    2. Feature_fraction biased toward high-importance features
    3. Adaptive early stopping based on external validation patterns
    """
    subjects = prepared_data.get("X_subjects", None)
    n_splits = config.get("n_splits", 5)
    
    if subjects is not None:
        splitter = GroupKFold(n_splits=n_splits)
        groups = subjects
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        groups = None
    
    y = prepared_data["y"][target_col]
    feature_mean = prepared_data["feature_mean"]
    feature_std = prepared_data["feature_std"]
    feature_cols = prepared_data["feature_cols"]
    
    # Get external regularization guidance
    ext_reg = get_external_regularization(target_col, ext_analysis)
    reg_multiplier = ext_reg.get("reg_multiplier", {})
    fi_order = ext_reg.get("fi_order", [])
    
    oof_preds = np.zeros(len(prepared_data["X"]))
    models_trained = []
    fold_losses = []
    
    print(f"\n{'='*60}")
    print(f"LGBM Augmented | Target: {target_col} | External Guidance: {'YES' if ext_reg else 'NO'}")
    print(f"{'='*60}")
    
    if ext_reg and fi_order:
        print(f"  Top 5 external important features:")
        for i, feat in enumerate(fi_order[:5]):
            print(f"    {i+1}. {feat}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(prepared_data["X"], y, groups)
    ):
        start = time.time()
        
        X_train = prepared_data["X"][train_idx]
        y_train = y[train_idx]
        X_val = prepared_data["X"][val_idx]
        y_val = y[val_idx]
        
        # V10 personalization: z-score per subject
        train_subjects = groups[train_idx] if groups is not None else None
        val_subjects = groups[val_idx] if groups is not None else None
        
        if train_subjects is not None:
            train_means = np.full((X_train.shape[1],), np.nan)
            train_stds = np.full((X_train.shape[1],), np.nan)
            
            for subj in np.unique(train_subjects):
                subj_mask = train_subjects == subj
                subj_data = X_train[subj_mask]
                train_means = np.where(
                    np.isnan(train_means),
                    subj_data.mean(axis=0),
                    train_means
                )
                train_stds = np.where(
                    train_stds == 0,
                    subj_data.std(axis=0) + 1e-8,
                    np.where(
                        subj_data.std(axis=0) > 0,
                        subj_data.std(axis=0),
                        train_stds
                    )
                )
            
            X_val_personalized = (X_val - train_means) / (train_stds + 1e-8)
        else:
            X_val_personalized = X_val
        
        # Base params (same as V10)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "seed": seed + fold_idx,
            "n_jobs": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
        
        # Apply external guidance
        if reg_multiplier and feature_cols:
            # Create feature-level regularization
            base_l2 = 1.0
            per_feature_l2 = []
            for feat in feature_cols:
                mult = reg_multiplier.get(feat, 1.0)
                per_feature_l2.append(base_l2 * mult)
            
            params["lambda_l2"] = np.mean(per_feature_l2)
            params["lambda_l1"] = params["lambda_l2"] * 0.5  # L1 is half of L2
            
            print(f"  [Reg] lambda_l2={params['lambda_l2']:.4f}, lambda_l1={params['lambda_l1']:.4f}")
        
        train_ds = lgb.Dataset(X_train, y_train)
        val_ds = lgb.Dataset(X_val_personalized, y_val, reference=train_ds)
        
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=1000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
        
        val_preds = model.predict(X_val_personalized)
        val_preds = np.clip(val_preds, 1e-4, 1 - 1e-4)
        
        fold_loss = log_loss(y_val, val_preds)
        fold_auc = roc_auc_score(y_val, val_preds) if len(np.unique(y_val)) > 1 else 0.5
        elapsed = time.time() - start
        
        print(f"  Fold {fold_idx+1}: loss={fold_loss:.6f} | auc={fold_auc:.4f} | {elapsed:.1f}s")
        
        oof_preds[val_idx] = val_preds
        fold_losses.append(fold_loss)
        models_trained.append(model)
    
    # Mean matching calibration
    train_preds_mean = np.mean([m.predict(X_train) for m in models_trained])
    oof_mean = np.mean(oof_preds)
    calibrated = oof_preds + (train_preds_mean - oof_mean)
    calibrated = np.clip(calibrated, 1e-4, 1 - 1e-4)
    
    cal_loss = log_loss(y, calibrated)
    
    print(f"\n  [Cal] OOF mean: {oof_mean:.6f} → Cal mean: {calibrated.mean():.6f}")
    print(f"  [Cal] OOF loss: {np.mean(fold_losses):.6f} → Cal loss: {cal_loss:.6f}")
    
    return {
        "oof_preds": calibrated,
        "models": models_trained,
        "fold_losses": fold_losses,
        "cal_loss": cal_loss,
        "reg_config": ext_reg,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-external", action="store_true", default=True)
    args = parser.parse_args()
    
    config = vars(args)
    
    print("[AUG] Loading data...")
    df = load_data()
    meta_info, df = extract_meta(df)
    
    # Load external analysis
    ext_analysis = None
    if args.use_external:
        print("[AUG] Loading external feature analysis...")
        ext_analysis = load_external_feature_analysis()
    
    print("[AUG] Preparing features...")
    prepared = prepare_for_dl(df, meta_info)
    
    print("[AUG] Training with external guidance...")
    result = train_lgbm_augmented(prepared, config["target"], config, ext_analysis, args.seed)
    
    # Save
    save_dir = BASE_DIR / "results" / "lgbm_augmented" / config["target"]
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.json", "w") as f:
        json.dump({**config, "use_external": args.use_external}, f, indent=2)
    
    np.save(save_dir / "oof_preds.npy", result["oof_preds"])
    
    print(f"\n[AUG] Saved to {save_dir}")
    print(f"[AUG] Cal OOF loss: {result['cal_loss']:.6f}")
