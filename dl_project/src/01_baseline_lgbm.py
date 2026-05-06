# =============================
# LGBM Baseline — Reimplement V10
# =============================
# Goal: Reproduce cal OOF ≈ 0.6038
# This serves as the baseline to beat with FT-Transformer

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


def train_lgbm_baseline(prepared_data, target_col, config, seed=42):
    """
    Train LightGBM with V10-like configuration.
    Per-subject personalization (z-score), mean matching calibration.
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
    
    oof_preds = np.zeros(len(prepared_data["X"]))
    models_trained = []
    fold_losses = []
    
    print(f"\n{'='*60}")
    print(f"LGBM Baseline | Target: {target_col} | Seed: {seed}")
    print(f"{'='*60}")
    
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
        
        # Compute per-subject stats on training data
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
            
            # Z-score for validation
            X_val_personalized = (X_val - train_means) / (train_stds + 1e-8)
        else:
            X_val_personalized = X_val
        
        # LGBM params (V10-like)
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
    
    # Mean matching calibration (V10 style)
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
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = vars(args)
    
    print("[LGBM] Loading data...")
    df = load_data()
    meta_info, df = extract_meta(df)
    
    if args.target not in meta_info["target_cols"]:
        if meta_info["target_cols"]:
            config["target"] = meta_info["target_cols"][0]
            print(f"[LGBM] Using first target: {config['target']}")
    
    print("[LGBM] Preparing features...")
    prepared = prepare_for_dl(df, meta_info)
    
    print("[LGBM] Training V10-like baseline...")
    result = train_lgbm_baseline(prepared, config["target"], config)
    
    # Save
    save_dir = BASE_DIR / "results" / "lgbm_baseline" / config["target"]
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    np.save(save_dir / "oof_preds.npy", result["oof_preds"])
    
    print(f"\n[LGBM] Saved to {save_dir}")
    print(f"[LGBM] Cal OOF loss: {result['cal_loss']:.6f}")
