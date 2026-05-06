# =============================
# FT-Transformer Training Script
# Based on pytabkit implementation
# =============================

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import importlib.util

BASE_DIR = Path(__file__).parent.parent
src_dir = BASE_DIR / "src"
sys.path.insert(0, str(src_dir))

# Import prepare module directly
spec = importlib.util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

load_data = prepare.load_data
extract_meta = prepare.extract_meta
prepare_for_dl = prepare.prepare_for_dl


def prepare_dataset(prepared_data, target_col, split_name="all"):
    """Create PyTorch DataLoader for a specific target."""
    X = torch.FloatTensor(prepared_data["X"])
    y = torch.FloatTensor(prepared_data["y"][target_col]).unsqueeze(1)
    
    split = prepared_data.get("split", None)
    subjects = prepared_data.get("X_subjects", None)
    
    if split is not None and split_name != "all":
        mask = split == split_name
        X, y = X[mask], y[mask]
        if subjects is not None:
            subjects = subjects[mask]
    
    dataset = TensorDataset(X, y)
    return dataset, subjects


def train_ft_transformer(prepared_data, target_col, config):
    """
    Train FT-Transformer for a single target.
    
    Uses pytabkit's FTTransformerTrainer internally via manual training.
    """
    from pytabkit.models.TabularML import TabularML
    from pytabkit.Models.TabularFTTransformer import get_default_architecture, TabularFTTransformer
    from pytabkit.Dataset import TabularDataset
    from pytabkit.Models.TabularFTTransformer import TabularFTTransformer
    
    # Create TabularDataset
    dataset = TabularDataset(
        X=prepared_data["X"],
        y=prepared_data["y"][target_col] if prepared_data["y"] else None,
        subjects=prepared_data.get("X_subjects", None),
    )
    
    # FT-Transformer architecture
    if config.get("pretrained", False):
        print(f"[FT] Using pretrained architecture")
        # Use prebuilt pretrained models
        model = TabularML.get_pretrained(model_type="ft-transformer", n_features=prepared_data["X"].shape[1])
        # For custom training, need to build architecture
    else:
        # Build custom architecture
        arch = get_default_architecture(
            n_features=prepared_data["X"].shape[1],
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 4),
            mlp_hidden_mult=config.get("mlp_hidden_mult", 2),
            dropout=config.get("dropout", 0.1),
        )
        
        model = TabularML(
            arch=arch,
            train_mode="binary-classification",  # or regression
            early_stopping_patience=config.get("early_stopping", 10),
            batch_size=config.get("batch_size", 256),
            optimizer="adamw",
            learning_rate=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
    
    return model, dataset


def run_experiment(prepared_data, config, fold=0, seed=42):
    """Run one fold experiment."""
    target_col = config["target"]
    n_splits = config.get("n_splits", 5)
    
    subjects = prepared_data.get("X_subjects", None)
    
    # Choose split strategy
    if subjects is not None:
        splitter = GroupKFold(n_splits=n_splits)
        groups = subjects
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        groups = None
    
    oof_preds = np.zeros(len(prepared_data["X"]))
    test_preds = None
    models_trained = []
    
    print(f"\n{'='*60}")
    print(f"Target: {target_col} | Seed: {seed} | Folds: {n_splits}")
    print(f"{'='*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(
        prepared_data["X"], prepared_data["y"][target_col], groups
    )):
        start = time.time()
        
        X_train = prepared_data["X"][train_idx]
        y_train = prepared_data["y"][target_col][train_idx]
        X_val = prepared_data["X"][val_idx]
        y_val = prepared_data["y"][target_col][val_idx]
        
        # Create fold-specific datasets
        train_dataset = TabularDataset(X=X_train, y=y_train)
        val_dataset = TabularDataset(X=X_val, y=y_val)
        
        # Build model
        arch = get_default_architecture(
            n_features=X_train.shape[1],
            n_layers=config.get("n_layers", 4),
            n_heads=config.get("n_heads", 4),
            mlp_hidden_mult=config.get("mlp_hidden_mult", 2),
            dropout=config.get("dropout", 0.1),
        )
        
        model_type = config.get("model_type", "binary-classification")
        model = TabularML(
            arch=arch,
            train_mode=model_type,
            early_stopping_patience=config.get("early_stopping", 10),
            batch_size=config.get("batch_size", 256),
            optimizer="adamw",
            learning_rate=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        
        # Train
        model.fit(train_dataset, val_dataset)
        
        # Predict
        val_preds = model.predict_proba(val_dataset) if model_type == "binary-classification" else model.predict(val_dataset)
        
        # Metrics
        if model_type == "binary-classification":
            fold_loss = log_loss(y_val, val_preds[:, 1])
            fold_auc = roc_auc_score(y_val, val_preds[:, 1]) if len(np.unique(y_val)) > 1 else 0.5
        else:
            fold_loss = np.mean((y_val - val_preds) ** 2)
            fold_auc = 0.5
        
        elapsed = time.time() - start
        print(f"  Fold {fold_idx+1}: loss={fold_loss:.6f} | auc={fold_auc:.4f} | {elapsed:.1f}s")
        
        oof_preds[val_idx] = val_preds[:, 1] if model_type == "binary-classification" else val_preds
        models_trained.append(model)
    
    # Average OOF
    avg_loss = np.mean([oof_preds[i] for i in range(len(oof_preds))])
    
    return {
        "oof_preds": oof_preds,
        "models": models_trained,
        "target": target_col,
        "fold_losses": oof_preds,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", type=str, default="binary-classification")
    args = parser.parse_args()
    
    config = vars(args)
    
    # Load data
    print("[FT] Loading data...")
    from prepare import load_data, extract_meta, prepare_for_dl
    
    df = load_data()
    meta_info, df = extract_meta(df)
    
    # If multiple targets, pick one
    if args.target not in meta_info["target_cols"]:
        print(f"[FT] Warning: {args.target} not found. Available: {meta_info['target_cols']}")
        if meta_info["target_cols"]:
            config["target"] = meta_info["target_cols"][0]
    
    print("[FT] Preparing features...")
    prepared = prepare_for_dl(df, meta_info)
    
    print("[FT] Training...")
    result = run_experiment(prepared, config)
    
    # Save results
    save_dir = BASE_DIR / "results" / "ft_transformer" / args.target
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    np.save(save_dir / "oof_preds.npy", result["oof_preds"])
    
    print(f"\n[FT] Results saved to {save_dir}")
    print(f"[FT] OOF mean: {result['oof_preds'].mean():.6f}")
