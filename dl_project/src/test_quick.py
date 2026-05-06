# ============================================================
# Quick Test: FT-Transformer on Dacon2 features.parquet
# ============================================================

import sys
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Load prepare module
src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
spec = importlib.util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

# Step 1: Load data
print("[1] Loading data...")
df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
print(f"    Shape: {df.shape}, Features: {len(meta_info['all_numeric_cols'])}, Targets: {meta_info['target_cols']}")

# Step 2: Prepare features
print("[2] Preparing features...")
prepared = prepare.prepare_for_dl(df, meta_info)
print(f"    X shape: {prepared['X'].shape}")

# Step 3: Quick test with FT-Transformer on ONE target
print("[3] Quick FT-Transformer test on Q1...")
from pytabkit import FTT_D_Classifier, LGBM_D_Classifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

target = "Q1"
y = prepared["y"][target]
X = prepared["X"]
groups = prepared["X_subjects"]

# GroupKFold by subject
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(X))
oof_labels = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # FT-Transformer
    model = FTT_D_Classifier(
        module_d_token=64,
        module_d_ffn_factor=2,
        module_n_layers=2,
        module_n_heads=4,
        verbose=0,
    )
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_val)[:, 1]
    fold_loss = log_loss(y_val, preds)
    fold_auc = roc_auc_score(y_val, preds)
    oof_preds[val_idx] = preds
    oof_labels[val_idx] = y_val
    print(f"    Fold {fold+1}: log_loss={fold_loss:.6f}, AUC={fold_auc:.4f}")

overall_auc = roc_auc_score(oof_labels, oof_preds)
print(f"\n    [RESULT] OOF AUC: {overall_auc:.4f}")

# Also test LGBM baseline for comparison
print("[4] Quick LGBM baseline on Q1...")
from pytabkit import LGBM_D_Classifier

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = LGBM_D_Classifier(verbose=0)
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_val)[:, 1]
    oof_preds_lgbm = np.zeros(len(X))
    oof_preds_lgbm[val_idx] = preds

lgbm_auc = roc_auc_score(oof_labels, oof_preds_lgbm)
print(f"    [RESULT] LGBM OOF AUC: {lgbm_auc:.4f}")

print(f"\n{'='*50}")
print(f"Q1 Comparison:")
print(f"  FT-Transformer (small): {overall_auc:.4f}")
print(f"  LGBM (default):         {lgbm_auc:.4f}")
print(f"  Gap: {overall_auc - lgbm_auc:+.4f}")
