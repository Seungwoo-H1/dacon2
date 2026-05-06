# ============================================================
# Quick Test: FT-Transformer on Dacon2 features.parquet (Memory-safe)
# ============================================================

import sys
import importlib.util
import warnings
import tracemalloc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

tracemalloc.start()

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
X = prepared["X"]
y = prepared["y"]["Q1"]
groups = prepared["X_subjects"]
print(f"    X shape: {X.shape}")

# Step 3: Simple MLP baseline (scikit-learn)
print("[3] Simple MLP (scikit-learn) baseline...")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    
    preds = model.predict_proba(X_val)[:, 1]
    fold_loss = log_loss(y_val, preds)
    fold_auc = roc_auc_score(y_val, preds)
    print(f"    Fold {fold+1}: log_loss={fold_loss:.6f}, AUC={fold_auc:.4f}")

# OOF AUC
oof_mlp = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    oof_mlp[val_idx] = model.predict_proba(X_val)[:, 1]

mlp_auc = roc_auc_score(y, oof_mlp)
print(f"\n    [MLP RESULT] OOF AUC: {mlp_auc:.4f}")

# Step 4: LGBM from pytabkit (no deep learning, fast baseline)
print("[4] LGBM from pytabkit...")
from pytabkit import LGBM_D_Classifier

oof_lgbm = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = LGBM_D_Classifier(verbose=0)
    model.fit(X_tr, y_tr)
    oof_lgbm[val_idx] = model.predict_proba(X_val)[:, 1]

lgbm_auc = roc_auc_score(y, oof_lgbm)
print(f"    [LGBM RESULT] OOF AUC: {lgbm_auc:.4f}")

# Step 5: FT-Transformer with minimal config
print("[5] FT-Transformer (minimal config, single fold)...")
from pytabkit import FTT_D_Classifier

# Just train 1 model on full data for speed test
model = FTT_D_Classifier(
    module_d_token=32,       # smaller token dim
    module_d_ffn_factor=1,   # smaller FFN
    module_n_layers=1,       # only 1 layer
    module_n_heads=2,        # fewer heads
    verbose=1,
)
print("    Fitting FT-Transformer (1 fold, minimal)...")
try:
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, preds)
    print(f"    [FT RESULT] Train AUC: {train_auc:.4f}")
except Exception as e:
    print(f"    [FT ERROR] {e}")

# Memory stats
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n[MEM] Current: {current/1024/1024:.1f} MB, Peak: {peak/1024/1024:.1f} MB")
print(f"\n{'='*50}")
print(f"Summary:")
print(f"  MLP (sklearn):     {mlp_auc:.4f} AUC")
print(f"  LGBM (pytabkit):   {lgbm_auc:.4f} AUC")
print(f"{'='*50}")
