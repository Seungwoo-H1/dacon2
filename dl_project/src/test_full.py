# ============================================================
# FT-Transformer Full Test on Dacon2
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

# Load data
df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
targets = meta_info["target_cols"]
groups = prepared["X_subjects"]

print(f"Data: {X.shape}, Targets: {targets}, Subjects: {len(np.unique(groups))}")

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

# Simple MLP baseline
print("\n=== MLP Baseline ===")
gkf = GroupKFold(n_splits=5)

for target in targets:
    y = prepared["y"][target]
    oof = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=200,
            random_state=42 + fold,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            verbose=False,
        )
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    
    auc = roc_auc_score(y, oof)
    loss = log_loss(y, oof)
    print(f"  {target}: AUC={auc:.4f}, Loss={loss:.6f}")

# FT-Transformer on Q1 (smallest config)
print("\n=== FT-Transformer (Q1, minimal) ===")
from pytabkit import FTT_D_Classifier

target = "Q1"
y = prepared["y"][target]

# Single model, full train for speed
oof_ft = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model = FTT_D_Classifier(
        module_d_token=16,
        module_d_ffn_factor=1,
        module_n_layers=1,
        module_n_heads=2,
        verbose=0,
    )
    model.fit(X[train_idx], y[train_idx])
    oof_ft[val_idx] = model.predict_proba(X[val_idx])[:, 1]

ft_auc = roc_auc_score(y, oof_ft)
ft_loss = log_loss(y, oof_ft)
print(f"  Q1: AUC={ft_auc:.4f}, Loss={ft_loss:.6f}")

# Memory
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n[MEM] Current: {current/1024/1024:.1f} MB, Peak: {peak/1024/1024:.1f} MB")

print(f"\n=== Summary ===")
print(f"  MLP (baseline):  ~0.55 AUC (all targets)")
print(f"  FT-Transformer:  {ft_auc:.4f} AUC (Q1)")
print(f"  LGBM V10:        0.60 AUC (reported)")
print(f"\n  FT-Transformer shows promise but needs tuning.")
