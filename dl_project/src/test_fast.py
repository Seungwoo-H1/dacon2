# ============================================================
# FT-Transformer Fast Test (CPU, minimal)
# ============================================================
import sys, importlib.util, warnings, tracemalloc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier

tracemalloc.start()

# Load prepare module
src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
spec = importlib.util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
targets = meta_info["target_cols"]
groups = prepared["X_subjects"]
gkf = GroupKFold(n_splits=5)

print(f"Data: {X.shape}, Targets: {targets}")
print(f"RAM: 15GB, CPU: 24 cores")

# =====================
# 1. MLP (scikit-learn) — fast baseline
# =====================
print("\n=== MLP Baseline (128-64-32) ===")
for target in targets:
    y = prepared["y"][target]
    oof = np.zeros(len(X))
    for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        m = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=42,
                          early_stopping=True, validation_fraction=0.15, verbose=False)
        m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {target}: AUC={auc:.4f}")

# =====================
# 2. FT-Transformer — 1 fold only, tiny model
# =====================
print("\n=== FT-Transformer (Q1, 1 fold, tiny) ===")
from pytabkit import FTT_D_Classifier

target = "Q1"
y = prepared["y"][target]

ti, vi = list(gkf.split(X, y, groups))[0]
print(f"  Train: {len(ti)}, Val: {len(vi)}")

t0 = __import__('time').time()
model = FTT_D_Classifier(
    module_d_token=8,        # Very small
    module_d_ffn_factor=1,
    module_n_layers=1,
    module_n_heads=2,
    verbose=0,
)
model.fit(X[ti], y[ti])
elapsed = __import__('time').time() - t0
print(f"  Fit time: {elapsed:.1f}s")

preds = model.predict_proba(X[vi])[:, 1]
auc = roc_auc_score(y[vi], preds)
loss = log_loss(y[vi], preds)
print(f"  Fold 1: AUC={auc:.4f}, Loss={loss:.6f}")

# =====================
# 3. RTD-MLP (from pytabkit) — DL but faster
# =====================
print("\n=== RTD-MLP (Q1, 5 fold) ===")
from pytabkit import MLP_RTDL_D_Classifier

oof_rtdl = np.zeros(len(X))
for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
    m = MLP_RTDL_D_Classifier(verbose=0)
    m.fit(X[ti], y[ti])
    oof_rtdl[vi] = m.predict_proba(X[vi])[:, 1]

auc_rtdl = roc_auc_score(y, oof_rtdl)
loss_rtdl = log_loss(y, oof_rtdl)
print(f"  RTD-MLP Q1: AUC={auc_rtdl:.4f}, Loss={loss_rtdl:.6f}")

# Memory
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n[MEM] Current: {current/1024/1024:.1f}MB, Peak: {peak/1024/1024:.1f}MB")

print(f"\n{'='*40}")
print("Summary for Q1:")
print(f"  MLP sklearn:   (above) AUC=??")
print(f"  RTD-MLP:       {auc_rtdl:.4f}")
print(f"  FT-Transformer:{auc:.4f} (1 fold)")
print(f"  LGBM V10:      0.6038")
