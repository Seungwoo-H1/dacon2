# ============================================================
# Test: MLP baseline (single target, very fast)
# ============================================================
import sys, warnings, tracemalloc
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier

print("Start...")

src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
spec = __import__('importlib.util').util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(prepare)

df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
targets = meta_info["target_cols"]
groups = prepared["X_subjects"]

print("Data loaded. X:", X.shape)

# Just Q1, 1 fold
y = prepared["y"][targets[0]]
gkf = GroupKFold(n_splits=5)
ti, vi = list(gkf.split(X, y, groups))[0]

print(f"Training MLP on Q1, train={len(ti)}, val={len(vi)}...")
m = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=50, random_state=42, verbose=False)
m.fit(X[ti], y[ti])
preds = m.predict_proba(X[vi])[:, 1]
auc = roc_auc_score(y[vi], preds)
print(f"Fold 1: AUC={auc:.4f}")
print("Done!")
