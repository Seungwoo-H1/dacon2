import sys, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

src_dir = Path("/home/mwoo423/.openclaw/workspace/dl_project/src")
sys.path.insert(0, str(src_dir))
spec = __import__('importlib.util').util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(prepare)

df = prepare.load_data()
meta_info, df = prepare.extract_meta(df)
prepared = prepare.prepare_for_dl(df, meta_info)
X = prepared["X"]
groups = prepared["X_subjects"]

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
gkf = GroupKFold(n_splits=5)
target = "Q1"
y = prepared["y"][target]

ti, vi = list(gkf.split(X, y, groups))[0]

# === 1. MLP_SKL_D_Classifier ===
print("1. MLP_SKL_D_Classifier...")
t0 = time.time()
from pytabkit import MLP_SKL_D_Classifier
m = MLP_SKL_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")

# === 2. FTT_D_Classifier (tiny) ===
print("2. FTT_D_Classifier (tiny)...")
t0 = time.time()
from pytabkit import FTT_D_Classifier
m = FTT_D_Classifier(module_d_token=8, module_d_ffn_factor=1, module_n_layers=1, module_n_heads=2)
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")

# === 3. MLP_RTDL_D_Classifier ===
print("3. MLP_RTDL_D_Classifier...")
t0 = time.time()
from pytabkit import MLP_RTDL_D_Classifier
m = MLP_RTDL_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")

# === 4. XRFM_D_Classifier ===
print("4. XRFM_D_Classifier...")
t0 = time.time()
from pytabkit import XRFM_D_Classifier
m = XRFM_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")
