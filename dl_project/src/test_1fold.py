import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

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
gkf = GroupKFold(n_splits=5)

target = targets[0]
y = prepared["y"][target]

print(f"Testing: {target}, X: {X.shape}")

# Test 1: MLP_SKL_D_Classifier
print("\n1. MLP_SKL_D_Classifier")
from pytabkit import MLP_SKL_D_Classifier
ti, vi = list(gkf.split(X, y, groups))[0]
m = MLP_SKL_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")

# Test 2: FTT_D_Classifier (tiny)
print("\n2. FTT_D_Classifier")
from pytabkit import FTT_D_Classifier
m = FTT_D_Classifier(module_d_token=8, module_d_ffn_factor=1, module_n_layers=1, module_n_heads=2)
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")

# Test 3: MLP_RTDL_D_Classifier
print("\n3. MLP_RTDL_D_Classifier")
from pytabkit import MLP_RTDL_D_Classifier
m = MLP_RTDL_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")

# Test 4: XRFM_D_Classifier
print("\n4. XRFM_D_Classifier")
from pytabkit import XRFM_D_Classifier
m = XRFM_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")

# Test 5: RealMLP_TD_Classifier
print("\n5. RealMLP_TD_Classifier")
from pytabkit import RealMLP_TD_Classifier
m = RealMLP_TD_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")

# Test 6: TabM_HPO_Classifier
print("\n6. TabM_HPO_Classifier")
from pytabkit import TabM_HPO_Classifier
m = TabM_HPO_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"   Fold 1 AUC: {roc_auc_score(y[vi], oof):.4f}")
