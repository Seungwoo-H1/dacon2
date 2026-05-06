import time, sys, warnings, os
warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
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
y = prepared["y"]["Q1"]
groups = prepared["X_subjects"]
gkf = GroupKFold(n_splits=5)
ti, vi = list(gkf.split(X, y, groups))[0]

# MLP_SKL_D_Classifier
print("MLP_SKL_D_Classifier (1 fold)...")
from pytabkit import MLP_SKL_D_Classifier
t0 = time.time()
m = MLP_SKL_D_Classifier()
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"  AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")

# FTT_D_Classifier (tiny)
print("FTT_D_Classifier (tiny, 1 fold)...")
from pytabkit import FTT_D_Classifier
t0 = time.time()
m = FTT_D_Classifier(module_d_token=8, module_d_ffn_factor=1, module_n_layers=1, module_n_heads=2)
m.fit(X[ti], y[ti])
oof = m.predict_proba(X[vi])[:, 1]
print(f"  AUC={roc_auc_score(y[vi], oof):.4f}, time={time.time()-t0:.1f}s")
