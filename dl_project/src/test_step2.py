import time, sys, warnings
warnings.filterwarnings("ignore")
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

models_no_params = ["MLP_SKL_D_Classifier", "FTT_D_Classifier", "MLP_RTDL_D_Classifier"]

for name in models_no_params:
    mod_class = __import__('pytabkit', fromlist=[name])
    ModelClass = getattr(mod_class, name)
    
    # FTT needs special params
    if name == "FTT_D_Classifier":
        params = {"module_d_token": 8, "module_d_ffn_factor": 1, "module_n_layers": 1, "module_n_heads": 2}
    else:
        params = {}
    
    print(f"\n{name}...")
    t0 = time.time()
    m = ModelClass(**params)
    print(f"  Create: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    m.fit(X[ti], y[ti])
    fit_time = time.time() - t0
    print(f"  Fit: {fit_time:.1f}s")
    
    t0 = time.time()
    oof = m.predict_proba(X[vi])[:, 1]
    print(f"  Predict: {time.time()-t0:.3f}s")
    
    auc = roc_auc_score(y[vi], oof)
    print(f"  AUC: {auc:.4f}")
