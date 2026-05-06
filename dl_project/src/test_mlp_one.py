import time, sys, warnings
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
y = prepared["y"]["Q1"]

# Just MLP_SKL_D_Classifier, 1 fold
print("MLP_SKL_D_Classifier fit (1 fold)...")
from pytabkit import MLP_SKL_D_Classifier
m = MLP_SKL_D_Classifier()
t0 = time.time()
m.fit(X[:360], y[:360])
print(f"Fit time: {time.time()-t0:.1f}s")
oof = m.predict_proba(X[360:361])[:, 1]
print(f"Pred: {oof}")
print("OK!")
