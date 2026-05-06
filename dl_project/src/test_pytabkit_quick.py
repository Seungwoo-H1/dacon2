# pytabkit DL Models Test — parameter-free, 1 target
import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss

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
target = targets[0]  # Q1
y = prepared["y"][target]

print(f"Data: {X.shape}, Target: {target}")

models = [
    ("MLP_SKL_D_Classifier", {}),
    ("MLP_RTDL_D_Classifier", {}),
    ("RealMLP_TD_Classifier", {}),
    ("FTT_D_Classifier", {"module_d_token": 8, "module_d_ffn_factor": 1, "module_n_layers": 1, "module_n_heads": 2}),
    ("TabM_HPO_Classifier", {}),
    ("XRFM_D_Classifier", {}),
]

print(f"\n{'Model':<30} {'AUC':>6} {'Loss':>8} {'Time/fold':>10}")
print(f"{'='*60}")

results = {}
for name, extra_params in models:
    mod_class = __import__('pytabkit', fromlist=[name])
    ModelClass = getattr(mod_class, name)
    
    oof = np.zeros(len(X))
    t0 = time.time()
    
    try:
        for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
            kwargs = dict(extra_params)
            m = ModelClass(**kwargs)
            m.fit(X[ti], y[ti])
            if hasattr(m, 'predict_proba'):
                oof[vi] = m.predict_proba(X[vi])[:, 1]
            else:
                oof[vi] = m.predict(X[vi])
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{name:<30} {'ERROR':>6} {'':>8} {elapsed:>9.1f}s")
        print(f"  Error: {e}")
        continue
    
    elapsed = time.time() - t0
    auc = roc_auc_score(y, oof)
    loss = log_loss(y, oof)
    results[name] = {"auc": auc, "loss": loss, "time": elapsed}
    print(f"{name:<30} {auc:>6.4f} {loss:>8.6f} {elapsed:>9.1f}s")

print(f"\n{'='*60}")
print("LGBM V10 baseline: cal OOF loss ≈ 0.6038")
