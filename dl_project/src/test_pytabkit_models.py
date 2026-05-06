# pytabkit DL Models Test — All targets, 5-fold GroupKFold
import sys, warnings, tracemalloc
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

print(f"Data: {X.shape}, Targets: {targets}")
print(f"RAM: 15GB, CPU: {__import__('os').cpu_count()} cores")
print()

models_to_try = [
    ("MLP_SKL_D_Classifier", {"verbose": 0}),
    ("MLP_RTDL_D_Classifier", {"verbose": 0}),
    ("RealMLP_TD_Classifier", {"verbose": 0}),
    ("FTT_D_Classifier", {"module_d_token": 8, "module_d_ffn_factor": 1, "module_n_layers": 1, "module_n_heads": 2, "verbose": 0}),
    ("TabM_HPO_Classifier", {"verbose": 0}),
    ("XRFM_D_Classifier", {"verbose": 0}),
]

results = {}

for model_name, params in models_to_try:
    print(f"\n=== {model_name} ===")
    try:
        mod_class = __import__('pytabkit', fromlist=[model_name])
        ModelClass = getattr(mod_class, model_name)
    except AttributeError:
        print(f"  NOT FOUND")
        continue
    
    t0 = time.time()
    t_total = 0
    
    for target in targets:
        y = prepared["y"][target]
        oof = np.zeros(len(X))
        model_times = []
        
        for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
            ft = time.time()
            m = ModelClass(**params)
            m.fit(X[ti], y[ti])
            model_times.append(time.time() - ft)
            
            # Get predictions
            if hasattr(m, 'predict_proba'):
                preds = m.predict_proba(X[vi])[:, 1]
            else:
                preds = m.predict(X[vi])
            
            oof[vi] = preds
        
        avg_time = np.mean(model_times)
        t_total += avg_time
        
        auc = roc_auc_score(y, oof)
        loss = log_loss(y, oof)
        results[f"{model_name}/{target}"] = {"auc": auc, "loss": loss, "time_per_fold": avg_time}
        print(f"  {target}: AUC={auc:.4f}, Loss={loss:.6f}, Time/fold={avg_time:.1f}s")
    
    print(f"  Total avg fold time: {t_total/len(targets):.1f}s")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for key in sorted(results.keys()):
    r = results[key]
    print(f"  {key}: AUC={r['auc']:.4f}, Loss={r['loss']:.6f}")
