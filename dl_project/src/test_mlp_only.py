# ============================================================
# 1. MLP Only — All 7 targets
# ============================================================
import sys, warnings, tracemalloc
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier

tracemalloc.start()

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

print(f"Data: {X.shape}, Targets: {targets}, CPU: {__import__('os').cpu_count()} cores, RAM: 15GB")
print()

results = {}
for target in targets:
    y = prepared["y"][target]
    oof = np.zeros(len(X))
    for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        m = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=150,
            random_state=42 + fold,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            verbose=False,
        )
        m.fit(X[ti], y[ti])
        oof[vi] = m.predict_proba(X[vi])[:, 1]
    auc = roc_auc_score(y, oof)
    loss = log_loss(y, oof)
    results[target] = {"auc": auc, "loss": loss}
    print(f"  {target}: AUC={auc:.4f}, Loss={loss:.6f}")

avg_auc = np.mean([v["auc"] for v in results.values()])
avg_loss = np.mean([v["loss"] for v in results.values()])
print(f"\n  AVG: AUC={avg_auc:.4f}, Loss={avg_loss:.6f}")

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n[MEM] Peak: {peak/1024/1024:.1f}MB")
