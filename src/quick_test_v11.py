"""Quick test: V11 Q1 — manual ranking, then CV on top features."""
import sys, warnings, logging, re, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
gc.collect()  # Clear LGBM global state
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

V10 = {'Q1': 0.6338}
META = {"subject_id", "lifelog_date", "sleep_date", "date"}
L = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum', 'wPedo_pedo_step_frequency_mean',
    'wPedo_pedo_step_frequency_sum', 'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum', 'wPedo_pedo_distance_mean',
    'wPedo_pedo_distance_sum', 'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

df = pd.read_parquet("data_processed/features_v11.parquet")
t = 'Q1'
base = [c for c in df.columns if c not in META | set(["Q1","Q2","Q3","S1","S2","S3","S4"])
        and not c.endswith('_zscore') and df[c].dtype in [np.float64, np.int64, float, int, bool]]
leak_free = [c for c in base if c not in L]
col2idx = {c: i for i, c in enumerate(base)}
leak_idx = [col2idx[c] for c in leak_free]
y = df[t].values
sid = df['subject_id'].values
X = df[base].fillna(0).values
X_leak = X[:, leak_idx]

log.info(f"Leak-free: {len(leak_free)}")

# Ranking with LGBM — use num_boost_round=1, then add trees one at a time
n_pos = max((y == 1).sum(), 1); n_neg = (y == 0).sum(); spw = n_neg / n_pos
params_rank = dict(objective='binary', metric='binary_logloss', verbose=-1,
                   num_leaves=10, max_depth=3, learning_rate=0.03, subsample=0.7,
                   colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
                   scale_pos_weight=spw, random_state=42, min_child_samples=10)

ds = lgb.Dataset(X_leak, label=y)
mdl = lgb.train(params_rank, ds, num_boost_round=30)
imp = mdl.feature_importance(importance_type="gain")
ranked = sorted(zip(leak_free, imp), key=lambda x: -x[1])
log.info(f"Top 5: {[r[0] for r in ranked[:5]]}")

# Now train CV models on TOP 30 features — this is the critical part
top30 = [r[0] for r in ranked[:30] if r[1] > 0]
top30_idx = [col2idx[c] for c in top30]
X30 = X[:, top30_idx]
s30 = [sanitize(c) for c in top30]
log.info(f"X30: {X30.shape}")

# Clear ranking model from memory
del mdl, ds
gc.collect()

# CV
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_leak, y, sid))
SEEDS = [42, 123, 456, 789, 1024]
cfg = dict(nl=10, md=3, lr=0.02, ne=200, ss=0.7, cst=0.7, ra=1.0, rl=3.0, mc=10)

n = len(y)
oof = np.zeros((n, len(SEEDS)))
for si, seed in enumerate(SEEDS):
    prms = dict(objective='binary', metric='binary_logloss', verbose=-1, **cfg,
                force_row_wise=True, n_jobs=8, scale_pos_weight=spw, random_state=seed)
    for fi, (tr_i, va_i) in enumerate(splits):
        tr = lgb.Dataset(X30[tr_i], label=y[tr_i])
        tr.set_feature_name(s30)
        va = lgb.Dataset(X30[va_i], label=y[va_i])
        va.set_feature_name(s30)
        m = lgb.train(prms, tr, num_boost_round=cfg['ne'], valid_sets=[va],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va_i, si] = m.predict(X30[va_i])
        del m, tr, va
    gc.collect()
    log.info(f"  Seed {si}/{len(SEEDS)} done")

oof_avg = oof.mean(1)
cv = log_loss(y, oof_avg, labels=[0, 1])
cal = np.clip(oof_avg + (y.mean() - oof_avg.mean()), 0.0001, 0.9999)
cal_loss = log_loss(y, cal, labels=[0, 1])
v10 = V10[t]
log.info(f"Q1 CV={cv:.4f} Cal={cal_loss:.4f} V10={v10:.4f} Δ={cal_loss - v10:+.4f}")
log.info(f"{'🎉 V11 beats V10' if cal_loss < v10 else 'V10 wins'}")
