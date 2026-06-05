"""
V363 — Key Insight: Reduce Student OOF

Problem: Q targets have student OOF ~0.70-0.78
V339's meta-learner reduces this but can't fix fundamental signal weakness.

New approach: Instead of adding features or models, 
try **target transformation** — change the target encoding to make it easier.

Methods:
1. Platt scaling per target (fit on OOF predictions)
2. Temperature scaling on probabilities
3. Quantile normalization of target distribution

Also try: **Ensemble of multiple LGBM configs** (same seeds, different hyperparams)
V339 uses one config per target. What if we ensemble 3 configs per target?
"""
import sys, gc, logging, re, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)
def get_feature_cols(df): return [c for c in df.columns if c not in META_COLS|set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]
def get_base_feats(df): return [c for c in df.columns if c not in META_COLS|set(TARGETS) and not c.endswith('_zscore') and not c.startswith('oof_') and np.issubdtype(df[c].dtype, np.number)]
def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

t_start = time.time()
log.info("V363 — Multi-Config Ensemble (3 configs × seeds)")

train = pd.read_parquet(DATA / "features.parquet")
test = pd.read_parquet(DATA / "test_features.parquet")
for df in [train, test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

base_feats = get_base_feats(train)
for col in base_feats:
    if col in test.columns:
        vals = train[col].fillna(0).values.astype(np.float64)
        mean_val = np.mean(vals)
        std_val = max(np.std(vals, ddof=0), 1e-8)
        zc = f'{col}_zscore'
        train[zc] = (vals - mean_val) / std_val
        test[zc] = (test[col].fillna(0).values.astype(np.float64) - mean_val) / std_val

train_cols = get_feature_cols(train)
test_cols = get_feature_cols(test)
log.info(f"Features: {len(train_cols)}")

gkf = GroupKFold(n_splits=5)
N_FOLDS = 5

# 3 different configs per target, ensemble them
CONFIGS = {
    'shallow': {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 500,
                'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'reg_lambda': 5.0},
    'deep':    {'num_leaves': 25, 'max_depth': 6, 'learning_rate': 0.01, 'n_estimators': 1500,
                'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    'medium':  {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
                'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
}

# V339 feature counts
FEAT_COUNTS = {
    'Q1': 19, 'Q2': 14, 'Q3': 11,
    'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20,
}

# Per config per target, generate OOF predictions
# Total: 3 configs × 15 seeds = 45 predictions per target
results = {}

for t in TARGETS:
    feat_clean = remove_leak(train_cols, t)
    y = train[t].values.astype(np.float64)
    grp = train['subject_id'].values
    n_tr = len(train)
    n_te = len(test)
    n_feat = FEAT_COUNTS[t]
    
    X_all = train[feat_clean].fillna(0).values.astype(np.float64)
    X_test = test[feat_clean].fillna(0).values.astype(np.float64)
    
    # Feature ranking
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    snames = [sanitize_col(c) for c in feat_clean]
    ds = lgb.Dataset(X_all, label=y, feature_name=snames)
    mr = lgb.train({'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':20,'max_depth':5,'learning_rate':0.05,'n_estimators':50,
                    'scale_pos_weight':spw,'random_state':42,'force_row_wise':True,'n_jobs':1},
                   ds, num_boost_round=50)
    ranked = sorted(zip(feat_clean, mr.feature_importance('gain')), key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:n_feat]]
    sel_idx = [feat_clean.index(c) for c in sel]
    sel_te = [c for c in sel if c in test_cols]
    
    # For each config, generate 15-seed OOF
    all_oofs = []
    all_tests = []
    
    for cfg_name, cfg in CONFIGS.items():
        cfg_oofs = np.zeros((n_tr, 15))
        cfg_tests = np.zeros((n_te, 15))
        
        for si in range(15):
            seed = 42 + si * 7 + hash(t) % 100  # Different seeds per target
            os_ = np.zeros(n_tr)
            ts_ = np.zeros(n_te)
            
            for fold, (tri, vai) in enumerate(gkf.split(train, y, grp)):
                spwf = max(((y[tri]==0).sum())/max((y[tri]==1).sum(),1), 0.1)
                pm = {**cfg, 'scale_pos_weight':spwf, 'random_state':seed,
                      'force_row_wise':True,'n_jobs':1,'verbose':-1}
                m = lgb.train(pm, lgb.Dataset(X_all[tri][:, sel_idx], label=y[tri],
                                               feature_name=[sanitize_col(c) for c in sel]))
                os_[vai] = m.predict(X_all[vai][:, sel_idx])
                ts_ += m.predict(test[sel_te].fillna(0).values.astype(np.float64)) / N_FOLDS
            
            cfg_oofs[:, si] = np.clip(os_, 0.001, 0.999)
            cfg_tests[:, si] = ts_
        
        all_oofs.append(cfg_oofs)
        all_tests.append(cfg_tests)
        log.info(f"    {t} {cfg_name}: student OOF = {log_loss(y, np.mean(cfg_oofs, axis=1)):.5f}")
    
    # Stack all configs
    stacked = np.hstack(all_oofs)  # (n_tr, 45)
    
    # Student average
    student_preds = np.mean(stacked, axis=1)
    student_ll = log_loss(y, student_preds)
    
    # Meta-learner
    meta = LogisticRegression(C=1, max_iter=1000, random_state=42)
    meta.fit(stacked, y)
    meta_ll = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
    
    results[t] = {'student': student_ll, 'meta': meta_ll, 'configs': 3, 'seeds': 15}
    log.info(f"  {t}: student={student_ll:.5f}, meta={meta_ll:.5f} (3 configs × 15 seeds = 45 preds)")

avg_s = np.mean([v['student'] for v in results.values()])
avg_m = np.mean([v['meta'] for v in results.values()])
log.info(f"\n{'='*70}")
log.info(f"V363: AVG student={avg_s:.5f}, AVG meta={avg_m:.5f} (V339=0.61244, Δ={avg_m-0.61244:+.5f})")
for t in TARGETS:
    log.info(f"  {t}: student={results[t]['student']:.5f}, meta={results[t]['meta']:.5f}")
log.info(f"Time: {time.time()-t_start:.0f}s")
