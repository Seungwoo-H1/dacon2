"""
V361 — Multi-Model Ensemble: LGBM + RF + ExtraTree

Hypothesis: LGBM-only ensemble이 최적은 아님.
서로 다른 모델(RF, ExtraTree)의 OOF predictions을 추가하면
보완적인 signal을 meta-learner가 포착할 수 있음.

Pipeline:
1. V339-style LGBM (15 seeds) OOF
2. RandomForest (10 seeds) OOF — 전 features 사용
3. ExtraTree (10 seeds) OOF — 전 features 사용
4. 총 35개 OOF predictions → LR meta-learner(C=1)
"""
import sys, gc, logging, json, re, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
log.info("V361 — Multi-Model Ensemble: LGBM + RF + ExtraTree")

train = pd.read_parquet(DATA / "features.parquet")
test = pd.read_parquet(DATA / "test_features.parquet")
for df in [train, test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

base_feats = get_base_feats(train)

# Z-score
for col in base_feats:
    if col in test.columns:
        vals = train[col].fillna(0).values.astype(np.float64)
        mean_val, std_val = np.mean(vals), max(np.std(vals, ddof=0), 1e-8)
        zc = f'{col}_zscore'
        train[zc] = (vals - mean_val) / std_val
        test[zc] = (test[col].fillna(0).values.astype(np.float64) - mean_val) / std_val

train_cols = get_feature_cols(train)
test_cols = get_feature_cols(test)
log.info(f"Features: {len(train_cols)}")

gkf = GroupKFold(n_splits=5)
N_FOLDS = 5

LGBM_CFG = {
    'Q1':  {'n_feat': 19, 'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'n_seeds': 15},
    'Q2':  {'n_feat': 14, 'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'n_seeds': 15},
    'Q3':  {'n_feat': 11, 'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'n_seeds': 15},
    'S1':  {'n_feat': 21, 'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'n_seeds': 15},
    'S2':  {'n_feat': 19, 'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'n_seeds': 15},
    'S3':  {'n_feat': 23, 'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'n_seeds': 15},
    'S4':  {'n_feat': 20, 'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'n_seeds': 15},
}

results = {}

for t in TARGETS:
    tc = LGBM_CFG[t]
    feat_clean = remove_leak(train_cols, t)
    y = train[t].values.astype(np.float64)
    grp = train['subject_id'].values
    n_tr = len(train)
    n_te = len(test)
    
    X_all_np = train[feat_clean].fillna(0).values.astype(np.float64)
    X_test_np = test[feat_clean].fillna(0).values.astype(np.float64)
    
    # Feature ranking (LGBM gain)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    snames = [sanitize_col(c) for c in feat_clean]
    ds = lgb.Dataset(X_all_np, label=y, feature_name=snames)
    mr = lgb.train({'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':20,'max_depth':5,'learning_rate':0.05,'n_estimators':50,
                    'scale_pos_weight':spw,'random_state':42,'force_row_wise':True,'n_jobs':1},
                   ds, num_boost_round=50)
    ranked = sorted(zip(feat_clean, mr.feature_importance('gain')), key=lambda x: -x[1])
    sel = [r[0] for r in ranked[:tc['n_feat']]]
    sel_idx = [feat_clean.index(c) for c in sel]
    sel_te = [c for c in sel if c in test_cols]
    
    # LGBM OOF
    l_oofs = np.zeros((n_tr, tc['n_seeds']))
    l_tests = np.zeros((n_te, tc['n_seeds']))
    for si in range(tc['n_seeds']):
        seed = 42 + si * 7
        os_ = np.zeros(n_tr)
        ts_ = np.zeros(n_te)
        for fold, (tri, vai) in enumerate(gkf.split(train, y, grp)):
            spwf = max(((y[tri]==0).sum())/max((y[tri]==1).sum(),1), 0.1)
            pm = {**{k:v for k,v in tc.items() if k not in ['n_feat','n_seeds']},
                  'scale_pos_weight':spwf,'random_state':seed,'force_row_wise':True,'n_jobs':1,'verbose':-1}
            m = lgb.train(pm, lgb.Dataset(X_all_np[tri][:, sel_idx], label=y[tri], 
                                           feature_name=[sanitize_col(c) for c in sel]))
            os_[vai] = m.predict(X_all_np[vai][:, sel_idx])
            ts_ += m.predict(test[sel_te].fillna(0).values.astype(np.float64)) / N_FOLDS
        l_oofs[:, si] = np.clip(os_, 0.001, 0.999)
        l_tests[:, si] = ts_
    
    # RF OOF (10 seeds)
    r_oofs = np.zeros((n_tr, 10))
    for si in range(10):
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, 
                                     random_state=100+si*13, max_features=0.5, n_jobs=1)
        os_ = np.zeros(n_tr)
        for fold, (tri, vai) in enumerate(gkf.split(train, y, grp)):
            rf.fit(X_all_np[tri], y[tri])
            os_[vai] = rf.predict_proba(X_all_np[vai])[:, 1]
        r_oofs[:, si] = np.clip(os_, 0.001, 0.999)
    
    # ExtraTree OOF (10 seeds)
    e_oofs = np.zeros((n_tr, 10))
    for si in range(10):
        et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   random_state=200+si*17, max_features=0.5, n_jobs=1)
        os_ = np.zeros(n_tr)
        for fold, (tri, vai) in enumerate(gkf.split(train, y, grp)):
            et.fit(X_all_np[tri], y[tri])
            os_[vai] = et.predict_proba(X_all_np[vai])[:, 1]
        e_oofs[:, si] = np.clip(os_, 0.001, 0.999)
    
    # Combined
    all_oofs = np.column_stack([l_oofs, r_oofs, e_oofs])
    
    meta = LogisticRegression(C=1, max_iter=1000, random_state=42)
    meta.fit(all_oofs, y)
    meta_ll = log_loss(y, np.clip(meta.predict_proba(all_oofs)[:, 1], 0.001, 0.999))
    student_ll = log_loss(y, np.clip(np.mean(all_oofs, axis=1), 0.001, 0.999))
    
    results[t] = {'student': student_ll, 'meta': meta_ll}
    log.info(f"  {t}: student={student_ll:.5f}, meta={meta_ll:.5f} ({all_oofs.shape[1]} preds)")

avg_s = np.mean([v['student'] for v in results.values()])
avg_m = np.mean([v['meta'] for v in results.values()])
log.info(f"\nV361: AVG student={avg_s:.5f}, AVG meta={avg_m:.5f} (V339=0.61244, Δ={avg_m-0.61244:+.5f})")
for t in TARGETS:
    log.info(f"  {t}: student={results[t]['student']:.5f}, meta={results[t]['meta']:.5f}")
log.info(f"Time: {time.time()-t_start:.0f}s")
