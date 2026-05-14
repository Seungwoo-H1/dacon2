"""V256: Diverse Feature Pipeline Ensemble — V127+

Hypothesis: 5 completely different feature pipelines trained separately 
and optimally weighted ensemble can capture orthogonal signal.

5 Pipelines:
- A: V127 baseline (base + z-score personalization) 
- B: Raw stats only (no personalization)
- C: Temporal features (dow, doe, sin/cos, weekend)
- D: Cross-feature interactions (pairwise ×, /, diff)
- E: Rolling window aggregates (multi-window per subject)
"""
import logging, sys, gc, time, warnings, json, re, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]
N_FOLDS = 5

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide':   {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5},
    'deep':   {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15},
    'v48':    {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10},
    'safety': {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20},
}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feat_cols(df):
    return [c for c in df.columns if c not in META | set(TARGETS) 
            and c not in ['subject_id','lifelog_date','sleep_date','date']
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

def add_zscore(df, feat_cols, stats=None, for_test=False):
    """Add per-subject z-score for each feature."""
    df = df.copy()
    all_stats = {}
    zcols = []
    for c in feat_cols:
        vals = df[c].fillna(0)
        grp = vals.groupby(df['subject_id']).agg(mean='mean', std='std').reset_index()
        grp.columns = ['subject_id', f'{c}_subj_mean', f'{c}_subj_std']
        df = df.merge(grp, on='subject_id', how='left')
        sm = df[f'{c}_subj_mean']
        ss = df[f'{c}_subj_std']
        if not for_test:
            all_stats[c] = {'mean': sm, 'std': ss}
        mask = (ss == 0) | df[c].isnull()
        df[f'{c}_z'] = np.where(mask, 0.0, (df[c].fillna(0) - sm) / np.maximum(ss, 1e-8))
        zcols.append(f'{c}_z')
        gc.collect()
    return df, zcols, all_stats

def rank_features_on_df(df, feat_cols, target, seed=42):
    """Rank features using LGBM importance on given dataframe."""
    y = df[target].values.astype(np.float64)
    X = df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x:-x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

def train_cv_on_df(df_feat, df_test, sel_cols, y, seeds, cfg, n_folds=5):
    """Train LGBM on given dataframe with GroupKFold."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    test_preds = np.zeros((len(df_test), len(seeds)))
    sn = [sanitize(c) for c in sel_cols]
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    X_full = df_feat[sel_cols].fillna(0).values.astype(np.float64)
    X_test = df_test[sel_cols].fillna(0).values.astype(np.float64)
    for si, seed in enumerate(seeds):
        cfg_full = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
            'min_child_samples':cfg['mc'],'random_state':seed,'scale_pos_weight':spw,
        }
        for tr_i, va_i in gkf.split(df_feat, y, df_feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(cfg_full, ds, num_boost_round=cfg['ne'], valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            test_preds[:, si] = m.predict(X_test)
            del ds, vd, m
            gc.collect()
    return np.clip(oof, 0.0001, 0.9999), np.clip(test_preds, 0.0001, 0.9999)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def optimize_weights(preds, y):
    """SLSQP weight optimization for ensemble blending."""
    n = preds.shape[1]
    def neg_ll(w):
        blended = np.clip(preds @ w, 0.0001, 0.9999)
        return log_loss(y, blended, labels=[0,1])
    cons = {'type':'eq','fun':lambda w: np.sum(w)-1}
    bnds = tuple((0.01, 0.99) for _ in range(n))
    w0 = np.ones(n)/n
    res = minimize(neg_ll, w0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':500})
    if res.success:
        return res.x, res.fun
    return w0, log_loss(y, np.clip(preds @ w0, 0.0001, 0.9999))

# ============================================================
# Build feature pipelines
# ============================================================
log.info("Loading data...")
feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")
for df in [feat, feat_test]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

y_train = {t: feat[t].values for t in TARGETS}
train_rates = {t: feat[t].mean() for t in TARGETS}

log.info("\n=== Building feature pipelines ===")

pipelines = {}
t0 = time.time()

# Pipeline A: base + zscore
log.info("Building A_base...")
feat_A = feat.copy()
feat_cols_raw = get_feat_cols(feat_A)
feat_A, zcols_A, stats_A = add_zscore(feat_A, feat_cols_raw)
feat_test_A = feat_test.copy()
feat_cols_raw_t = get_feat_cols(feat_test)
feat_test_A, _, _ = add_zscore(feat_test_A, feat_cols_raw_t, stats_A, for_test=True)
pipelines['A_base'] = {'train': feat_A, 'test': feat_test_A, 'feat_cols': feat_cols_raw + zcols_A}
log.info(f"  A_base: {len(pipelines['A_base']['feat_cols'])} features")

# Pipeline B: raw stats only (no personalization)
log.info("Building B_stat...")
feat_B = feat.copy()
feat_cols_B = get_feat_cols(feat_B)
feat_test_B = feat_test.copy()
feat_cols_B_t = get_feat_cols(feat_test_B)
pipelines['B_stat'] = {'train': feat_B, 'test': feat_test_B, 'feat_cols': feat_cols_B}
log.info(f"  B_stat: {len(pipelines['B_stat']['feat_cols'])} features")

# Pipeline C: temporal features
log.info("Building C_temporal...")
feat_C = feat.copy()
feat_cols_C = get_feat_cols(feat_C)
feat_C['doe'] = pd.to_datetime(feat_C['sleep_date']).dt.dayofyear
feat_C['dow'] = pd.to_datetime(feat_C['sleep_date']).dt.dayofweek
feat_C['is_weekend'] = (feat_C['dow'] >= 5).astype(int)
feat_C['sin_doe'] = np.sin(2*np.pi * feat_C['doe']/365.25)
feat_C['cos_doe'] = np.cos(2*np.pi * feat_C['doe']/365.25)
feat_C['sin_dow'] = np.sin(2*np.pi * feat_C['dow']/7)
feat_C['cos_dow'] = np.cos(2*np.pi * feat_C['dow']/7)
# Weekend ratio per subject
wr = feat_C.groupby('subject_id')['is_weekend'].mean().reset_index().rename(columns={'is_weekend':'weekend_ratio'})
feat_C = feat_C.merge(wr, on='subject_id', how='left')
new_temporal = ['doe','dow','is_weekend','sin_doe','cos_doe','sin_dow','cos_dow','weekend_ratio']
feat_C_t = feat_test.copy()
feat_C_t['doe'] = pd.to_datetime(feat_C_t['sleep_date']).dt.dayofyear
feat_C_t['dow'] = pd.to_datetime(feat_C_t['sleep_date']).dt.dayofweek
feat_C_t['is_weekend'] = (feat_C_t['dow'] >= 5).astype(int)
feat_C_t['sin_doe'] = np.sin(2*np.pi * feat_C_t['doe']/365.25)
feat_C_t['cos_doe'] = np.cos(2*np.pi * feat_C_t['doe']/365.25)
feat_C_t['sin_dow'] = np.sin(2*np.pi * feat_C_t['dow']/7)
feat_C_t['cos_dow'] = np.cos(2*np.pi * feat_C_t['dow']/7)
wr_t = feat_C_t.groupby('subject_id')['is_weekend'].mean().reset_index().rename(columns={'is_weekend':'weekend_ratio'})
feat_C_t = feat_C_t.merge(wr_t, on='subject_id', how='left')
pipelines['C_temporal'] = {'train': feat_C, 'test': feat_C_t, 'feat_cols': feat_cols_C + new_temporal}
log.info(f"  C_temporal: {len(pipelines['C_temporal']['feat_cols'])} features")

# Pipeline D: cross-feature interactions
log.info("Building D_interaction...")
feat_D = feat.copy()
feat_cols_D = get_feat_cols(feat_D)
key_feats = [c for c in feat_cols_D if any(x in c for x in ['activity','pedo_step','pedo_distance','screen_use','light_mean','ambience'])]
key_feats = key_feats[:10]
added_d = []
for i in range(min(len(key_feats), 6)):
    for j in range(i+1, min(len(key_feats), 6)):
        f1, f2 = key_feats[i], key_feats[j]
        if f1 not in feat_D.columns or f2 not in feat_D.columns: continue
        v1, v2 = feat_D[f1].fillna(0).values, feat_D[f2].fillna(0).values
        feat_D[f'{f1}_x_{f2}'] = v1 * v2
        feat_D[f'{f1}_diff_{f2}'] = np.abs(v1 - v2)
        added_d.extend([f'{f1}_x_{f2}', f'{f1}_diff_{f2}'])
for f in key_feats[:8]:
    if f in feat_D.columns:
        vals = feat_D[f].fillna(0).values
        feat_D[f'{f}_log1p'] = np.log1p(np.maximum(vals, 0))
        added_d.append(f'{f}_log1p')
feat_D_t = feat_test.copy()
feat_cols_D_t = get_feat_cols(feat_D_t)
for i in range(min(len(key_feats), 6)):
    for j in range(i+1, min(len(key_feats), 6)):
        f1, f2 = key_feats[i], key_feats[j]
        if f1 not in feat_D_t.columns or f2 not in feat_D_t.columns: continue
        v1, v2 = feat_D_t[f1].fillna(0).values, feat_D_t[f2].fillna(0).values
        feat_D_t[f'{f1}_x_{f2}'] = v1 * v2
        feat_D_t[f'{f1}_diff_{f2}'] = np.abs(v1 - v2)
for f in key_feats[:8]:
    if f in feat_D_t.columns:
        vals = feat_D_t[f].fillna(0).values
        feat_D_t[f'{f}_log1p'] = np.log1p(np.maximum(vals, 0))
pipelines['D_interaction'] = {'train': feat_D, 'test': feat_D_t, 'feat_cols': feat_cols_D + added_d}
log.info(f"  D_interaction: {len(pipelines['D_interaction']['feat_cols'])} features")

# Pipeline E: rolling window aggregates
log.info("Building E_rolling...")
feat_E = feat.copy()
feat_cols_E = get_feat_cols(feat_E)
for col in feat_cols_E[:8]:
    vals = feat_E[col].fillna(0).values
    for w in [3, 7]:
        r = pd.Series(vals).rolling(w, min_periods=1)
        feat_E[f'{col}_r{w}_mean'] = r.mean().values
        feat_E[f'{col}_r{w}_std'] = r.std().fillna(0).values
    for alpha in [0.2, 0.5, 0.8]:
        ema = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean()
        feat_E[f'{col}_ema{alpha}'] = ema.values

feat_E_t = feat_test.copy()
feat_cols_E_t = get_feat_cols(feat_E_t)
for col in feat_cols_E_t[:8]:
    vals = feat_E_t[col].fillna(0).values
    for w in [3, 7]:
        r = pd.Series(vals).rolling(w, min_periods=1)
        feat_E_t[f'{col}_r{w}_mean'] = r.mean().values
        feat_E_t[f'{col}_r{w}_std'] = r.std().fillna(0).values
    for alpha in [0.2, 0.5, 0.8]:
        ema = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean()
        feat_E_t[f'{col}_ema{alpha}'] = ema.values

rolling_added = []
for col in feat_cols_E[:8]:
    for w in [3, 7]:
        rolling_added.extend([f'{col}_r{w}_mean', f'{col}_r{w}_std'])
    for alpha in [0.2, 0.5, 0.8]:
        rolling_added.append(f'{col}_ema{alpha}')
pipelines['E_rolling'] = {'train': feat_E, 'test': feat_E_t, 'feat_cols': feat_cols_E + rolling_added}
log.info(f"  E_rolling: {len(pipelines['E_rolling']['feat_cols'])} features")

log.info(f"Pipeline build time: {time.time()-t0:.0f}s")

# ============================================================
# Train and evaluate
# ============================================================
log.info("\n=== Training pipelines ===")
results = {}
pipeline_oofs = {}

for target in TARGETS:
    log.info(f"\n--- Target {target} ---")
    y = y_train[target]
    t_cfg = V53_SWEEP[target]
    cfg_name = t_cfg['cfg']
    n_feat = t_cfg['n_feat']
    cfg = CFGS[cfg_name]
    
    oof_dict = {}
    test_dict = {}
    
    for pname, pdata in pipelines.items():
        train_df = pdata['train']
        test_df = pdata['test']
        feat_cols = pdata['feat_cols']
        log.info(f"  {pname}: {len(feat_cols)} features")
        
        # Rank features on THIS pipeline's dataframe
        ranked = rank_features_on_df(train_df, feat_cols, target)
        sel_cols = ranked[:n_feat]
        
        # Train
        oof, test_p = train_cv_on_df(train_df, test_df, sel_cols, y, SEEDS, cfg)
        oof_avg = oof.mean(axis=1)
        test_avg = test_p.mean(axis=1)
        
        # Mean match
        oof_cal = mean_match(oof_avg, train_rates[target])
        test_cal = mean_match(test_avg, train_rates[target])
        
        ll = log_loss(y, oof_cal, labels=[0,1])
        oof_dict[pname] = oof_cal
        test_dict[pname] = test_cal
        pipeline_oofs[f"{pname}_{target}"] = ll
        log.info(f"    {pname}: OOF={ll:.5f} test_mean={test_cal.mean():.4f}")
    
    # Ensemble: mean blending
    pred_matrix = np.column_stack([oof_dict[p] for p in sorted(oof_dict.keys())])
    best_w, best_ll = optimize_weights(pred_matrix, y)
    ens_oof = mean_match(pred_matrix @ best_w, train_rates[target])
    ens_ll = log_loss(y, ens_oof, labels=[0,1])
    test_ens = mean_match(pred_matrix @ best_w, train_rates[target])
    
    # Rank blending
    ranks = {p: pd.Series(oof_dict[p]).rank() for p in oof_dict}
    rank_avg = pd.DataFrame(ranks).mean(axis=1).values
    rank_cal = mean_match(rank_avg, train_rates[target])
    rank_ll = log_loss(y, rank_cal, labels=[0,1])
    
    results[target] = {
        'pipeline_oofs': {k: round(float(v.mean()) if hasattr(v,'mean') else v, 5) for k,v in oof_dict.items()},
        'ensemble_mean': {'ll': round(ens_ll, 5), 'weights': [round(float(w),3) for w in best_w]},
        'ensemble_rank': {'ll': round(rank_ll, 5)},
        'best_method': 'mean' if ens_ll < rank_ll else 'rank',
        'test_preds': test_ens if ens_ll < rank_ll else rank_cal,
    }
    best_ll = min(ens_ll, rank_ll)
    log.info(f"  {target}: mean_blend={ens_ll:.5f} (w={best_w.round(3)}), rank_blend={rank_ll:.5f}, best={best_ll:.5f}")

# ============================================================
# Summary
# ============================================================
avg_oof = 0
for t in TARGETS:
    avg_oof += min(results[t]['ensemble_mean']['ll'], results[t]['ensemble_rank']['ll'])
avg_oof /= len(TARGETS)

log.info(f"\n{'='*70}")
log.info("V256: DIVERSE PIPELINE ENSEMBLE — SUMMARY")
log.info(f"{'='*70}")

for t in TARGETS:
    r = results[t]
    log.info(f"  {t}: pipe={r['pipeline_oofs']} ens={r['ensemble_mean']} rank={r['ensemble_rank']}")
log.info(f"  AVG_OOF: {avg_oof:.5f}")
log.info(f"  V127 baseline AVG_OOF: 0.53731")
log.info(f"  Delta: {avg_oof - 0.53731:+.5f}")

# Submission
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({t: results[t]['test_preds'] for t in TARGETS})
sub.insert(0, 'subject_id', feat_test['subject_id'].values)
sub.insert(1, 'sleep_date', feat_test['sleep_date'].values)
sub.insert(2, 'lifelog_date', feat_test['lifelog_date'].values)
sub_path = SUBMIT / f'submission_v256_{ts}.csv'
sub.to_csv(sub_path, index=False)

exp_log = {
    'version': 'V256',
    'timestamp': ts,
    'description': '5 diverse feature pipeline ensemble + optimal blending',
    'pipelines': list(pipelines.keys()),
    'pipeline_features': {k: len(v['feat_cols']) for k,v in pipelines.items()},
    'pipeline_build_time_s': round(time.time()-t0, 0),
    'per_target': {t: {k: v for k,v in r.items() if k != 'test_preds'} for t,r in results.items()},
    'avg_oof': round(avg_oof, 5),
    'baseline_v127_oof': 0.53731,
    'delta': round(avg_oof - 0.53731, 5),
    'submission_file': str(sub_path),
    'total_time_s': round(time.time()-t0, 0),
}
exp_path = EXPERIMENTS / f'v256_{ts}.json'
with open(exp_path, 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved: {sub_path}, log: {exp_path}")
log.info(f"Done in {time.time()-t0:.0f}s")
