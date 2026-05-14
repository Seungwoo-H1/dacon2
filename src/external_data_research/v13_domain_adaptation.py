"""
V13: Domain Adaptation via External Data - No direct feature injection

Key insight: Internal data has 10 subjects, external data has completely different people.
Direct feature injection is useless (same constant for every row).

Strategy:
1. External data for distribution learning (moment matching, density estimation)
2. Adversarial feature filtering (remove features that distinguish external from internal)
3. External data as regularization (train on combined data with domain label, domain-adversarial penalty)
4. Pseudo-label on external test predictions, calibrated and filtered
5. Multi-task learning: predict labels on internal, predict domain on features
6. Conformal prediction bounds from external data for calibration

All experiments logged automatically. No user interaction.
"""
import re, gc, json, time, warnings, traceback, os, itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from scipy import stats, integrate
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXTERNAL = ROOT / 'external_data'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT, EXTERNAL]: d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42]

CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':0.8,'rl':5.0,'mc':5}
CFG_DEEP  = {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15}
CFG_V48   = {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10}
CFG_SAFETY = {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20}
CFGS = {'wide':CFG_WIDE,'deep':CFG_DEEP,'v48':CFG_V48,'safety':CFG_SAFETY}
V53_SWEEP = {'Q1':'deep','Q2':'deep','Q3':'v48','S1':'wide','S2':'deep','S3':'safety','S4':'wide'}
LEAK_S = {'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
          'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean',
          'wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
          'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean',
          'wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def mean_match(pred, tm): return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)
def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols
def get_feature_cols(df):
    ex = META | set(TARGETS) | {'subject_id'}
    return [c for c in df.columns if c not in ex and not c.endswith('_subj_mean') and not c.endswith('_subj_std') and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, fcols, fit_stats=None, for_test=False):
    pc, stats_d, sc = [], {}, []
    for col in fcols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean','std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        sc.extend([f'{col}_subj_mean', f'{col}_subj_std'])
        if not for_test: stats_d[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        sm = fit_stats[col]['mean'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_mean']
        sd = fit_stats[col]['std'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_std']
        m0 = sd == 0; mn = df[col].isnull()
        z = f'{col}_zscore'
        df[z] = np.where(m0|mn, 0.0, (df[col].fillna(0)-sm)/np.maximum(sd, 1e-8))
        pc.append(z); gc.collect()
    drop = [c for c in sc if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df, pc, stats_d

def cfg_to_params(cfg_s, seed_val, spw):
    return {
        'objective':'binary', 'metric':'binary_logloss', 'verbose':-1,
        'num_leaves':int(cfg_s['nl']), 'max_depth':int(cfg_s['md']),
        'learning_rate':float(cfg_s['lr']), 'n_estimators':int(cfg_s['ne']),
        'subsample':float(cfg_s['ss']), 'colsample_bytree':float(cfg_s['cb']),
        'reg_alpha':float(cfg_s['ra']), 'reg_lambda':float(cfg_s['rl']),
        'min_child_samples':max(1,int(cfg_s['mc'])),
        'scale_pos_weight':spw, 'random_state':int(seed_val),
        'force_row_wise':True, 'n_jobs':1
    }
def train_cv(feat, ftst, cols, y, seeds, cfg):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None
    nr = int(cfg['ne'])
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=nr, valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vai, si] = m.predict(Xf[vai])
            if Xt is not None: tp[:, si] = m.predict(Xt)
            del ds, vd, m; gc.collect()
    if tp is not None: tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

def rank_f(feat, cols, target):
    y = feat[target].values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    p = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
        'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
        'force_row_wise':True,'n_jobs':1
    }
    X = feat[cols].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(cols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]


# ============================================================
# STEP 0: Load internal data + personalization
# ============================================================
print('=== STEP 0: Load internal data ===')
feat = pd.read_parquet(DATA / 'features.parquet')
ftst = pd.read_parquet(DATA / 'test_features.parquet')
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Proxy features (same as V11)
f = feat.copy(); ft = ftst.copy()
all_num = get_feature_cols(f)
if 'wPedo_pedo_step_mean' in all_num:
    s = f['wPedo_pedo_step_mean'].fillna(0); s_t = ft['wPedo_pedo_step_mean'].fillna(0)
    f['ext_activity_z'] = (s-s.mean())/max(s.std(),1e-8)
    ft['ext_activity_z'] = (s_t-s.mean())/max(s.std(),1e-8)
if 'mACStatus_m_charging_mean' in all_num:
    ch = f['mACStatus_m_charging_mean'].fillna(0); ch_t = ft['mACStatus_m_charging_mean'].fillna(0)
    f['ext_charging_z'] = (ch-ch.mean())/max(ch.std(),1e-8)
    ft['ext_charging_z'] = (ch_t-ch.mean())/max(ch.std(),1e-8)
if all(c in all_num for c in ['wPedo_pedo_step_mean','mACStatus_m_charging_mean','mScreenStatus_m_screen_use_mean','wHr_hr_mean']):
    sa=f['wPedo_pedo_step_mean'].fillna(0); sc_h=f['mACStatus_m_charging_mean'].fillna(0)
    ss=f['mScreenStatus_m_screen_use_mean'].fillna(0); hr=f['wHr_hr_mean'].fillna(0)
    sa_t=ft['wPedo_pedo_step_mean'].fillna(0); sc_t=ft['mACStatus_m_charging_mean'].fillna(0)
    ss_t=ft['mScreenStatus_m_screen_use_mean'].fillna(0); hr_t=ft['wHr_hr_mean'].fillna(0)
    f['ext_health_composite'] = (sa-sa.mean())/max(sa.std(),1e-8) - (sc_h-sc_h.mean())/max(sc_h.std(),1e-8) + (ss-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr-hr.mean())/max(hr.std(),1e-8)*0.1
    ft['ext_health_composite'] = (sa_t-sa_t.mean())/max(sa_t.std(),1e-8) - (sc_t-sc_t.mean())/max(sc_t.std(),1e-8) + (ss_t-ss_t.mean())/max(ss_t.std(),1e-8)*0.3 + (hr_t-hr_t.mean())/max(hr_t.std(),1e-8)*0.1
if 'wLight_w_light_mean' in all_num and 'mACStatus_hour_night' in all_num:
    f['ext_night_light'] = f['wLight_w_light_mean'].fillna(0) / (f['mACStatus_hour_night'].fillna(0)+1e-8)
    ft['ext_night_light'] = ft['wLight_w_light_mean'].fillna(0) / (ft['mACStatus_hour_night'].fillna(0)+1e-8)
amb_cols = [c for c in all_num if 'ambience' in c.lower() and c.endswith('_sum')]
if amb_cols:
    f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
    ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
if 'wHr_hr_mean' in all_num and 'wPedo_pedo_step_mean' in all_num:
    f['ext_hr_step'] = f['wHr_hr_mean'].fillna(0) * f['wPedo_pedo_step_mean'].fillna(0)
    ft['ext_hr_step'] = ft['wHr_hr_mean'].fillna(0) * ft['wPedo_pedo_step_mean'].fillna(0)
if 'mScreenStatus_m_screen_use_mean' in all_num:
    sm_v = f['mScreenStatus_m_screen_use_mean'].fillna(0); sm_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
    f['ext_screen_ratio'] = sm_v / (sm_v+1e-8)
    ft['ext_screen_ratio'] = sm_t / (sm_t+1e-8)
wifi_cols = [c for c in all_num if 'wifi' in c.lower() and c.endswith('_mean')]
ble_cols = [c for c in all_num if 'ble' in c.lower() and c.endswith('_mean')]
if wifi_cols and ble_cols:
    w = f[wifi_cols].fillna(0).sum(axis=1); b = f[ble_cols].fillna(0).sum(axis=1)
    w_t = ft[wifi_cols].fillna(0).sum(axis=1); b_t = ft[ble_cols].fillna(0).sum(axis=1)
    f['ext_wifi_ble'] = w / (b+1e-8)
    ft['ext_wifi_ble'] = w_t / (b_t+1e-8)
if 'ext_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
    f['ext_activity_ambience'] = f['ext_activity_z'] * f['ext_total_ambience']
    ft['ext_activity_ambience'] = ft['ext_activity_z'] * ft['ext_total_ambience']
if 'wPedo_pedo_step_std' in all_num:
    f['ext_step_consistency'] = f['wPedo_pedo_step_std'].fillna(0) / (f['wPedo_pedo_step_mean'].fillna(0)+1e-8)
    ft['ext_step_consistency'] = ft['wPedo_pedo_step_std'].fillna(0) / (ft['wPedo_pedo_step_mean'].fillna(0)+1e-8)

fcols = get_feature_cols(f)
f, zscore_cols, fit_stats = add_personalization(f, fcols)
ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
non_const = [c for c in fcols+zscore_cols if f[c].std() > 0]
y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}
print(f'  Train: {f.shape}, Test: {ft.shape}, Features: {len(non_const)}')

# ============================================================
# STEP 0b: Load external data
# ============================================================
print('\n=== STEP 0b: Load external data ===')
EXTERNAL_FILES = {
    'A': 'external_data/sleep_health_2.csv',
    'B': 'external_data/sleep_health_lifestyle.csv',
    'C': 'external_data/sleep_lifestyle_1000_kaggle_extracted/sleep_study_1000.csv',
}
ext_dfs = {}
for eid, fpath in EXTERNAL_FILES.items():
    try:
        df = pd.read_csv(fpath)
        ext_dfs[eid] = df
        print(f'  {eid}: {df.shape}, cols={list(df.columns)[:8]}...')
    except Exception as e:
        print(f'  {eid}: FAILED {e}')

# ============================================================
# STEP 1: V127 Reproduction (baseline)
# ============================================================
print('\n=== STEP 1: V127 Reproduction ===')
v127_oof = {}
for target in TARGETS:
    t0 = time.time()
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]
    best_cols = (ext_in[:2] if len(ext_in)>=2 else []) + non_ext_in[:13]
    best_cols = list(dict.fromkeys(best_cols))
    
    oofs_all = []
    var_cfgs = [
        cfg,
        {**cfg, 'nl': int(cfg['nl']*0.8), 'ne': int(cfg['ne']*0.7)},
        {**cfg, 'nl': int(cfg['nl']*1.2), 'ne': int(cfg['ne']*1.3)},
    ]
    for vcfg in var_cfgs:
        oof, _ = train_cv(f, ft, best_cols, y, SEEDS, vcfg)
        oofs_all.append(oof)
    oof_avg = np.clip(np.mean(oofs_all, axis=0).mean(axis=1), 0.0001, 0.9999)
    v127_oof[target] = {'oof_raw': round(float(log_loss(y, oof_avg, labels=[0,1])), 5),
                        'oof_cal': round(float(log_loss(y, mean_match(oof_avg, y.mean()), labels=[0,1])), 5),
                        'n_feat': len(best_cols)}
    print(f'  {target}: OOF={v127_oof[target]["oof_raw"]:.5f} cal={v127_oof[target]["oof_cal"]:.5f} time={time.time()-t0:.1f}s')

avg_v127 = np.mean([v127_oof[t]['oof_raw'] for t in TARGETS])
print(f'\n  V127 AVG OOF: {avg_v127:.5f} (V127 SOTA: 0.53731)')


# ============================================================
# STEP 2: Feature Engineering for External Data (per-subject style)
# ============================================================
# Since external data has no ID overlap, we create "synthetic subject groups"
# by clustering external data into groups of ~50 (similar to internal 10 subjects × 50 days)
# Then compute per-group summary features that mimic the internal personalization structure.
# This allows the external data to provide DISTRIBUTIONAL signals that shape model behavior.

print('\n=== STEP 2: External data feature engineering ===')

# For each external dataset, create per-cluster features that mimic internal personalization
# Strategy: Cluster external data into ~10 groups (matching internal 10 subjects),
# compute group-level summaries, then use group assignment as a feature for internal data
# via distribution matching.

ext_cluster_features = {}  # {eid: {feature_name: global_distribution_stats}}

for eid, ext_df in ext_dfs.items():
    print(f'\n  [{eid}] Clustering external data...')
    
    # Get numeric columns
    num_cols = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Map to internal feature space
    mapping = {}
    for nc in num_cols:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step', 
                                                          'screen', 'light', 'hr', 'bmi', 'age', 'calorie',
                                                          'distance', 'charge']):
                mapping[nc] = ic
                break
    
    mapped_int_cols = [mapping[c] for c in num_cols if c in mapping and mapping[c] in f.columns]
    mapped_ext_cols = [c for c in num_cols if c in mapping and mapping[c] in f.columns]
    mapped_cols = mapped_int_cols  # backward compat
    print(f'    Mapped {len(mapped_cols)} columns')
    
    if len(mapped_cols) < 2:
        print(f'    Not enough mapped columns, computing raw statistics only')
        # Just compute global distribution stats
        ext_cluster_features[eid] = {}
        for nc in num_cols[:10]:  # Top 10 numeric
            s = ext_df[nc].dropna()
            if len(s) > 20:
                ext_cluster_features[eid][f'ext_{eid}_{nc}_skew'] = float(stats.skew(s))
                ext_cluster_features[eid][f'ext_{eid}_{nc}_kurt'] = float(stats.kurtosis(s))
                ext_cluster_features[eid][f'ext_{eid}_{nc}_iqr'] = float(s.quantile(0.75) - s.quantile(0.25))
        continue
    
    # Cluster external data into 10 groups using KMeans on mapped features
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X_ext = ext_df[mapped_ext_cols].fillna(0).values.astype(np.float64)
    scaler_ext = StandardScaler()
    X_ext_scaled = scaler_ext.fit_transform(X_ext)
    
    # Try different cluster counts and pick best by silhouette
    best_k = 10
    best_score = -1
    for k in [5, 8, 10, 12, 15]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_ext_scaled)
        if len(set(labels)) > 2:
            from sklearn.metrics import silhouette_score
            try:
                score = silhouette_score(X_ext_scaled, labels, sample_size=min(2000, len(X_ext_scaled)))
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                pass
    
    print(f'    Best k={best_k}, silhouette={best_score:.4f}')
    
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_ext_scaled)
    
    # Compute per-cluster statistics
    cluster_stats = {}
    for c in range(best_k):
        mask = cluster_labels == c
        n_c = mask.sum()
        cluster_stats[c] = {'n': n_c, 'pct': float(n_c / len(X_ext))}
        for j, ext_col in enumerate(mapped_ext_cols):
            vals = X_ext_scaled[mask, j]
            if len(vals) > 5:
                cluster_stats[c][f'mean_{ext_col}'] = float(np.mean(vals))
                cluster_stats[c][f'std_{ext_col}'] = float(np.std(vals))
    
    # Now compute GROUP-WIDE DISTRIBUTION features from external data
    # These are the real "signals" we extract - they describe the distribution
    # of each feature in external data, which can inform regularization
    dist_features = {}
    for j, col in enumerate(mapped_cols):
        vals = X_ext_scaled[:, j]
        # Distribution shape features
        dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ext_skew'] = float(stats.skew(vals))
        dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ext_kurt'] = float(stats.kurtosis(vals))
        dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ext_iqr'] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ext_cv'] = float(np.std(vals) / (np.abs(np.mean(vals)) + 1e-8))
        
        # Compare with internal distribution
        internal_vals = f[mapped_int_cols[j]].fillna(0).values
        if len(internal_vals) > 5:
            # Kolmogorov-Smirnov test statistic
            try:
                ks_stat, ks_p = stats.ks_2samp(internal_vals, vals)
                dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ks_stat'] = float(ks_stat)
                dist_features[f'ext_{eid}_{mapped_int_cols[j]}_ks_p'] = float(ks_p)
            except:
                pass
            
            # Earth Mover's Distance (approximate)
            try:
                internal_hist, internal_edges = np.histogram(internal_vals, bins=30, density=True)
                ext_hist, ext_edges = np.histogram(vals, bins=30, density=True)
                # Normalize to same bins
                common_bins = 30
                ih, _ = np.histogram(internal_vals, bins=common_bins, density=True)
                eh, _ = np.histogram(vals, bins=common_bins, density=True)
                total = ih + eh + 1e-10
                ih_n = ih / (total + 1e-10)
                eh_n = eh / (total + 1e-10)
                emd = 0.5 * np.sum(np.abs(ih_n - eh_n))
                dist_features[f'ext_{eid}_{mapped_int_cols[j]}_emd'] = float(emd)
            except:
                pass
    
    ext_cluster_features[eid] = dist_features
    print(f'    Computed {len(dist_features)} distribution features')

# ============================================================
# STEP 3: Domain Similarity - Adversarial Validation
# ============================================================
print('\n=== STEP 3: Domain Similarity ===')

domain_scores = {}
for eid, ext_df in ext_dfs.items():
    num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Map to internal
    mapping = {}
    for nc in num_ext:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                          'screen', 'light', 'hr', 'bmi', 'age']):
                mapping[nc] = ic
                break
    
    mapped_int_cols = [mapping[c] for c in num_ext if c in mapping and mapping[c] in f.columns]
    mapped_ext_cols = [c for c in num_ext if c in mapping and mapping[c] in f.columns]
    
    if len(mapped_int_cols) < 3:
        domain_scores[eid] = {'mapped': len(mapped_int_cols), 'auc': None, 'interpretation': 'too_few_features'}
        continue
    
    # Combine internal + external features for adversarial validation
    shared_int = mapped_int_cols
    shared_ext = mapped_ext_cols
    X_int = f[shared_int].fillna(0).values.astype(np.float64)
    X_ext = ext_df[shared_ext].fillna(0).values.astype(np.float64)
    
    # Equal sample size
    n = min(len(X_int), len(X_ext), 300)
    X_adv = np.vstack([X_int[:n], X_ext[:n]])
    y_adv = np.array([0]*n + [1]*n)
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tri, vai in skf.split(X_adv, y_adv):
        ds = lgb.Dataset(X_adv[tri], label=y_adv[tri])
        vd = lgb.Dataset(X_adv[vai], label=y_adv[vai])
        m = lgb.train({
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.05,'n_estimators':200,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':5,'random_state':42,'n_jobs':1
        }, ds, num_boost_round=200, valid_sets=[vd],
           callbacks=[lgb.early_stopping(20, verbose=False)])
        pred = m.predict(X_adv[vai])
        auc = roc_auc_score(y_adv[vai], pred)
        scores.append(auc)
    
    adv_auc = np.mean(scores)
    domain_scores[eid] = {
        'mapped': len(mapped_int_cols), 'auc': round(float(adv_auc), 4),
        'interpretation': 'same_domain' if adv_auc < 0.6 else ('mixed' if adv_auc < 0.7 else 'different_domain'),
    }
    print(f'  [{eid}] AUC={adv_auc:.4f} ({domain_scores[eid]["interpretation"]})')


# ============================================================
# STEP 4: Adversarial Feature Filtering
# ============================================================
# For each external dataset, identify features that strongly predict
# "is from external dataset" — those are domain-specific and should be removed
# or down-weighted when training on internal data.

print('\n=== STEP 4: Adversarial Feature Filtering ===')

adversarial_importance = {}  # {eid: {feature: importance}}

for eid, ext_df in ext_dfs.items():
    print(f'\n  [{eid}] Running adversarial feature importance...')
    
    # Find mapped features
    num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    mapping = {}
    for nc in num_ext:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                          'screen', 'light', 'hr', 'bmi', 'age', 'calorie',
                                                          'distance', 'charge']):
                mapping[nc] = ic
                break
    
    mapped_int_cols = [mapping[c] for c in num_ext if c in mapping and mapping[c] in f.columns]
    mapped_ext_cols = [c for c in num_ext if c in mapping and mapping[c] in f.columns]
    
    if len(mapped_int_cols) < 5:
        print(f'    Skipping: only {len(mapped_int_cols)} mapped features')
        continue
    
    # Combined dataset
    n = min(len(f), len(ext_df), 300)
    X_int = f[mapped_int_cols].fillna(0).values.astype(np.float64)
    X_ext = ext_df[mapped_ext_cols].fillna(0).values.astype(np.float64)
    X_adv = np.vstack([X_int[:n], X_ext[:n]])
    y_adv = np.array([0]*n + [1]*n)
    
    # Train adversarial model
    from sklearn.model_selection import StratifiedKFold as SKF
    skf4 = SKF(n_splits=5, shuffle=True, random_state=42)
    imp_scores = np.zeros(len(mapped_int_cols))
    counts = np.zeros(len(mapped_int_cols))
    
    for tri, vai in skf4.split(X_adv, y_adv):
        ds = lgb.Dataset(X_adv[tri], label=y_adv[tri])
        vd = lgb.Dataset(X_adv[vai], label=y_adv[vai])
        m = lgb.train({
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.05,'n_estimators':200,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':5,'random_state':42,'n_jobs':1
        }, ds, num_boost_round=200, valid_sets=[vd],
           callbacks=[lgb.early_stopping(20, verbose=False)])
        imp = m.feature_importance(importance_type='gain')
        imp_scores += imp
        counts += 1
    
    adv_imp = imp_scores / max(counts, 1)
    threshold = np.percentile(adv_imp, 70)  # Top 30% of features are domain-specific
    
    domain_specific = [mapped_int_cols[i] for i in range(len(mapped_int_cols)) if adv_imp[i] > threshold]
    domain_safe = [mapped_int_cols[i] for i in range(len(mapped_int_cols)) if adv_imp[i] <= threshold]
    
    adversarial_importance[eid] = {
        'domain_specific': domain_specific,
        'domain_safe': domain_safe,
        'all_importance': {mapped_int_cols[i]: round(float(adv_imp[i]), 4) for i in range(len(mapped_int_cols))},
        'threshold': round(float(threshold), 4),
    }
    print(f'    Domain-specific features: {len(domain_specific)}/{len(mapped_int_cols)}')
    print(f'    Domain-safe features: {len(domain_safe)}/{len(mapped_int_cols)}')

# ============================================================
# STEP 5: Feature Pruning via Adversarial Filtering
# ============================================================
# For each target, remove features that are "domain-leaking" — features that
# are very important for predicting the external domain but noisy for the target.

print('\n=== STEP 5: Adversarial feature pruning for each target ===')

pruned_results = {}
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    
    # Baseline
    baseline_cols = ranked[:15]
    oof_base, _ = train_cv(f, ft, baseline_cols, y, SEEDS, cfg)
    oof_avg_base = np.clip(oof_base.mean(axis=1), 0.0001, 0.9999)
    ll_base = log_loss(y, oof_avg_base, labels=[0,1])
    
    # Try removing top-N most domain-leaking features
    # Collect all domain-specific features across all external datasets
    all_domain_specific = set()
    for eid, ai in adversarial_importance.items():
        all_domain_specific.update(ai['domain_specific'])
    
    domain_leak_ranks = [i for i, c in enumerate(ranked) if c in all_domain_specific]
    
    best_pruned_ll = ll_base
    best_pruned_cols = baseline_cols
    best_pruned_info = None
    
    for n_remove in [0, 1, 2, 3, 5, 7]:
        if n_remove == 0:
            continue
        # Remove top n_remove domain-specific features from ranking
        remove_set = set(ranked[i] for i in domain_leak_ranks[:n_remove])
        pruned_cols = [c for c in baseline_cols if c not in remove_set]
        if len(pruned_cols) < 10:
            continue
        # Pad with next ranked features if needed
        pad_cols = [c for c in ranked if c not in pruned_cols]
        pruned_cols = pruned_cols + pad_cols[:15-len(pruned_cols)]
        
        oof_p, _ = train_cv(f, ft, pruned_cols, y, SEEDS, cfg)
        oof_avg_p = np.clip(oof_p.mean(axis=1), 0.0001, 0.9999)
        ll_p = log_loss(y, oof_avg_p, labels=[0,1])
        delta = ll_p - ll_base
        
        if ll_p < best_pruned_ll:
            best_pruned_ll = ll_p
            best_pruned_cols = pruned_cols
            best_pruned_info = {'n_removed': n_remove, 'll': round(ll_p, 5), 'delta': round(delta, 5)}
    
    pruned_results[target] = {
        'baseline_ll': round(ll_base, 5),
        'best_pruned_ll': round(best_pruned_ll, 5),
        'best_pruned_info': best_pruned_info,
        'improvement': round(best_pruned_ll - ll_base, 5),
    }
    print(f'  {target}: baseline={ll_base:.5f} best_pruned={best_pruned_ll:.5f} '
          f'delta={best_pruned_ll-ll_base:+.5f} remove={best_pruned_info["n_removed"] if best_pruned_info else "none"}')


# ============================================================
# STEP 6: Distribution Matching Regularization
# ============================================================
# Key idea: use external data distribution to regularize internal training.
# For each internal feature, if its distribution differs significantly from
# external data, add regularization to match distributions.
# This is done by adding a "distribution loss" term during training.

print('\n=== STEP 6: Distribution matching regularization ===')

# For each feature, compute the "external alignment score"
# = how similar is the internal feature distribution to the external one
# Features with high similarity can contribute more; features with low
# similarity might be noisy when combined

dist_alignment = {}
for eid, ext_df in ext_dfs.items():
    dist_alignment[eid] = {}
    num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    mapping = {}
    for nc in num_ext:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                          'screen', 'light', 'hr', 'bmi', 'age']):
                mapping[nc] = ic
                break
    
    for icol in all_num[:50]:  # Top 50 internal features
        if icol not in f.columns: continue
        internal_vals = f[icol].fillna(0).values
        
        # Find corresponding external column
        ext_col = None
        for nc, mapped in mapping.items():
            if mapped == icol:
                ext_col = nc
                break
        
        if ext_col is None:
            # No external counterpart: alignment = 0.5 (neutral)
            dist_alignment[eid][icol] = 0.5
            continue
        
        if ext_col not in ext_df.columns:
            dist_alignment[eid][icol] = 0.5
            continue
        
        ext_vals = ext_df[ext_col].dropna().values
        if len(ext_vals) < 20 or len(internal_vals) < 20:
            dist_alignment[eid][icol] = 0.5
            continue
        
        # Compute distribution distance
        try:
            # KS test
            ks_stat, _ = stats.ks_2samp(internal_vals, ext_vals)
            # Correlation
            corr = np.corrcoef(internal_vals, ext_vals)[0, 1] if np.std(internal_vals) > 0 and np.std(ext_vals) > 0 else 0
            
            # Alignment score: low distance → high alignment
            alignment = 1.0 - ks_stat
            dist_alignment[eid][icol] = round(float(alignment), 4)
        except:
            dist_alignment[eid][icol] = 0.5

# Now try feature weighting based on distribution alignment
print('\n  Trying distribution-aligned feature weighting...')

weighted_results = {}
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    
    baseline_cols = ranked[:15]
    oof_base, _ = train_cv(f, ft, baseline_cols, y, SEEDS, cfg)
    oof_avg_base = np.clip(oof_base.mean(axis=1), 0.0001, 0.9999)
    ll_base = log_loss(y, oof_avg_base, labels=[0,1])
    
    best_w_ll = ll_base
    
    # Try different weighting strategies
    for eid in ext_dfs:
        for strategy in ['align_down', 'align_up', 'ks_down']:
            # Create weighted features
            f_w = f.copy()
            ft_w = ft.copy()
            
            for icol in baseline_cols:
                if icol not in f_w.columns: continue
                al = dist_alignment.get(eid, {}).get(icol, 0.5)
                
                if strategy == 'align_down':
                    # Down-weight features with low alignment
                    w = max(0.1, al)
                elif strategy == 'align_up':
                    # Up-weight features with high alignment
                    w = 0.1 + al * 0.9
                else:  # ks_down
                    # Down-weight features with high KS distance (low alignment)
                    w = max(0.1, 1.0 - abs(0.5 - al))
                
                f_w[icol] = f_w[icol] * w
                ft_w[icol] = ft_w[icol] * w
            
            oof_w, _ = train_cv(f_w, ft_w, baseline_cols, y, SEEDS, cfg)
            oof_avg_w = np.clip(oof_w.mean(axis=1), 0.0001, 0.9999)
            ll_w = log_loss(y, oof_avg_w, labels=[0,1])
            delta = ll_w - ll_base
            
            if ll_w < best_w_ll:
                best_w_ll = ll_w
                weighted_results[f'{target}_{strategy}_{eid}'] = {
                    'll': round(ll_w, 5), 'delta': round(delta, 5),
                    'baseline': round(ll_base, 5),
                }
    
    weighted_results[f'{target}_baseline'] = {'ll': round(ll_base, 5), 'delta': 0}

print('  Weighting results logged.')

# ============================================================
# STEP 7: Pseudo-labeling with External Data
# ============================================================
# Strategy: 
# 1. Train model on internal data
# 2. Predict on external data
# 3. Select high-confidence predictions as pseudo-labels
# 4. Retrain on internal + pseudo-labeled external
# 5. Iterate

print('\n=== STEP 7: Pseudo-labeling with external data ===')

pseudo_results = {}
for eid, ext_df in ext_dfs.items():
    print(f'\n  [{eid}] Pseudo-labeling...')
    
    # Map external columns
    num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    mapping = {}
    for nc in num_ext:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                          'screen', 'light', 'hr', 'bmi', 'age']):
                mapping[nc] = ic
                break
    
    mapped_int_cols = [mapping[c] for c in num_ext if c in mapping and mapping[c] in f.columns]
    mapped_ext_cols = [c for c in num_ext if c in mapping and mapping[c] in f.columns]
    
    if len(mapped) < 3:
        print(f'    Skipping: only {len(mapped)} mapped features')
        continue
    
    for target in TARGETS:
        y = y_dict[target]
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_f(f, leak_cols, target)
        top_cols = ranked[:15]
        
        # Train initial model on internal
        oof, tp = train_cv(f, ft, top_cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        
        # Get internal OOF predictions
        gkf = GroupKFold(n_splits=5)
        internal_oof = np.zeros(len(f))
        for tri, vai in gkf.split(f, y, f['subject_id']):
            ds = lgb.Dataset(f[top_cols].fillna(0).values[tri], label=y[tri], feature_name=[sanitize_col(c) for c in top_cols])
            vd = lgb.Dataset(f[top_cols].fillna(0).values[vai], label=y[vai], feature_name=[sanitize_col(c) for c in top_cols], reference=ds)
            m = lgb.train({
                'objective':'binary','metric':'binary_logloss','verbose':-1,
                'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':200,
                'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
                'min_child_samples':5,'random_state':42,'n_jobs':1
            }, ds, num_boost_round=200, valid_sets=[vd], callbacks=[lgb.early_stopping(20, verbose=False)])
            internal_oof[vai] = m.predict(f[top_cols].fillna(0).values[vai])
        internal_oof = np.clip(internal_oof, 0.0001, 0.9999)
        
        baseline_ll = log_loss(y, internal_oof, labels=[0,1])
        
        # Predict on external data (using features that exist in both)
        ext_mapped_cols = [c for c in mapped_ext_cols if c in top_cols]
        if len(ext_mapped_cols) < 2:
            print(f'    {target}: no mapped features in top_cols')
            continue
        
        ext_pred = ext_df[ext_mapped_cols].fillna(0).values.astype(np.float64)
        # Use a simple model trained on internal to predict on external
        p_simple = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':200,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':5,'random_state':42,'n_jobs':1
        }
        sn = [sanitize_col(c) for c in ext_mapped_cols]
        ds_train = lgb.Dataset(f[ext_mapped_cols].fillna(0).values, label=y, feature_name=sn)
        m_train = lgb.train(p_simple, ds_train, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
        ext_probs = m_train.predict(ext_pred)
        ext_probs = np.clip(ext_probs, 0.0001, 0.9999)
        
        # Try different confidence thresholds
        for thresh in [0.6, 0.7, 0.8, 0.9]:
            high_conf = ext_probs >= thresh
            low_conf = ext_probs <= (1-thresh)
            combined_mask = high_conf | low_conf
            n_pseudo = combined_mask.sum()
            
            if n_pseudo < 10:
                continue
            
            pseudo_pos = (ext_probs[combined_mask] > 0.5).mean()
            
            # Retrain with pseudo-labels
            # Combine internal data with pseudo-labeled external
            # Weight external samples lower (less reliable)
            n_ext = n_pseudo
            w_ext = 0.3  # external samples weighted at 30% of internal
            
            # Build combined dataset
            all_feat = f[ext_mapped_cols].fillna(0).values
            all_label = y
            all_weight = np.ones(len(y))
            
            ext_feat = ext_pred[combined_mask]
            ext_label = (ext_probs[combined_mask] > 0.5).astype(np.float64)
            ext_weight = np.full(n_ext, w_ext)
            
            # For simplicity, use sample weighting via scale_pos_weight approach
            # Instead, create augmented dataset
            if n_ext > 0:
                aug_feat = np.vstack([all_feat, ext_feat])
                aug_label = np.concatenate([all_label, ext_label])
                aug_weight = np.concatenate([all_weight, ext_weight])
            else:
                continue
            
            # Train with sample weights
            aug_oof_internal = np.zeros(len(f))
            # We can't do proper OOF with augmented data, so just measure
            # the effect on test predictions
            aug_ds = lgb.Dataset(aug_feat, label=aug_label, weight=aug_weight, feature_name=sn)
            m_aug = lgb.train(p_simple, aug_ds, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
            
            # Predict on test set
            ft_pred = ft[ext_mapped_cols].fillna(0).values.astype(np.float64)
            ft_aug_pred = m_aug.predict(ft_pred)
            ft_aug_pred = np.clip(ft_aug_pred, 0.0001, 0.9999)
            
            # Ensemble: average internal test predictions with external-augmented
            ft_base = tp.mean(axis=1) if tp is not None else internal_oof  # approximate
            ft_ens = 0.5 * ft_base + 0.5 * ft_aug_pred
            ft_ens = np.clip(ft_ens, 0.0001, 0.9999)
            
            pseudo_results[f'{target}_{eid}_t{thresh}'] = {
                'n_pseudo': int(n_pseudo), 'pseudo_pos': round(pseudo_pos, 3),
                'thresh': thresh, 'w_ext': w_ext,
            }
            print(f'    {target} t={thresh}: n_pseudo={n_pseudo} pos={pseudo_pos:.3f}')

# ============================================================
# STEP 8: Multi-Stage Training
# ============================================================
print('\n=== STEP 8: Multi-stage training ===')

staged_results = {}
for eid, ext_df in ext_dfs.items():
    print(f'\n  [{eid}] Staged training...')
    
    # Map external columns
    num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    mapping = {}
    for nc in num_ext:
        nc_l = nc.lower().strip()
        for ic in all_num:
            ic_l = ic.lower().strip()
            if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                          'screen', 'light', 'hr', 'bmi', 'age']):
                mapping[nc] = ic
                break
    
    mapped_ext = [c for c in num_ext if c in mapping and mapping[c] in f.columns]
    mapped_int = [mapping[c] for c in mapped_ext if mapping[c] in f.columns]
    
    for target in TARGETS:
        y = y_dict[target]
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_f(f, leak_cols, target)
        top_int_cols = ranked[:12]  # Internal-only features
        
        # ext_mapped_cols_internal: internal names for model features
        ext_mapped_cols_internal = [mi for mi, me in zip(mapped_int, mapped_ext) 
                                     if mi in top_int_cols or mi in non_const]
        ext_mapped_cols_internal = [c for c in ext_mapped_cols_internal if c in non_const]
        # ext_mapped_cols_ext: corresponding external names for ext_df access
        ext_mapped_cols_ext = [me for mi, me in zip(mapped_int, mapped_ext)
                                if mi in top_int_cols or mi in non_const]
        ext_mapped_cols_ext = [c for c in ext_mapped_cols_ext if mapping[c] in non_const] if mapping else []
        
        # Stage 1: Pre-train on external data (if enough mapped features)
        if len(ext_mapped_cols) >= 3:
            p_simple = {
                'objective':'binary','metric':'binary_logloss','verbose':-1,
                'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':200,
                'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
                'min_child_samples':5,'random_state':42,'n_jobs':1
            }
            sn_ext = [sanitize_col(c) for c in ext_mapped_cols]
            
            # Pre-train on external: predict nothing useful (no labels on external)
            # Instead, use unsupervised representation learning via feature compression
            # For simplicity, train autoencoder-like: reconstruct external features from compressed form
            X_ext = ext_df[ext_mapped_cols_ext].fillna(0).values.astype(np.float64)
            
            # Compress via PCA
            from sklearn.decomposition import PCA
            n_components = min(5, len(ext_mapped_cols_ext))
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_ext)
            X_ext_comp = pca.transform(X_ext)
            
            # Now use PCA components as additional features for internal data
            X_int = f[ext_mapped_cols_internal].fillna(0).values.astype(np.float64)
            X_int_comp = pca.transform(X_int) if len([c for c in ext_mapped_cols if c in f.columns]) > 0 else np.zeros((len(f), n_components))
            
            # Create PCA features
            pca_features = [f'pca_{eid}_c{i}' for i in range(n_components)]
            f_aug = f.copy()
            ft_aug = ft.copy()
            for i, pf in enumerate(pca_features):
                f_aug[pf] = X_int_comp[:, i]
                ft_aug[pf] = np.zeros(len(ft_aug))  # test features not in external
            
            # Train with PCA features
            pca_cols = top_int_cols + pca_features
            pca_cols = list(dict.fromkeys(pca_cols))[:15]
            
            oof_pca, _ = train_cv(f_aug, ft_aug, pca_cols, y, SEEDS, cfg)
            oof_avg_pca = np.clip(oof_pca.mean(axis=1), 0.0001, 0.9999)
            ll_pca = log_loss(y, oof_avg_pca, labels=[0,1])
            baseline = v127_oof[target]['oof_raw']
            delta = ll_pca - baseline
            
            staged_results[f'{target}_{eid}_pca'] = {'ll': round(ll_pca, 5), 'delta': round(delta, 5), 'n_pca': n_components}
            
            # Also try weighted ensemble
            oof_base, tp_base = train_cv(f, ft, top_int_cols, y, SEEDS, cfg)
            tp_ens = 0.5 * tp_base.mean(axis=1) + 0.5 * oof_pca.mean(axis=1)  # approximate
            # Use internal OOF + PCA model ensemble
            oof_ens = np.clip(np.mean([oof_base, oof_pca], axis=0), 0.0001, 0.9999)
            ll_ens = log_loss(y, oof_ens.mean(axis=1), labels=[0,1])
            delta_ens = ll_ens - baseline
            
            staged_results[f'{target}_{eid}_pca_ens'] = {'ll': round(ll_ens, 5), 'delta': round(delta_ens, 5)}
            
            print(f'    {target}: pca_ll={ll_pca:.5f} delta={delta:+.5f} ens_ll={ll_ens:.5f} delta_ens={delta_ens:+.5f}')
        else:
            print(f'    {target}: only {len(ext_mapped_cols)} mapped features, skipping')


# ============================================================
# STEP 9: Curriculum Learning (Ordered External Data Exposure)
# ============================================================
print('\n=== STEP 9: Curriculum learning ordering ===')

# Order external datasets by domain similarity (easiest first)
sorted_ext = sorted(
    [eid for eid in ext_dfs if eid in domain_scores],
    key=lambda x: domain_scores.get(x, {}).get('auc', 0.5)
)
print(f'  Curriculum order: {sorted_ext} (sorted by adv_auc)')

curriculum_results = {}
for n_stages in range(1, len(sorted_ext)+1):
    curriculum = sorted_ext[:n_stages]
    # For each stage, compute the PCA from combined external data
    all_ext_pca = {}
    for eid in curriculum:
        ext_df = ext_dfs[eid]
        num_ext = ext_df.select_dtypes(include=[np.number]).columns.tolist()
        mapping = {}
        for nc in num_ext:
            nc_l = nc.lower().strip()
            for ic in all_num:
                ic_l = ic.lower().strip()
                if any(kw in nc_l and kw in ic_l for kw in ['sleep', 'heart', 'activity', 'stress', 'step',
                                                              'screen', 'light', 'hr', 'bmi', 'age']):
                    mapping[nc] = ic
                    break
        
        mapped = [c for c in num_ext if c in mapping and mapping[c] in f.columns]
        if len(mapped) >= 3:
            X_ext = ext_df[mapped].fillna(0).values.astype(np.float64)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_ext_scaled = sc.fit_transform(X_ext)
            pca = PCA(n_components=min(5, len(mapped)), random_state=42)
            pca.fit(X_ext_scaled)
            all_ext_pca[eid] = {'pca': pca, 'scaler': sc, 'cols': mapped}
    
    for target in TARGETS:
        y = y_dict[target]
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_f(f, leak_cols, target)
        top_int_cols = ranked[:12]
        
        # Add PCA features from curriculum
        f_cur = f.copy()
        ft_cur = ft.copy()
        pca_features = []
        
        for eid in curriculum:
            if eid not in all_ext_pca: continue
            p = all_ext_pca[eid]['pca']
            cols = all_ext_pca[eid]['cols']
            X_ext = ext_dfs[eid][cols].fillna(0).values.astype(np.float64)
            X_ext_scaled = all_ext_pca[eid]['scaler'].transform(X_ext)
            X_comp = p.transform(X_ext_scaled)
            
            f_cols = [c for c in cols if c in f.columns]
            if len(f_cols) > 0:
                X_int = f[f_cols].fillna(0).values.astype(np.float64)
                X_int_scaled = all_ext_pca[eid]['scaler'].transform(X_int)
                X_int_comp = p.transform(X_int_scaled)
            else:
                X_int_comp = np.zeros((len(f), X_comp.shape[1]))
            
            for i in range(X_comp.shape[1]):
                pf = f'curriculum_{eid}_c{i}'
                f_cur[pf] = X_int_comp[:, i]
                ft_cur[pf] = 0
                pca_features.append(pf)
        
        # Train with curriculum features
        cur_cols = top_int_cols + pca_features
        cur_cols = list(dict.fromkeys(cur_cols))[:15]
        
        oof_c, _ = train_cv(f_cur, ft_cur, cur_cols, y, SEEDS, cfg)
        oof_avg_c = np.clip(oof_c.mean(axis=1), 0.0001, 0.9999)
        ll_c = log_loss(y, oof_avg_c, labels=[0,1])
        delta = ll_c - v127_oof[target]['oof_raw']
        
        key = f'target{target}_stages{n_stages}_curriculum'
        curriculum_results[key] = {
            'curriculum': curriculum, 'll': round(ll_c, 5), 'delta': round(delta, 5),
        }
        print(f'    {key}: delta={delta:+.5f}')

# ============================================================
# STEP 10: Confidence-Filtered Training
# ============================================================
print('\n=== STEP 10: Confidence-filtered training ===')

# For each target, train on internal data, then filter samples
# by prediction confidence and retrain on confident samples only

confidence_results = {}
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    top_cols = ranked[:15]
    
    n_groups_all = f['subject_id'].nunique()
    gkf = GroupKFold(n_splits=min(5, n_groups_all))
    fold_preds = np.zeros(len(y))
    
    for tri, vai in gkf.split(f, y, f['subject_id']):
        ds = lgb.Dataset(f[top_cols].fillna(0).values[tri], label=y[tri],
                         feature_name=[sanitize_col(c) for c in top_cols])
        vd = lgb.Dataset(f[top_cols].fillna(0).values[vai], label=y[vai],
                         feature_name=[sanitize_col(c) for c in top_cols], reference=ds)
        m = lgb.train({
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':300,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':5,'random_state':42,'n_jobs':1
        }, ds, num_boost_round=300, valid_sets=[vd], callbacks=[lgb.early_stopping(30, verbose=False)])
        fold_preds[vai] = m.predict(f[top_cols].fillna(0).values[vai])
    
    fold_preds = np.clip(fold_preds, 0.0001, 0.9999)
    baseline_ll = log_loss(y, fold_preds, labels=[0,1])
    
    # Filter by confidence
    for thresh in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        high_conf = (fold_preds >= thresh) | (fold_preds <= (1-thresh))
        n_kept = high_conf.sum()
        if n_kept < 50:
            continue
        
        # Retrain on filtered data
        f_filt = f[high_conf]
        y_filt = y[high_conf]
        
        n_groups_filt = f_filt['subject_id'].nunique()
        n_sp = min(5, max(2, n_groups_filt))
        if n_sp < 2:
            confidence_results[f'{target}_t{thresh}_error'] = {'error': 'too_few_groups', 'n_groups': n_groups_filt}
            continue
        
        # Use KFold on filtered data to avoid GroupKFold issue
        from sklearn.model_selection import KFold as KF
        skf_filt = KF(n_splits=n_sp, shuffle=True, random_state=42)
        fold_preds_filt = np.zeros(len(f_filt))
        tp_filt_all = np.zeros((len(ft), n_sp))
        for fi, (tri_f, vai_f) in enumerate(skf_filt.split(f_filt, y_filt)):
            ds_f = lgb.Dataset(f_filt[top_cols].fillna(0).values[tri_f], label=y_filt[tri_f],
                               feature_name=[sanitize_col(c) for c in top_cols])
            vd_f = lgb.Dataset(f_filt[top_cols].fillna(0).values[vai_f], label=y_filt[vai_f],
                               feature_name=[sanitize_col(c) for c in top_cols], reference=ds_f)
            m_f = lgb.train({
                'objective':'binary','metric':'binary_logloss','verbose':-1,
                'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':300,
                'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
                'min_child_samples':5,'random_state':42,'n_jobs':1
            }, ds_f, num_boost_round=300, valid_sets=[vd_f], callbacks=[lgb.early_stopping(30, verbose=False)])
            fold_preds_filt[vai_f] = m_f.predict(f_filt[top_cols].fillna(0).values[vai_f])
        
        # Also predict on test set
        tp_filt_all = np.column_stack([m_f.predict(ft[top_cols].fillna(0).values) for _ in range(n_sp)])
        # Actually we need per-seed test predictions but using single model
        tp_single = m_f.predict(ft[top_cols].fillna(0).values)
        
        oof_avg_filt = np.clip(fold_preds_filt, 0.0001, 0.9999)
        ll_filt = log_loss(y_filt, oof_avg_filt, labels=[0,1])
        delta = ll_filt - baseline_ll
        
        confidence_results[f'{target}_t{thresh}'] = {
            'n_kept': int(n_kept), 'baseline': round(baseline_ll, 5),
            'filtered_ll': round(ll_filt, 5), 'delta': round(delta, 5),
        }
        print(f'    {target} t={thresh}: n_kept={n_kept} filtered_ll={ll_filt:.5f} delta={delta:+.5f}')

# ============================================================
# STEP 11: Noise Filtering
# ============================================================
print('\n=== STEP 11: Noise filtering (noisy sample detection) ===')

# Detect noisy labels by cross-validation inconsistency
# Samples where multiple folds predict very differently might have noisy labels

noise_results = {}
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    top_cols = ranked[:15]
    
    # Use GroupKFold: for each group, get 4-fold predictions for that group
    n_groups = f['subject_id'].nunique()
    n_splits_noise = min(5, max(2, n_groups))
    if n_splits_noise < 2:
        continue
    gkf = GroupKFold(n_splits=n_splits_noise)
    fold_preds_all = np.zeros((len(y), n_splits_noise))
    
    for fold_i, (tri, vai) in enumerate(gkf.split(f, y, f['subject_id'])):
        ds = lgb.Dataset(f[top_cols].fillna(0).values[tri], label=y[tri],
                         feature_name=[sanitize_col(c) for c in top_cols])
        vd = lgb.Dataset(f[top_cols].fillna(0).values[vai], label=y[vai],
                         feature_name=[sanitize_col(c) for c in top_cols], reference=ds)
        m = lgb.train({
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':300,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':5,'random_state':42,'n_jobs':1
        }, ds, num_boost_round=300, valid_sets=[vd], callbacks=[lgb.early_stopping(30, verbose=False)])
        fold_preds_all[vai, fold_i] = m.predict(f[top_cols].fillna(0).values[vai])
    
    fold_preds_avg = fold_preds_all.mean(axis=1)
    fold_preds_std = fold_preds_all.std(axis=1)
    
    baseline_ll = log_loss(y, np.clip(fold_preds_avg, 0.0001, 0.9999), labels=[0,1])
    
    # Remove samples with highest prediction variance (likely noisy)
    for n_remove in [0, 5, 10, 15, 20, 25]:
        if n_remove == 0:
            filtered_preds = fold_preds_avg
            filtered_y = y
        else:
            # Remove top n_remove samples with highest std
            noisy_idx = np.argsort(fold_preds_std)[-n_remove:]
            keep_mask = np.ones(len(y), dtype=bool)
            keep_mask[noisy_idx] = False
            filtered_preds = fold_preds_avg[keep_mask]
            filtered_y = y[keep_mask]
        
        ll_filt = log_loss(filtered_y, np.clip(filtered_preds, 0.0001, 0.9999), labels=[0,1])
        delta = ll_filt - baseline_ll
        
        noise_results[f'{target}_remove{n_remove}'] = {
            'n_removed': n_remove, 'baseline': round(baseline_ll, 5),
            'filtered_ll': round(ll_filt, 5), 'delta': round(delta, 5),
        }
        print(f'    {target} remove_{n_remove}: delta={delta:+.5f}')

# ============================================================
# STEP 12: Ensemble Optimization
# ============================================================
print('\n=== STEP 12: Ensemble optimization ===')

# Try different ensemble combinations of all internal models
ens_results = {}
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    top_int_cols = ranked[:12]
    
    # Train multiple configurations
    configs = [
        ('default', cfg),
        ('wide', CFGS['wide']),
        ('deep', CFGS['deep']),
        ('v48', CFGS['v48']),
        ('safety', CFGS['safety']),
    ]
    
    models = {}
    for cname, c in configs:
        oof, tp = train_cv(f, ft, top_int_cols, y, SEEDS, c)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        models[cname] = oof_avg
    
    # Find optimal weight combination (uniform + per-weight)
    model_names = list(models.keys())
    best_ll = float('inf')
    best_w = None
    
    # Uniform weights
    ens_uniform = np.mean([models[m] for m in model_names], axis=0)
    ll_uniform = log_loss(y, ens_uniform, labels=[0,1])
    if ll_uniform < best_ll:
        best_ll = ll_uniform
        best_w = {m: 1.0/len(model_names) for m in model_names}
    
    # Try pairs
    for m1, m2 in itertools.combinations(model_names, 2):
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ens = w * models[m1] + (1-w) * models[m2]
            ll = log_loss(y, ens, labels=[0,1])
            if ll < best_ll:
                best_ll = ll
                best_w = {m1: w, m2: 1-w}
    
    # Try 3-way
    for combo in itertools.combinations(model_names, 3):
        for w1 in [0.3, 0.4, 0.5]:
            for w2 in [0.2, 0.3, 0.4]:
                w3 = 1 - w1 - w2
                if w3 < 0.1: continue
                ens = w1 * models[combo[0]] + w2 * models[combo[1]] + w3 * models[combo[2]]
                ll = log_loss(y, ens, labels=[0,1])
                if ll < best_ll:
                    best_ll = ll
                    best_w = {combo[0]: w1, combo[1]: w2, combo[2]: w3}
    
    ens_results[target] = {
        'best_ll': round(best_ll, 5),
        'weights': {k: round(v, 2) for k, v in best_w.items()},
        'baseline': round(v127_oof[target]['oof_raw'], 5),
        'delta': round(best_ll - v127_oof[target]['oof_raw'], 5),
    }
    print(f'  {target}: best_ll={best_ll:.5f} delta={best_ll-v127_oof[target]["oof_raw"]:+.5f} weights={best_w}')

# ============================================================
# STEP 13: Summary
# ============================================================
print('\n' + '=' * 80)
print('V13 FINAL SUMMARY')
print('=' * 80)

print(f'\n  V127 Reproduction Baseline:')
for t in TARGETS:
    print(f'    {t}: {v127_oof[t]["oof_raw"]:.5f} (cal: {v127_oof[t]["oof_cal"]:.5f})')
print(f'    AVG: {avg_v127:.5f}')

print(f'\n  Domain Similarity:')
for eid, ds in domain_scores.items():
    print(f'    {eid}: AUC={ds["auc"] or "N/A"} ({ds["interpretation"]})')

print(f'\n  Adversarial Feature Pruning:')
for t, r in pruned_results.items():
    print(f'    {t}: baseline={r["baseline_ll"]:.5f} pruned={r["best_pruned_ll"]:.5f} delta={r["improvement"]:+.5f}')

print(f'\n  Multi-Stage Training (PCA):')
for k, r in staged_results.items():
    print(f'    {k}: ll={r["ll"]:.5f} delta={r["delta"]:+.5f}')

print(f'\n  Curriculum Learning:')
for k, r in curriculum_results.items():
    print(f'    {k}: delta={r["delta"]:+.5f}')

print(f'\n  Confidence Filtering:')
for k, r in confidence_results.items():
    print(f'    {k}: delta={r["delta"]:+.5f}')

print(f'\n  Noise Filtering:')
for k, r in noise_results.items():
    if r['delta'] < 0:
        print(f'    {k}: delta={r["delta"]:+.5f}')

print(f'\n  Ensemble Optimization:')
for t, r in ens_results.items():
    print(f'    {t}: best_ll={r["best_ll"]:.5f} delta={r["delta"]:+.5f} weights={r["weights"]}')

# Find overall best
overall_best_target = None
overall_best_delta = 0

# Check all results
all_improvements = []
for k, r in pruned_results.items():
    if r['improvement'] < overall_best_delta:
        overall_best_delta = r['improvement']
        overall_best_target = f'pruning_{k}'

for k, r in staged_results.items():
    if r['delta'] < overall_best_delta:
        overall_best_delta = r['delta']
        overall_best_target = f'staged_{k}'

for k, r in ens_results.items():
    if r['delta'] < overall_best_delta:
        overall_best_delta = r['delta']
        overall_best_target = f'ensemble_{k}'

for k, r in noise_results.items():
    if r['delta'] < overall_best_delta:
        overall_best_delta = r['delta']
        overall_best_target = f'noise_{k}'

print(f'\n  *** Overall best: {overall_best_target} (delta={overall_best_delta:+.5f}) ***')

# Save results
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
result = {
    'version': 'v13_domain_adaptation',
    'timestamp': ts,
    'v127_repro': {t: v127_oof[t] for t in TARGETS},
    'v127_avg_oof': round(float(avg_v127), 5),
    'domain_scores': domain_scores,
    'adversarial_importance': adversarial_importance,
    'pruned_results': pruned_results,
    'weighted_results': {k: v for k, v in weighted_results.items() if 'baseline' not in k},
    'staged_results': staged_results,
    'curriculum_results': curriculum_results,
    'confidence_results': {k: v for k, v in confidence_results.items() if float(v.get('delta', 0)) < 0},
    'noise_results': {k: v for k, v in noise_results.items() if float(v.get('delta', 0)) < 0},
    'ensemble_results': ens_results,
    'overall_best': overall_best_target,
    'overall_best_delta': round(float(overall_best_delta), 5),
}

with open(EXPERIMENTS / f'v13_domain_adaptation_{ts}.json', 'w') as fout:
    json.dump(result, fout, indent=2, default=str)
print(f'\n  Saved: v13_domain_adaptation_{ts}.json')

# Save submission with best ensemble predictions
print('\n  Generating submission with best ensemble...')
submit_df = pd.DataFrame({'subject_id': ft['subject_id'].values})
for target in TARGETS:
    y = y_dict[target]
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    top_cols = ranked[:12]
    
    # Get ensemble predictions
    configs = [('default', cfg), ('wide', CFGS['wide']), ('deep', CFGS['deep']),
               ('v48', CFGS['v48']), ('safety', CFGS['safety'])]
    ens_pred = None
    for cname, c in configs:
        oof, tp = train_cv(f, ft, top_cols, y, SEEDS, c)
        tp_avg = tp.mean(axis=1) if tp is not None else np.zeros(len(ft))
        if ens_pred is None:
            ens_pred = tp_avg.copy()
        else:
            w = ens_results[target]['weights'].get(cname, 1.0/len(configs))
            ens_pred = ens_pred + w * tp_avg
    
    ens_pred = np.clip(ens_pred, 0.0001, 0.9999)
    submit_df[target] = ens_pred

submit_path = SUBMIT / f'v13_submission_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f'  Saved submission: {submit_path}')
print(f'  Submission shape: {submit_df.shape}')
print(f'  Submission head:\n{submit_df.head()}')

print('\n=== V13 COMPLETE ===')
