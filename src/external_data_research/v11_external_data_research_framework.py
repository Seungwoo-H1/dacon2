"""
V11: External Data Research Framework - automated external data exploration loop

V127 fixed baseline + external data collection / quality evaluation / combination search

Execution order:
1. V127 reproduce (baseline OOF verification)
2. External data collection + quality assessment
3. Domain similarity measurement (adversarial validation)
4. Data combination exploration (single -> pair -> triple -> quad)
5. Weighting / Filtering / Curriculum / Staged Training automation
"""
import re, gc, json, time, warnings, traceback, os, itertools, urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from scipy import stats
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

CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5}
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

# STEP 0: Load data
print('=== STEP 0: Load baseline data ===')
feat = pd.read_parquet(DATA / 'features.parquet')
ftst = pd.read_parquet(DATA / 'test_features.parquet')
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Proxy features
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
print(f'  Features after personalization: {len(non_const)}')

# STEP 1: V127 Reproduction
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
    test_preds_all = []
    var_cfgs = [
        cfg,
        {**cfg, 'nl': int(cfg['nl']*0.8), 'ne': int(cfg['ne']*0.7)},
        {**cfg, 'nl': int(cfg['nl']*1.2), 'ne': int(cfg['ne']*1.3)},
    ]
    for vcfg in var_cfgs:
        oof, tp = train_cv(f, ft, best_cols, y, SEEDS, vcfg)
        oofs_all.append(oof)
        test_preds_all.append(tp)
    oof_avg = np.clip(np.mean(oofs_all, axis=0).mean(axis=1), 0.0001, 0.9999)
    v127_oof[target] = {'oof_raw': round(float(log_loss(y, oof_avg, labels=[0,1])), 5),
                        'oof_cal': round(float(log_loss(y, mean_match(oof_avg, y.mean()), labels=[0,1])), 5),
                        'n_feat': len(best_cols)}
    print(f'  {target}: OOF={v127_oof[target]["oof_raw"]:.5f} (cal={v127_oof[target]["oof_cal"]:.5f}) n_feat={len(best_cols)} time={time.time()-t0:.1f}s')

avg_v127_oof = np.mean([v127_oof[t]['oof_raw'] for t in TARGETS])
print(f'\n  V127 AVG OOF: {avg_v127_oof:.5f} (target: 0.53731)')

# STEP 2: External Data Collection
print('\n=== STEP 2: External Data Collection ===')

EXTERNAL_DATASETS = {
    'sleep_health_kaggle': {
        'name': 'Sleep Health & Lifestyle Dataset (Kaggle)',
        'url': 'https://raw.githubusercontent.com/markrojkamp/Sleep-Health-Lifestyle-Dataset/master/SleepHealthandLifestyle.csv',
    },
    'sleep_healthcare_kaggle': {
        'name': 'Sleep Health and Healthcare Dataset (Kaggle)',
        'url': 'https://raw.githubusercontent.com/anishLernAi/Python-100/main/Datasets/sleep_health_and_healthcare_dataset.csv',
    },
}

external_data_store = {}
for eid, edata in EXTERNAL_DATASETS.items():
    path = EXTERNAL / f'{eid}.csv'
    url = edata['url']
    print(f'\n  [{eid}] {edata["name"]}')
    print(f'    URL: {url}')
    try:
        if not path.exists():
            print(f'    Downloading...')
            urllib.request.urlretrieve(url, str(path))
        df_ext = pd.read_csv(path)
        print(f'    Shape: {df_ext.shape}')
        print(f'    Columns: {list(df_ext.columns)}')
        missing = df_ext.isnull().mean().mean()
        numeric_cols = df_ext.select_dtypes(include=[np.number]).columns.tolist()
        external_data_store[eid] = {
            'name': edata['name'], 'path': str(path), 'shape': df_ext.shape,
            'columns': list(df_ext.columns), 'numeric_cols': numeric_cols,
            'missing_rate': float(missing), 'df': df_ext,
        }
        print(f'    Missing: {missing:.3%}, Numeric cols: {len(numeric_cols)}')
    except Exception as e:
        print(f'    FAIL: {e}')
        external_data_store[eid] = {'error': str(e), 'df': None}

# STEP 3: Domain Similarity - Adversarial Validation
print('\n=== STEP 3: Domain Similarity (Adversarial Validation) ===')

domain_scores = {}
for eid, info in external_data_store.items():
    if info.get('df') is None:
        continue
    df_ext = info['df']
    num_ext = df_ext.select_dtypes(include=[np.number]).columns.tolist()
    
    # Map external features to our internal feature space
    # Find similar patterns (mean/std/sum/etc patterns)
    mapped = {}
    for ecol in num_ext:
        ecol_l = ecol.lower()
        for icol in all_num:
            icol_l = icol.lower()
            # Match by semantic similarity
            if any(kw in ecol_l and kw in icol_l for kw in ['sleep', 'heart', 'hr', 'activity', 'pedo',
                                                              'step', 'screen', 'usage', 'light', 'light_',
                                                              'ambience', 'gps', 'wifi', 'ble', 'charging',
                                                              'calories', 'distance', 'sleep_duration',
                                                              'sleep_quality', 'stress', 'bmi', 'age',
                                                              'hour_night']):
                mapped[icol] = ecol
    
    if not mapped:
        print(f'  [{eid}] No feature mapping found')
        domain_scores[eid] = {'mapped_features': 0, 'adversarial_auc': None}
        continue
    
    print(f'  [{eid}] Mapped {len(mapped)} internal features to external')
    for k, v in list(mapped.items())[:5]:
        print(f'    {k} -> {v}')
    
    # Build a combined dataset for adversarial validation
    # Use only mapped features that exist in both
    shared_features = [fc for fc in mapped if fc in f.columns and fc in df_ext.columns]
    # Also use zscore versions
    zshared = [f'{fc}_zscore' for fc in shared_features if f'{fc}_zscore' in f.columns]
    all_shared = shared_features + zshared
    
    if len(shared_features) < 3:
        print(f'  [{eid}] Only {len(shared_features)} shared features, skipping adversarial validation')
        domain_scores[eid] = {'shared_features': len(shared_features), 'adversarial_auc': None}
        continue
    
    # Sample equal sizes for fairness
    n_train = min(len(f), 200)
    n_ext = min(len(df_ext), 200)
    
    X_train = f[shared_features].fillna(0).values.astype(np.float64)[:n_train]
    X_ext = df_ext[shared_features].fillna(0).values.astype(np.float64)[:n_ext]
    
    X_adv = np.vstack([X_train, X_ext])
    y_adv = np.array([0]*n_train + [1]*n_ext)
    
    # Simple adversarial: can we distinguish train vs external?
    # If AUC ~0.5: same domain. If AUC >0.7: different domain
    gkf_adv = GroupKFold(n_splits=5)
    scores = []
    for tri, vai in gkf_adv.split(X_adv):
        ds = lgb.Dataset(X_adv[tri], label=y_adv[tri])
        vd = lgb.Dataset(X_adv[vai], label=y_adv[vai])
        m = lgb.train({'objective':'binary','metric':'binary_logloss','verbose':-1,
                       'num_leaves':10,'max_depth':3,'learning_rate':0.05,'n_estimators':100,
                       'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
                       'min_child_samples':10,'random_state':42,'n_jobs':1},
                      ds, num_boost_round=100, valid_sets=[vd], callbacks=[lgb.early_stopping(10, verbose=False)])
        pred = m.predict(X_adv[vai])
        auc = roc_auc_score(y_adv[vai], pred)
        scores.append(auc)
    adv_auc = np.mean(scores)
    
    domain_scores[eid] = {
        'shared_features': len(shared_features), 'mapped_features': len(mapped),
        'adversarial_auc': round(float(adv_auc), 4),
        'interpretation': 'same_domain' if adv_auc < 0.6 else ('mixed' if adv_auc < 0.7 else 'different_domain'),
    }
    print(f'  [{eid}] Adversarial AUC: {adv_auc:.4f} ({domain_scores[eid]["interpretation"]})')

# STEP 4: External Feature Evaluation
print('\n=== STEP 4: External Feature Evaluation ===')

# Strategy: for each external dataset, create summary statistics as features
# and evaluate their contribution to OOF improvement
ext_feature_scores = {}
for eid, info in external_data_store.items():
    if info.get('df') is None:
        continue
    df_ext = info['df']
    num_ext = df_ext.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create external summary features
    ext_summaries = {}
    for col in num_ext:
        s = df_ext[col].dropna()
        if len(s) >= 20:
            ext_summaries[f'ext_sum_{col}'] = s.mean()
            ext_summaries[f'ext_std_{col}'] = s.std()
    
    print(f'\n  [{eid}] External summary features: {len(ext_summaries)}')
    
    # For each target, try adding external summaries as features
    for target in TARGETS:
        y = y_dict[target]
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_f(f, leak_cols, target)
        
        # Baseline: top N features
        n_base = 15
        best_cols_base = ranked[:n_base]
        oof_base, _ = train_cv(f, ft, best_cols_base, y, SEEDS, cfg)
        oof_avg_base = np.clip(oof_base.mean(axis=1), 0.0001, 0.9999)
        ll_base = log_loss(y, oof_avg_base, labels=[0,1])
        
        # Try adding external summaries (one at a time)
        best_ext_add = None
        best_delta = 0
        for ecol, evalue in ext_summaries.items():
            f_try = f.copy()
            ft_try = ft.copy()
            f_try[f'{ecol}_feat'] = evalue
            ft_try[f'{ecol}_feat'] = evalue
            
            try_cols = best_cols_base + [f'{ecol}_feat']
            oof_try, _ = train_cv(f_try, ft_try, try_cols, y, SEEDS, cfg)
            oof_avg_try = np.clip(oof_try.mean(axis=1), 0.0001, 0.9999)
            ll_try = log_loss(y, oof_avg_try, labels=[0,1])
            delta = ll_try - ll_base
            
            f_try.drop(columns=[f'{ecol}_feat'], inplace=True)
            ft_try.drop(columns=[f'{ecol}_feat'], inplace=True)
            
            if delta < best_delta:
                best_delta = delta
                best_ext_add = ecol
        
        if best_ext_add:
            ext_feature_scores[f'{eid}_{target}'] = {'best_add': best_ext_add, 'delta': round(best_delta, 5)}
            print(f'    {target}: best_ext={best_ext_add} delta={best_delta:+.5f}')
        else:
            ext_feature_scores[f'{eid}_{target}'] = {'best_add': 'none', 'delta': 0}

# STEP 5: Combination Exploration
print('\n=== STEP 5: Combination Exploration ===')

available_eids = [eid for eid in external_data_store if external_data_store[eid].get('df') is not None]
combinations = []
for r in range(1, min(len(available_eids)+1, 5)):
    for combo in itertools.combinations(available_eids, r):
        combinations.append(combo)

print(f'  Combinations: {len(combinations)}')

# For each combination, evaluate multi-external-feature addition
for combo in combinations[:5]:  # Test first 5 to save time
    combo_str = '+'.join(combo)
    print(f'\n  Testing: {combo_str}')
    
    # Build combined external features
    f_try = f.copy()
    ft_try = ft.copy()
    ext_col_names = []
    
    for eid in combo:
        info = external_data_store[eid]
        df_ext = info['df']
        num_ext = df_ext.select_dtypes(include=[np.number]).columns.tolist()
        
        # Find mapped features
        for ecol in num_ext:
            for icol in all_num:
                if any(kw in ecol.lower() and kw in icol.lower() for kw in ['sleep', 'heart', 'hr', 'activity',
                    'step', 'screen', 'light', 'ambience', 'stress', 'bmi', 'age']):
                    evalue = df_ext[ecol].mean()
                    fname = f'ext_{eid}_{ecol}'
                    f_try[fname] = evalue
                    ft_try[fname] = evalue
                    ext_col_names.append(fname)
                    break
            if len(ext_col_names) >= 5:  # Cap external features
                break
        if len(ext_col_names) >= 5:
            break
    
    if not ext_col_names:
        print(f'    No features added, skipping')
        continue
    
    print(f'    Added {len(ext_col_names)} external features: {ext_col_names[:5]}...')
    
    # Evaluate each target
    for target in TARGETS:
        y = y_dict[target]
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_f(f_try, leak_cols, target)
        best_cols = ranked[:12]
        
        # Add external features
        try_cols = best_cols + [c for c in ext_col_names if c not in best_cols[:5]]
        
        oof, _ = train_cv(f_try, ft_try, try_cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        
        baseline = baseline_oofs[target] if 'baseline_oofs' in dir() else v127_oof[target]['oof_raw']
        delta = ll - baseline
        
        print(f'    {target}: OOF={ll:.5f} delta={delta:+.5f}')

# STEP 6: Save results
print('\n=== V11 SUMMARY ===')
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

result = {
    'version': 'v11_external_data_research_framework',
    'timestamp': ts,
    'v127_repro': {t: v127_oof[t] for t in TARGETS},
    'v127_avg_oof': round(float(avg_v127_oof), 5),
    'external_datasets': {eid: {k: v for k, v in info.items() if k != 'df'} for eid, info in external_data_store.items()},
    'domain_scores': domain_scores,
    'ext_feature_scores': ext_feature_scores,
    'combinations_tested': len(combinations),
}

with open(EXPERIMENTS / f'v11_framework_{ts}.json', 'w') as fout:
    json.dump(result, fout, indent=2, default=str)
print(f'Saved: v11_framework_{ts}.json')
