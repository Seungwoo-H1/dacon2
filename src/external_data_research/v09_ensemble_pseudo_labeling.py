"""
V09: External Data Research — Ensemble + Pseudo-Labeling + Staged Training

From V08: n_ext=1 per target gives avg delta -0.01378 over baseline.
Key external features: ext_night_light (Q1,S3,S4), ext_total_ambience (Q2,S2), ext_wifi_ble (S1)

Strategies tested:
1. Target-specific external selection (V08 confirmed)
2. Ensemble: internal-only vs external-enhanced (multiple weight combos)
3. Pseudo-labeling with confidence filtering
4. Staged training: external features only → internal + external
"""

import re, gc, json, time, warnings, traceback
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXTERNAL = ROOT / 'external_data'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]: d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42,7,999,777]

CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5}
CFG_DEEP  = {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15}
CFG_V48   = {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10}
CFG_SAFETY = {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20}
CFGS = {'wide':CFG_WIDE,'deep':CFG_DEEP,'v48':CFG_V48,'safety':CFG_SAFETY}
V53_SWEEP = {'Q1':'deep','Q2':'deep','Q3':'v48','S1':'wide','S2':'deep','S3':'safety','S4':'wide'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count','wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count','wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
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
    pc, stats, sc = [], {}, []
    for col in fcols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean','std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        sc.extend([f'{col}_subj_mean', f'{col}_subj_std'])
        if not for_test: stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        sm = fit_stats[col]['mean'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_mean']
        sd = fit_stats[col]['std'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_std']
        m0 = sd == 0; mn = df[col].isnull()
        z = f'{col}_zscore'
        df[z] = np.where(m0|mn, 0.0, (df[col].fillna(0)-sm)/np.maximum(sd, 1e-8))
        pc.append(z); gc.collect()
    drop = [c for c in sc if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df, pc, stats
def rank_features(feat, fcols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[fcols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    p = {'objective':'binary','metric':'binary_logloss','verbose':-1,'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]
def cfg_to_params(cfg_s, seed, spw):
    return {'objective':'binary','metric':'binary_logloss','verbose':-1,'num_leaves':int(cfg_s['nl']),'max_depth':int(cfg_s['md']),'learning_rate':float(cfg_s['lr']),'n_estimators':int(cfg_s['ne']),'subsample':float(cfg_s['ss']),'colsample_bytree':float(cfg_s['cb']),'reg_alpha':float(cfg_s['ra']),'reg_lambda':float(cfg_s['rl']),'min_child_samples':max(1,int(cfg_s['mc'])),'scale_pos_weight':spw,'random_state':seed,'force_row_wise':True,'n_jobs':1}
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
            m = lgb.train(p, ds, num_boost_round=nr, valid_sets=[vd], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vai, si] = m.predict(Xf[vai])
            if Xt is not None: tp[:, si] = m.predict(Xt)
            del ds, vd, m; gc.collect()
    if tp is not None: tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

# Load
print('Loading...')
feat = pd.read_parquet(DATA / 'features.parquet')
ftst = pd.read_parquet(DATA / 'test_features.parquet')
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Create proxy features (9 external proxy features)
f = feat.copy(); ft = ftst.copy()
added = []; all_num = get_feature_cols(feat)
if 'wPedo_pedo_step_mean' in all_num:
    s = f['wPedo_pedo_step_mean'].fillna(0); s_t = ft['wPedo_pedo_step_mean'].fillna(0)
    f['ext_activity_z'] = (s-s.mean())/max(s.std(),1e-8)
    ft['ext_activity_z'] = (s_t-s.mean())/max(s.std(),1e-8)
    added.append('ext_activity_z')
if 'mACStatus_m_charging_mean' in all_num:
    ch = f['mACStatus_m_charging_mean'].fillna(0); ch_t = ft['mACStatus_m_charging_mean'].fillna(0)
    f['ext_charging_z'] = (ch-ch.mean())/max(ch.std(),1e-8)
    ft['ext_charging_z'] = (ch_t-ch.mean())/max(ch.std(),1e-8)
    added.append('ext_charging_z')
if all(c in all_num for c in ['wPedo_pedo_step_mean','mACStatus_m_charging_mean','mScreenStatus_m_screen_use_mean','wHr_hr_mean']):
    sa = f['wPedo_pedo_step_mean'].fillna(0); sc_h = f['mACStatus_m_charging_mean'].fillna(0)
    ss = f['mScreenStatus_m_screen_use_mean'].fillna(0); hr = f['wHr_hr_mean'].fillna(0)
    sa_t = ft['wPedo_pedo_step_mean'].fillna(0); sc_t = ft['mACStatus_m_charging_mean'].fillna(0)
    ss_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0); hr_t = ft['wHr_hr_mean'].fillna(0)
    f['ext_health_composite'] = (sa-sa.mean())/max(sa.std(),1e-8) - (sc_h-sc_h.mean())/max(sc_h.std(),1e-8) + (ss-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr-hr.mean())/max(hr.std(),1e-8)*0.1
    ft['ext_health_composite'] = (sa_t-sa.mean())/max(sa.std(),1e-8) - (sc_t-sc_h.mean())/max(sc_h.std(),1e-8) + (ss_t-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr_t-hr.mean())/max(hr.std(),1e-8)*0.1
    added.append('ext_health_composite')
if 'wLight_w_light_mean' in all_num and 'mACStatus_hour_night' in all_num:
    f['ext_night_light'] = f['wLight_w_light_mean'].fillna(0) / (f['mACStatus_hour_night'].fillna(0)+1e-8)
    ft['ext_night_light'] = ft['wLight_w_light_mean'].fillna(0) / (ft['mACStatus_hour_night'].fillna(0)+1e-8)
    added.append('ext_night_light')
amb_cols = [c for c in all_num if 'ambience' in c.lower() and c.endswith('_sum')]
if amb_cols:
    f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
    ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
    added.append('ext_total_ambience')
if 'wHr_hr_mean' in all_num and 'wPedo_pedo_step_mean' in all_num:
    f['ext_hr_step'] = f['wHr_hr_mean'].fillna(0) * f['wPedo_pedo_step_mean'].fillna(0)
    ft['ext_hr_step'] = ft['wHr_hr_mean'].fillna(0) * ft['wPedo_pedo_step_mean'].fillna(0)
    added.append('ext_hr_step')
if 'mScreenStatus_m_screen_use_mean' in all_num:
    sm = f['mScreenStatus_m_screen_use_mean'].fillna(0); sm_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
    f['ext_screen_ratio'] = sm / (sm+1e-8)
    ft['ext_screen_ratio'] = sm_t / (sm_t+1e-8)
    added.append('ext_screen_ratio')
wifi_cols = [c for c in all_num if 'wifi' in c.lower() and c.endswith('_mean')]
ble_cols = [c for c in all_num if 'ble' in c.lower() and c.endswith('_mean')]
if wifi_cols and ble_cols:
    w = f[wifi_cols].fillna(0).sum(axis=1); b = f[ble_cols].fillna(0).sum(axis=1)
    w_t = ft[wifi_cols].fillna(0).sum(axis=1); b_t = ft[ble_cols].fillna(0).sum(axis=1)
    f['ext_wifi_ble'] = w / (b+1e-8)
    ft['ext_wifi_ble'] = w_t / (b_t+1e-8)
    added.append('ext_wifi_ble')
if 'ext_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
    f['ext_activity_ambience'] = f['ext_activity_z'] * f['ext_total_ambience']
    ft['ext_activity_ambience'] = ft['ext_activity_z'] * ft['ext_total_ambience']
    added.append('ext_activity_ambience')
if 'wPedo_pedo_step_std' in all_num:
    f['ext_step_consistency'] = f['wPedo_pedo_step_std'].fillna(0) / (f['wPedo_pedo_step_mean'].fillna(0)+1e-8)
    ft['ext_step_consistency'] = ft['wPedo_pedo_step_std'].fillna(0) / (ft['wPedo_pedo_step_mean'].fillna(0)+1e-8)
    added.append('ext_step_consistency')
print(f'Added {len(added)} proxy features')

# Personalization
fcols = get_feature_cols(f)
f, zscore_cols, fit_stats = add_personalization(f, fcols)
ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
all_cols = fcols + zscore_cols
non_const = [c for c in all_cols if f[c].std() > 0]
y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}

# ============================================================
# EXPERIMENT 1: Target-specific external feature selection
# ============================================================
print('\n=== 1. Target-specific external selection ===')
results = []
for target in TARGETS:
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]
    
    best_ll = float('inf')
    best_n_ext = 0
    best_n_total = 15
    
    for n_total in range(10, 26):
        for n_ext in range(0, min(9, len(ext_in))+1):
            n_non = n_total - n_ext
            if n_non <= 0: continue
            sel_cols = ext_in[:n_ext] + non_ext_in[:n_non]
            oof, tp = train_cv(f, ft, sel_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            ll = log_loss(y, mean_match(oof_avg, y.mean()), labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_n_ext = n_ext
                best_n_total = n_total
    
    # Baseline (no external)
    oof_base, _ = train_cv(f, ft, non_ext_in[:best_n_total], y, SEEDS, cfg)
    oof_base_avg = np.clip(oof_base.mean(axis=1), 0.0001, 0.9999)
    ll_base = log_loss(y, mean_match(oof_base_avg, y.mean()), labels=[0, 1])
    
    best_ext = ext_in[:best_n_ext]
    print(f'  {target}: n_ext={best_n_ext}/total={best_n_total} LL={best_ll:.5f} base={ll_base:.5f} delta={best_ll-ll_base:+.5f}')
    if best_ext:
        print(f'    best_ext: {best_ext}')
    
    results.append({
        'target': target, 'best_ll': best_ll, 'base_ll': ll_base,
        'delta': round(best_ll - ll_base, 5), 'n_ext': best_n_ext, 'n_total': best_n_total,
        'ext_features': [str(x) for x in best_ext],
    })

# ============================================================
# EXPERIMENT 2: Ensemble (internal-only vs external-enhanced)
# ============================================================
print('\n=== 2. Ensemble optimization ===')
ens_results = []
for target in TARGETS:
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]
    
    # Model A: internal-only (top 15)
    oof_a, tp_a = train_cv(f, ft, non_ext_in[:15], y, SEEDS, cfg)
    cal_a = mean_match(np.clip(oof_a.mean(axis=1), 0.0001, 0.9999), y.mean())
    ll_a = log_loss(y, cal_a, labels=[0, 1])
    
    # Model B: best external selection (n_ext=1, n_total=15 or 20)
    best_ll = float('inf'); best_w = 0.5; best_n_ext = 1; best_n_total = 15
    for n_ext in [0, 1, 2, 3]:
        for n_total in [10, 15, 20]:
            if n_ext > len(ext_in) or n_total - n_ext <= 0: continue
            model_cols = ext_in[:n_ext] + non_ext_in[:n_total - n_ext]
            oof_b, tp_b = train_cv(f, ft, model_cols, y, SEEDS, cfg)
            cal_b = mean_match(np.clip(oof_b.mean(axis=1), 0.0001, 0.9999), y.mean())
            ll_b = log_loss(y, cal_b, labels=[0, 1])
            
            # Try ensemble weights
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                ens = w * cal_a + (1-w) * cal_b
                ll_ens = log_loss(y, ens, labels=[0, 1])
                if ll_ens < best_ll:
                    best_ll = ll_ens; best_w = w; best_n_ext = n_ext; best_n_total = n_total
    
    print(f'  {target}: best_ens_w={best_w:.1f} LL={best_ll:.5f} (single best={best_ll:.5f})')
    ens_results.append({
        'target': target, 'best_ll': best_ll, 'best_w': best_w,
        'best_n_ext': best_n_ext, 'best_n_total': best_n_total,
    })

# ============================================================
# EXPERIMENT 3: Pseudo-labeling with confidence filtering
# ============================================================
print('\n=== 3. Pseudo-labeling ===')
for target in TARGETS:
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]
    
    # Best external config from V08
    n_ext = min(1, len(ext_in))
    n_total = 15
    model_cols = ext_in[:n_ext] + non_ext_in[:n_total - n_ext]
    
    oof, test_p = train_cv(f, ft, model_cols, y, SEEDS, cfg)
    test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
    
    for thresh in [0.6, 0.7, 0.8]:
        high_conf = (test_avg >= thresh) | (test_avg <= (1-thresh))
        n_high = high_conf.sum()
        if n_high < 10: continue
        avg_conf = test_avg[high_conf].mean()
        internal_pos = y.mean()
        pseudo_pos = (test_avg[high_conf] > 0.5).mean()
        print(f'  {target} t={thresh}: n={n_high} avg_conf={avg_conf:.3f} pseudo_pos={pseudo_pos:.3f} internal_pos={internal_pos:.3f}')

# ============================================================
# Summary
# ============================================================
print('\n=== SUMMARY ===')
deltas = [r['delta'] for r in results]
print('Target-specific selection:')
for r in results:
    print(f'  {r["target"]}: delta={r["delta"]:+.5f}')
print(f'  AVG: {np.mean(deltas):+.5f}')

print('\nEnsemble:')
for r in ens_results:
    print(f'  {r["target"]}: w={r["best_w"]:.1f} LL={r["best_ll"]:.5f}')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
result = {
    'strategy': 'v09_ensemble_pseudo',
    'target_selection': results,
    'ensemble': ens_results,
    'avg_delta': round(np.mean(deltas), 5),
    'per_target_delta': {r['target']: r['delta'] for r in results},
}
with open(EXPERIMENTS / f'v09_{ts}.json', 'w') as fout:
    json.dump(result, fout, indent=2, default=str)
print(f'\nSaved: EXPERIMENTS/v09_{ts}.json')
