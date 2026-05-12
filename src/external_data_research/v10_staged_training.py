"""
V10: Comparison experiments
- Calibration-aware pseudo-labeling
- Staged training (external pretrain -> internal finetune)
- Pseudo-label augmentation
"""

import re, gc, json, time, warnings
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
EXPERIMENTS = ROOT / 'experiments'
for d in [EXPERIMENTS]: d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42]
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

# Load
print('Loading...')
feat = pd.read_parquet(DATA / 'features.parquet')
ftst = pd.read_parquet(DATA / 'test_features.parquet')
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Proxy features
f = feat.copy(); ft = ftst.copy()
all_num = get_feature_cols(feat)
if 'wPedo_pedo_step_mean' in all_num:
    s = f['wPedo_pedo_step_mean'].fillna(0); s_t = ft['wPedo_pedo_step_mean'].fillna(0)
    f['ext_activity_z'] = (s-s.mean())/max(s.std(),1e-8)
    ft['ext_activity_z'] = (s_t-s.mean())/max(s.std(),1e-8)
if 'mACStatus_m_charging_mean' in all_num:
    ch = f['mACStatus_m_charging_mean'].fillna(0); ch_t = ft['mACStatus_m_charging_mean'].fillna(0)
    f['ext_charging_z'] = (ch-ch.mean())/max(ch.std(),1e-8)
    ft['ext_charging_z'] = (ch_t-ch.mean())/max(ch.std(),1e-8)
if all(c in all_num for c in ['wPedo_pedo_step_mean','mACStatus_m_charging_mean','mScreenStatus_m_screen_use_mean','wHr_hr_mean']):
    sa = f['wPedo_pedo_step_mean'].fillna(0); sc_h = f['mACStatus_m_charging_mean'].fillna(0)
    ss = f['mScreenStatus_m_screen_use_mean'].fillna(0); hr = f['wHr_hr_mean'].fillna(0)
    sa_t = ft['wPedo_pedo_step_mean'].fillna(0); sc_t = ft['mACStatus_m_charging_mean'].fillna(0)
    ss_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0); hr_t = ft['wHr_hr_mean'].fillna(0)
    f['ext_health_composite'] = (sa-sa.mean())/max(sa.std(),1e-8) - (sc_h-sc_h.mean())/max(sc_h.std(),1e-8) + (ss-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr-hr.mean())/max(hr.std(),1e-8)*0.1
    ft['ext_health_composite'] = (sa_t-sa.mean())/max(sa.std(),1e-8) - (sc_t-sc_h.mean())/max(sc_h.std(),1e-8) + (ss_t-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr_t-hr.mean())/max(hr.std(),1e-8)*0.1
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
    sm = f['mScreenStatus_m_screen_use_mean'].fillna(0); sm_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
    f['ext_screen_ratio'] = sm / (sm+1e-8)
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

# Personalization
fcols = get_feature_cols(f)
f, zscore_cols, fit_stats = add_personalization(f, fcols)
ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
non_const = [c for c in fcols+zscore_cols if f[c].std() > 0]
y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}

# Best configs from V09
BEST = {
    'Q1': {'ne':1,'nt':15,'cfg':'deep'},
    'Q2': {'ne':1,'nt':20,'cfg':'deep'},
    'Q3': {'ne':0,'nt':12,'cfg':'v48'},
    'S1': {'ne':2,'nt':20,'cfg':'wide'},
    'S2': {'ne':2,'nt':15,'cfg':'deep'},
    'S3': {'ne':1,'nt':12,'cfg':'safety'},
    'S4': {'ne':2,'nt':15,'cfg':'wide'},
}

# ============================================================
# V10 EXP1: Calibration + Pseudo-label analysis
# ============================================================
print('\n=== V10 EXP1: Calibration + Pseudo-labeling ===')
exp1 = {}
for target in TARGETS:
    t0 = time.time()
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]

    cb = BEST[target]
    best_cols = ext_in[:cb['ne']] + non_ext_in[:cb['nt']-cb['ne']]
    oof, tp = train_cv(f, ft, best_cols, y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    test_avg = np.clip(tp.mean(axis=1), 0.0001, 0.9999)
    train_mean = y.mean()

    ll_orig = log_loss(y, oof_avg, labels=[0,1])
    ll_cal = log_loss(y, mean_match(oof_avg, train_mean), labels=[0,1])

    cal_test = mean_match(test_avg, train_mean)
    pseudo_info = {}
    for thresh in [0.55, 0.6, 0.65]:
        hc = (cal_test >= thresh) | (cal_test <= (1-thresh))
        nh = hc.sum()
        if nh < 10: continue
        pseudo_info[thresh] = {
            'n': int(nh), 'avg_conf': float(cal_test[hc].mean()),
            'pseudo_pos': float((cal_test[hc] > train_mean).mean())
        }

    exp1[target] = {
        'll_orig': round(ll_orig, 5), 'll_cal': round(ll_cal, 5),
        'delta': round(ll_cal - ll_orig, 5),
        'pseudo': pseudo_info, 'time': round(time.time()-t0, 1)
    }
    print(f'  {target}: ll_orig={ll_orig:.5f} ll_cal={ll_cal:.5f} delta={ll_cal-ll_orig:+.5f} time={time.time()-t0:.1f}s')

# ============================================================
# V10 EXP2: Staged Training - simulate by training with fewer trees first
# Strategy: train with ALL features but fewer trees, then continue
# ============================================================
print('\n=== V10 EXP2: Staged Training ===')
exp2 = {}
for target in TARGETS:
    t0 = time.time()
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]

    cb = BEST[target]
    best_cols = ext_in[:cb['ne']] + non_ext_in[:cb['nt']-cb['ne']]
    gkf = GroupKFold(n_splits=5)

    # Baseline: all features, 1 pass
    oof_base, _ = train_cv(f, ft, best_cols, y, SEEDS, cfg)
    ll_base = log_loss(y, mean_match(np.clip(oof_base.mean(axis=1), 0.0001, 0.9999), y.mean()), labels=[0,1])

    # Staged: train with all features in 2 stages
    # Stage 1: learn from external features with high LR (100 trees)
    # Stage 2: fine-tune with all features with lower LR (remaining trees)
    oof_staged = np.zeros((len(y), len(SEEDS)))
    sn_all = [sanitize_col(c) for c in best_cols]
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    Xf = f[best_cols].fillna(0).values.astype(np.float64)
    Xt = ft[best_cols].fillna(0).values.astype(np.float64)
    nr = int(cfg['ne'])

    for si, seed in enumerate(SEEDS):
        p1 = {
            'objective':'binary', 'metric':'binary_logloss', 'verbose':-1,
            'num_leaves':int(cfg['nl']), 'max_depth':int(cfg['md']),
            'learning_rate':float(cfg['lr'])*1.5, 'n_estimators':100,
            'subsample':float(cfg['ss']), 'colsample_bytree':float(cfg['cb']),
            'reg_alpha':float(cfg['ra'])*0.5, 'reg_lambda':float(cfg['rl'])*0.5,
            'min_child_samples':max(1,int(cfg['mc'])),
            'scale_pos_weight':spw, 'random_state':int(seed),
            'force_row_wise':True, 'n_jobs':1
        }
        p2 = {
            'objective':'binary', 'metric':'binary_logloss', 'verbose':-1,
            'num_leaves':int(cfg['nl']), 'max_depth':int(cfg['md']),
            'learning_rate':float(cfg['lr'])*0.5, 'n_estimators':nr,
            'subsample':float(cfg['ss']), 'colsample_bytree':float(cfg['cb']),
            'reg_alpha':float(cfg['ra']), 'reg_lambda':float(cfg['rl']),
            'min_child_samples':max(1,int(cfg['mc'])),
            'scale_pos_weight':spw, 'random_state':int(seed),
            'force_row_wise':True, 'n_jobs':1
        }
        for tri, vai in gkf.split(f, y, f['subject_id']):
            ds1 = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn_all)
            vd1 = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn_all, reference=ds1)
            m1 = lgb.train(p1, ds1, num_boost_round=100, valid_sets=[vd1], callbacks=[lgb.log_evaluation(0)])
            # Continue training with lower LR
            ds2 = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn_all, reference=ds1)
            vd2 = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn_all, reference=ds1)
            m2 = lgb.train(p2, ds2, num_boost_round=100+(nr-100), valid_sets=[vd2],
                           init_model=m1, callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_staged[vai, si] = m2.predict(Xf[vai])
            del ds1, vd1, ds2, vd2, m1, m2; gc.collect()

    ll_staged = log_loss(y, mean_match(np.clip(oof_staged.mean(axis=1), 0.0001, 0.9999), y.mean()), labels=[0,1])
    exp2[target] = {'ll_base': round(ll_base, 5), 'll_staged': round(ll_staged, 5), 'delta': round(ll_staged-ll_base, 5), 'time': round(time.time()-t0, 1)}
    print(f'  {target}: base={ll_base:.5f} staged={ll_staged:.5f} delta={ll_staged-ll_base:+.5f} time={time.time()-t0:.1f}s')

# ============================================================
# V10 EXP3: Weighted ensemble of multiple model configs
# ============================================================
print('\n=== V10 EXP3: Multi-config ensemble ===')
exp3 = {}
all_configs = [
    ('wide', CFG_WIDE), ('deep', CFG_DEEP), ('v48', CFG_V48), ('safety', CFG_SAFETY)
]
for target in TARGETS:
    t0 = time.time()
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_f(f, leak_cols, target)
    ext_in = [c for c in ranked if c.startswith('ext_')]
    non_ext_in = [c for c in ranked if not c.startswith('ext_')]

    cb = BEST[target]
    best_cols = ext_in[:cb['ne']] + non_ext_in[:cb['nt']-cb['ne']]

    # Get all model predictions
    oofs = {}
    for cname, ccfg in all_configs:
        oof, tp = train_cv(f, ft, best_cols, y, SEEDS, ccfg)
        oofs[cname] = np.clip(oof.mean(axis=1), 0.0001, 0.9999)

    # Baseline
    ll_base = log_loss(y, mean_match(oofs[cb['cfg']], y.mean()), labels=[0,1])

    # Try all weight combos of models that improve over baseline
    models = {}
    for cname, ccfg in all_configs:
        ll = log_loss(y, mean_match(oofs[cname], y.mean()), labels=[0,1])
        if ll <= ll_base + 0.005:  # within 0.005 of baseline
            models[cname] = oofs[cname]

    best_ens_ll = ll_base
    best_weights = {}
    if len(models) >= 2:
        model_names = list(models.keys())
        n = len(model_names)
        if n == 2:
            for w1 in np.arange(0, 1.05, 0.1):
                w2 = 1.0 - w1
                ens = w1*models[model_names[0]] + w2*models[model_names[1]]
                ens = mean_match(ens, y.mean())
                ll = log_loss(y, ens, labels=[0,1])
                if ll < best_ens_ll:
                    best_ens_ll = ll
                    best_weights = {model_names[0]:round(w1,2), model_names[1]:round(w2,2)}
        elif n == 3:
            for w1 in np.arange(0, 1.05, 0.1):
                for w2 in np.arange(0, 1.05-w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < -0.01: continue
                    w3 = max(0, w3)
                    ens = w1*models[model_names[0]] + w2*models[model_names[1]] + w3*models[model_names[2]]
                    ens = mean_match(ens, y.mean())
                    ll = log_loss(y, ens, labels=[0,1])
                    if ll < best_ens_ll:
                        best_ens_ll = ll
                        best_weights = {model_names[0]:round(w1,2), model_names[1]:round(w2,2), model_names[2]:round(w3,2)}
        else:
            for w1 in np.arange(0, 1.05, 0.15):
                for w2 in np.arange(0, 1.05-w1, 0.15):
                    for w3 in np.arange(0, 1.05-w1-w2, 0.15):
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < -0.01: continue
                        w4 = max(0, w4)
                        ens = w1*models[model_names[0]] + w2*models[model_names[1]] + w3*models[model_names[2]] + w4*models[model_names[3]]
                        ens = mean_match(ens, y.mean())
                        ll = log_loss(y, ens, labels=[0,1])
                        if ll < best_ens_ll:
                            best_ens_ll = ll
                            best_weights = {model_names[0]:round(w1,2), model_names[1]:round(w2,2), model_names[2]:round(w3,2), model_names[3]:round(w4,2)}

    exp3[target] = {
        'll_base': round(ll_base, 5), 'll_ensemble': round(best_ens_ll, 5),
        'delta': round(best_ens_ll-ll_base, 5), 'weights': best_weights,
        'time': round(time.time()-t0, 1)
    }
    print(f'  {target}: base={ll_base:.5f} ens={best_ens_ll:.5f} delta={best_ens_ll-ll_base:+.5f} weights={best_weights} time={time.time()-t0:.1f}s')

# ============================================================
# Summary
# ============================================================
print('\n=== V10 SUMMARY ===')
print('Calibration: EXP1')
print('Staged Training: EXP2')
print('Multi-config Ensemble: EXP3')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
result = {
    'strategy': 'v10_comparison',
    'exp1_calibration': exp1,
    'exp2_staged': exp2,
    'exp3_ensemble': exp3,
}
with open(EXPERIMENTS / f'v10_{ts}.json', 'w') as fout:
    json.dump(result, fout, indent=2, default=str)
print(f'\nSaved: v10_{ts}.json')
