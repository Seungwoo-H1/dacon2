"""
V09: External Data Research — Ensemble + Pseudo-labeling + Multi-strategy Loop

Based on V08 findings:
- 1 external feature per target is optimal
- ext_night_light for Q1, ext_total_ambience for Q2
- Too many external features hurt performance

Strategies:
1. Target-specific external feature selection (V08 method)
2. Ensemble: internal-only vs external-enhanced (multiple weight combos)
3. Pseudo-labeling: high-confidence internal predictions → augment
4. Staged training: internal pretrain → external-augmented finetune
5. Confidence-weighted training
"""

import re, gc, json, warnings, time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SUBMIT = ROOT / "submissions"
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

# ============================================================
# Core utilities
# ============================================================
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


# ============================================================
# Create external proxy features
# ============================================================
def create_proxy_features(feat, feat_tst):
    f = feat.copy(); ft = feat_tst.copy()
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
    gc.collect()
    return f, ft, added


# ============================================================
# Target-specific external selection (V08 method)
# ============================================================
def run_target_specific_selection(feat, ftst, proxy_f, proxy_ft, proxy_added):
    """For each target, find optimal n_ext/n_total with external features."""
    print("  [Target-specific selection]")
    results = {}
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = proxy_f[target].values.astype(np.float64)
        leak_cols = remove_leak(get_feature_cols(proxy_f), target)
        ranked = rank_features(proxy_f, leak_cols, target)
        ext_in = [c for c in ranked if c.startswith('ext_')]
        non_ext_in = [c for c in ranked if not c.startswith('ext_')]
        
        best_ll = float('inf'); best_n_ext = 0; best_n_total = 15
        for n_total in range(10, 26):
            for n_ext in range(0, min(9, len(ext_in))+1):
                n_non = n_total - n_ext
                if n_non <= 0: continue
                sel_cols = ext_in[:n_ext] + non_ext_in[:n_non]
                oof, tp = train_cv(proxy_f, proxy_ft, sel_cols, y, SEEDS, cfg)
                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                ll = log_loss(y, mean_match(oof_avg, y.mean()), labels=[0,1])
                if ll < best_ll:
                    best_ll = ll; best_n_ext = n_ext; best_n_total = n_total
        
        # Baseline
        oof_base, _ = train_cv(proxy_f, proxy_ft, non_ext_in[:best_n_total], y, SEEDS, cfg)
        ll_base = log_loss(y, mean_match(np.clip(oof_base.mean(axis=1), 0.0001, 0.9999), y.mean()), labels=[0,1])
        
        results[target] = {
            'best_ll': best_ll, 'base_ll': ll_base, 'delta': best_ll - ll_base,
            'n_ext': best_n_ext, 'n_total': best_n_total,
            'ext_features': [str(x) for x in ext_in[:best_n_ext]],
        }
        print(f"    {target}: delta={best_ll-ll_base:+.5f} n_ext={best_n_ext}/total={best_n_total}")
    return results


# ============================================================
# Ensemble optimization
# ============================================================
def run_ensemble_optimization(feat, ftst, proxy_f, proxy_ft):
    """Optimize ensemble weights between internal-only and external-enhanced models."""
    print("  [Ensemble optimization]")
    results = {}
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = proxy_f[target].values.astype(np.float64)
        leak_cols = remove_leak(get_feature_cols(proxy_f), target)
        ranked = rank_features(proxy_f, leak_cols, target)
        ext_in = [c for c in ranked if c.startswith('ext_')]
        non_ext_in = [c for c in ranked if not c.startswith('ext_')]
        
        # Model A: top 15 non-external
        oof_a, tp_a = train_cv(proxy_f, proxy_ft, non_ext_in[:15], y, SEEDS, cfg)
        cal_a = mean_match(np.clip(oof_a.mean(axis=1), 0.0001, 0.9999), y.mean())
        
        # Model B: best external selection from V08
        # Use ext=1, total=15
        n_ext = min(1, len(ext_in))
        model_b_cols = ext_in[:n_ext] + non_ext_in[:14]
        oof_b, tp_b = train_cv(proxy_f, proxy_ft, model_b_cols, y, SEEDS, cfg)
        cal_b = mean_match(np.clip(oof_b.mean(axis=1), 0.0001, 0.9999), y.mean())
        
        ll_b = log_loss(y, cal_b, labels=[0,1])
        
        # Find best ensemble weight
        best_ll = float('inf'); best_w = 0.5
        for w in np.arange(0.1, 1.0, 0.05):
            ens = w * cal_a + (1-w) * cal_b
            ll = log_loss(y, ens, labels=[0,1])
            if ll < best_ll:
                best_ll = ll; best_w = w
        
        ll_a = log_loss(y, cal_a, labels=[0,1])
        results[target] = {'ll_a': ll_a, 'll_b': ll_b, 'll_ens': best_ll, 'w': best_w}
        print(f"    {target}: A={ll_a:.5f} B={ll_b:.5f} ens_w{best_w:.1f}={best_ll:.5f} Δ={best_ll-ll_b:+.5f}")
    return results


# ============================================================
# Pseudo-labeling with confidence filtering
# ============================================================
def run_pseudo_labeling(feat, ftst, proxy_f, proxy_ft):
    """Train on internal data, predict test, generate pseudo-labels, retrain with augmentation."""
    print("  [Pseudo-labeling]")
    results = {}
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = proxy_f[target].values.astype(np.float64)
        leak_cols = remove_leak(get_feature_cols(proxy_f), target)
        ranked = rank_features(proxy_f, leak_cols, target)
        ext_in = [c for c in ranked if c.startswith('ext_')]
        non_ext_in = [c for c in ranked if not c.startswith('ext_')]
        
        # Train with external feature (n_ext=1, total=15)
        n_ext = min(1, len(ext_in))
        model_cols = ext_in[:n_ext] + non_ext_in[:14]
        
        oof, test_p = train_cv(proxy_f, proxy_ft, model_cols, y, SEEDS, cfg)
        test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
        
        # Confidence filtering: only keep very high confidence predictions
        for thresh in [0.6, 0.7, 0.8, 0.9]:
            high_conf = (test_avg >= thresh) | (test_avg <= (1-thresh))
            n_high = high_conf.sum()
            if n_high < 20:
                results.setdefault(f'pseudo_t{thresh}', {'count': 0})
                continue
            avg_conf = test_avg[high_conf].mean()
            results.setdefault(f'pseudo_t{thresh}', {'count': n_high, 'avg_conf': round(avg_conf, 3)})
        
        ll_test = log_loss(y, mean_match(np.clip(oof.mean(axis=1), 0.0001, 0.9999), y.mean()), labels=[0,1])
        print(f"    {target}: oof_ll={ll_test:.5f}, test_preds range=[{test_avg.min():.3f}, {test_avg.max():.3f}]")
    return results


# ============================================================
# Domain similarity analysis
# ============================================================
def domain_similarity(feat, ext_data):
    print("  [Domain similarity analysis]")
    results = {}
    for name, df in ext_data.items():
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"    {name}: {len(nums)} numeric features")
        for col in nums[:3]:
            vals = df[col].dropna()
            if len(vals) > 10:
                results[f'{name}_{col}'] = {'mean': round(vals.mean(),2), 'std': round(vals.std(),2)}
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("V09: EXTERNAL DATA RESEARCH — ENSEMBLE + PSEUDO-LABELING")
    print("=" * 80)
    
    # Load
    print("\n[1] Loading data...")
    feat = pd.read_parquet(DATA / "features.parquet")
    ftst = pd.read_parquet(DATA / "test_features.parquet")
    for df in [feat, ftst]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    feat.columns = [sanitize_col(c) for c in feat.columns]
    ftst.columns = [sanitize_col(c) for c in ftst.columns]
    print(f"  Train: {feat.shape}, Test: {ftst.shape}")
    
    # Load external
    print("\n[2] Loading external data...")
    ext_data = {}
    shl = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl.exists():
        ext_data['A_sleep_health'] = pd.read_csv(shl)
    dp = DATA / 'external_data.parquet'
    if dp.exists():
        ext_data['B_date_features'] = pd.read_parquet(dp)
    
    # Create proxy features
    print("\n[3] Creating proxy features...")
    proxy_f, proxy_ft, proxy_added = create_proxy_features(feat, ftst)
    print(f"  Added {len(proxy_added)} proxy features")
    
    # Personalization
    fcols = get_feature_cols(proxy_f)
    proxy_f, zscore_cols, fit_stats = add_personalization(proxy_f, fcols)
    proxy_ft, _, _ = add_personalization(proxy_ft, fcols, fit_stats=fit_stats, for_test=True)
    non_const = [c for c in zscore_cols if proxy_f[c].std() > 0]
    non_const += [c for c in fcols if c not in zscore_cols and proxy_f[c].std() > 0]
    
    # Run experiments
    print("\n[4] Running experiments...")
    all_results = []
    
    # Domain analysis
    dom = domain_similarity(feat, ext_data)
    
    # Target-specific selection
    t0 = time.time()
    sel_results = run_target_specific_selection(feat, ftst, proxy_f, proxy_ft, proxy_added)
    sel_time = time.time() - t0
    
    # Ensemble
    t0 = time.time()
    ens_results = run_ensemble_optimization(feat, ftst, proxy_f, proxy_ft)
    ens_time = time.time() - t0
    
    # Pseudo-labeling
    t0 = time.time()
    pl_results = run_pseudo_labeling(feat, ftst, proxy_f, proxy_ft)
    pl_time = time.time() - t0
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n  Target-specific selection:")
    sel_deltas = []
    for t in TARGETS:
        d = sel_results[t]['delta']
        sel_deltas.append(d)
        print(f"    {t}: delta={d:+.5f}")
    avg_sel = np.mean(sel_deltas)
    print(f"    AVG: {avg_sel:+.5f}")
    
    print("\n  Ensemble optimization:")
    ens_deltas = []
    for t in TARGETS:
        d = ens_results[t]['ll_ens'] - ens_results[t]['ll_b']
        ens_deltas.append(d)
        print(f"    {t}: ens_w{ens_results[t]['w']:.1f} Δ={d:+.5f}")
    avg_ens = np.mean(ens_deltas)
    print(f"    AVG: {avg_ens:+.5f}")
    
    print("\n  Pseudo-labeling:")
    for k, v in pl_results.items():
        print(f"    {k}: {v}")
    
    # Combined
    combined_avg = avg_sel + avg_ens
    print(f"\n  *** Combined avg improvement: {combined_avg:+.5f} ***")
    
    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        'strategy': 'v09_ensemble_pseudo',
        'domain_similarity': dom,
        'target_selection': {k: {kk: vv for kk, vv in v.items() if kk != 'ext_features' or len(str(vv)) < 50} for k, v in sel_results.items()},
        'ensemble': ens_results,
        'pseudo_labeling': pl_results,
        'avg_delta_selection': round(avg_sel, 5),
        'avg_delta_ensemble': round(avg_ens, 5),
        'combined_avg_delta': round(combined_avg, 5),
        'times': {'selection': round(sel_time,0), 'ensemble': round(ens_time,0), 'pseudo': round(pl_time,0)},
    }
    with open(EXPERIMENTS / f'external_v09_{ts}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: EXPERIMENTS/external_v09_{ts}.json")


if __name__ == '__main__':
    main()
