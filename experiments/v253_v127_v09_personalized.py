"""
V253: V127 Ensemble + V09 Personalized External Features

Migrates V09 proxy features to the V127 (features_clean_v60) pipeline.
Creates 9 personalized external features, then tests:
1. V115_base + ext
2. V123_pair + ext
3. V121_pair+rank + ext

Compared against V252 base results.
"""

import re, gc, json, warnings, os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date'}
W121, W123, W115 = 0.35, 0.25, 0.40
SEEDS = [42, 7, 999, 777]

CFGS = {
    'wide':   {'num_leaves':30,'max_depth':3,'learning_rate':0.05,'n_estimators':300,
               'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,'min_child_samples':5},
    'deep':   {'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_samples':15},
    'v48':    {'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10},
    'safety': {'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':3.0,'reg_lambda':10.0,'min_child_samples':20},
}
V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}
LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

import lightgbm as lgb

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META | set(TARGETS)
    if exclude: ex |= exclude
    return [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_] and c not in ex]

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def mean_match(pred, tm):
    return np.clip(pred + (tm - np.clip(pred, 0.0001, 0.9999).mean()), 0.0001, 0.9999)

def isotonic_calibrate(oof_preds, y_true):
    try:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0001, y_max=0.9999)
        iso.fit(oof_preds, y_true)
        cal = iso.predict(oof_preds)
        cal = mean_match(cal, float(y_true.mean()))
        return cal, True
    except:
        return np.clip(oof_preds, 0.0001, 0.9999), False

def cfg_to_params(cfg_s, seed, spw):
    p = dict(cfg_s)
    p.update({'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1})
    return p

def rank_features(feat_df, fcols, target, seed=42):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[fcols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    p = cfg_to_params({'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
                        'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                        'min_child_samples':10}, seed, spw)
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]

def add_pairwise(feat_df, top_feats):
    top_s = [sanitize_col(c) for c in top_feats]
    new_df = feat_df.copy()
    added = []
    for i, f1 in enumerate(top_s[:8]):
        if f1 not in new_df.columns: continue
        for f2 in top_s[i+1:8]:
            if f2 not in new_df.columns: continue
            c1 = new_df[f1].fillna(0)
            c2 = new_df[f2].fillna(0)
            nc = f1 + '_diff_' + f2
            new_df[nc] = c1 - c2
            new_df[nc + '_sum'] = c1 + c2
            new_df[nc + '_ratio'] = c1 / (c2.abs() + 1e-10)
            added.extend([nc, nc+'_sum', nc+'_ratio'])
    return new_df, added

def train_cv(feat_df, ftst_df, cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst_df), len(seeds))) if ftst_df is not None else None
    sn = [sanitize_col(c) for c in cols]
    Xf = feat_df[cols].fillna(0).values.astype(np.float64)
    Xt = ftst_df[cols].fillna(0).values.astype(np.float64) if ftst_df is not None else None
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for fi, (tri, vai) in enumerate(gkf.split(feat_df, y, feat_df['subject_id'])):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            if Xt is not None:
                vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
                tp[:, si] = m.predict(Xt)
                del vd
            else:
                m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                             callbacks=[lgb.log_evaluation(0)])
                oof[vai, si] = m.predict(Xf[vai])
            del ds, m; gc.collect()
    if tp is not None: tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp

# ============================================================
# 1. Load data
# ============================================================
print("="*60)
print("V253: V127 + V09 Personalized External Features")
print("="*60)

print("\n[1] Loading data...")
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
ftst.columns = [sanitize_col(c) for c in ftst.columns]
y_dict = {t: feat[t].values for t in TARGETS}

base_numeric = get_numeric_cols(feat)
print(f"  feat: {feat.shape}, base_numeric: {len(base_numeric)}")
print(f"  ftst: {ftst.shape}")

# ============================================================
# 2. Create V09 personalized external features
# ============================================================
print("\n[2] Creating V09 personalized external features...")

ext_proxy = {}  # dict of (train_series, test_series)
proxy_names = []

# Helper: z-score proxy
def make_zscore(col_name, f, ft):
    if col_name not in f.columns: return None
    s = f[col_name].fillna(0)
    s_t = ft[col_name].fillna(0)
    z = (s - s.mean()) / max(s.std(), 1e-8)
    z_t = (s_t - s.mean()) / max(s.std(), 1e-8)
    return z, z_t

# Proxy 1: ext_activity_z (step z-score)
proxy_names.append('ext_activity_z')
v = make_zscore('wpedo_pedo_step_mean', feat, ftst)
if v: ext_proxy['ext_activity_z'] = v

# Proxy 2: ext_charging_z
proxy_names.append('ext_charging_z')
v = make_zscore('macstatus_m_charging_mean', feat, ftst)
if v: ext_proxy['ext_charging_z'] = v

# Proxy 3: ext_health_composite
if all(c in feat.columns for c in ['wpedo_pedo_step_mean','macstatus_m_charging_mean',
                                     'mscreenstatus_m_screen_use_mean','whr_hr_mean']):
    proxy_names.append('ext_health_composite')
    sa = feat['wpedo_pedo_step_mean'].fillna(0); sc = feat['macstatus_m_charging_mean'].fillna(0)
    ss = feat['mscreenstatus_m_screen_use_mean'].fillna(0); hr = feat['whr_hr_mean'].fillna(0)
    sa_t = ftst['wpedo_pedo_step_mean'].fillna(0); sc_t = ftst['macstatus_m_charging_mean'].fillna(0)
    ss_t = ftst['mscreenstatus_m_screen_use_mean'].fillna(0); hr_t = ftst['whr_hr_mean'].fillna(0)
    ext_proxy['ext_health_composite'] = (
        (sa - sa.mean())/max(sa.std(),1e-8) -
        (sc - sc.mean())/max(sc.std(),1e-8) +
        (ss - ss.mean())/max(ss.std(),1e-8)*0.3 +
        (hr - hr.mean())/max(hr.std(),1e-8)*0.1,
        (sa_t - sa.mean())/max(sa.std(),1e-8) -
        (sc_t - sc.mean())/max(sc.std(),1e-8) +
        (ss_t - ss.mean())/max(ss.std(),1e-8)*0.3 +
        (hr_t - hr.mean())/max(hr.std(),1e-8)*0.1,
    )

# Proxy 4: ext_night_light (light / hour_night)
if 'wlight_w_light_mean' in feat.columns and 'macstatus_hour_night' in feat.columns:
    proxy_names.append('ext_night_light')
    l = feat['wlight_w_light_mean'].fillna(0); h = feat['macstatus_hour_night'].fillna(0)
    lt = ftst['wlight_w_light_mean'].fillna(0); ht = ftst['macstatus_hour_night'].fillna(0)
    ext_proxy['ext_night_light'] = (l / (h+1e-8), lt / (ht+1e-8))

# Proxy 5: ext_total_ambience (sum of ambience_sum cols)
amb_cols = [c for c in feat.columns if 'ambience' in c.lower() and c.endswith('_sum')]
if amb_cols:
    proxy_names.append('ext_total_ambience')
    ext_proxy['ext_total_ambience'] = (
        feat[amb_cols].fillna(0).sum(axis=1),
        ftst[amb_cols].fillna(0).sum(axis=1),
    )

# Proxy 6: ext_hr_step (HR mean * step mean)
if all(c in feat.columns for c in ['whr_hr_mean','wpedo_pedo_step_mean']):
    proxy_names.append('ext_hr_step')
    ext_proxy['ext_hr_step'] = (
        feat['whr_hr_mean'].fillna(0) * feat['wpedo_pedo_step_mean'].fillna(0),
        ftst['whr_hr_mean'].fillna(0) * ftst['wpedo_pedo_step_mean'].fillna(0),
    )

# Proxy 7: ext_screen_ratio
if 'mscreenstatus_m_screen_use_mean' in feat.columns:
    proxy_names.append('ext_screen_ratio')
    sm = feat['mscreenstatus_m_screen_use_mean'].fillna(0); sm_t = ftst['mscreenstatus_m_screen_use_mean'].fillna(0)
    ext_proxy['ext_screen_ratio'] = (sm / (sm+1e-8), sm_t / (sm_t+1e-8))

# Proxy 8: ext_wifi_ble (wifi_sum / ble_sum)
wifi_cols = [c for c in feat.columns if 'wifi' in c.lower() and c.endswith('_mean')]
ble_cols = [c for c in feat.columns if 'ble' in c.lower() and c.endswith('_mean')]
if wifi_cols and ble_cols:
    proxy_names.append('ext_wifi_ble')
    w = feat[wifi_cols].fillna(0).sum(axis=1); b = feat[ble_cols].fillna(0).sum(axis=1)
    w_t = ftst[wifi_cols].fillna(0).sum(axis=1); b_t = ftst[ble_cols].fillna(0).sum(axis=1)
    ext_proxy['ext_wifi_ble'] = (w / (b+1e-8), w_t / (b_t+1e-8))

# Proxy 9: ext_activity_ambience (activity_z * total_ambience)
if 'ext_activity_z' in ext_proxy and 'ext_total_ambience' in ext_proxy:
    proxy_names.append('ext_activity_ambience')
    ext_proxy['ext_activity_ambience'] = (
        ext_proxy['ext_activity_z'][0] * ext_proxy['ext_total_ambience'][0],
        ext_proxy['ext_activity_z'][1] * ext_proxy['ext_total_ambience'][1],
    )

# Add to dataframes
for name, (s_train, s_test) in ext_proxy.items():
    feat[name] = s_train
    ftst[name] = s_test

ext_feature_names = [n for n in proxy_names if n in feat.columns]
print(f"  Added {len(ext_feature_names)} proxy features: {ext_feature_names}")

# Stats
for name in ext_feature_names:
    print(f"    {name}: train mean={feat[name].mean():.3f} std={feat[name].std():.3f} na={feat[name].isna().sum()}")

# ============================================================
# 3. Personalization (z-score per subject)
# ============================================================
print("\n[3] Personalization (subject z-score)...")

# Add subject-level z-scores for all external proxy features
for name in ext_feature_names:
    grp = feat[name].fillna(0).groupby(feat['subject_id']).agg(['mean','std'])
    grp.columns = [f'{name}_subj_mean', f'{name}_subj_std']
    grp = grp.reset_index()
    feat = feat.merge(grp, on='subject_id', how='left')

ftst_fit = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
ftst_fit.columns = [sanitize_col(c) for c in ftst_fit.columns]
for name in ext_feature_names:
    grp = ftst_fit[name].fillna(0).groupby(ftst_fit['subject_id']).agg(['mean','std'])
    grp.columns = [f'{name}_subj_mean', f'{name}_subj_std']
    grp = grp.reset_index()
    ftst_fit = ftst_fit.merge(grp, on='subject_id', how='left')

# Now merge the ext features from ftst into ftst_fit (ftst already has them)
for name in ext_feature_names:
    ftst_fit[name] = ftst[name]

# Subject z-scores for proxy features
for name in ext_feature_names:
    m_col = f'{name}_subj_mean'
    s_col = f'{name}_subj_std'
    if m_col in ftst_fit.columns and s_col in ftst_fit.columns:
        ftst_fit[f'{name}_z'] = np.where(
            (ftst_fit[s_col] == 0) | ftst_fit[name].isna(),
            0.0,
            (ftst_fit[name].fillna(0) - ftst_fit[m_col]) / ftst_fit[s_col].clip(lower=1e-8)
        )

# Do same for train
for name in ext_feature_names:
    m_col = f'{name}_subj_mean'
    s_col = f'{name}_subj_std'
    if m_col in feat.columns and s_col in feat.columns:
        feat[f'{name}_z'] = np.where(
            (feat[s_col] == 0) | feat[name].isna(),
            0.0,
            (feat[name].fillna(0) - feat[m_col]) / feat[s_col].clip(lower=1e-8)
        )

zscore_cols = [f'{name}_z' for name in ext_feature_names if f'{name}_z' in feat.columns]
print(f"  Z-score columns: {zscore_cols}")

# ============================================================
# 4. Feature sets
# ============================================================
print("\n[4] Feature sets...")

# Full numeric columns after adding external features
all_numeric = get_numeric_cols(feat)
ext_base_numeric = get_numeric_cols(feat, exclude=set(ext_feature_names) | set(zscore_cols))
print(f"  All numeric: {len(all_numeric)}")
print(f"  Base + external: {len(all_numeric)} (base={len(ext_base_numeric)}, ext={len(ext_feature_names)+len(zscore_cols)})")

# External feature columns (raw + z-score)
all_ext_cols = ext_feature_names + zscore_cols
print(f"  External cols: {all_ext_cols}")

# ============================================================
# 5. Train strategies (V252 structure, with external features)
# ============================================================
print("\n[5] Training 3 strategies x 2 feature sets")

results = {}

for fs_name, feat_df, ftst_df in [('base', feat.drop(columns=ext_feature_names + zscore_cols, errors='ignore'), ftst.drop(columns=ext_feature_names + zscore_cols, errors='ignore')),
                                     ('ext', feat, ftst_fit)]:
    n_ext = len(all_ext_cols) if fs_name == 'ext' else 0
    print(f"\n  === {fs_name} ({n_ext} external features) ===")
    
    # Get base numeric columns for this df
    base_nc = get_numeric_cols(feat_df)
    print(f"    Base cols: {len(base_nc)}")

    for strat_name, do_pair, do_trans in [
        ('V115_base', False, False),
        ('V123_pair', True, False),
        ('V121_p+t', True, True),
    ]:
        tag = f"v253_{fs_name}_{strat_name}"
        for target in TARGETS:
            y = feat_df[target].values.astype(np.float64)
            cfg = CFGS[V53_SWEEP[target]['cfg']]

            base_cols = get_numeric_cols(feat_df)
            cols = remove_leak(base_cols, target)
            working_df = feat_df.copy()

            if do_pair:
                ranked = rank_features(feat_df, cols, target)
                top8 = ranked[:8]
                working_df, added = add_pairwise(working_df, top8)
                post_pair = get_numeric_cols(working_df)
                cols = remove_leak(post_pair, target)

            if do_trans and do_pair:
                ranked = rank_features(working_df, cols, target)
                top10 = ranked[:10]
                for f in top10:
                    if f in working_df.columns:
                        working_df[f + '_rank'] = pd.Series(working_df[f].fillna(0)).rank(pct=True).values
                cols = [c for c in get_numeric_cols(working_df) if c not in META | set(TARGETS)]
                cols = remove_leak(cols, target)

            oof, _ = train_cv(working_df, None, cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            ll = log_loss(y, iso_cal, labels=[0,1])

            results[(tag, target)] = {'iso_cal': iso_cal, 'll': ll, 'n_feat': len(cols)}
            print(f"    {target}: LL={ll:.5f} (n_feat={len(cols)})")

# ============================================================
# 6. Ensemble + Compare with V252
# ============================================================
print("\n[6] Ensemble Results (0.35*V121 + 0.25*V123 + 0.40*V115)")
print("  Compared to V252 base (0.64471)")

for fs_name in ['base', 'ext']:
    print(f"\n  --- {fs_name} Ensemble ---")
    ens_oof = {}
    for target in TARGETS:
        ens = (W121*results[f"v253_{fs_name}_V121_p+t", target]['iso_cal'] +
               W123*results[f"v253_{fs_name}_V123_pair", target]['iso_cal'] +
               W115*results[f"v253_{fs_name}_V115_base", target]['iso_cal'])
        ens_oof[target] = ens
        ll = log_loss(y_dict[target], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
        print(f"    {target}: {ll:.5f}")
    avg = np.mean([log_loss(y_dict[t], np.clip(ens_oof[t], 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
    print(f"    AVG: {avg:.5f}")
    if fs_name == 'base':
        base_avg = avg
    else:
        ext_avg = avg
        delta_vs_base = ext_avg - base_avg
        delta_vs_v252 = ext_avg - 0.64471
        print(f"    vs base: {delta_vs_base:+.5f}")
        print(f"    vs V252 base (0.64471): {delta_vs_v252:+.5f}")

# ============================================================
# 7. Save
# ============================================================
print(f"\n[7] Summary")
print(f"  V253 Base AVG:  {base_avg:.5f}")
print(f"  V253 Ext AVG:   {ext_avg:.5f}")
print(f"  Delta (ext-base): {delta_vs_base:+.5f}")
print(f"  vs V252 Base:   {delta_vs_v252:+.5f}")

v252_base = 0.64471
print(f"\n  Comparison:")
print(f"    V251 (calendar ext):     0.69288  (delta +0.04817)")
print(f"    V252 (calendar ext):     0.64177  (delta -0.00294)")
print(f"    V253 (V09 personalized): {ext_avg:.5f}  (delta {delta_vs_base:+.5f})")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = {
    'name': 'V253_V127_V09_personalized',
    'timestamp': ts,
    'v252_base_oof': 0.64471,
    'v253_base_oof': float(base_avg),
    'v253_ext_oof': float(ext_avg),
    'delta_vs_v252_base': float(delta_vs_base),
    'delta_vs_v252_v252': float(delta_vs_v252),
    'ext_features': ext_feature_names,
    'zscore_features': zscore_cols,
}
for t in TARGETS:
    log[f'base_{t}'] = float(log_loss(y_dict[t], np.clip(
        W121*results[f"v253_base_V121_p+t", t]['iso_cal'] +
        W123*results[f"v253_base_V123_pair", t]['iso_cal'] +
        W115*results[f"v253_base_V115_base", t]['iso_cal'], 0.0001, 0.9999), labels=[0,1]))
    log[f'ext_{t}'] = float(log_loss(y_dict[t], np.clip(
        W121*results[f"v253_ext_V121_p+t", t]['iso_cal'] +
        W123*results[f"v253_ext_V123_pair", t]['iso_cal'] +
        W115*results[f"v253_ext_V115_base", t]['iso_cal'], 0.0001, 0.9999), labels=[0,1]))

with open(EXPERIMENTS / f'v253_{ts}.json', 'w') as f:
    json.dump(log, f, indent=2, default=str)
print(f"  Saved: experiments/v253_{ts}.json")
print(f"\nV253 COMPLETE ✓")
