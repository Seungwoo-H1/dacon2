"""
V252: V127 True Ensemble Verification + External Test
- Part 1: Verify V127 OOF using saved OOF files
- Part 2: Re-train from scratch and compare base vs external
- Part 3: Ensemble calculation + submission
"""

import os, sys, gc, re, json, warnings
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
META_COLS = {'subject_id','lifelog_date','sleep_date'}
W121, W123, W115 = 0.35, 0.25, 0.40
SEEDS = [42, 7, 999, 777]

def sanitize_col(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude: ex |= exclude
    return [c for c in df.columns
            if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
            and c not in ex]

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

import lightgbm as lgb

def cfg_to_params(cfg_s, seed, spw):
    p = dict(cfg_s)
    p.update({'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1})
    return p

def mean_match(pred, tm):
    if isinstance(tm, float):
        return np.clip(pred + (tm - np.clip(pred, 0.0001, 0.9999).mean()), 0.0001, 0.9999)
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

# Leak columns
LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

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

print("="*60)
print("V252: V127 Ensemble Verification + External Test")
print("="*60)

# ============================================================
# Part 1: Verify V127 from saved OOF files
# ============================================================
oof_files = {
    'V121': pd.read_csv(DATA / 'oof_v121_20260511_221621.csv'),
    'V123': pd.read_csv(DATA / 'oof_v123_20260511_223008.csv'),
    'V115': pd.read_csv(DATA / 'oof_v115_20260511_125245.csv'),
}
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
y_dict = {t: feat[t].values for t in TARGETS}

print("\n[Part 1] V127 Ensemble from saved OOFs")
v127_ens = {}
for t in TARGETS:
    ens = W121*oof_files['V121'][t].values + W123*oof_files['V123'][t].values + W115*oof_files['V115'][t].values
    v127_ens[t] = ens
    ll = log_loss(y_dict[t], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
    print(f"  {t}: {ll:.5f}")
v127_avg = np.mean([log_loss(y_dict[t], np.clip(v127_ens[t], 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
print(f"  AVG OOF: {v127_avg:.5f} (expected: 0.53731)")
assert abs(v127_avg - 0.53731) < 0.001, f"V127 mismatch: {v127_avg}"
print("  [OK] V127 verified!")

# ============================================================
# Part 2: Load data + external features
# ============================================================
print("\n[Part 2] Data Loading")
ext = pd.read_parquet(DATA / 'external_data.parquet')
ext['lifelog_date'] = pd.to_datetime(ext['lifelog_date']).dt.normalize()
ext['is_holiday'] = ext['is_holiday'].astype(int)
ext['is_school_term'] = ext['is_school_term'].astype(int)
ext['is_exam_period'] = ext['is_exam_period'].astype(int)
ext['is_lunar_holiday'] = ext['is_lunar_holiday'].astype(int)

ext_merge = ext[['lifelog_date','is_holiday','month','is_school_term','is_exam_period',
                  'is_lunar_holiday','daylight_hours','daylight_ratio','season_index']]

ext_cols = ['is_holiday','month','is_school_term','is_exam_period','is_lunar_holiday',
            'daylight_hours','daylight_ratio','season_index']

# Merge into feat (datetime columns)
feat_ext = feat.copy()
for _, row in ext_merge.iterrows():
    mask = pd.to_datetime(feat_ext['lifelog_date']).dt.normalize() == row['lifelog_date']
    for c in ext_merge.columns:
        if c != 'lifelog_date':
            feat_ext.loc[mask, c] = row[c]
print(f"  feat_ext merged: {feat_ext.shape}, ext cols NA: {feat_ext[ext_cols].isna().sum().sum()}")

# Merge into test (object columns)
ftst = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
ftst.columns = [sanitize_col(c) for c in ftst.columns]
ftst_ext = ftst.copy()
ftst_l = pd.to_datetime(ftst['lifelog_date']).dt.normalize()
ext_merge_t = ext_merge.copy()
ext_merge_t['lifelog_date'] = pd.to_datetime(ext_merge_t['lifelog_date']).dt.normalize()
for _, row in ext_merge_t.iterrows():
    mask = ftst_l == row['lifelog_date']
    for c in ext_merge_t.columns:
        if c != 'lifelog_date':
            ftst_ext.loc[mask, c] = row[c]
print(f"  ftst_ext merged: {ftst_ext.shape}, ext cols NA: {ftst_ext[ext_cols].isna().sum().sum()}")

# Base feature columns
base_numeric = get_numeric_cols(feat_ext)
base_ext_numeric = get_numeric_cols(feat_ext)
print(f"  Base features: {len(base_ext_numeric)}")

# ============================================================
# Part 3: Train strategies
# ============================================================
print("\n[Part 3] Training 3 strategies x 2 feature sets")

results = {}

for fs_name, feat_df, ftst_df in [('base', feat, ftst), ('ext', feat_ext, ftst_ext)]:
    n_ext = len(ext_cols) if fs_name == 'ext' else 0
    print(f"\n  === {fs_name} ({n_ext} external features, {len(get_numeric_cols(feat_df))} base cols) ===")

    for strat_name, do_pair, do_trans in [
        ('V115_base', False, False),
        ('V123_pair', True, False),
        ('V121_p+t', True, True),
    ]:
        tag = f"v252_{fs_name}_{strat_name}"
        for target in TARGETS:
            y = feat_df[target].values.astype(np.float64)
            cfg = CFGS[V53_SWEEP[target]['cfg']]

            # Start with base feature columns
            base_cols = get_numeric_cols(feat_df)
            cols = remove_leak(base_cols, target)

            working_df = feat_df.copy()

            if do_pair:
                ranked = rank_features(feat_df, cols, target)
                top8 = ranked[:8]
                working_df, added = add_pairwise(working_df, top8)
                # Get numeric columns after pairwise
                post_pair = get_numeric_cols(working_df)
                cols = remove_leak(post_pair, target)

            if do_trans and do_pair:
                # Only apply rank transform if we haven't already done pairwise+rank together
                ranked = rank_features(working_df, cols, target)
                top10 = ranked[:10]
                for f in top10:
                    if f in working_df.columns:
                        working_df[f + '_rank'] = pd.Series(working_df[f].fillna(0)).rank(pct=True).values
                cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
                cols = remove_leak(cols, target)

            oof, _ = train_cv(working_df, None, cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            ll = log_loss(y, iso_cal, labels=[0,1])

            results[(tag, target)] = {'iso_cal': iso_cal, 'll': ll, 'n_feat': len(cols)}
            print(f"    {target}: LL={ll:.5f} (n_feat={len(cols)})")

# ============================================================
# Part 4: Ensemble
# ============================================================
print("\n[Part 4] Ensemble Results (0.35*V121 + 0.25*V123 + 0.40*V115)")
for fs_name in ['base', 'ext']:
    print(f"\n  --- {fs_name} Ensemble ---")
    ens_oof = {}
    for target in TARGETS:
        ens = (W121*results[f"v252_{fs_name}_V121_p+t", target]['iso_cal'] +
               W123*results[f"v252_{fs_name}_V123_pair", target]['iso_cal'] +
               W115*results[f"v252_{fs_name}_V115_base", target]['iso_cal'])
        ens_oof[target] = ens
        ll = log_loss(y_dict[target], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
        print(f"    {target}: {ll:.5f}")
    avg = np.mean([log_loss(y_dict[t], np.clip(ens_oof[t], 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
    print(f"    AVG: {avg:.5f}")
    if fs_name == 'base':
        base_avg = avg
    else:
        ext_avg = avg
        delta = ext_avg - base_avg
        print(f"    vs base: {delta:+.5f}")
        print(f"    V127 saved: {v127_avg:.5f}")

# ============================================================
# Part 5: Save
# ============================================================
print(f"\n[Part 5] Summary")
print(f"  Base AVG: {base_avg:.5f}")
print(f"  Ext AVG:  {ext_avg:.5f}")
print(f"  Delta:    {delta:+.5f}")
print(f"  Best:     {'ext' if delta < 0 else 'base'}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')

log = {
    'name': 'V252_V127_ensemble',
    'timestamp': ts,
    'v127_ensemble_oof': float(v127_avg),
    'base_avg_oof': float(base_avg),
    'ext_avg_oof': float(ext_avg),
    'delta': float(delta),
    'best_fs': 'ext' if delta < 0 else 'base',
}
for t in TARGETS:
    log[f'base_{t}'] = float(log_loss(y_dict[t], np.clip(
        W121*results[f"v252_base_V121_p+t", t]['iso_cal'] +
        W123*results[f"v252_base_V123_pair", t]['iso_cal'] +
        W115*results[f"v252_base_V115_base", t]['iso_cal'], 0.0001, 0.9999), labels=[0,1]))
    log[f'ext_{t}'] = float(log_loss(y_dict[t], np.clip(
        W121*results[f"v252_ext_V121_p+t", t]['iso_cal'] +
        W123*results[f"v252_ext_V123_pair", t]['iso_cal'] +
        W115*results[f"v252_ext_V115_base", t]['iso_cal'], 0.0001, 0.9999), labels=[0,1]))

with open(EXPERIMENTS / f'v252_{ts}.json', 'w') as f:
    json.dump(log, f, indent=2, default=str)

# Save per-target detailed results
detail_log = {}
for tag_key in sorted(results.keys()):
    detail_log[str(tag_key)] = results[tag_key]
with open(EXPERIMENTS / f'v252_detail_{ts}.json', 'w') as f:
    json.dump(detail_log, f, indent=2, default=str)

print(f"  Log saved: experiments/v252_{ts}.json")
print(f"  Detail saved: experiments/v252_detail_{ts}.json")
print(f"\nV252 COMPLETE ✓")
