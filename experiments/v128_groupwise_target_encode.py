"""
V128: Group-wise Target Encoding (Leave-One-Out) + Enhanced Feature Engineering

Building on V127 ensemble pipeline:
- V121: pairwise + rank transform (weight 0.35)
- V123: pairwise only (weight 0.25)
- V115: base (weight 0.40)

New features:
1. Group-wise LOO target encoding per subject_id
2. Cross-group interactions between existing z-scored features
3. Subject-level aggregate stats for target groups
4. Temporal trend features (slope of target-relevant features over time)

Key: Group-wise LOO encoding avoids leakage by using leave-one-out within each group.
"""

import re, gc, json, warnings, os, sys
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

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude: ex |= exclude
    result = []
    for c in df.columns:
        if c in ex: continue
        # Skip object/string columns entirely
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            continue
        try:
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().sum() > 0:
                result.append(c)
        except:
            pass
    return result

CFGS_MAP = CFGS
V53_SWEEP_MAP = V53_SWEEP

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
print("V128: Group-wise Target Encoding (LOO) + Enhanced Features")
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
# 2. Group-wise Leave-One-Out Target Encoding
# ============================================================
print("\n[2] Group-wise LOO Target Encoding...")

# For each subject_id, compute LOO target stats per target
# This avoids leakage: when encoding subject_id=X, we use mean of all OTHER subjects
loo_targets = {}
for target in TARGETS:
    y = feat[target].values.astype(np.float64)
    subject_ids = feat['subject_id'].values
    
    # Group-wise mean (leave-one-out style)
    # For each sample, compute mean of target for all other subjects (excluding self)
    group_mean = feat.groupby('subject_id')[target].transform('mean')
    group_count = feat.groupby('subject_id')[target].transform('count')
    
    # LOO: (total_sum - self) / (total_count - 1)
    total_sum = group_mean * group_count
    loo_mean = (total_sum - y) / (group_count - 1)
    
    # Also compute std
    group_std = feat.groupby('subject_id')[target].transform('std')
    loo_std = group_std
    
    # Also: proportion of same-group samples (activity level)
    group_n = group_count
    
    loo_col_mean = f'loo_{target}_mean'
    loo_col_std = f'loo_{target}_std'
    loo_col_n = f'loo_{target}_n'
    
    feat[loo_col_mean] = loo_mean.clip(0.0001, 0.9999)
    feat[loo_col_std] = loo_std.fillna(0)
    feat[loo_col_n] = group_n
    
    loo_targets[target] = {
        'mean_col': loo_col_mean,
        'std_col': loo_col_std,
        'n_col': loo_col_n,
    }
    
print(f"  Created LOO features for {len(loo_targets)} targets")
print(f"  LOO feature names: {[(t, d['mean_col']) for t, d in loo_targets.items()]}")

# For test set: use overall mean (no other subjects to compute from)
overall_mean = feat[[t for t in TARGETS]].mean()
for target in TARGETS:
    mc = loo_targets[target]['mean_col']
    sc = loo_targets[target]['std_col']
    nc = loo_targets[target]['n_col']
    ftst[mc] = overall_mean[target]
    ftst[sc] = 0
    ftst[nc] = feat.groupby('subject_id')['lifelog_date'].transform('count').groupby(feat['subject_id']).first().to_dict()

# ============================================================
# 3. Enhanced Feature Engineering
# ============================================================
print("\n[3] Enhanced Feature Engineering...")

enhanced_cols = []

# 3a. Cross-group interaction: activity * environment
# These combine orthogonal signal groups
activity_feats = ['wPedo_pedo_step_mean', 'mActivity_m_activity_mean',
                  'wHr_hr_mean', 'wPedo_pedo_distance_mean']
env_feats = ['mAmbience_ambience_top5_sum', 'mAmbience_ambience_speech_sum',
             'mWifi_wifi_avg_rssi_mean', 'mBle_ble_device_count_mean']

for af in activity_feats:
    for ef in env_feats:
        if af in feat.columns and ef in feat.columns:
            col_name = f'xact_{af.split("_")[-1]}_x_{ef.split("_")[-1]}'
            feat[col_name] = feat[af].fillna(0) * feat[ef].fillna(0)
            ftst[col_name] = ftst[af].fillna(0) * ftst[ef].fillna(0)
            enhanced_cols.append(col_name)

# 3b. Subject-level temporal trends (slope per subject)
print("  [3b] Subject-level temporal trends...")
trend_feats = ['wPedo_pedo_step_mean', 'mActivity_m_activity_mean',
               'mScreenStatus_m_screen_use_mean', 'wHr_hr_mean',
               'mLight_m_light_mean', 'mAmbience_ambience_top5_sum']

for feat_name in trend_feats:
    if feat_name in feat.columns:
        # Compute slope per subject over time
        feat_sorted = feat.sort_values(['subject_id', 'lifelog_date']).copy()
        ftst_sorted = ftst.sort_values(['subject_id', 'lifelog_date']).copy()
        
        def compute_slope(sub_df, col):
            if len(sub_df) < 2:
                return np.zeros(len(sub_df))
            x = np.arange(len(sub_df), dtype=np.float64)
            y = sub_df[col].fillna(0).values.astype(np.float64)
            # Simple linear regression: slope = cov(x,y) / var(x)
            x_mean = x.mean()
            y_mean = y.mean()
            cov_xy = ((x - x_mean) * (y - y_mean)).sum()
            var_x = ((x - x_mean) ** 2).sum()
            slope = cov_xy / max(var_x, 1e-10)
            # Also compute recent trend (last 3 vs first 3)
            if len(y) >= 6:
                recent_mean = y[-3:].mean()
                early_mean = y[:3].mean()
                recent_trend = recent_mean - early_mean
            else:
                recent_trend = slope * 3
            return slope, recent_trend
        
        # Apply per subject to train
        slope_vals = np.zeros(len(feat_sorted))
        trend_vals = np.zeros(len(feat_sorted))
        for sid, group in feat_sorted.groupby('subject_id'):
            idx = group.index
            slope_vals[idx], trend_vals[idx] = compute_slope(group, feat_name)
        
        train_slope_name = f'trend_{feat_name}_slope'
        train_trend_name = f'trend_{feat_name}_recent'
        feat[train_slope_name] = slope_vals
        feat[train_trend_name] = trend_vals
        enhanced_cols.extend([train_slope_name, train_trend_name])
        
        # For test: compute within test set
        test_slope_vals = np.zeros(len(ftst_sorted))
        test_trend_vals = np.zeros(len(ftst_sorted))
        for sid, group in ftst_sorted.groupby('subject_id'):
            idx = group.index
            ts, tt = compute_slope(group, feat_name)
            test_slope_vals[idx] = ts
            test_trend_vals[idx] = tt
        ftst[train_slope_name] = test_slope_vals
        ftst[train_trend_name] = test_trend_vals

print(f"  Added {len(enhanced_cols)} enhanced features")

# 3c. Subject-level aggregate stats of base features
print("  [3c] Subject-level aggregate stats...")
subject_agg_feats = [
    'wPedo_pedo_step_mean', 'mActivity_m_activity_mean',
    'wHr_hr_mean', 'mLight_m_light_mean',
    'mScreenStatus_m_screen_use_mean',
    'mAmbience_ambience_top5_sum',
    'mWifi_wifi_avg_rssi_mean', 'mBle_ble_device_count_mean',
    'mACStatus_m_charging_mean',
]

for feat_name in subject_agg_feats:
    if feat_name in feat.columns:
        # Mean, std, min, max per subject (computed from train set)
        grp_stats = feat.groupby('subject_id')[feat_name].agg(['mean','std','min','max'])
        grp_stats.columns = [f'{feat_name}_subj_{s}' for s in ['mean','std','min','max']]
        
        feat = feat.merge(grp_stats, on='subject_id', how='left')
        
        # Test: use same subject-level stats from train
        grp_stats_t = ftst.groupby('subject_id')[feat_name].agg(['mean','std','min','max'])
        grp_stats_t.columns = [f'{feat_name}_subj_{s}' for s in ['mean','std','min','max']]
        ftst = ftst.merge(grp_stats_t, on='subject_id', how='left')
        
        for s in ['mean','std','min','max']:
            enhanced_cols.append(f'{feat_name}_subj_{s}')

# 3d. Deviation from subject mean (personalized residual)
print("  [3d] Deviation from subject mean...")
for feat_name in subject_agg_feats:
    if feat_name in feat.columns:
        subj_mean_col = f'{feat_name}_subj_mean'
        subj_std_col = f'{feat_name}_subj_std'
        if subj_mean_col in feat.columns and subj_std_col in feat.columns:
            dev_name = f'{feat_name}_dev'
            feat[dev_name] = (feat[feat_name] - feat[subj_mean_col]) / feat[subj_std_col].clip(lower=1e-8)
            ftst[dev_name] = (ftst[feat_name] - ftst[subj_mean_col]) / ftst[subj_std_col].clip(lower=1e-8)
            enhanced_cols.append(dev_name)

print(f"  Total enhanced cols so far: {len(enhanced_cols)}")

# ============================================================
# 4. Prepare feature sets for training
# ============================================================
print("\n[4] Preparing feature sets...")

# Feature sets to evaluate:
# V128_base: V127 base features (no external) 
# V128_ext: V127 + external features
# V128_loo: V128_base + LOO target encoding
# V128_enhanced: V128_loo + enhanced features
# V128_full: V128_enhanced + external

# Load external features
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

# Merge external features into feat via direct assignment
feat_ext = feat.copy()
for ec in ext_cols:
    if ec in ext_merge.columns:
        feat_ext[ec] = pd.NA  # fill default
    else:
        feat_ext[ec] = pd.NA

feat_dt = pd.to_datetime(feat_ext['lifelog_date']).dt.normalize()
for _, row in ext_merge.iterrows():
    mask = feat_dt == row['lifelog_date']
    for c in ext_cols:
        if c in ext_merge.columns:
            feat_ext.loc[mask, c] = row[c]

# Merge external features into ftst via direct assignment
ftst_ext = ftst.copy()
for ec in ext_cols:
    ftst_ext[ec] = pd.NA

ftst_dt = pd.to_datetime(ftst_ext['lifelog_date']).dt.normalize()
ext_merge_t = ext_merge.copy()
for _, row in ext_merge_t.iterrows():
    mask = ftst_dt == row['lifelog_date']
    for c in ext_cols:
        if c in ext_merge_t.columns:
            ftst_ext.loc[mask, c] = row[c]

# Feature set definitions
# Use get_numeric_cols to filter out non-numeric columns (like 'date', 'mAmbience_ambience_max_cat')
all_numeric_in_feat = get_numeric_cols(feat)

# LOO features
loo_feature_cols = []
for t in TARGETS:
    d = loo_targets[t]
    loo_feature_cols.extend([d['mean_col'], d['std_col'], d['n_col']])

# Enhanced features (excluding LOO and base)
enhanced_feature_cols = []
for c in feat.columns:
    if c.startswith('trend_') or c.endswith('_dev') or '_subj_' in c or '_x_' in c:
        if c in all_numeric_in_feat and c not in loo_feature_cols:
            enhanced_feature_cols.append(c)

# Base set: numeric columns minus LOO and enhanced
base_feature_cols = [c for c in all_numeric_in_feat 
                     if c not in loo_feature_cols and c not in enhanced_feature_cols]

# All LOO + enhanced + base = full internal
full_internal_cols = base_feature_cols + loo_feature_cols + enhanced_feature_cols

# External features
all_ext_cols = full_internal_cols + ext_cols

print(f"  Base features: {len(base_feature_cols)}")
print(f"  LOO features: {len(loo_feature_cols)}")
print(f"  Enhanced features: {len(enhanced_feature_cols)}")
print(f"  Full internal: {len(full_internal_cols)}")
print(f"  External cols: {len(ext_cols)}")

# ============================================================
# 5. Train strategies
# ============================================================
print("\n[5] Training strategies...")

strategies = [
    ('V115_base', False, False),
    ('V123_pair', True, False),
    ('V121_p+t', True, True),
]

results = {}

# Feature set configurations to test
fs_configs = [
    ('base', base_feature_cols, feat, ftst),
    ('full_internal', full_internal_cols, feat, ftst),
    ('full_external', all_ext_cols, feat_ext, ftst_ext),
]

for fs_name, feat_cols, feat_df, ftst_df in fs_configs:
    print(f"\n  === {fs_name} ({len(feat_cols)} features) ===")
    
    for strat_name, do_pair, do_trans in strategies:
        tag = f"v128_{fs_name}_{strat_name}"
        for target in TARGETS:
            y = feat_df[target].values.astype(np.float64)
            cfg = CFGS[V53_SWEEP[target]['cfg']]
            
            cols = list(feat_cols)
            cols = remove_leak(cols, target)
            working_df = feat_df.copy()
            
            if do_pair:
                ranked = rank_features(working_df, cols, target)
                top8 = ranked[:8]
                working_df, added = add_pairwise(working_df, top8)
                post_pair = get_numeric_cols(working_df)
                cols = remove_leak(post_pair, target)
                
                # Add pairwise features to working feature list
                for a in added:
                    if a not in cols and a in working_df.columns:
                        cols.append(a)
            
            if do_trans and do_pair:
                ranked = rank_features(working_df, cols, target)
                top10 = ranked[:10]
                for f_name in top10:
                    if f_name in working_df.columns:
                        working_df[f_name + '_rank'] = pd.Series(working_df[f_name].fillna(0)).rank(pct=True).values
                cols = [c for c in get_numeric_cols(working_df) if c not in META_COLS | set(TARGETS)]
                cols = remove_leak(cols, target)
                for f_name in top10:
                    rank_col = f_name + '_rank'
                    if rank_col not in cols:
                        cols.append(rank_col)
            
            oof, _ = train_cv(working_df, None, cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            ll = log_loss(y, iso_cal, labels=[0,1])
            
            results[(tag, target)] = {'iso_cal': iso_cal, 'll': ll, 'n_feat': len(cols)}
            print(f"    {tag} / {target}: LL={ll:.5f} (n_feat={len(cols)})")

# ============================================================
# 6. Ensemble
# ============================================================
print("\n[6] Ensemble Results (0.35*V121 + 0.25*V123 + 0.40*V115)")

for fs_name in ['base', 'full_internal', 'full_external']:
    print(f"\n  --- {fs_name} Ensemble ---")
    ens_oof = {}
    for target in TARGETS:
        ens = (W121*results[f"v128_{fs_name}_V121_p+t", target]['iso_cal'] +
               W123*results[f"v128_{fs_name}_V123_pair", target]['iso_cal'] +
               W115*results[f"v128_{fs_name}_V115_base", target]['iso_cal'])
        ens_oof[target] = ens
        ll = log_loss(y_dict[target], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
        print(f"    {target}: {ll:.5f}")
    avg = np.mean([log_loss(y_dict[t], np.clip(ens_oof[t], 0.0001, 0.9999), labels=[0,1]) for t in TARGETS])
    print(f"    AVG: {avg:.5f}")
    results[f'_ens_{fs_name}_avg'] = avg
    print(f"    vs V127 (0.53731): {avg - 0.53731:+.5f}")

# ============================================================
# 7. Compare with V252 results (for reference)
# ============================================================
print("\n[7] Comparison with V252")
v252_ext_avg = 0.64471  # From V252 ext ensemble
for fs_name in ['base', 'full_internal', 'full_external']:
    avg = results[f'_ens_{fs_name}_avg']
    print(f"  V128 {fs_name}: {avg:.5f} (vs V252: {avg - v252_ext_avg:+.5f})")

# ============================================================
# 8. Save results
# ============================================================
print(f"\n[8] Saving results...")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')

log = {
    'name': 'V128_Groupwise_Target_Encode',
    'timestamp': ts,
    'v127_baseline_oof': 0.53731,
}
for fs_name in ['base', 'full_internal', 'full_external']:
    log[f'{fs_name}_avg'] = float(results[f'_ens_{fs_name}_avg'])
    for t in TARGETS:
        ens = (W121*results[f"v128_{fs_name}_V121_p+t", t]['iso_cal'] +
               W123*results[f"v128_{fs_name}_V123_pair", t]['iso_cal'] +
               W115*results[f"v128_{fs_name}_V115_base", t]['iso_cal'])
        ll = log_loss(y_dict[t], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
        log[f'{fs_name}_{t}'] = float(ll)

with open(EXPERIMENTS / f'v128_{ts}.json', 'w') as f:
    json.dump(log, f, indent=2, default=str)
print(f"  Saved: experiments/v128_{ts}.json")

# Per-target detailed results
detail_log = {}
for tag_key in sorted(results.keys()):
    if isinstance(tag_key, tuple):
        detail_log[str(tag_key)] = {
            'll': results[tag_key]['ll'],
            'n_feat': results[tag_key]['n_feat'],
        }
with open(EXPERIMENTS / f'v128_detail_{ts}.json', 'w') as f:
    json.dump(detail_log, f, indent=2, default=str)
print(f"  Detail saved: experiments/v128_detail_{ts}.json")

# ============================================================
# 9. Best configuration analysis
# ============================================================
print(f"\n[9] Best configuration analysis")
best_fs = min(['base', 'full_internal', 'full_external'],
              key=lambda x: results[f'_ens_{x}_avg'])
print(f"  Best feature set: {best_fs} (AVG LL: {results[f'_ens_{best_fs}_avg']:.5f})")

# Per-target improvements over V127
print(f"\n  Per-target improvement over V127 (0.53731):")
for fs_name in ['base', 'full_internal', 'full_external']:
    if fs_name == best_fs:
        print(f"\n  ** {fs_name} ** (best):")
        ens_oof = {}
        for target in TARGETS:
            ens = (W121*results[f"v128_{fs_name}_V121_p+t", target]['iso_cal'] +
                   W123*results[f"v128_{fs_name}_V123_pair", target]['iso_cal'] +
                   W115*results[f"v128_{fs_name}_V115_base", target]['iso_cal'])
            ll = log_loss(y_dict[target], np.clip(ens, 0.0001, 0.9999), labels=[0,1])
            delta = ll - 0.53731
            print(f"    {target}: {ll:.5f} (delta: {delta:+.5f})")

print(f"\nV128 COMPLETE ✓")
print(f"Best OOF: {results[f'_ens_{best_fs}_avg']:.5f}")
print(f"Improvement over V127: {results[f'_ens_{best_fs}_avg'] - 0.53731:+.5f}")
