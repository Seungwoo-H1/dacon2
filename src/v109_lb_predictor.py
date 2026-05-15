"""V109: LB Predictor + Meta-analysis + Multi-Direction Experiments

Goal: Build LB prediction model from submission history, then use it to guide experiments.

Key insights from data:
- V53: train=0.54793, LB=0.65358, gap=0.10565
- All mean-matched submissions: OOF mean = train mean
- LB depends on test-set predictions, not train-set predictions
- Key factors: prediction diversity, calibration error on test, shift patterns

Approach:
1. Build LB predictor from known/predicted data points
2. Analyze all submissions for patterns
3. Run multiple experiment directions in parallel
4. Ensemble optimization
"""
import sys, re, gc, time, warnings, logging, json, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_AGGRESSIVE = {'nl': 5, 'md': 2, 'lr': 0.01, 'ne': 2000, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 15.0, 'mc': 30}
CFG_REGULARIZED = {'nl': 40, 'md': 3, 'lr': 0.1, 'ne': 100, 'ss': 0.9, 'cb': 0.9, 'ra': 0.5, 'rl': 2.0, 'mc': 3}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48,
        'safety': CFG_SAFETY, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP,
        'aggressive': CFG_AGGRESSIVE, 'regularized': CFG_REGULARIZED}

SEEDS = [42, 123, 7, 999, 777, 2026, 111, 555]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

# ============================================================
# STEP 0: LB Predictor Analysis
# ============================================================
log.info("=" * 60)
log.info("STEP 0: LB Predictor - Meta-analysis of submission history")
log.info("=" * 60)

train_df = pd.read_parquet(DATA / "features.parquet")
y_train = {t: train_df[t].values for t in TARGETS}
y_means = {t: y_train[t].mean() for t in TARGETS}

# Known data points:
# V53: train=0.54793, LB=0.65358
# V53 CV=0.54793, LB=0.65358 → gap=0.10565
# V102: predicted LB~0.64
# V99: predicted LB~0.73
# V106: no LB yet (not submitted)

# Analyze all OOF files for meta-features
oof_files = sorted([f for f in Path('data_processed').glob('oof_v*.csv')])
submission_files = sorted([f for f in SUBMIT.glob('*.csv')])

# For each OOF file, compute prediction characteristics on train
meta_features = []
for of in oof_files:
    oof = pd.read_csv(of)
    missing = [t for t in TARGETS if t not in oof.columns]
    if missing:
        continue
    
    name = of.stem.replace('oof_', '')
    oof_vals = {t: oof[t].values for t in TARGETS}
    
    # CV log_loss
    cv_ll = np.mean([log_loss(y_train[t], oof_vals[t], labels=[0,1]) for t in TARGETS])
    
    # Per-target OOF std
    oof_stds = {t: float(np.std(oof_vals[t])) for t in TARGETS}
    mean_oof_std = np.mean(list(oof_stds.values()))
    std_oof_std = np.std(list(oof_stds.values()))
    
    # Entropy
    eps = 1e-10
    entropies = {t: -(np.clip(oof_vals[t], eps, 1-eps) * np.log(np.clip(oof_vals[t], eps, 1-eps)) +
                       (1-np.clip(oof_vals[t], eps, 1-eps)) * np.log(1-np.clip(oof_vals[t], eps, 1-eps))).mean()
                 for t in TARGETS}
    mean_entropy = np.mean(list(entropies.values()))
    
    # Calibration: how close OOF mean is to train mean (should be ~0 due to mean_match)
    cal_error = np.mean([abs(oof_vals[t].mean() - y_means[t]) for t in TARGETS])
    
    # Prediction range (max - min)
    ranges = {t: oof_vals[t].max() - oof_vals[t].min() for t in TARGETS}
    mean_range = np.mean(list(ranges.values()))
    
    # Brier score
    brier = np.mean([((oof_vals[t] - y_train[t])**2).mean() for t in TARGETS])
    
    meta_features.append({
        'name': name,
        'cv_ll': cv_ll,
        'mean_oof_std': mean_oof_std,
        'std_oof_std': std_oof_std,
        'mean_entropy': mean_entropy,
        'cal_error': cal_error,
        'mean_range': mean_range,
        'brier': brier,
        'num_models': 1,
    })

# For known LB points, add to training set
# V53: train=0.54793, LB=0.65358
# V99 predicted LB=0.73 (from earlier analysis)
# We need to estimate LB from submission characteristics

# Key insight from V101 analysis:
# LB = mean(log_loss per target) on test
# The gap between train LL and LB is driven by:
# 1. Test set distribution shift from train
# 2. Model overfitting to train patterns
# 3. Calibration mismatch on test

# LB predictor: use submission std and entropy as proxies
# Higher entropy + lower std → better test generalization
# But too low std = overconfident = bad

# For now, let's predict LB using:
# LB ≈ cv_ll + gap
# gap correlates with: (test_mean - train_mean) for each target
# Since test means are unknown, we use:
# - Submission std (lower = more uniform = potentially worse generalization)
# - Entropy (higher = more informative predictions)

log.info(f"Collected {len(meta_features)} OOF data points")
for mf in sorted(meta_features, key=lambda x: x['cv_ll']):
    log.info(f"  {mf['name']:20s} cv={mf['cv_ll']:.5f} entropy={mf['mean_entropy']:.4f} std={mf['mean_oof_std']:.4f} brier={mf['brier']:.5f}")

# ============================================================
# STEP 1: Load data + personalization
# ============================================================
log.info("\n" + "=" * 60)
log.info("STEP 1: Loading data + personalization")
log.info("=" * 60)

feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")

for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

# Base features
feature_cols = [c for c in feat.columns
                if c not in META | set(TARGETS)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
log.info(f"Base features: {len(feature_cols)}")

# Personalization
log.info("Adding personalization...")
feat_all = feat.copy()
personal_cols_added = []
for col in feature_cols:
    col_filled = feat_all[col].fillna(0)
    grp = col_filled.groupby(feat_all['subject_id']).agg(['mean', 'std'])
    grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
    grp = grp.reset_index()
    feat_all = feat_all.merge(grp, on='subject_id', how='left')
    mask_zero = feat_all[f'{col}_subj_std'] == 0
    mask_null = feat_all[col].isnull()
    feat_all[f'{col}_zscore'] = np.where(
        mask_zero | mask_null, 0.0,
        (feat_all[col].fillna(0) - feat_all[f'{col}_subj_mean']) /
        np.maximum(feat_all[f'{col}_subj_std'], 1e-8))
    personal_cols_added.append(f'{col}_zscore')
    gc.collect()

# Test personalization
test_feat = feat_test.copy()
for col in feature_cols:
    col_filled = feat_all[col].fillna(0)
    grp = col_filled.groupby(feat_all['subject_id']).agg(['mean', 'std'])
    grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
    grp = grp.reset_index()
    test_feat = test_feat.merge(grp, on='subject_id', how='left')
    test_feat[f'{col}_zscore'] = np.where(
        (test_feat[f'{col}_subj_std'] == 0) | test_feat[col].isnull(), 0.0,
        (test_feat[col].fillna(0) - test_feat[f'{col}_subj_mean']) /
        np.maximum(test_feat[f'{col}_subj_std'], 1e-8))

all_cols = feature_cols + personal_cols_added
log.info(f"After personalization: {feat_all.shape}, personal_cols={len(personal_cols_added)}")

# ============================================================
# STEP 2: Build multiple experiment pipelines
# ============================================================
# Strategy: Run 4 experiment directions simultaneously
# 
# Exp A: Aggressive regularization (CFG_AGGRESSIVE)
# Exp B: Regularized wide (CFG_REGULARIZED) 
# Exp C: Stacking ensemble (base models → meta-learner)
# Exp D: Target-aware calibration (per-target post-processing)

log.info("\n" + "=" * 60)
log.info("STEP 2: Running 4 experiment directions")
log.info("=" * 60)

results = {}  # name → {oof, test, metrics}

for exp_name, cfgs_to_use in [
    ('a_aggressive', ['aggressive']),
    ('b_regularized', ['regularized']),
    ('c_multi_seed_v108', ['wide', 'deep', 'v48', 'safety']),
    ('d_shallow_ensemble', ['wide', 'deep', 'v48']),
]:
    t_start_exp = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"EXP: {exp_name} (configs: {cfgs_to_use})")
    log.info(f"{'='*60}")
    
    oof_all = {t: [] for t in TARGETS}
    test_all = {t: [] for t in TARGETS}
    
    for target in TARGETS:
        y = feat_all[target].values.astype(np.float64)
        train_rate = y.mean()
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        # Remove leak columns
        leak = LEAK_S if target.startswith('S') else LEAK_Q
        safe_cols = [c for c in all_cols if c not in leak]
        
        # For aggressive: use ALL safe cols
        # For others: use top feature selection based on importance
        if exp_name in ['a_aggressive', 'b_regularized']:
            # Use all safe columns (no feature selection)
            sel_cols = safe_cols
        else:
            # Use feature selection: top 50 from simple ranking
            # Simple variance-based feature selection (fast)
            variances = feat_all[safe_cols].var()
            # Remove near-constant features
            variances = variances[variances > 1e-10]
            top_feats = variances.nlargest(80).index.tolist()
            # Also include top personalization features
            sel_cols = top_feats[:50] + [c for c in personal_cols_added if c in feat_all.columns][:30]
            sel_cols = [c for c in sel_cols if c in safe_cols]
        
        sn = [sanitize(c) for c in sel_cols]
        gkf = GroupKFold(n_splits=5)
        
        for cfg_name in cfgs_to_use:
            cfg = CFGS[cfg_name]
            
            for seed in SEEDS:
                oof_fold = np.zeros(450)
                
                for tr_i, va_i in gkf.split(feat_all, y, feat_all['subject_id']):
                    X_tr = feat_all.iloc[tr_i][sel_cols].fillna(0).values.astype(np.float64)
                    X_va = feat_all.iloc[va_i][sel_cols].fillna(0).values.astype(np.float64)
                    
                    ds_train = lgb.Dataset(X_tr, label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
                    ds_val = lgb.Dataset(X_va, label=y[va_i], feature_name=sn, reference=ds_train, params={'verbose': '-1'})
                    
                    cfg_full = {
                        'objective': 'binary', 'metric': 'binary_logloss',
                        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                        'min_child_samples': cfg['mc'],
                        'random_state': seed, 'scale_pos_weight': spw,
                    }
                    
                    model = lgb.train(cfg_full, ds_train, num_boost_round=cfg['ne'],
                                     valid_sets=[ds_val],
                                     callbacks=[lgb.early_stopping(30, verbose=False),
                                               lgb.log_evaluation(0)])
                    
                    oof_fold[va_i] = model.predict(X_va)
                    
                    del ds_train, ds_val, model
                    gc.collect()
                
                oof_all[target].append(oof_fold)
                
                # Test prediction: retrain on all data
                X_all_feat = feat_all[sel_cols].fillna(0).values.astype(np.float64)
                X_test_feat = test_feat[sel_cols].fillna(0).values.astype(np.float64)
                
                ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
                cfg_all = {**cfg_full, 'n_estimators': cfg['ne']}
                model_test = lgb.train(cfg_all, ds_all, num_boost_round=cfg['ne'])
                test_pred = model_test.predict(X_test_feat)
                test_all[target].append(test_pred)
                
                del ds_all, model_test
                gc.collect()
        
        # Average across all models for this target
        oof_arr = np.array(oof_all[target])
        test_arr = np.array(test_all[target])
        oof_avg = oof_arr.mean(axis=0)
        test_avg = test_arr.mean(axis=0)
    
    # Compute metrics
    exp_oof = {}
    exp_test = {}
    exp_metrics = {}
    
    for target in TARGETS:
        oof_arr = np.array(oof_all[target])
        test_arr = np.array(test_all[target])
        oof_avg = oof_arr.mean(axis=0)
        test_avg = test_arr.mean(axis=0)
        
        oof_cal = mean_match(oof_avg, y_means[target])
        test_cal = mean_match(test_avg, y_means[target])
        
        val_ll = log_loss(y_train[target], oof_cal, labels=[0,1])
        
        exp_oof[target] = oof_cal
        exp_test[target] = test_cal
        exp_metrics[target] = {
            'oof_ll': round(val_ll, 5),
            'oof_mean': round(oof_avg.mean(), 4),
            'test_mean': round(test_avg.mean(), 4),
            'oof_std': round(float(np.std(oof_cal)), 4),
            'test_std': round(float(np.std(test_cal)), 4),
            'n_models': len(oof_all[target]),
        }
    
    avg_oof = np.mean([exp_metrics[t]['oof_ll'] for t in TARGETS])
    
    exp_metrics['avg_oof_ll'] = round(avg_oof, 5)
    exp_metrics['total_time_s'] = round(time.time() - t_start_exp, 0)
    
    results[exp_name] = {
        'oof': exp_oof,
        'test': exp_test,
        'metrics': exp_metrics,
        'avg_oof': avg_oof,
    }
    
    log.info(f"\n{'='*60}")
    log.info(f"EXP {exp_name} COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"{'Target':<10} {'OOF LL':>8} {'OOF mean':>10} {'Test mean':>10} {'OOF std':>8} {'Models':>8}")
    log.info(f"{'-'*70}")
    for t in TARGETS:
        m = exp_metrics[t]
        log.info(f"{t:<10} {m['oof_ll']:>8.5f} {m['oof_mean']:>10.4f} {m['test_mean']:>10.4f} {m['oof_std']:>8.4f} {m['n_models']:>8}")
    log.info(f"{'AVG':<10} {avg_oof:>8.5f}")

# ============================================================
# STEP 3: Multi-experiment ensemble
# ============================================================
log.info("\n" + "=" * 60)
log.info("STEP 3: Multi-experiment ensemble optimization")
log.info("=" * 60)

# Try different ensemble combinations of the 4 experiments
exp_names = list(results.keys())
log.info(f"Experiments: {exp_names}")

# Compute pairwise correlation between experiments
log.info("\nExperiment pairwise OOF correlations:")
corr_matrix = {}
for e1 in exp_names:
    corr_matrix[e1] = {}
    for e2 in exp_names:
        oof_e1 = np.array([results[e1]['oof'][t] for t in TARGETS])
        oof_e2 = np.array([results[e2]['oof'][t] for t in TARGETS])
        corr = np.corrcoef(oof_e1, oof_e2)[0, 1] if len(oof_e1) > 1 else 0
        corr_matrix[e1][e2] = round(corr, 3)
        if e1 != e2:
            log.info(f"  {e1} vs {e2}: r={corr:.3f}")

# Try all subsets and find best ensemble
# For each subset, use equal weights and evaluate
from itertools import combinations

best_oof = float('inf')
best_combo = None
best_weights = None

# Single experiments
for en in exp_names:
    avg = results[en]['avg_oof']
    if avg < best_oof:
        best_oof = avg
        best_combo = [en]
        best_weights = {en: 1.0}

# Pairs
for combo in combinations(exp_names, 2):
    # Equal weight ensemble
    combined_oof = np.zeros(450)
    combined_test = np.zeros(250)
    for en in combo:
        for t in TARGETS:
            combined_oof += results[en]['oof'][t]
            combined_test += results[en]['test'][t]
    n = len(combo)
    combined_oof /= n
    combined_test /= n
    
    train_rate_avg = np.mean([y_means[t] for t in TARGETS])
    combined_oof_cal = mean_match(combined_oof, train_rate_avg)
    combined_test_cal = mean_match(combined_test, train_rate_avg)
    
    avg_ll = np.mean([log_loss(y_train[t], combined_oof_cal, labels=[0,1]) for t in TARGETS])
    
    if avg_ll < best_oof:
        best_oof = avg_ll
        best_combo = list(combo)
        best_weights = {en: 1.0/len(combo) for en in combo}
        log.info(f"  NEW BEST: {combo} → OOF={avg_ll:.5f}")

# Triplets
for combo in combinations(exp_names, 3):
    combined_oof = np.zeros(450)
    combined_test = np.zeros(250)
    for en in combo:
        for t in TARGETS:
            combined_oof += results[en]['oof'][t]
            combined_test += results[en]['test'][t]
    n = len(combo)
    combined_oof /= n
    combined_test /= n
    
    train_rate_avg = np.mean([y_means[t] for t in TARGETS])
    combined_oof_cal = mean_match(combined_oof, train_rate_avg)
    combined_test_cal = mean_match(combined_test, train_rate_avg)
    
    avg_ll = np.mean([log_loss(y_train[t], combined_oof_cal, labels=[0,1]) for t in TARGETS])
    
    if avg_ll < best_oof:
        best_oof = avg_ll
        best_combo = list(combo)
        best_weights = {en: 1.0/len(combo) for en in combo}
        log.info(f"  NEW BEST: {combo} → OOF={avg_ll:.5f}")

# All four
combo = tuple(exp_names)
combined_oof = np.zeros(450)
combined_test = np.zeros(250)
for en in combo:
    for t in TARGETS:
        combined_oof += results[en]['oof'][t]
        combined_test += results[en]['test'][t]
n = len(combo)
combined_oof /= n
combined_test /= n

train_rate_avg = np.mean([y_means[t] for t in TARGETS])
combined_oof_cal = mean_match(combined_oof, train_rate_avg)
combined_test_cal = mean_match(combined_test, train_rate_avg)

avg_ll = np.mean([log_loss(y_train[t], combined_oof_cal, labels=[0,1]) for t in TARGETS])

if avg_ll < best_oof:
    best_oof = avg_ll
    best_combo = list(combo)
    best_weights = {en: 1.0/len(combo) for en in combo}
    log.info(f"  NEW BEST: {combo} → OOF={avg_ll:.5f}")

log.info(f"\nBest ensemble: {best_combo}")
log.info(f"Best OOF: {best_oof:.5f}")
log.info(f"Weights: {best_weights}")

# Per-target breakdown
log.info("\nPer-target ensemble OOF:")
for t in TARGETS:
    ll = log_loss(y_train[t], combined_oof_cal, labels=[0,1])
    log.info(f"  {t}: {ll:.5f}")

# ============================================================
# STEP 4: Save best submission + logs
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save combined OOF
oof_df = pd.DataFrame({t: combined_oof_cal for t in TARGETS})
oof_df.insert(0, 'subject_id', feat_all['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat_all['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat_all['lifelog_date'].values)
oof_path = DATA / f'oof_v109_ensemble_{ts}.csv'
oof_df.to_csv(oof_path, index=False)
log.info(f"\nSaved OOF: {oof_path}")

# Save test submission
sub_df = pd.DataFrame({t: combined_test_cal for t in TARGETS})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v109_ensemble_{ts}.csv'
sub_df.to_csv(sub_path, index=False)
log.info(f"Saved submission: {sub_path}")
log.info(f"Test means: { {t: round(sub_df[t].mean(), 4) for t in TARGETS} }")
log.info(f"Test stds:  { {t: round(sub_df[t].std(), 4) for t in TARGETS} }")

# Save experiment log
exp_log = {
    'version': 'V109',
    'timestamp': ts,
    'experiments': exp_names,
    'experiment_configs': {en: CFGS[c] for en in exp_names for c in CFGS if c in results[en].get('metrics', {}).get('target_configs', {})},
    'pairwise_correlations': corr_matrix,
    'ensemble_results': {
        'best_combo': best_combo,
        'best_weights': best_weights,
        'best_oof': round(best_oof, 5),
        'per_target_oof': {t: round(log_loss(y_train[t], combined_oof_cal, labels=[0,1]), 5) for t in TARGETS},
    },
    'per_experiment_oof': {en: round(results[en]['avg_oof'], 5) for en in exp_names},
    'test_submission': str(sub_path.name),
    'oof_file': str(oof_path.name),
    'test_means': {t: round(sub_df[t].mean(), 4) for t in TARGETS},
    'test_stds': {t: round(sub_df[t].std(), 4) for t in TARGETS},
    'total_time_s': round(time.time() - t_start_exp, 0),
}
with open(EXPERIMENTS / f'v109_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)
log.info(f"\nLog: {EXPERIMENTS / f'v109_{ts}.json'}")
log.info(f"Done in {time.time()-t_start_exp:.0f}s")
