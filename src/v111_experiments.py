"""V111: V108 baseline reproduction + personalization variants + 8 seeds

Strategy:
1. Reproduce V108 exactly (baseline)
2. Add personalization (z-score per subject) 
3. Add pairwise interactions on top of personalization
4. Compare baseline vs personalization vs interactions
5. Optimize ensemble across all variants

Key from V109/V110: personalization + feature selection = worse.
So this time: use ALL features (no top-20 selection), test personalization effect directly.
"""
import sys, re, gc, time, warnings, logging, json, os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
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
CFG_V111A = {'nl': 25, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.75, 'cb': 0.75, 'ra': 1.5, 'rl': 4.0, 'mc': 6}
CFG_V111B = {'nl': 18, 'md': 4, 'lr': 0.015, 'ne': 1200, 'ss': 0.65, 'cb': 0.65, 'ra': 2.0, 'rl': 6.0, 'mc': 10}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48,
        'safety': CFG_SAFETY, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP,
        'v111a': CFG_V111A, 'v111b': CFG_V111B}

SEEDS = [42, 123, 7, 999]
SEEDS_8 = [42, 123, 7, 999, 777, 2026, 111, 555]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

# ============================================================
# Load data
# ============================================================
t_start = time.time()
log.info("Loading features...")

train_df = pd.read_parquet(DATA / "features.parquet")
y_train = {t: train_df[t].values for t in TARGETS}
feat = train_df.copy()
feat_test = pd.read_parquet(DATA / "test_features.parquet")

for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feature_cols = [c for c in feat.columns
                if c not in META | set(TARGETS)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
log.info(f"Base features: {len(feature_cols)}")

# ============================================================
# Run 3 experiment directions
# A: V108 baseline (raw features) × 8 seeds
# B: V108 + personalization × 8 seeds  
# C: V108 + personalization + pairwise interactions × 8 seeds
# ============================================================
log.info("\n" + "=" * 60)
log.info("V111: Running experiment directions")
log.info("=" * 60)

results = {}

for exp_name, use_personal, use_interact, desc in [
    ('A_baseline', False, False, 'V108 baseline (raw features)'),
    ('B_personal', True, False, 'Personalization (z-score)'),
    ('C_personal_inter', True, True, 'Personalization + pairwise interactions'),
]:
    t_exp = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"EXP {exp_name}: {desc}")
    log.info(f"{'='*60}")
    
    # Build featureset
    feat_use = feat.copy()
    feat_test_use = feat_test.copy()
    cols_used = list(feature_cols)
    
    if use_personal:
        log.info("  Adding personalization...")
        personal_cols_added = []
        for col in feature_cols:
            col_filled = feat_use[col].fillna(0)
            grp = col_filled.groupby(feat_use['subject_id']).agg(['mean', 'std'])
            grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
            grp = grp.reset_index()
            feat_use = feat_use.merge(grp, on='subject_id', how='left')
            mask_zero = feat_use[f'{col}_subj_std'] == 0
            mask_null = feat_use[col].isnull()
            feat_use[f'{col}_zscore'] = np.where(
                mask_zero | mask_null, 0.0,
                (feat_use[col].fillna(0) - feat_use[f'{col}_subj_mean']) /
                np.maximum(feat_use[f'{col}_subj_std'], 1e-8))
            personal_cols_added.append(f'{col}_zscore')
            gc.collect()
        
        # Test personalization
        for col in feature_cols:
            grp = feat_use[col].fillna(0).groupby(feat_use['subject_id']).agg(['mean', 'std'])
            grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
            grp = grp.reset_index()
            feat_test_use = feat_test_use.merge(grp, on='subject_id', how='left')
            feat_test_use[f'{col}_zscore'] = np.where(
                (feat_test_use[f'{col}_subj_std'] == 0) | feat_test_use[col].isnull(), 0.0,
                (feat_test_use[col].fillna(0) - feat_test_use[f'{col}_subj_mean']) /
                np.maximum(feat_test_use[f'{col}_subj_std'], 1e-8))
        
        cols_used = feature_cols + personal_cols_added
        log.info(f"  Personal cols added: {len(personal_cols_added)}")
    
    if use_interact:
        log.info("  Adding pairwise interactions...")
        interact_cols = []
        # Top 10 most important features from a quick ranking
        # Use variance as proxy for importance
        variances = feat_use[cols_used].var()
        top_feats = variances.nlargest(15).index.tolist()
        
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                f1, f2 = top_feats[i], top_feats[j]
                if f1 not in feat_use.columns or f2 not in feat_use.columns:
                    continue
                feat_use[f'inter_{f1}_{f2}'] = feat_use[f1].fillna(0) * feat_use[f2].fillna(0)
                interact_cols.append(f'inter_{f1}_{f2}')
                feat_test_use[f'inter_{f1}_{f2}'] = feat_test_use[f1].fillna(0) * feat_test_use[f2].fillna(0)
        
        cols_used = cols_used + interact_cols
        log.info(f"  Interact cols added: {len(interact_cols)}")
    
    log.info(f"  Total features: {len(cols_used)}")
    
    # Train models
    oof_all = {t: [] for t in TARGETS}
    test_all = {t: [] for t in TARGETS}
    
    for target in TARGETS:
        y = feat_use[target].values.astype(np.float64)
        train_rate = y.mean()
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        # Remove leak columns
        leak = LEAK_S if target.startswith('S') else LEAK_Q
        safe_cols = [c for c in cols_used if c not in leak]
        
        sn = [sanitize(c) for c in safe_cols]
        gkf = GroupKFold(n_splits=5)
        
        # Use all configs + all seeds
        for cfg_name in CFGS:
            cfg = CFGS[cfg_name]
            
            for seed in SEEDS_8:
                oof_fold = np.zeros(450)
                
                for tr_i, va_i in gkf.split(feat_use, y, feat_use['subject_id']):
                    X_tr = feat_use.iloc[tr_i][safe_cols].fillna(0).values.astype(np.float64)
                    X_va = feat_use.iloc[va_i][safe_cols].fillna(0).values.astype(np.float64)
                    
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
                
                # Test prediction
                X_all_feat = feat_use[safe_cols].fillna(0).values.astype(np.float64)
                X_test_feat = feat_test_use[safe_cols].fillna(0).values.astype(np.float64)
                
                ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
                cfg_all = {**cfg_full, 'n_estimators': cfg['ne']}
                model_test = lgb.train(cfg_all, ds_all, num_boost_round=cfg['ne'])
                test_pred = model_test.predict(X_test_feat)
                test_all[target].append(test_pred)
                
                del ds_all, model_test
                gc.collect()
    
    # Evaluate
    avg_oof = 0
    target_metrics = {}
    for t in TARGETS:
        oof_arr = np.array(oof_all[t])
        test_arr = np.array(test_all[t])
        oof_avg = oof_arr.mean(axis=0)
        test_avg = test_arr.mean(axis=0)
        
        oof_cal = mean_match(oof_avg, y_train[t].mean())
        test_cal = mean_match(test_avg, y_train[t].mean())
        
        ll = log_loss(y_train[t], oof_cal, labels=[0,1])
        target_metrics[t] = {
            'oof_ll': round(ll, 5),
            'oof_mean': round(float(oof_cal.mean()), 4),
            'test_mean': round(float(test_cal.mean()), 4),
            'oof_std': round(float(np.std(oof_cal)), 4),
            'n_models': len(oof_all[t]),
        }
        avg_oof += ll
    
    avg_oof /= 7
    
    results[exp_name] = {
        'oof': {t: np.array(oof_all[t]).mean(axis=0) for t in TARGETS},
        'test': {t: np.array(test_all[t]).mean(axis=0) for t in TARGETS},
        'oof_raw': {t: oof_all[t] for t in TARGETS},  # Keep raw arrays for ensemble
        'test_raw': {t: test_all[t] for t in TARGETS},
        'metrics': target_metrics,
        'avg_oof': avg_oof,
        'time': time.time() - t_exp,
        'features': f"raw={not use_personal}, personal={use_personal}, interact={use_interact}",
        'n_features': len(cols_used),
    }
    
    log.info(f"\n{'Target':<10} {'OOF LL':>8} {'Models':>8} {'OOF mean':>10} {'Test mean':>10} {'OOF std':>8}")
    log.info(f"{'-'*70}")
    for t in TARGETS:
        m = target_metrics[t]
        log.info(f"{t:<10} {m['oof_ll']:>8.5f} {m['n_models']:>8} {m['oof_mean']:>10.4f} {m['test_mean']:>10.4f} {m['oof_std']:>8.4f}")
    log.info(f"{'AVG':<10} {avg_oof:>8.5f} (time: {time.time()-t_exp:.0f}s)")

# ============================================================
# Ensemble optimization: cross-experiment
# ============================================================
log.info("\n" + "=" * 60)
log.info("ENSEMBLE OPTIMIZATION")
log.info("=" * 60)

exp_names = list(results.keys())

log.info(f"Individual experiment OOFs:")
for en in exp_names:
    log.info(f"  {en}: {results[en]['avg_oof']:.5f} ({results[en]['features']}, feats={results[en]['n_features']})")

# Try all subsets with equal weights
from itertools import combinations as comb

best_oof = float('inf')
best_combo = None

for r in range(1, len(exp_names) + 1):
    for combo in comb(exp_names, r):
        combined_oof = np.zeros((7, 450))
        combined_test = np.zeros((7, 250))
        for en in combo:
            for j, t in enumerate(TARGETS):
                combined_oof[j] += results[en]['oof'][t]
                combined_test[j] += results[en]['test'][t]
        n = len(combo)
        combined_oof /= n
        combined_test /= n
        
        # Mean match per target
        for j, t in enumerate(TARGETS):
            combined_oof[j] = mean_match(combined_oof[j], y_train[t].mean())
            combined_test[j] = mean_match(combined_test[j], y_train[t].mean())
        
        avg_ll = np.mean([log_loss(y_train[t], combined_oof[j], labels=[0,1]) for j, t in enumerate(TARGETS)])
        
        if avg_ll < best_oof:
            best_oof = avg_ll
            best_combo = list(combo)

log.info(f"\nBest ensemble: {best_combo}")
log.info(f"Best OOF: {best_oof:.5f}")

# Per-target best
log.info("\nPer-target ensemble OOF:")
for j, t in enumerate(TARGETS):
    ll = log_loss(y_train[t], combined_oof[j], labels=[0,1])
    log.info(f"  {t}: {ll:.5f}")

# ============================================================
# Save
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

oof_df = pd.DataFrame({t: combined_oof[j] for j, t in enumerate(TARGETS)})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v111_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

sub_df = pd.DataFrame({t: combined_test[j] for j, t in enumerate(TARGETS)})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v111_{ts}.csv'
sub_df.to_csv(sub_path, index=False)

exp_log = {
    'version': 'V111',
    'timestamp': ts,
    'experiments': {en: {
        'oof': round(results[en]['avg_oof'], 5),
        'features': results[en]['features'],
        'n_features': results[en]['n_features'],
        'time_s': round(results[en]['time'], 0),
    } for en in exp_names},
    'ensemble': {
        'best_combo': best_combo,
        'best_oof': round(best_oof, 5),
        'per_target_oof': {t: round(log_loss(y_train[t], combined_oof[j], labels=[0,1]), 5) for j, t in enumerate(TARGETS)},
    },
    'test_means': {t: round(sub_df[t].mean(), 4) for t in TARGETS},
    'test_stds': {t: round(sub_df[t].std(), 4) for t in TARGETS},
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v111_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved OOF: {oof_path}")
log.info(f"Saved submission: {sub_path}")
log.info(f"Log: {EXPERIMENTS / f'v111_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
