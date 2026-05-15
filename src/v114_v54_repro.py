"""V114: V54 pipeline reproduction (no personalization, 8 seeds, 8 configs)

V54 OOF = 0.53971 (confirmed from oof_v54.csv)
V54 code uses personalization + feature ranking + pairwise + transformed features
V114 strips personalization and feature selection, uses raw features only

Key learning from V112: V112 OOF = 0.65420 with 6 configs × 8 seeds
Need to reproduce V54 exactly first to understand the 0.53971 gap
"""
import sys, re, gc, time, warnings, logging, json
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

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
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
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

# V54 configs
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_EXTRA_DEEP = {'nl': 25, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.6, 'cb': 0.5, 'ra': 0.1, 'rl': 1.0, 'mc': 25}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V53SAFE = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}

CFGS = {
    'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48, 'safety': CFG_SAFETY,
    'extra_deep': CFG_EXTRA_DEEP, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP, 'v53safe': CFG_V53SAFE
}

SEEDS = [42, 123, 7, 999, 777, 2026, 111, 555]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def train_cv_model(feat, feat_cols, target, seeds, cfg, n_folds=5):
    """Train LGBM with CV, return OOF and test predictions."""
    y = feat[target].values.astype(np.float64)
    train_rate = y.mean()
    
    # Remove leak
    leak = LEAK_S if target.startswith('S') else LEAK_Q
    cols = [c for c in feat_cols if c not in leak]
    
    sn = [sanitize(c) for c in cols]
    gkf = GroupKFold(n_splits=n_folds)
    
    oof = np.zeros((len(y), len(seeds)))
    test_preds = np.zeros(len(feat_test))
    
    for s_idx, seed in enumerate(seeds):
        for fold_i, (tr_i, va_i) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            X_tr = feat.iloc[tr_i][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va_i][cols].fillna(0).values.astype(np.float64)
            
            spw = max(((y[tr_i] == 0).sum()) / max((y[tr_i] == 1).sum(), 1), 0.1)
            
            ds_train = lgb.Dataset(X_tr, label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            ds_val = lgb.Dataset(X_va, label=y[va_i], feature_name=sn, reference=ds_train, params={'verbose': '-1'})
            
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'],
                'random_state': seed, 'scale_pos_weight': spw,
            }
            
            model = lgb.train(params, ds_train, num_boost_round=cfg['ne'],
                             valid_sets=[ds_val],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                       lgb.log_evaluation(0)])
            
            oof[va_i, s_idx] = model.predict(X_va)
            
            del ds_train, ds_val, model
            gc.collect()
        
        # Test prediction: train on full data
        X_full = feat[cols].fillna(0).values.astype(np.float64)
        ds_full = lgb.Dataset(X_full, label=y, feature_name=sn, params={'verbose': '-1'})
        model_test = lgb.train(params, ds_full, num_boost_round=cfg['ne'])
        test_preds = model_test.predict(feat_test[cols].fillna(0).values.astype(np.float64))
        del ds_full, model_test
        gc.collect()
    
    return oof, test_preds, train_rate, cols

# ============================================================
# Load data
# ============================================================
t_start = time.time()
log.info("Loading data...")

feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")

for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feature_cols = get_feature_cols(feat)
log.info(f"Feature cols: {len(feature_cols)}")
y_train = {t: feat[t].values for t in TARGETS}

# ============================================================
# Run V54 reproduction: 8 configs × 8 seeds
# ============================================================
log.info(f"\nV114: {len(CFGS)} configs × {len(SEEDS)} seeds = {len(CFGS)*len(SEEDS)} models per target")

# Per config + per seed results
all_oof = {}  # {(cfg, seed): {target: oof_array}}
all_test = {}

for cfg_name, cfg in CFGS.items():
    log.info(f"\n--- Config: {cfg_name} (nl={cfg['nl']}, md={cfg['md']}, lr={cfg['lr']}, ne={cfg['ne']}) ---")
    for seed in SEEDS:
        target_oof = {}
        target_test = {}
        
        for target in TARGETS:
            oof, test_pred, train_rate, cols_used = train_cv_model(
                feat, feature_cols, target, [seed], cfg)
            
            # Mean match calibration
            oof_mm = mean_match(oof[:, 0], train_rate)
            test_mm = mean_match(test_pred, train_rate)
            
            target_oof[target] = oof_mm
            target_test[target] = test_mm
        
        key = (cfg_name, seed)
        all_oof[key] = target_oof
        all_test[key] = target_test

# ============================================================
# Evaluate all configs + seeds individually
# ============================================================
log.info("\n" + "=" * 80)
log.info("INDIVIDUAL MODEL EVALUATION")
log.info("=" * 80)

model_scores = {}
for (cfg_name, seed), oof_dict in all_oof.items():
    avg_ll = 0
    for j, target in enumerate(TARGETS):
        ll = log_loss(y_train[target], oof_dict[target], labels=[0, 1])
        avg_ll += ll
    avg_ll /= 7
    model_scores[(cfg_name, seed)] = avg_ll

# Sort by OOF
sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

log.info(f"\nBest 10 individual models:")
for (cfg_name, seed), score in sorted_models[:10]:
    log.info(f"  {cfg_name:15s} seed={seed:4d}  OOF={score:.5f}")

# ============================================================
# Ensemble: top-k models, all k from 1 to N
# ============================================================
log.info(f"\n{'='*80}")
log.info("ENSEMBLE OPTIMIZATION (top-k)")
log.info(f"{'='*80}")

best_oof = float('inf')
best_k = 0
best_combo = []
best_oof_targets = None
best_test_targets = None

for k in range(1, len(sorted_models) + 1):
    top_k = sorted_models[:k]
    top_k_keys = [(cfg_name, seed) for (cfg_name, seed), _ in sorted_models[:k]]
    
    combined_oof = np.zeros((7, 450))
    combined_test = np.zeros((7, 250))
    
    for (cfg_name, seed) in top_k_keys:
        for j, target in enumerate(TARGETS):
            combined_oof[j] += all_oof[(cfg_name, seed)][target]
            combined_test[j] += all_test[(cfg_name, seed)][target]
    
    n = len(top_k_keys)
    combined_oof /= n
    combined_test /= n
    
    for j, target in enumerate(TARGETS):
        combined_oof[j] = mean_match(combined_oof[j], y_train[target].mean())
        combined_test[j] = mean_match(combined_test[j], y_train[target].mean())
    
    avg_ll = np.mean([log_loss(y_train[t], combined_oof[j], labels=[0, 1]) for j, t in enumerate(TARGETS)])
    
    if avg_ll < best_oof:
        best_oof = avg_ll
        best_k = k
        best_combo = top_k_keys
        best_oof_targets = combined_oof.copy()
        best_test_targets = combined_test.copy()

log.info(f"\nBest ensemble: top-{best_k} models")
log.info(f"Best OOF: {best_oof:.5f}")
log.info("Per-target OOF:")
for j, t in enumerate(TARGETS):
    ll = log_loss(y_train[t], best_oof_targets[j], labels=[0, 1])
    log.info(f"  {t}: {ll:.5f} (mean={best_oof_targets[j].mean():.4f})")

# Also test: equal-weight ensemble of ALL models
log.info(f"\n{'='*80}")
log.info("ALL MODELS EQUAL WEIGHT")
log.info(f"{'='*80}")

all_combined_oof = np.zeros((7, 450))
all_combined_test = np.zeros((7, 250))
for key in all_oof:
    for j, target in enumerate(TARGETS):
        all_combined_oof[j] += all_oof[key][target]
        all_combined_test[j] += all_test[key][target]

all_n = len(all_oof)
all_combined_oof /= all_n
all_combined_test /= all_n

for j, target in enumerate(TARGETS):
    all_combined_oof[j] = mean_match(all_combined_oof[j], y_train[target].mean())
    all_combined_test[j] = mean_match(all_combined_test[j], y_train[target].mean())

all_avg = np.mean([log_loss(y_train[t], all_combined_oof[j], labels=[0, 1]) for j, t in enumerate(TARGETS)])
log.info(f"All {all_n} models ensemble OOF: {all_avg:.5f}")

# ============================================================
# Compare best_k vs all
# ============================================================
log.info(f"\n{'='*80}")
log.info("COMPARISON")
log.info(f"{'='*80}")
log.info(f"Single best model OOF:    {sorted_models[0][1]:.5f}")
log.info(f"Top-{best_k} ensemble OOF:  {best_oof:.5f}")
log.info(f"All models ensemble OOF:  {all_avg:.5f}")

# ============================================================
# Save
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save best ensemble OOF
oof_df = pd.DataFrame({t: best_oof_targets[j] for j, t in enumerate(TARGETS)})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v114_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

# Save best ensemble submission
sub_df = pd.DataFrame({t: best_test_targets[j] for j, t in enumerate(TARGETS)})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v114_{ts}.csv'
sub_df.to_csv(sub_path, index=False)

# Also save all models ensemble
oof_df_all = pd.DataFrame({t: all_combined_oof[j] for j, t in enumerate(TARGETS)})
oof_df_all.insert(0, 'subject_id', feat['subject_id'].values)
oof_df_all.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df_all.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path_all = DATA / f'oof_v114_all_{ts}.csv'
oof_df_all.to_csv(oof_path_all, index=False)

sub_df_all = pd.DataFrame({t: all_combined_test[j] for j, t in enumerate(TARGETS)})
sub_df_all.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path_all = SUBMIT / f'submission_v114_all_{ts}.csv'
sub_df_all.to_csv(sub_path_all, index=False)

exp_log = {
    'version': 'V114',
    'timestamp': ts,
    'configs': list(CFGS.keys()),
    'seeds': SEEDS,
    'n_models_per_target': len(CFGS) * len(SEEDS),
    'results': {
        'single_best': {
            'model': f"{sorted_models[0][0][0]}_{sorted_models[0][0][1]}",
            'oof': round(sorted_models[0][1], 5),
        },
        'top_k_ensemble': {
            'k': best_k,
            'oof': round(best_oof, 5),
            'models': [f"{k[0]}_{k[1]}" for k in best_combo],
        },
        'all_ensemble': {
            'n_models': all_n,
            'oof': round(all_avg, 5),
        },
    },
    'per_target_best': {t: round(log_loss(y_train[t], best_oof_targets[j], labels=[0, 1]), 5) for j, t in enumerate(TARGETS)},
    'per_target_all': {t: round(log_loss(y_train[t], all_combined_oof[j], labels=[0, 1]), 5) for j, t in enumerate(TARGETS)},
    'test_means': {t: round(sub_df[t].mean(), 4) for t in TARGETS},
    'test_stds': {t: round(sub_df[t].std(), 4) for t in TARGETS},
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v114_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved:")
log.info(f"  OOF (best): {oof_path}")
log.info(f"  Submission (best): {sub_path}")
log.info(f"  OOF (all): {oof_path_all}")
log.info(f"  Submission (all): {sub_path_all}")
log.info(f"  Log: {EXPERIMENTS / f'v114_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
