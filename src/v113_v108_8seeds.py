"""V113 (V108 with 8 seeds): Multi-config LGBM ensemble replicating V54_re2 pipeline.

Approach:
- Use ALL base features (like V54_re2, NOT top-20 selection)
- Multi-config: wide, deep, v48, safety × each target
- Multi-seed: 4 seeds × 5 folds
- Ensemble all models → OOF → calibrate → submit

This is the proven V54_re2 approach with diversity from:
1. Different configs (wide/deep/v48/safety)  
2. Different seeds
3. Multi-target ensemble (different configs per target)
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
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48,
        'safety': CFG_SAFETY, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP}

SEEDS = [42, 123, 7, 999, 777, 2026, 111, 555]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

# ============================================================
# Load data
# ============================================================
t_start = time.time()
log.info("Loading features...")

feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")
log.info(f"Train: {feat.shape}, Test: {feat_test.shape}")

# Clean date columns
for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

# Feature columns (141 raw features from parquet)
feature_cols = get_feature_cols(feat)
log.info(f"Feature cols: {len(feature_cols)}")

# ============================================================
# Phase 1: Train multi-config models
# ============================================================
log.info("\n=== Phase 1: Multi-config training ===")

oof_all = {t: [] for t in TARGETS}
test_all = {t: [] for t in TARGETS}
model_count = {t: 0 for t in TARGETS}
config_log = {t: {} for t in TARGETS}

for target in TARGETS:
    log.info(f"\n{'='*60}")
    log.info(f"Target: {target}")
    log.info(f"{'='*60}")
    
    y = feat[target].values.astype(np.float64)
    train_rate = y.mean()
    
    # Remove leak columns for this target
    leak = LEAK_S if target.startswith('S') else LEAK_Q
    feat_cols = [c for c in feature_cols if c not in leak]
    log.info(f"  Features: {len(feat_cols)} (after leak removal)")
    
    sn = [sanitize(c) for c in feat_cols]
    gkf = GroupKFold(n_splits=5)
    
    # Use all configs for this target (like V53: all configs per target)
    for cfg_name, cfg in CFGS.items():
        for seed in SEEDS:
            oof_fold = np.zeros(450)
            
            for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
                X_tr = feat.iloc[tr_i][feat_cols].fillna(0).values.astype(np.float64)
                X_va = feat.iloc[va_i][feat_cols].fillna(0).values.astype(np.float64)
                
                spw = max(((y[tr_i] == 0).sum()) / max((y[tr_i] == 1).sum(), 1), 0.1)
                
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
                                 callbacks=[lgb.early_stopping(50, verbose=False),
                                           lgb.log_evaluation(0)])
                
                oof_fold[va_i] = model.predict(X_va)
                
                del ds_train, ds_val, model
                gc.collect()
            
            oof_all[target].append(oof_fold)
            model_count[target] += 1
            
            # Test prediction: retrain on all data
            X_all_feat = feat[feat_cols].fillna(0).values.astype(np.float64)
            X_test_feat = feat_test[feat_cols].fillna(0).values.astype(np.float64)
            
            ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
            cfg_all = {**cfg_full, 'n_estimators': cfg['ne']}
            model_test = lgb.train(cfg_all, ds_all, num_boost_round=cfg['ne'])
            test_pred = model_test.predict(X_test_feat)
            test_all[target].append(test_pred)
            
            del ds_all, model_test
            gc.collect()
    
    # Average across all models
    oof_arr = np.array(oof_all[target])
    test_arr = np.array(test_all[target])
    
    oof_avg = oof_arr.mean(axis=0)
    test_avg = test_arr.mean(axis=0)
    
    # Calibrate: mean matching
    oof_cal = mean_match(oof_avg, train_rate)
    test_cal = mean_match(test_avg, train_rate)
    
    # Evaluate
    val_ll = log_loss(y, oof_cal, labels=[0,1])
    
    n_models = model_count[target]
    configs_used = list(CFGS.keys())
    
    log.info(f"  Models: {n_models} ({len(CFGS)} configs × {len(SEEDS)} seeds)")
    log.info(f"  OOF LL: {val_ll:.5f} (calibrated)")
    log.info(f"  Train rate: {train_rate:.4f}, OOF mean: {oof_avg.mean():.4f}, Test mean: {test_avg.mean():.4f}")
    log.info(f"  Δ vs V54(0.53971): {val_ll - 0.53971:+.5f}")
    
    # Per-config contribution
    config_breakdown = {}
    for c_idx, c_name in enumerate(sorted(set(f'cfg_{c_name}_seed_{seed}' 
                                                for cfg_name in CFGS 
                                                for seed in SEEDS)[:n_models])):
        config_breakdown[c_name] = True  # Just track count
    
    config_log[target] = {
        'model_count': n_models,
        'oof_ll': round(val_ll, 5),
        'oof_mean': round(oof_avg.mean(), 4),
        'test_mean': round(test_avg.mean(), 4),
        'train_rate': round(train_rate, 4),
        'configs': list(CFGS.keys()),
        'seeds': SEEDS,
    }

# ============================================================
# Phase 2: Summary + Save
# ============================================================
log.info(f"\n{'='*60}")
log.info("V108 SUMMARY")
log.info(f"{'='*60}")
log.info(f"{'Target':<10} {'OOF LL':>8} {'Models':>8} {'OOF mean':>10} {'Test mean':>10} {'Δ vs V54':>10}")
log.info(f"{'-'*70}")

avg_v108 = 0
for t in TARGETS:
    oof_arr = np.array(oof_all[t])
    test_arr = np.array(test_all[t])
    oof_avg = oof_arr.mean(axis=0)
    test_avg = test_arr.mean(axis=0)
    train_rate = feat[t].mean()
    oof_cal = mean_match(oof_avg, train_rate)
    val_ll = log_loss(feat[t].values, oof_cal, labels=[0,1])
    delta = val_ll - 0.53971
    avg_v108 += val_ll
    log.info(f"{t:<10} {val_ll:>8.5f} {model_count[t]:>8} {oof_avg.mean():>10.4f} {test_avg.mean():>10.4f} {delta:>+10.5f}")
avg_v108 /= 7

log.info(f"{'AVG':<10} {avg_v108:>8.5f}")
log.info(f"V53 avg: 0.54793, V54 avg: 0.53971")

# ============================================================
# Phase 3: Save OOF and test submission
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

oof_df = pd.DataFrame({t: mean_match(np.array(oof_all[t]).mean(axis=0), feat[t].mean()) 
                        for t in TARGETS})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v108_{ts}.csv'
oof_df.to_csv(oof_path, index=False)
log.info(f"\nSaved OOF: {oof_path}")

sub_df = pd.DataFrame({t: mean_match(np.array(test_all[t]).mean(axis=0), feat[t].mean())
                       for t in TARGETS})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v108_{ts}.csv'
sub_df.to_csv(sub_path, index=False)
log.info(f"Saved submission: {sub_path}")
log.info(f"Test means: { {t: round(sub_df[t].mean(), 4) for t in TARGETS} }")

# Save experiment log
exp_log = {
    'version': 'V113',
    'timestamp': ts,
    'oof_lls': {t: config_log[t]['oof_ll'] for t in TARGETS},
    'avg_oof_ll': round(avg_v108, 5),
    'model_counts': {t: config_log[t]['model_count'] for t in TARGETS},
    'configs': list(CFGS.keys()),
    'seeds': SEEDS,
    'test_submission': str(sub_path.name),
    'oof_file': str(oof_path.name),
    'total_models': sum(model_count.values()),
    'total_time_s': time.time() - t_start,
}
with open(EXPERIMENTS / f'v108_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2)
log.info(f"\nLog: {EXPERIMENTS / f'v108_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
