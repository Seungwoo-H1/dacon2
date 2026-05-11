"""V108: Multi-strategy LGBM with proper test predictions.

Pipeline:
1. Load features.parquet (train) + test_features.parquet (test)
2. Add personalization features
3. Per-target feature ranking → top-20
4. Per target: train 2 strategies (base, personal) × N configs × 4 seeds × 5 folds
5. Ensemble → calibrate → submit

Simplification: only use 'base' and 'personal' strategies to avoid
complexity of interaction features on mismatched column sets.
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

TARGET_CFGS = {
    'Q1': ['wide', 'v53wide'],
    'Q2': ['wide', 'v48'],
    'Q3': ['v48', 'wide'],
    'S1': ['wide', 'v53wide', 'v48'],
    'S2': ['v48', 'wide'],
    'S3': ['safety', 'wide'],
    'S4': ['safety', 'v48'],
}

SEEDS = [42, 123, 7, 999]

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def remove_leak(cols, target):
    leak = LEAK_S if target.startswith('S') else LEAK_Q
    return [c for c in cols if c not in leak]

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(feat_df, feature_cols):
    feat_df = feat_df.copy()
    added = []
    for col in feature_cols:
        col_filled = feat_df[col].fillna(0)
        grp = col_filled.groupby(feat_df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        feat_df = feat_df.merge(grp, on='subject_id', how='left')
        mask_zero = feat_df[f'{col}_subj_std'] == 0
        mask_null = feat_df[col].isnull()
        feat_df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (feat_df[col].fillna(0) - feat_df[f'{col}_subj_mean']) /
            np.maximum(feat_df[f'{col}_subj_std'], 1e-8))
        added.append(f'{col}_zscore')
        gc.collect()
    return feat_df, added

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

# Base features (from original parquet, before personalization)
feature_cols = get_feature_cols(feat)
log.info(f"Base feature cols: {len(feature_cols)}")

# Personalization
log.info("Adding personalization...")
feat_all, personal_cols = add_personalization(feat, feature_cols)
test_personal = feat_test.copy()
test_personal = test_personal.merge(
    feat_all[['subject_id'] + personal_cols].drop_duplicates('subject_id'),
    on='subject_id', how='left')
test_personal[personal_cols] = test_personal[personal_cols].fillna(0)
log.info(f"After personalization: train={feat_all.shape}, test={test_personal.shape}")

all_base_cols = feature_cols + personal_cols

# ============================================================
# Phase 1: Feature importance ranking (on base features only)
# ============================================================
log.info("\n=== Phase 1: Feature importance ranking ===")
all_feat_importance = {}

for target in TARGETS:
    y = feat_all[target].values.astype(np.float64)
    leak_cols = remove_leak(all_base_cols, target)
    
    p_rank = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True,
        'num_leaves': 31, 'max_depth': 8, 'learning_rate': 0.05,
        'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 20,
        'random_state': 42, 'n_jobs': -1,
    }
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    p_rank['scale_pos_weight'] = spw
    
    X = feat_all[leak_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in leak_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    
    m_rank = lgb.train(p_rank, ds, num_boost_round=100)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
    all_feat_importance[target] = ranked
    
    top20 = [r[0] for r in ranked[:20]]
    log.info(f"  {target}: top5 = {[r[0] for r in ranked[:5]]}")

# Save feature importance
with open(EXPERIMENTS / "v108_feature_importance.json", 'w') as f:
    json.dump({t: {r[0]: r[1] for r in ranked} for t, ranked in all_feat_importance.items()}, f, indent=2)

# ============================================================
# Phase 2: Multi-strategy training with proper test predictions
# ============================================================
log.info("\n=== Phase 2: Multi-strategy training ===")

oof_all = {t: [] for t in TARGETS}
test_all = {t: [] for t in TARGETS}
model_count = {t: 0 for t in TARGETS}
config_log = {t: {} for t in TARGETS}

for target in TARGETS:
    log.info(f"\n{'='*60}")
    log.info(f"Target: {target}")
    log.info(f"{'='*60}")
    
    y = feat_all[target].values.astype(np.float64)
    train_rate = y.mean()
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    gkf = GroupKFold(n_splits=5)
    
    top_feats = [r[0] for r in all_feat_importance[target][:20]]
    
    # Strategy 1: base — top 20 from ranking
    base_feats = remove_leak(top_feats, target)
    base_feats = [c for c in base_feats if c in feat_all.columns]
    
    # Strategy 2: personal — top 20 + personal zscore cols
    personal_feats = remove_leak(top_feats + personal_cols[:20], target)
    personal_feats = [c for c in personal_feats if c in feat_all.columns]
    
    strategies = {
        'base': ('feat_all', base_feats),
        'personal': ('feat_all', personal_feats),
    }
    
    configs_for_target = TARGET_CFGS[target]
    
    for strat_name, (feat_source_name, sel_cols) in strategies.items():
        feat_source = feat_all  # both use feat_all since personal is already added
        
        for cfg_name in configs_for_target:
            cfg = CFGS[cfg_name]
            
            for seed in SEEDS:
                sn = [sanitize(c) for c in sel_cols]
                
                # OOF via GroupKFold
                oof_fold = np.zeros(450)
                
                for tr_i, va_i in gkf.split(feat_source, y, feat_source['subject_id']):
                    X_tr = feat_source.iloc[tr_i][sel_cols].fillna(0).values.astype(np.float64)
                    X_va = feat_source.iloc[va_i][sel_cols].fillna(0).values.astype(np.float64)
                    
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
                X_all_feat = feat_source[sel_cols].fillna(0).values.astype(np.float64)
                
                # Build test feature set - ensure all sel_cols are available
                test_cols = test_personal.columns.tolist() if 'personal' in sel_cols else list(feat_test.columns)
                # For any sel_col ending with _zscore, it must come from test_personal
                needs_personal = any(c.endswith('_zscore') for c in sel_cols)
                if needs_personal or strat_name == 'personal':
                    X_test_feat = test_personal[sel_cols].fillna(0).values.astype(np.float64)
                else:
                    X_test_feat = feat_test[sel_cols].fillna(0).values.astype(np.float64)
                
                ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
                cfg_all = {**cfg_full, 'n_estimators': cfg['ne']}
                model_test = lgb.train(cfg_all, ds_all, num_boost_round=cfg['ne'])
                test_pred = model_test.predict(X_test_feat)
                test_all[target].append(test_pred)
                
                del ds_all, model_test
                gc.collect()
            
            gc.collect()
        
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
    
    log.info(f"  Models: {model_count[target]}")
    log.info(f"  OOF LL: {val_ll:.5f} (calibrated)")
    log.info(f"  Train rate: {train_rate:.4f}, OOF mean: {oof_avg.mean():.4f}, Test mean: {test_avg.mean():.4f}")
    log.info(f"  OOF cal mean: {oof_cal.mean():.4f}, Test cal mean: {test_cal.mean():.4f}")
    log.info(f"  Δ vs V54(0.53971): {val_ll - 0.53971:+.5f}")
    
    config_log[target] = {
        'model_count': model_count[target],
        'oof_ll': round(val_ll, 5),
        'oof_mean': round(oof_avg.mean(), 4),
        'test_mean': round(test_avg.mean(), 4),
        'train_rate': round(train_rate, 4),
    }

# ============================================================
# Phase 3: Summary + Save
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
    train_rate = feat_all[t].mean()
    oof_cal = mean_match(oof_avg, train_rate)
    val_ll = log_loss(feat_all[t].values, oof_cal, labels=[0,1])
    delta = val_ll - 0.53971
    avg_v108 += val_ll
    log.info(f"{t:<10} {val_ll:>8.5f} {model_count[t]:>8} {oof_avg.mean():>10.4f} {test_avg.mean():>10.4f} {delta:>+10.5f}")
avg_v108 /= 7

log.info(f"{'AVG':<10} {avg_v108:>8.5f}")
log.info(f"V53 avg: 0.54793, V54 avg: 0.53971")

# Save OOF
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
oof_df = pd.DataFrame({t: mean_match(np.array(oof_all[t]).mean(axis=0), feat_all[t].mean()) 
                        for t in TARGETS})
oof_df.insert(0, 'subject_id', feat_all['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat_all['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat_all['lifelog_date'].values)
oof_path = DATA / f'oof_v108_{ts}.csv'
oof_df.to_csv(oof_path, index=False)
log.info(f"\nSaved OOF: {oof_path}")

# Save test submission
sub_df = pd.DataFrame({t: mean_match(np.array(test_all[t]).mean(axis=0), feat_all[t].mean())
                       for t in TARGETS})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v108_{ts}.csv'
sub_df.to_csv(sub_path, index=False)
log.info(f"Saved submission: {sub_path}")
log.info(f"Test means: { {t: round(sub_df[t].mean(), 4) for t in TARGETS} }")

# Save experiment log
exp_log = {
    'version': 'V108',
    'timestamp': ts,
    'oof_lls': {t: config_log[t]['oof_ll'] for t in TARGETS},
    'avg_oof_ll': round(avg_v108, 5),
    'model_counts': {t: config_log[t]['model_count'] for t in TARGETS},
    'strategies': ['base', 'personal'],
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
