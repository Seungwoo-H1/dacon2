"""V112: V54_re2 Pipeline Exact Reproduction + 8 Seeds + Ensemble

V54_re2 OOF = 0.54761, LB = 0.65358 (V53)
V54 OOF = 0.53971
V83 OOF = 0.54575
V106 Ensemble (V54+V83+V54_re2+V55+V53+V55_re2) = 0.5250

V112:
- Exact V54_re2 pipeline: 6 configs × 8 seeds (V108 방식 but 8 seeds)
- 8 seeds: [42, 123, 7, 999, 777, 2026, 111, 555]
- Test with 4 seeds and 8 seeds to see seed count effect
- No personalization (V109/V110 showed it hurts)
- No feature selection (use ALL features after leak removal)
- Configs: wide, deep, v48, safety, v53wide, v53deep
- Early stopping patience: 50 (same as V108)
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

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

def run_pipeline(feat, feat_test, seeds, t_start, label=""):
    """Run V108 pipeline with given seeds."""
    log.info(f"\n{'='*60}")
    log.info(f"V112 {label} | Seeds: {len(seeds)} | Configs: {len(CFGS)}")
    log.info(f"{'='*60}")
    
    oof_all = {t: [] for t in TARGETS}
    test_all = {t: [] for t in TARGETS}
    model_count = {t: 0 for t in TARGETS}
    
    for target in TARGETS:
        y = feat[target].values.astype(np.float64)
        train_rate = y.mean()
        
        leak = LEAK_S if target.startswith('S') else LEAK_Q
        feat_cols = [c for c in feat.columns if c not in META | set(TARGETS) | leak
                     and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
        
        sn = [sanitize(c) for c in feat_cols]
        gkf = GroupKFold(n_splits=5)
        
        for cfg_name, cfg in CFGS.items():
            for seed in seeds:
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
                
                # Test prediction
                X_all_feat = feat[feat_cols].fillna(0).values.astype(np.float64)
                X_test_feat = feat_test[feat_cols].fillna(0).values.astype(np.float64)
                
                ds_all = lgb.Dataset(X_all_feat, label=y, feature_name=sn, params={'verbose': '-1'})
                model_test = lgb.train(cfg_full, ds_all, num_boost_round=cfg['ne'])
                test_pred = model_test.predict(X_test_feat)
                test_all[target].append(test_pred)
                del ds_all, model_test
                gc.collect()
    
    # Evaluate
    target_metrics = {}
    avg_oof = 0
    for t in TARGETS:
        oof_avg = np.array(oof_all[t]).mean(axis=0)
        test_avg = np.array(test_all[t]).mean(axis=0)
        oof_cal = mean_match(oof_avg, feat[t].mean())
        test_cal = mean_match(test_avg, feat[t].mean())
        ll = log_loss(y_train[t], oof_cal, labels=[0,1])
        target_metrics[t] = {
            'oof_ll': round(ll, 5),
            'oof_mean': round(float(oof_cal.mean()), 4),
            'test_mean': round(float(test_cal.mean()), 4),
            'oof_std': round(float(np.std(oof_cal)), 4),
            'test_std': round(float(np.std(test_cal)), 4),
            'n_models': len(oof_all[t]),
        }
        avg_oof += ll
    avg_oof /= 7
    
    elapsed = time.time() - t_start
    log.info(f"\n{'Target':<10} {'OOF LL':>8} {'Models':>8} {'OOF mean':>10} {'Test mean':>10} {'OOF std':>8}")
    for t in TARGETS:
        m = target_metrics[t]
        log.info(f"{t:<10} {m['oof_ll']:>8.5f} {m['n_models']:>8} {m['oof_mean']:>10.4f} {m['test_mean']:>10.4f} {m['oof_std']:>8.4f}")
    log.info(f"{'AVG':<10} {avg_oof:>8.5f} | time: {elapsed:.0f}s | {label}")
    
    return {
        'oof': {t: np.array(oof_all[t]).mean(axis=0) for t in TARGETS},
        'test': {t: np.array(test_all[t]).mean(axis=0) for t in TARGETS},
        'oof_raw': {t: oof_all[t] for t in TARGETS},
        'test_raw': {t: test_all[t] for t in TARGETS},
        'metrics': target_metrics,
        'avg_oof': avg_oof,
        'time': elapsed,
        'label': label,
    }

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

SEEDS_4 = [42, 123, 7, 999]
SEEDS_8 = [42, 123, 7, 999, 777, 2026, 111, 555]

# Run 4 seeds
r1 = run_pipeline(feat, feat_test, SEEDS_4, t_start, "4 seeds baseline")

# Run 8 seeds
r2 = run_pipeline(feat, feat_test, SEEDS_8, r1['time'] + t_start, "8 seeds")

# ============================================================
# Ensemble optimization
# ============================================================
log.info("\n" + "=" * 60)
log.info("ENSEMBLE OPTIMIZATION")
log.info("=" * 60)

results = {'4seeds': r1, '8seeds': r2}

log.info(f"Individual results:")
for en, r in results.items():
    log.info(f"  {en}: OOF={r['avg_oof']:.5f} (models={r['metrics']['Q1']['n_models']})")

# Try all subsets
from itertools import combinations as comb

best_oof = float('inf')
best_combo = None
combined_oof = None
combined_test = None

for r in range(1, len(results) + 1):
    for combo in comb(results.keys(), r):
        combined = np.zeros((7, 450))
        for en in combo:
            for j, t in enumerate(TARGETS):
                combined[j] += results[en]['oof'][t]
        n = len(combo)
        combined /= n
        
        for j, t in enumerate(TARGETS):
            combined[j] = mean_match(combined[j], y_train[t].mean())
        
        avg_ll = np.mean([log_loss(y_train[t], combined[j], labels=[0,1]) for j, t in enumerate(TARGETS)])
        
        if avg_ll < best_oof:
            best_oof = avg_ll
            best_combo = list(combo)

# Compute best ensemble predictions
combined_oof = np.zeros((7, 450))
combined_test = np.zeros((7, 250))
for en in best_combo:
    for j, t in enumerate(TARGETS):
        combined_oof[j] += results[en]['oof'][t]
        combined_test[j] += results[en]['test'][t]
n = len(best_combo)
combined_oof /= n
combined_test /= n

for j, t in enumerate(TARGETS):
    combined_oof[j] = mean_match(combined_oof[j], y_train[t].mean())
    combined_test[j] = mean_match(combined_test[j], y_train[t].mean())

log.info(f"\nBest ensemble: {best_combo}")
log.info(f"Best OOF: {best_oof:.5f}")
log.info("Per-target OOF:")
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
oof_path = DATA / f'oof_v112_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

sub_df = pd.DataFrame({t: combined_test[j] for j, t in enumerate(TARGETS)})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_path = SUBMIT / f'submission_v112_{ts}.csv'
sub_df.to_csv(sub_path, index=False)

exp_log = {
    'version': 'V112',
    'timestamp': ts,
    'results': {en: {
        'oof': round(results[en]['avg_oof'], 5),
        'n_models_per_target': results[en]['metrics']['Q1']['n_models'],
        'time_s': round(results[en]['time'], 0),
        'per_target_oof': {t: results[en]['metrics'][t]['oof_ll'] for t in TARGETS},
    } for en in results},
    'ensemble': {
        'best_combo': best_combo,
        'best_oof': round(best_oof, 5),
        'per_target_oof': {t: round(log_loss(y_train[t], combined_oof[j], labels=[0,1]), 5) for j, t in enumerate(TARGETS)},
    },
    'test_means': {t: round(sub_df[t].mean(), 4) for t in TARGETS},
    'test_stds': {t: round(sub_df[t].std(), 4) for t in TARGETS},
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v112_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved OOF: {oof_path}")
log.info(f"Saved submission: {sub_path}")
log.info(f"Log: {EXPERIMENTS / f'v112_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
