"""
V41 GPU — V10 strategy fully on GPU
- XGBoost GPU (tree_method='hist', gpu_id=0) for feature ranking + training
- Single thread (no n_jobs parallelism), max GPU utilization
- Per-target: 6 configs × 2 feat counts × 20 seeds = 240 models/target × 7 targets = 1680 models
- Pre-computed features_v11_personalized.parquet → only top-20 selected for training
- Memory-safe: process one target at a time, rank→select→train→clear GPU cache
"""

import sys, re, gc, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import torch

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from config import TARGETS

DATA_PROCESSED = ROOT / "data_processed"
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ── GPU setup ──
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log.info(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    """Mean-matching calibration."""
    return np.clip(p + (r.mean() - p.mean()), 0.0001, 0.9999)

def clear_gpu():
    """Free GPU memory."""
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

# ── Leakage columns ──
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

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

# ── Hyperparameters ──
# V10-style 6 configs, tuned on GPU
XGB_CONFIGS = [
    # (name, n_estimators, max_depth, lr, subsample, colsample, reg_alpha, reg_lambda, min_child)
    {'name': 'C1', 'n_est': 300, 'md': 3, 'lr': 0.05, 'ss': 0.8, 'cb': 0.8, 'ra': 0.5, 'rl': 1.0, 'mc': 5},
    {'name': 'C2', 'n_est': 300, 'md': 4, 'lr': 0.03, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 8},
    {'name': 'C3', 'n_est': 400, 'md': 4, 'lr': 0.02, 'ss': 0.8, 'cb': 0.8, 'ra': 0.5, 'rl': 2.0, 'mc': 10},
    {'name': 'C4', 'n_est': 500, 'md': 4, 'lr': 0.03, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C5', 'n_est': 300, 'md': 5, 'lr': 0.02, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    {'name': 'C6', 'n_est': 500, 'md': 3, 'lr': 0.05, 'ss': 0.9, 'cb': 0.9, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

# ── Feature ranking with XGB GPU ──
def rank_features_gpu(feat, zscore_cols, target, seed=42):
    """Rank features using XGB GPU and return top-20."""
    y = feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    z_leak = remove_leak(zscore_cols, target)
    X = feat[z_leak].fillna(0).values.astype(np.float32)

    # Small rank tree → low memory
    params = {
        'objective': 'binary',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda:0',
        'max_depth': 4,
        'learning_rate': 0.03,
        'n_estimators': 30,  # Very few rounds for ranking
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'scale_pos_weight': spw,
        'random_state': seed,
        'min_child_weight': 3,
        'verbosity': 0,
        # Single thread for memory
        'process_count': 1,
    }

    dtrain = xgb.DMatrix(X, label=y, feature_names=[sanitize(c) for c in z_leak])
    model = xgb.train(params, dtrain, num_boost_round=30)

    gain = model.get_score(importance_type='gain')
    # Fill zeros for missing features
    all_gain = {f: gain.get(f, 0.0) for f in z_leak}
    ranked = sorted(z_leak, key=lambda f: -all_gain[f])

    dtrain.delete()
    del model
    clear_gpu()

    return ranked

# ── Train with XGB GPU (single thread, no parallelism) ──
def train_gpu(feat, cols, target, seeds, fold_cfg):
    """Train XGB ensemble on GPU. One fold at a time, no parallel."""
    y = feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]

    for si, seed in enumerate(seeds):
        cfg = {
            'objective': 'binary',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda:0',
            'max_depth': fold_cfg['md'],
            'learning_rate': fold_cfg['lr'],
            'n_estimators': fold_cfg['n_est'],
            'subsample': fold_cfg['ss'],
            'colsample_bytree': fold_cfg['cb'],
            'reg_alpha': fold_cfg['ra'],
            'reg_lambda': fold_cfg['rl'],
            'min_child_weight': fold_cfg['mc'],
            'scale_pos_weight': spw,
            'random_state': seed,
            'verbosity': 0,
            'process_count': 1,  # Single thread
        }

        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float32)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float32)

            dtrain = xgb.DMatrix(X_tr, label=y[tr], feature_names=sn)
            dval = xgb.DMatrix(X_va, label=y[va], feature_names=sn)

            evals = [(dval, 'eval')]
            model = xgb.train(cfg, dtrain, evals=evals,
                              early_stopping_rounds=50, verbose_eval=False)

            # Predict only the best iteration
            n_best = model.best_iteration + 1
            oof[va, si] = model.predict(dval, iteration_range=(0, n_best))

            dtrain.delete()
            dval.delete()
            del model
            del X_tr, X_va
            clear_gpu()

    return oof

# ── Main ──
def main():
    t_start = time.time()

    log.info("=" * 70)
    log.info("V41 GPU — XGBoost GPU (single thread, max GPU utilization)")
    log.info("Per-target: 6 configs × 2 feat counts × 20 seeds = 240 models")
    log.info("=" * 70)

    # Load pre-computed features
    feat_path = DATA_PROCESSED / "features_v11_personalized.parquet"
    log.info(f"Loading {feat_path}")
    t0 = time.time()
    feat = pd.read_parquet(feat_path)
    load_time = time.time() - t0
    log.info(f"  Loaded: {feat.shape}, {feat.memory_usage(deep=True).sum()/1024**2:.1f}MB ({load_time:.1f}s)")

    # Identify z-score columns
    all_numeric = [c for c in feat.columns
                   if c not in META | set(TARGET_COLS)
                   and feat[c].dtype in [np.float64, np.int64, float, int, bool]]
    zscore_cols = [c for c in all_numeric if '_zscore' in c]
    basic_cols = [c for c in all_numeric if '_zscore' not in c]
    log.info(f"  Z-score: {len(zscore_cols)}, Basic: {len(basic_cols)}")

    clear_gpu()
    t0 = time.time()

    # Per-target: rank → tune → final
    all_results = {}

    for target in TARGET_COLS:
        tgt_t = time.time()
        train_rate = feat[target].mean()
        log.info(f"\n{'='*50}")
        log.info(f"--- {target} (rate={train_rate:.3f}) ---")

        y = feat[target].values.astype(np.float64)
        z_leak = remove_leak(zscore_cols, target)
        log.info(f"  Leak-free z-score: {len(z_leak)}")

        # Step 1: Feature ranking (GPU)
        log.info(f"  [1/3] Feature ranking on GPU...")
        ranked = rank_features_gpu(feat, zscore_cols, target)
        log.info(f"  Top-5: {ranked[:5]}")

        # Step 2: Per-config tuning
        log.info(f"  [2/3] Config tuning (5-fold × 20 seeds)...")
        best_cv = float('inf')
        best_cfg = None
        best_n = None

        for n_feat in [10, 20]:
            sel = ranked[:n_feat]
            for cfg in XGB_CONFIGS:
                oof = train_gpu(feat, sel, target, SEEDS, cfg)
                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                cv = log_loss(y, oof_avg, labels=[0, 1])

                if cv < best_cv:
                    best_cv = cv
                    best_cfg = cfg
                    best_n = n_feat
                    log.info(f"    NEW BEST: {cfg['name']} n={n_feat} cv={cv:.4f}")

        clear_gpu()

        # Step 3: Final model with best config
        sel_final = ranked[:best_n]
        log.info(f"  [3/3] Final model: {best_cfg['name']} n={best_n}")
        oof_final = train_gpu(feat, sel_final, target, SEEDS, best_cfg)
        oof_avg_final = np.clip(oof_final.mean(axis=1), 0.0001, 0.9999)
        cal_final = mm(oof_avg_final, y)

        cal_loss = log_loss(y, cal_final, labels=[0, 1])
        oof_loss = log_loss(y, oof_avg_final, labels=[0, 1])

        all_results[target] = {
            'cal_oof': cal_final,
            'oof_oof': oof_avg_final,
            'config': best_cfg['name'],
            'n_feat': best_n,
            'cv': cal_loss,
        }

        log.info(f"  RESULT: Config={best_cfg['name']} n={best_n}, OOF={oof_loss:.4f}, Cal={cal_loss:.4f}")
        log.info(f"  Time: {time.time()-tgt_t:.0f}s")

        del oof_final, oof_avg_final, cal_final
        clear_gpu()

    # ── Summary ──
    log.info(f"\n{'='*70}")
    log.info("V41 GPU SUMMARY")
    log.info(f"{'='*70}")

    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: Config={r['config']} n={r['n_feat']} Cal={r['cv']:.4f}")

    avg_cal = np.mean([log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
                       for t in TARGET_COLS])
    avg_oof = np.mean([log_loss(feat[t].values, all_results[t]['oof_oof'], labels=[0, 1])
                       for t in TARGET_COLS])

    log.info(f"\n  V41 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V41 Avg OOF: {avg_oof:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save submission ──
    log.info(f"\nSaving submission...")

    # For submission, we'd need test features.
    # Using OOF as proxy for now
    submit = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        submit[target] = all_results[target]['cal_oof']

    submit_path = ROOT / "submissions" / f"submission_v41_gpu_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    submit.to_csv(submit_path, index=False)
    log.info(f"  Saved: {submit_path}")

    return all_results

if __name__ == "__main__":
    main()
