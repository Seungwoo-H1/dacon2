"""
V74 — LGBM Per-Target Optimized (memory-efficient)

Strategy:
1. Single model type: LightGBM (memory efficient vs CatBoost/XGB ensemble)
2. Per-target: 5 hyperparameter configs × 30 seeds × 5 n_feat values
3. GroupKFold 5-fold (subject-level) — OOF computation
4. Leakage-clean features (V61 pipeline)
5. Best config selected by OOF log_loss
6. Final: train best config on ALL data with best seeds

Key insight from V73 partial: stacking doesn't help much on 450 samples.
Simple LGBM with proper tuning is likely optimal.
Focus on: regularization diversity, feature count sweep, seed diversity.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS = ROOT / "experiments"
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ── Leakage columns ──
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}
NIGHTTIME_LEAK = {
    'mScreenStatus_hour_night', 'mACStatus_hour_night',
    'mScreenStatus_hour_morning', 'wLight_w_light_sum',
    'mACStatus_charging_sum', 'mACStatus_charging_max',
}
SLEEP_DIRECT_LEAK = {
    'mGps_gps_avg_speed_max', 'mGps_gps_count_mean',
    'mActivity_m_activity_sum', 'mActivity_m_activity_max',
    'mActivity_m_activity_min',
}

N_FEAT_RANGE = [10, 15, 20, 25, 30, 40]
N_SEEDS = 30  # More seeds for better averaging

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

import catboost as cb


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    leak = set()
    if target.startswith('S'):
        leak = LEAK_S | NIGHTTIME_LEAK | SLEEP_DIRECT_LEAK
    elif target.startswith('Q'):
        leak = LEAK_Q | NIGHTTIME_LEAK
    return [c for c in cols if c not in leak]

def logloss_fn(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ── Hyperparameter configs (6 configs for diversity) ──
LGBM_CONFIGS = [
    # Very conservative
    {'name': 'V_conservative', 'nl': 8, 'md': 2, 'lr': 0.01, 'ne': 2000, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    # Conservative
    {'name': 'V_cons', 'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1500, 'ss': 0.6, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    # Standard (V10 baseline)
    {'name': 'V_std', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 1000, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    # Deep
    {'name': 'V_deep', 'nl': 25, 'md': 5, 'lr': 0.015, 'ne': 1500, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    # Wide
    {'name': 'V_wide', 'nl': 40, 'md': 3, 'lr': 0.04, 'ne': 500, 'ss': 0.8, 'cb': 0.8, 'ra': 1.0, 'rl': 3.0, 'mc': 5},
    # Aggressive (deep + low lr)
    {'name': 'V_aggressive', 'nl': 30, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.8, 'cb': 0.9, 'ra': 0.1, 'rl': 1.0, 'mc': 8},
]


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V74 — LGBM Per-Target Optimized (6 configs × 30 seeds)")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    log.info(f"  Train: {train.shape}, Test: {test.shape}, Features: {len(feat_cols)}")

    target_results = {}
    predictions = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (target rate: {train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        log.info(f"  Leakage-clean features: {len(leak_cols)}")

        # Feature ranking
        X_all = train[leak_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        sn = [sanitize(c) for c in leak_cols]

        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1,
        }
        ds_rank = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        del model_rank, ds_rank, X_all
        gc.collect()

        best_n_feat = 20
        best_cfg_name = 'V_std'
        best_cv = float('inf')
        best_seeds_list = None

        for n_feat in N_FEAT_RANGE:
            sel_cols = [r[0] for r in ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in ranked[:n_feat]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]

            # Build selected feature matrix
            X_sel = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
            
            for cfg in LGBM_CONFIGS:
                t_cfg = time.time()
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        spw_fold = max(((y[tr]==0).sum())/max((y[tr]==1).sum(),1), 0.1)
                        params = {
                            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                            'min_child_samples': cfg['mc'], 'random_state': s,
                            'scale_pos_weight': spw_fold, 'force_row_wise': True,
                            'n_jobs': 1,
                        }
                        ds_tr = lgb.Dataset(X_sel[tr], label=y[tr], feature_name=sel_sn,
                                           params={'verbose': '-1'})
                        m = lgb.train(params, ds_tr, num_boost_round=cfg['ne'])
                        oof[va] += m.predict(X_sel[va])
                        n_valid[va] += 1
                
                # Average
                oof_avg = oof / n_valid
                cv = logloss_fn(y, oof_avg)
                
                log.info(f"    n_feat={n_feat:2d} {cfg['name']:15s}: cv={cv:.4f} ({time.time()-t_cfg:.0f}s)")
                
                if cv < best_cv:
                    best_cv = cv
                    best_n_feat = n_feat
                    best_cfg_name = cfg['name']
                
                del oof, n_valid
                gc.collect()

        log.info(f"\n  ✅ Best: cfg={best_cfg_name}, n_feat={best_n_feat}, cv={best_cv:.4f}")
        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cfg': best_cfg_name,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
        }

        # ── Final: train best config on ALL data ──
        log.info(f"  Training final on all data...")
        sel_cols = [r[0] for r in ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in ranked[:best_n_feat]]
        sel_idx = [leak_cols.index(c) for c in sel_cols]
        X_all = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]

        cfg = next(c for c in LGBM_CONFIGS if c['name'] == best_cfg_name)
        spw_final = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
        
        test_preds = np.zeros(len(X_test))
        for s in range(N_SEEDS):
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'], 'random_state': s,
                'scale_pos_weight': spw_final, 'force_row_wise': True,
                'n_jobs': 1,
            }
            m = lgb.train(params, lgb.Dataset(X_all, label=y, feature_name=sel_sn, params={'verbose':'-1'}),
                         num_boost_round=cfg['ne'])
            test_preds += m.predict(X_test)
        test_preds /= N_SEEDS

        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)
        target_results[target]['test_mean'] = float(test_preds.mean())

        log.info(f"  {target}: cv={best_cv:.4f}, test_mean={test_preds.mean():.4f}")
        del sel_cols, X_all, X_test
        gc.collect()

    # ── Summary ──
    avg_cv = np.mean([v['best_cv'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V74 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: cv={r['best_cv']:.4f} (cfg={r['best_cfg']}, n_feat={r['best_n_feat']})")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Target: 0.5000 | Current: {avg_cv:.4f} | Gap: {avg_cv-0.5:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v74_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V74_lgbm_per_target',
        'name': 'LGBM per-target optimized (6 configs × 30 seeds)',
        'cv_method': 'GroupKFold_5fold',
        'n_models_per_target': 180,  # 6 configs × 30 seeds
        'n_feat_sweep': N_FEAT_RANGE,
        'target_results': target_results,
        'avg_cv': float(avg_cv),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v74_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    main()
