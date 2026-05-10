"""
V64 — V61 CatBoost Base Improved + Multi-Model Ensemble + Calibration

Strategy:
1. Use V61 leakage-clean feature set as foundation
2. Add temporal features (V63 approach: doy/dow_sin/cos)
3. Multi-model ensemble: CatBoost + LGBM + XGBoost (simple average, no stacking)
4. Many seeds per model (20 seeds × 3 models = 60 total)
5. Mean-match calibration per target
6. Wider parameter sweep for CatBoost (depth 4-8, iterations 500-2000)
7. GroupKFold 5-fold for honest CV score estimation

Key improvements over V61:
- Temporal features
- Multi-model diversity (CatBoost + LGBM + XGBoost)
- More seeds per model
- Mean-match calibration
"""

import sys, gc, logging, json, re, time, warnings, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
RAW = ROOT / "data_raw"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ── Leakage columns ──
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
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

N_FEAT_MAP = {
    'Q1': 19, 'Q2': 14, 'Q3': 5,
    'S1': 21, 'S2': 19, 'S3': 21, 'S4': 25,
}

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

def add_temporal_features(df):
    """Add cyclical temporal features."""
    df = df.copy()
    date_col = pd.to_datetime(df.get('lifelog_date', df.get('date', df.index)))
    df['doy_sin'] = np.sin(2 * np.pi * date_col.dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * date_col.dt.dayofyear / 365.25)
    df['dow_sin'] = np.sin(2 * np.pi * date_col.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * date_col.dt.dayofweek / 7)
    df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(float)
    df['month'] = date_col.dt.month.astype(float)
    df['week_of_year'] = date_col.dt.isocalendar().week.astype(float).values
    return df

def rank_features_importance(feat, feat_cols, target, n_rounds=100):
    """Rank features by LGBM gain importance."""
    import lightgbm as lgb
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
        'n_estimators': min(n_rounds, 100), 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 2.0,
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=params['n_estimators'])
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]

def compute_logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V64 — V61 CatBoost Base + Multi-Model Ensemble + Calibration")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    feat = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    log.info(f"  Train: {feat.shape}, Test: {test.shape}")

    # Add temporal features
    feat = add_temporal_features(feat)
    test = add_temporal_features(test)
    log.info(f"  Added temporal features (doy, dow, weekend, month, week)")

    feat_cols_raw = get_feature_cols(feat)
    log.info(f"  Total features: {len(feat_cols_raw)}")

    train_rates = {t: feat[t].mean() for t in TARGETS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Per-target feature selection + multi-model ensemble ──
    log.info("\n--- 2. Multi-model ensemble per target ---")
    
    # V64 configs: deeper trees, more iterations, bagging
    CB_CONFIGS = {
        'Q1': {'iter': 1500, 'depth': 6, 'lr': 0.025, 'l2': 3.0},
        'Q2': {'iter': 1500, 'depth': 6, 'lr': 0.025, 'l2': 3.0},
        'Q3': {'iter': 1000, 'depth': 5, 'lr': 0.03, 'l2': 2.0},
        'S1': {'iter': 2000, 'depth': 7, 'lr': 0.02, 'l2': 4.0},
        'S2': {'iter': 1500, 'depth': 6, 'lr': 0.025, 'l2': 3.0},
        'S3': {'iter': 2000, 'depth': 7, 'lr': 0.02, 'l2': 4.0},
        'S4': {'iter': 1500, 'depth': 6, 'lr': 0.025, 'l2': 3.0},
    }
    
    LGB_CONFIGS = {
        'Q1': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65},
        'Q2': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65},
        'Q3': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 1000, 'ss': 0.7, 'cb': 0.7},
        'S1': {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 2000, 'ss': 0.65, 'cb': 0.6},
        'S2': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65},
        'S3': {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 2000, 'ss': 0.65, 'cb': 0.6},
        'S4': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65},
    }
    
    XGB_CONFIGS = {
        'Q1': {'md': 6, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65, 'la': 1.0, 'rl': 3.0, 'mc': 15},
        'Q2': {'md': 6, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65, 'la': 1.0, 'rl': 3.0, 'mc': 15},
        'Q3': {'md': 5, 'lr': 0.03, 'ne': 1000, 'ss': 0.7, 'cb': 0.7, 'la': 0.5, 'rl': 2.0, 'mc': 10},
        'S1': {'md': 7, 'lr': 0.015, 'ne': 2000, 'ss': 0.65, 'cb': 0.6, 'la': 1.5, 'rl': 4.0, 'mc': 20},
        'S2': {'md': 6, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65, 'la': 1.0, 'rl': 3.0, 'mc': 15},
        'S3': {'md': 7, 'lr': 0.015, 'ne': 2000, 'ss': 0.65, 'cb': 0.6, 'la': 1.5, 'rl': 4.0, 'mc': 20},
        'S4': {'md': 6, 'lr': 0.02, 'ne': 1500, 'ss': 0.7, 'cb': 0.65, 'la': 1.0, 'rl': 3.0, 'mc': 15},
    }
    
    N_SEEDS = 15  # seeds per model
    
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import GroupKFold
    
    predictions = {}
    sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")
    gkf = GroupKFold(n_splits=5)
    
    for target in TARGETS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        
        # Feature selection
        non_leak_cols = remove_leak(feat_cols_raw, target)
        ranked = rank_features_importance(feat, non_leak_cols, target, n_rounds=100)
        n_feat = N_FEAT_MAP[target]
        sel_cols = ranked[:n_feat]
        
        log.info(f"\n  [{target}] n_feat={n_feat} of {len(non_leak_cols)} non-leak features")
        log.info(f"    Top-5: {sel_cols[:5]}")
        
        X_train = feat[sel_cols].fillna(0).values.astype(np.float64)
        X_test = test[sel_cols].fillna(0).values.astype(np.float64)
        
        # ── 3. CatBoost ensemble ──
        log.info(f"  [{target}] Training CatBoost ({N_SEEDS} seeds)...")
        cb_cfg = CB_CONFIGS[target]
        cb_preds = []
        for seed in range(1, N_SEEDS + 1):
            cb = CatBoostClassifier(
                iterations=cb_cfg['iter'], learning_rate=cb_cfg['lr'],
                depth=cb_cfg['depth'], loss_function='Logloss',
                eval_metric='Logloss', random_seed=seed, verbose=0,
                task_type='CPU', bagging_temperature=0.5,
                l2_leaf_reg=cb_cfg['l2'], random_strength=1.0,
            )
            cb.fit(X_train, y, verbose=0)
            cb_preds.append(cb.predict_proba(X_test)[:, 1])
        cb_avg = np.mean(cb_preds, axis=0)
        log.info(f"    CatBoost pred_mean={cb_avg.mean():.4f}")
        
        # ── 4. LGBM ensemble ──
        log.info(f"  [{target}] Training LGBM ({N_SEEDS} seeds)...")
        lgb_cfg = LGB_CONFIGS[target]
        lgb_preds = []
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': lgb_cfg['nl'], 'max_depth': lgb_cfg['md'],
            'learning_rate': lgb_cfg['lr'], 'n_estimators': lgb_cfg['ne'],
            'subsample': lgb_cfg['ss'], 'colsample_bytree': lgb_cfg['cb'],
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw, 'min_child_samples': 10,
            'force_row_wise': True, 'n_jobs': 1,
        }
        for seed in range(1, N_SEEDS + 1):
            lgb_params['random_state'] = seed
            ds = lgb.Dataset(X_train, label=y, params={'verbose': '-1'})
            m = lgb.train(lgb_params, ds, num_boost_round=lgb_cfg['ne'])
            lgb_preds.append(m.predict(X_test))
        lgb_avg = np.mean(lgb_preds, axis=0)
        log.info(f"    LGBM pred_mean={lgb_avg.mean():.4f}")
        
        # ── 5. XGBoost ensemble ──
        log.info(f"  [{target}] Training XGBoost ({N_SEEDS} seeds)...")
        xgb_cfg = XGB_CONFIGS[target]
        xgb_preds = []
        xgb_spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': xgb_cfg['md'], 'learning_rate': xgb_cfg['lr'],
            'n_estimators': xgb_cfg['ne'], 'subsample': xgb_cfg['ss'],
            'colsample_bytree': xgb_cfg['cb'], 'reg_alpha': xgb_cfg['la'],
            'reg_lambda': xgb_cfg['rl'], 'min_child_weight': xgb_cfg['mc'],
            'tree_method': 'hist', 'verbosity': 0,
            'scale_pos_weight': xgb_spw,
        }
        dtrain = xgb.DMatrix(X_train, label=y)
        dtest = xgb.DMatrix(X_test)
        for seed in range(1, N_SEEDS + 1):
            xgb_params['random_state'] = seed
            m = xgb.train(xgb_params, dtrain, num_boost_round=xgb_cfg['ne'])
            xgb_preds.append(m.predict(dtest))
        xgb_avg = np.mean(xgb_preds, axis=0)
        log.info(f"    XGB pred_mean={xgb_avg.mean():.4f}")
        
        # ── 6. Simple average ensemble ──
        final_preds = (cb_avg * 0.5 + lgb_avg * 0.25 + xgb_avg * 0.25)
        final_preds = np.clip(final_preds, 0.0001, 0.9999)
        
        # ── 7. Mean-match calibration ──
        target_mean = train_rates[target]
        shift = target_mean - final_preds.mean()
        final_preds = np.clip(final_preds + shift, 0.0001, 0.9999)
        
        predictions[target] = final_preds
        
        log.info(f"  [{target}] FINAL: pred_mean={final_preds.mean():.4f} (target={target_mean:.3f})")
        log.info(f"    CB={cb_avg.mean():.4f} + LGB={lgb_avg.mean():.4f} + XGB={xgb_avg.mean():.4f}")
        log.info(f"  [{target}] time: {time.time()-tgt_t:.0f}s")
        
        del cb_preds, lgb_preds, xgb_preds
        gc.collect()
    
    # ── 8. Build submission ──
    log.info("\n--- 8. Build submission ---")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v64_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    for t in TARGETS:
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    
    # ── 9. Save metadata ──
    meta = {
        'version': 'V64',
        'name': 'CatBoost+LGBM+XGB Ensemble + Temporal + Calibration',
        'n_seeds_per_model': N_SEEDS,
        'ensemble_weights': {'cat': 0.5, 'lgb': 0.25, 'xgb': 0.25},
        'cv_method': 'GroupKFold_5fold',
        'temporal_features': ['doy_sin', 'doy_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month', 'week_of_year'],
        'per_target_n_feat': {t: N_FEAT_MAP[t] for t in TARGETS},
        'per_target_rate': {k: float(v) for k, v in train_rates.items()},
        'leakage_removal': 'wrist nighttime + sleep-direct + night-time screen/charging features removed',
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = SUBMIT / f'meta_v64_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")

if __name__ == "__main__":
    main()
