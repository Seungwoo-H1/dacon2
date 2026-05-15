"""
V65 — Leakage-Clean CatBoost with z-score features + temporal + parameter sweep

Key insight from V61-V64 analysis:
1. V61 (CatBoost on features_clean_v60 with z-score) = avg CV 0.583 — BEST
2. V63/V64 used features.parquet WITHOUT z-score = worse results (0.608)
3. Leakage removal (nighttime + sleep-direct) was the single biggest improvement
4. Stacking was counter-productive — CatBoost single model wins
5. Z-score personalization features are critical (142 base + 134 zscore = 276)

V65 Strategy:
- Use features_clean_v60 (WITH z-score) as base
- Add temporal features to base features (doy_sin/cos, dow_sin/cos)
- CatBoost single model (V61's winning approach)
- Parameter sweep: depth 4-8, iterations 500-2000, lr 0.015-0.05
- Multiple seeds for ensemble diversity
- Mean-match calibration
- Feature selection: rank by importance, use fewer features to reduce overfitting
"""

import sys, gc, logging, json, re, time, warnings
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
    """Rank features by CatBoost gain importance."""
    from catboost import CatBoostClassifier
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    
    cb = CatBoostClassifier(
        iterations=100, learning_rate=0.03, depth=6,
        loss_function='Logloss', eval_metric='Logloss',
        random_seed=42, verbose=0, task_type='CPU',
        bagging_temperature=0.5, l2_leaf_reg=3.0, random_strength=1.0,
    )
    cb.fit(X, y, verbose=0)
    imp = cb.get_feature_importance()
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]

def train_catboost(X, y, X_test, config, seed):
    """Train CatBoost and return predictions."""
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=config['iter'], learning_rate=config['lr'],
        depth=config['depth'], loss_function='Logloss',
        eval_metric='Logloss', random_seed=seed, verbose=0,
        task_type='CPU', bagging_temperature=config.get('bagging', 0.5),
        l2_leaf_reg=config.get('l2', 3.0), random_strength=config.get('rs', 1.0),
        subsample=config.get('subsample', 0.8),
        colsample_bylevel=config.get('colsample', 0.85),
    )
    cb.fit(X, y, verbose=0)
    return cb.predict_proba(np.where(np.isnan(X_test), 0, X_test))[:, 1]

def compute_logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V65 — Leakage-Clean CatBoost + z-score + temporal + param sweep")
    log.info("=" * 70)

    # ── 1. Load data with z-score features ──
    log.info("\n--- 1. Load data ---")
    feat = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    log.info(f"  Train: {feat.shape}, Test: {test.shape}")

    # Add temporal features
    feat = add_temporal_features(feat)
    test = add_temporal_features(test)
    log.info(f"  Added temporal features (doy, dow, weekend, month, week)")

    feat_cols_raw = get_feature_cols(feat)
    log.info(f"  Total features (base + zscore + temporal): {len(feat_cols_raw)}")

    train_rates = {t: feat[t].mean() for t in TARGETS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Feature selection per target ──
    log.info("\n--- 2. Feature selection ---")
    
    # Parameter sweep configs
    CB_PARAM_SWEEP = [
        {'iter': 1000, 'depth': 5, 'lr': 0.03, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0, 'subsample': 0.8, 'colsample': 0.85},
        {'iter': 1500, 'depth': 6, 'lr': 0.025, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0, 'subsample': 0.75, 'colsample': 0.8},
        {'iter': 2000, 'depth': 7, 'lr': 0.02, 'l2': 4.0, 'bagging': 0.5, 'rs': 1.5, 'subsample': 0.7, 'colsample': 0.75},
        {'iter': 1000, 'depth': 6, 'lr': 0.03, 'l2': 2.0, 'bagging': 0.7, 'rs': 1.0, 'subsample': 0.85, 'colsample': 0.9},
        {'iter': 1500, 'depth': 4, 'lr': 0.05, 'l2': 5.0, 'bagging': 0.3, 'rs': 2.0, 'subsample': 0.9, 'colsample': 0.95},
    ]
    
    N_SEEDS = 20  # seeds per best config
    
    predictions = {}
    sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")
    
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
        
        # ── 3. Parameter sweep on small subset ──
        log.info(f"  [{target}] Parameter sweep (5 configs x 5 seeds each)...")
        best_cv_loss = float('inf')
        best_config = CB_PARAM_SWEEP[0]
        
        for i, cfg in enumerate(CB_PARAM_SWEEP):
            cv_losses = []
            # Simple 3-fold CV for parameter selection
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
                X_tr = X_train[tr_idx]; y_tr = y[tr_idx]
                X_va = X_train[va_idx]; y_va = y[va_idx]
                
                cv_preds = np.zeros(len(y_va))
                for seed in range(1, 6):  # 5 seeds per config
                    cv_preds += train_catboost(X_tr, y_tr, X_va, cfg, seed)
                cv_preds /= 5
                
                cv_losses.append(compute_logloss(y_va, cv_preds))
            
            avg_cv = np.mean(cv_losses)
            log.info(f"    Config {i}: CV={avg_cv:.4f}")
            
            if avg_cv < best_cv_loss:
                best_cv_loss = avg_cv
                best_config = cfg
        
        log.info(f"  [{target}] Best config: {best_config}")
        log.info(f"  [{target}] Best CV: {best_cv_loss:.4f}")
        
        # ── 4. Train final model with best config ──
        log.info(f"  [{target}] Training final model ({N_SEEDS} seeds)...")
        final_preds = np.zeros(len(X_test))
        for seed in range(1, N_SEEDS + 1):
            final_preds += train_catboost(X_train, y, X_test, best_config, seed)
        final_preds /= N_SEEDS
        
        # ── 5. Mean-match calibration ──
        target_mean = train_rates[target]
        shift = target_mean - final_preds.mean()
        final_preds = np.clip(final_preds + shift, 0.0001, 0.9999)
        
        predictions[target] = final_preds
        
        log.info(f"  [{target}] FINAL: pred_mean={final_preds.mean():.4f} (target={target_mean:.3f})")
        log.info(f"  [{target}] time: {time.time()-tgt_t:.0f}s")
        
        del sel_cols, y, X_train, X_test
        gc.collect()
    
    # ── 6. Build submission ──
    log.info("\n--- 6. Build submission ---")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v65_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    for t in TARGETS:
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    
    # ── 7. Save metadata ──
    meta = {
        'version': 'V65',
        'name': 'Leakage-Clean CatBoost + z-score + temporal + param sweep',
        'n_seeds': N_SEEDS,
        'data_source': 'features_clean_v60.parquet (with z-score)',
        'temporal_features': ['doy_sin', 'doy_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month', 'week_of_year'],
        'per_target_n_feat': {t: N_FEAT_MAP[t] for t in TARGETS},
        'per_target_rate': {k: float(v) for k, v in train_rates.items()},
        'leakage_removal': 'wrist nighttime + sleep-direct + night-time screen/charging features removed',
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = SUBMIT / f'meta_v65_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")

if __name__ == "__main__":
    main()
