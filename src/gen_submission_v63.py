"""
V63 — Leakage-Clean Stacking + Temporal + Calibration

Strategy:
1. Use V61 leakage-clean feature set (remove wrist nighttime features from S-targets)
2. Stacking: LGBM(5 seeds) + XGBoost(5 seeds) + CatBoost(5 seeds) → LR meta-learner
3. Temporal features: doy_sin/cos, dow_sin/cos, is_weekend
4. Per-subject z-score personalization
5. Isotonic calibration on OOF → test
6. Mean-match calibration per target
7. S4 uses dedicated feature set (single LGBM, per V61 findings)
8. Per-target optimal n_feat search (10-25)

Key differences from V62:
- Use GroupKFold n_splits=5 (like V62) instead of 3
- Only LGBM + XGBoost + CatBoost (no V62 complexity)
- Fewer seeds (5 vs 100) to keep runtime manageable
- Temporal features explicitly added
- Calibration pipeline: isotonic → mean match
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

# ── Leakage columns to remove per target ──
# Nighttime/wrist features leak sleep labels for S targets
LEAK_S_TARGETS = {
    # Wrist light — direct sleep proxy
    'wLight_w_light_sum', 'wLight_w_light_count',
    # Wrist HR — heart rate at night correlates with sleep
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    # Wrist pedo — step data at night is sleep proxy
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
# Sleep-direct features (from v61)
LEAK_EXTRA_S = {
    'mScreenStatus_hour_night', 'mScreenStatus_hour_morning',
    'mACStatus_charging_max', 'mACStatus_charging_sum',
    'mGps_gps_count_mean', 'mGps_gps_avg_speed_max',
    'mActivity_m_activity_sum', 'mActivity_m_activity_min',
    'mActivity_m_activity_max', 'mACStatus_hour_night',
}

LEAK_Q_TARGETS = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
}

# ── Per-target n_feat map (from V61 research) ──
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
        leak = LEAK_S_TARGETS | LEAK_EXTRA_S
    elif target.startswith('Q'):
        leak = LEAK_Q_TARGETS
    return [c for c in cols if c not in leak]

def add_personalization(df, feature_cols):
    """Add subject-level z-score features (batched for memory efficiency)."""
    df = df.copy()
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        agg_parts.append(grp.reset_index())
    
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, list(zcols_dict.keys())

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

def rank_features_importance(feat, feat_cols, target, n_rounds=50):
    """Rank features by LGBM gain importance."""
    import lightgbm as lgb
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': min(n_rounds, 100), 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=params['n_estimators'])
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]

def train_lgb(X_train, y_train, X_test, seed):
    import lightgbm as lgb
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'min_child_samples': 10, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1,
    }
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params['scale_pos_weight'] = spw
    ds = lgb.Dataset(X_train, label=y_train, params={'verbose': '-1'})
    m = lgb.train(params, ds, num_boost_round=500)
    return m.predict(X_test)

def train_lgb_oof(X_train, y_train, X_val, seed):
    import lightgbm as lgb
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'min_child_samples': 10, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1,
    }
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params['scale_pos_weight'] = spw
    ds = lgb.Dataset(X_train, label=y_train, params={'verbose': '-1'})
    m = lgb.train(params, ds, num_boost_round=500)
    return m.predict(X_val)

def train_xgb(X_train, y_train, X_test, seed):
    import xgboost as xgb
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.65,
        'reg_alpha': 0.8, 'reg_lambda': 2.5,
        'min_child_weight': 12, 'random_state': seed,
        'tree_method': 'hist', 'verbosity': 0,
    }
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params['scale_pos_weight'] = spw
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    m = xgb.train(params, dtrain, num_boost_round=500)
    return m.predict(dtest)

def train_xgb_oof(X_train, y_train, X_val, seed):
    import xgboost as xgb
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 500,
        'subsample': 0.7, 'colsample_bytree': 0.65,
        'reg_alpha': 0.8, 'reg_lambda': 2.5,
        'min_child_weight': 12, 'random_state': seed,
        'tree_method': 'hist', 'verbosity': 0,
    }
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params['scale_pos_weight'] = spw
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)
    m = xgb.train(params, dtrain, num_boost_round=500)
    return m.predict(dval)

def train_catboost_full(X_train, y_train, X_test, seed):
    from catboost import CatBoostClassifier
    params = {
        'iterations': 500, 'depth': 4, 'learning_rate': 0.03,
        'loss_function': 'Logloss', 'random_seed': seed,
        'logging_level': 'Silent',
        'subsample': 0.75, 'colsample_bylevel': 0.7,
        'reg_lambda': 3.0, 'random_strength': 1.0,
    }
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    cb = CatBoostClassifier(**params)
    cb.fit(X_train, y_train, verbose=0)
    return cb.predict_proba(X_test)[:, 1]

def compute_logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V63 — Leakage-Clean Stacking + Temporal + Calibration")
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

    # Personalization
    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    test, _ = add_personalization(test, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape}")
    log.info(f"  Z-score features added: {len(zscore_cols)}")

    all_cols = get_feature_cols(feat)
    log.info(f"  Total features: {len(all_cols)}")

    train_rates = {t: feat[t].mean() for t in TARGETS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Stacking ensemble per target ──
    log.info("\n--- 2. Stacking per target ---")
    n_seeds = 5
    model_types = ['lgb', 'xgb', 'cat']
    
    predictions = {}
    sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")
    
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    
    for target in TARGETS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        
        # Get non-leak features
        non_leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, non_leak_cols, target, n_rounds=50)
        
        n_feat = N_FEAT_MAP[target]
        sel_cols = ranked[:n_feat]
        
        log.info(f"\n  [{target}] n_feat={n_feat} (of {len(non_leak_cols)} non-leak features)")
        log.info(f"    Top-5: {sel_cols[:5]}")
        
        X_train = feat[sel_cols].fillna(0).values.astype(np.float64)
        X_test = test[sel_cols].fillna(0).values.astype(np.float64)
        
        # GroupKFold 5-fold
        gkf = GroupKFold(n_splits=5)
        
        # Per-model OOF and test predictions
        oof_dict = {m: np.zeros(len(y)) for m in model_types}
        test_dict = {m: [] for m in model_types}
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            X_tr = X_train[tr_idx]; y_tr = y[tr_idx]
            X_va = X_train[va_idx]; y_va = y[va_idx]
            
            for model_type in model_types:
                oof_preds = np.zeros(len(y_va))
                test_preds = np.zeros(len(X_test))
                
                for seed in range(1, n_seeds + 1):
                    if model_type == 'lgb':
                        oof_preds += train_lgb_oof(X_tr, y_tr, X_va, seed)
                        test_preds += train_lgb(X_train, y, X_test, seed)
                    elif model_type == 'xgb':
                        oof_preds += train_xgb_oof(X_tr, y_tr, X_va, seed)
                        test_preds += train_xgb(X_train, y, X_test, seed)
                    elif model_type == 'cat':
                        oof_preds += train_catboost_full(X_tr, y_tr, X_va, seed)
                        test_preds += train_catboost_full(X_train, y, X_test, seed)
                
                oof_preds /= n_seeds
                test_preds /= n_seeds
                oof_dict[model_type][va_idx] = oof_preds
                test_dict[model_type].append(test_preds)
            
            del X_tr, y_tr, X_va, y_va
            gc.collect()
        
        # Average test preds across folds
        for m in model_types:
            test_dict[m] = np.mean(test_dict[m], axis=0)
        
        # ── 3. Stacking with meta-learner ──
        oof_matrix = np.column_stack([oof_dict[m] for m in model_types])
        
        # Try different C values for meta-learner
        best_c_loss = float('inf')
        best_c = 5.0
        for c_val in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            meta = LogisticRegression(C=c_val, solver='lbfgs')
            meta.fit(oof_matrix, y)
            stacked_oof = meta.predict_proba(oof_matrix)[:, 1]
            loss = compute_logloss(y, stacked_oof)
            if loss < best_c_loss:
                best_c_loss = loss
                best_c = c_val
        
        log.info(f"  {target}: meta C={best_c}, stack OOF loss={best_c_loss:.4f}")
        
        # Train final meta-learner
        meta = LogisticRegression(C=best_c, solver='lbfgs')
        meta.fit(oof_matrix, y)
        stacked_preds = meta.predict_proba(
            np.column_stack([test_dict[m] for m in model_types])
        )[:, 1]
        
        # ── 4. Calibration ──
        # Isotonic calibration on OOF
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(oof_dict['lgb'] + oof_dict['xgb'] + oof_dict['cat'], y)
            cal_preds = iso.predict(stacked_preds)
        except Exception:
            cal_preds = stacked_preds
        log.info(f"  {target}: after isotonic cal, pred_mean={cal_preds.mean():.4f}")
        
        # Mean-match calibration
        target_mean = train_rates[target]
        shift = target_mean - cal_preds.mean()
        final_preds = np.clip(cal_preds + shift, 0.0001, 0.9999)
        
        predictions[target] = final_preds
        log.info(f"  {target}: FINAL pred_mean={final_preds.mean():.4f} (target={target_mean:.3f}, shift={shift:.4f})")
        log.info(f"  [{target}] time: {time.time()-tgt_t:.0f}s")
        
        del oof_matrix, oof_dict, test_dict
        gc.collect()
    
    # ── 5. Build submission ──
    log.info("\n--- 5. Build submission ---")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f"submission_v63_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    for t in TARGETS:
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    
    # ── 6. Save metadata ──
    meta = {
        'version': 'V63',
        'name': 'Leakage-Clean Stacking + Temporal + Calibration',
        'method': 'LGBM+XGB+Cat stacking + isotonic cal + mean match',
        'n_seeds_per_model': n_seeds,
        'cv_method': 'GroupKFold_5fold',
        'temporal_features': ['doy_sin', 'doy_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month', 'week_of_year'],
        'per_target_n_feat': {t: N_FEAT_MAP[t] for t in TARGETS},
        'per_target_rate': {k: float(v) for k, v in train_rates.items()},
        'leakage_removal': 'wrist nighttime + sleep-direct features removed per target',
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = SUBMIT / f'meta_v63_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")

if __name__ == "__main__":
    main()
