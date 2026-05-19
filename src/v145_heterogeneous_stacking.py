"""
V145 — Heterogeneous Multi-Model Stacking

Hypothesis: V140 uses only LGBM (3 seeds). All students same family.
V145 adds CatBoost + XGBoost: 2 LGBM + 2 CB + 2 XGB = 6 students.
Same features, same CV, different algorithms = orthogonal errors.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

SEED = 42
N_FOLDS = 5
N_SEEDS = 2  # per model family
META_C = 0.1


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat_df, feat_cols, target, seed=SEED):
    import lightgbm as lgb
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def lgbm_to_catboost(lgbm_params):
    """Convert LightGBM params to CatBoost params.
    CatBoost: num_leaves only works with lossguide tree type (not default).
    Use max_depth instead (which we already have).
    """
    return {
        'iterations': lgbm_params.get('n_estimators', 300),
        'learning_rate': lgbm_params.get('learning_rate', 0.05),
        'max_depth': lgbm_params.get('max_depth', 3),
        'subsample': lgbm_params.get('subsample', 0.8),
        'l2_leaf_reg': lgbm_params.get('reg_alpha', 2.0),
        'min_data_in_leaf': lgbm_params.get('min_child_samples', 5),
        'objective': 'Logloss',
        'eval_metric': 'Logloss',
    }


def lgbm_to_xgb(lgbm_params):
    """Convert LightGBM params to XGBoost params."""
    return {
        'n_estimators': lgbm_params.get('n_estimators', 300),
        'learning_rate': lgbm_params.get('learning_rate', 0.05),
        'max_depth': lgbm_params.get('max_depth', 3),
        'min_child_weight': lgbm_params.get('min_child_samples', 5),
        'subsample': lgbm_params.get('subsample', 0.8),
        'colsample_bytree': lgbm_params.get('colsample_bytree', 0.8),
        'reg_alpha': lgbm_params.get('reg_alpha', 2.0),
        'reg_lambda': lgbm_params.get('reg_lambda', 5.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
    }


if __name__ == '__main__':
    t_start = time.time()
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    log.info("=" * 70)
    log.info("V145 — Heterogeneous Multi-Model Stacking")
    log.info("2 LGBM + 2 CatBoost + 2 XGBoost = 6 students per target")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
    
    # Configs per target (V140's mapping)
    CFGS = {
        'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
        'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
                   'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
        'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }
    V53_SWEEP = {
        'Q1':  {'cfg': 'deep',   'n_feat': 19},
        'Q2':  {'cfg': 'deep',   'n_feat': 14},
        'Q3':  {'cfg': 'v48',    'n_feat': 11},
        'S1':  {'cfg': 'wide',   'n_feat': 21},
        'S2':  {'cfg': 'deep',   'n_feat': 19},
        'S3':  {'cfg': 'safety', 'n_feat': 23},
        'S4':  {'cfg': 'wide',   'n_feat': 20},
    }
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros(len(test_df)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n  --- {t} ---")
        y = train_df[t].values.astype(np.float64)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg_base = CFGS[cfg_name]
        
        fc_leaked = remove_leak(feat_cols, t)
        ranked = rank_features(train_df, fc_leaked, t)
        selected_cols = ranked[:n_feat]
        
        X_all = train_df[selected_cols].fillna(0).values.astype(np.float64)
        X_test = test_df[selected_cols].fillna(0).values.astype(np.float64)
        y_arr = y
        
        oof_preds_list = []
        test_preds_list = []
        
        # === 2 LGBM seeds ===
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                spw = max(((y_arr[tr_idx] == 0).sum()) / max((y_arr[tr_idx] == 1).sum(), 1), 0.1)
                params = {**cfg_base, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in selected_cols]
                ds_tr = lgb.Dataset(X_all[tr_idx], label=y_arr[tr_idx], feature_name=sn)
                m = lgb.train(params, ds_tr, num_boost_round=cfg_base['n_estimators'])
                seed_oof[va_idx] += m.predict(X_all[va_idx])
                # Test: refit on fold train, predict test
                m_full = lgb.train(params, ds_tr, num_boost_round=cfg_base['n_estimators'])
                seed_test += m_full.predict(X_test)
            seed_oof /= N_FOLDS
            seed_test /= N_FOLDS
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test = np.clip(seed_test, 0.001, 0.999)
            oof_preds_list.append(seed_oof)
            test_preds_list.append(seed_test)
            log.info(f"    LGBM s{seed}: OOF={log_loss(y, seed_oof):.5f}")
        
        # === 2 CatBoost seeds ===
        for si in range(N_SEEDS):
            seed = SEED + si * 7 + 100
            cb_cfg = lgbm_to_catboost(cfg_base)
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                cb_params = {**cb_cfg, 'random_seed': seed}
                m = cb.CatBoostClassifier(**cb_params, auto_class_weights='Balanced')
                m.fit(X_all[tr_idx], y_arr[tr_idx], eval_set=(X_all[va_idx], y_arr[va_idx]),
                      early_stopping_rounds=50)
                seed_oof[va_idx] += m.predict_proba(X_all[va_idx])[:, 1]
                # Test: refit on fold train, predict test
                m_full = cb.CatBoostClassifier(**cb_params, auto_class_weights='Balanced')
                m_full.fit(X_all[tr_idx], y_arr[tr_idx], eval_set=(X_all[va_idx], y_arr[va_idx]),
                           early_stopping_rounds=50)
                seed_test += m_full.predict_proba(X_test)[:, 1]
            seed_oof /= N_FOLDS
            seed_test /= N_FOLDS
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test = np.clip(seed_test, 0.001, 0.999)
            oof_preds_list.append(seed_oof)
            test_preds_list.append(seed_test)
            log.info(f"    CatBoost s{seed}: OOF={log_loss(y, seed_oof):.5f}")
        
        # === 2 XGBoost seeds ===
        for si in range(N_SEEDS):
            seed = SEED + si * 7 + 200
            xgb_cfg = lgbm_to_xgb(cfg_base)
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                spw = max(((y_arr[tr_idx] == 0).sum()) / max((y_arr[tr_idx] == 1).sum(), 1), 0.1)
                xgb_params = {**xgb_cfg, 'scale_pos_weight': spw, 'random_state': seed}
                m = xgb.XGBClassifier(**xgb_params)
                m.fit(X_all[tr_idx], y_arr[tr_idx], eval_set=[(X_all[va_idx], y_arr[va_idx])], verbose=False)
                seed_oof[va_idx] += m.predict_proba(X_all[va_idx])[:, 1]
                # Test: refit on fold train, predict test
                m_full = xgb.XGBClassifier(**xgb_params)
                m_full.fit(X_all[tr_idx], y_arr[tr_idx], eval_set=[(X_all[va_idx], y_arr[va_idx])], verbose=False)
                seed_test += m_full.predict_proba(X_test)[:, 1]
            seed_oof /= N_FOLDS
            seed_test /= N_FOLDS
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test = np.clip(seed_test, 0.001, 0.999)
            oof_preds_list.append(seed_oof)
            test_preds_list.append(seed_test)
            log.info(f"    XGBoost s{seed}: OOF={log_loss(y, seed_oof):.5f}")
        
        # Stack: LR meta-learner
        stacked = np.column_stack(oof_preds_list)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        oof_pred = meta.predict_proba(stacked)[:, 1]
        test_stacked = np.column_stack(test_preds_list)
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        
        ll = log_loss(y, np.clip(oof_pred, 0.001, 0.999))
        log.info(f"  Stacking OOF: {ll:.5f}")
        
        train_oof[t] = oof_pred
        test_preds[t] = test_pred
    
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) for t in TARGETS])
    
    log.info(f"\n{'='*70}")
    log.info(f"AVG OOF: {avg_oof:.5f}")
    log.info(f"V140 AVG OOF: 0.64110")
    log.info(f"Δ vs V140: {avg_oof - 0.64110:+.5f}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v145_hetero_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    meta_data = {
        'version': 'V145',
        'name': 'Heterogeneous Multi-Model Stacking (2 LGBM + 2 CB + 2 XGB = 6 students)',
        'avg_oof': round(float(avg_oof), 5),
        'meta_C': META_C,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v145_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
