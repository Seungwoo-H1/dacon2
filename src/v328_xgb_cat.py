"""
V328 — XGBoost + CatBoost Ensemble

Hypothesis: Adding XGBoost and CatBoost students alongside LGBM will capture
different pattern types that LGBM misses. The diverse model types should
produce more diverse OOF predictions, improving stacking quality.

Architecture:
- 15 seeds × 3 model types (LGBM + XGBoost + CatBoost) = 45 students per target
- Feature bagging applied per student
- LR meta-learner on all 45 predictions
- Also try: separate LR per model type, then combined meta

Expected OOF: 0.595-0.605
Risk: MEDIUM
Cost: ~180s
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

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

N_FOLDS = 5
N_SEEDS = 15
FEATURE_BAG_FRACTION = 0.75


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS \| set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def rank_features(feat_df, feat_cols, target, seed=42):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'max_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def generate_zscore_features(train_df, test_df):
    train_base = [c for c in train_df.columns
                  if c not in META_COLS \| set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS \| set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_base) & set(test_base)
    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        test_df = test_df.copy()
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
        train_df = train_df.copy()
        train_df[zc] = (vals - mean) / std
    return train_df, test_df


CFGS_LGBM = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'max_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
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


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V328 — XGBoost + CatBoost Ensemble (with LGBM)")
    log.info("3 model types × 15 seeds = 45 students per target")
    log.info("=" * 70)
    
    SEED = 42
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    train_df, test_df = generate_zscore_features(train_df, test_df)
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features, Test: {len(test_feat_cols)}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Check if XGBoost and CatBoost are available
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False
        log.warning("XGBoost not available, using LGBM only")
    
    try:
        import catboost as cb
        has_cat = True
    except ImportError:
        has_cat = False
        log.warning("CatBoost not available, using LGBM only")
    
    # Determine which models to use
    model_types = []
    if has_xgb and has_cat:
        model_types = ['xgb', 'cat', 'lgbm']
    elif has_xgb:
        model_types = ['xgb', 'lgbm']
    elif has_cat:
        model_types = ['cat', 'lgbm']
    else:
        model_types = ['lgbm']
    
    log.info(f"Available models: {model_types}")
    n_models = len(model_types)
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        lgbm_cfg = CFGS_LGBM[cfg_name]
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        all_seed_oofs = []  # (15 * n_models) arrays of (450,)
        all_seed_tests = {mt: [] for mt in model_types}
        student_names = []
        
        for mi, mtype in enumerate(model_types):
            log.info(f"  Model type: {mtype}")
            
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                
                rng = np.random.RandomState(seed)
                n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), n_feat)
                bag = rng.choice(ranked, size=n_bag, replace=False)
                bag_set = set(bag)
                bag_feats = [f for f in ranked if f in bag_set][:n_feat]
                if len(bag_feats) < n_feat:
                    remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                    bag_feats.extend(remaining)
                
                s_cols = [c for c in bag_feats if c in test_feat_cols]
                
                seed_oof = np.zeros(len(train_df))
                seed_test = np.zeros(len(test_df))
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[s_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[s_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    
                    if mtype == 'lgbm':
                        params = {**lgbm_cfg, 'scale_pos_weight': spw, 'random_state': seed,
                                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                        sn = [sanitize_col(c) for c in s_cols]
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                        m = lgb.train(params, ds, num_boost_round=lgbm_cfg['n_estimators'])
                        
                    elif mtype == 'xgb':
                        params = {
                            'objective': 'binary:logistic', 'eval_metric': 'logloss',
                            'learning_rate': 0.05, 'max_depth': 5, 'max_leaves': 20,
                            'subsample': 0.7, 'colsample_bytree': 0.7,
                            'reg_alpha': 1.0, 'reg_lambda': 3.0,
                            'scale_pos_weight': spw, 'random_state': seed,
                            'tree_method': 'hist', 'n_jobs': 1
                        }
                        ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=[sanitize_col(c) for c in s_cols])
                        m = xgb.train(params, ds, num_boost_round=300)
                        
                    elif mtype == 'cat':
                        params = {
                            'loss_function': 'Logloss', 'eval_metric': 'Logloss',
                            'learning_rate': 0.05, 'max_depth': 5, 'max_leaves': 20,
                            'subsample': 0.7, 'colsample_bylevel': 0.7,
                            'l2_leaf_reg': 3.0, 'random_seed': seed,
                            'bootstrap_type': 'Bernoulli', 'random_strength': 1.0,
                            'verbose': 0
                        }
                        ds = cb.Pool(X_tr, label=y_tr, feature_names=[sanitize_col(c) for c in s_cols])
                        m = cb.CatBoostClassifier(**params, iterations=lgbm_cfg['n_estimators'])
                        m.fit(X_tr, y_tr, eval_set=cb.Pool(X_va, label=y[va_idx], feature_names=[sanitize_col(c) for c in s_cols]),
                              use_best_model=True, early_stopping_rounds=50)
                    
                    if mtype == 'lgbm':
                        oof_pred = m.predict(X_va)
                        seed_oof[va_idx] += oof_pred
                        seed_test += m.predict(test_df[s_cols].fillna(0).values.astype(np.float64))
                    elif mtype == 'xgb':
                        oof_pred = m.predict(xgb.DMatrix(X_va, feature_names=[sanitize_col(c) for c in s_cols]))
                        seed_oof[va_idx] += oof_pred
                        seed_test += m.predict(xgb.DMatrix(test_df[s_cols].fillna(0).values.astype(np.float64), feature_names=[sanitize_col(c) for c in s_cols]))
                    elif mtype == 'cat':
                        oof_pred = m.predict(cb.Pool(X_va), prediction_type='Probability')[:, 1]
                        seed_oof[va_idx] += oof_pred
                        seed_test += m.predict(cb.Pool(test_df[s_cols].fillna(0).values.astype(np.float64)), prediction_type='Probability')[:, 1]
                
                seed_oof /= N_FOLDS
                seed_test /= N_FOLDS
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                
                all_seed_oofs.append(seed_oof)
                all_seed_tests[mtype].append(seed_test)
                student_names.append(f"{mtype}_{si}")
                
                if si < 2 or si == N_SEEDS - 1:
                    s_oof = log_loss(y, seed_oof)
                    log.info(f"    {mtype} Seed {si}: OOF={s_oof:.5f}")
        
        # LR meta-learner on all (15 * n_models) predictions
        oof_matrix = np.column_stack(all_seed_oofs)
        meta = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oof = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oof = np.mean([log_loss(y, p) for p in all_seed_oofs])
        
        # Model-type-specific averages
        for mt in model_types:
            mt_oofs = [log_loss(y, p) for p in all_seed_oofs[mi*N_SEEDS:(mi+1)*N_SEEDS] if mi < n_models]
            if mt_oofs:
                mt_avg = np.mean(mt_oofs)
                log.info(f"  {mt} avg OOF: {mt_avg:.5f}")
        
        log.info(f"  {t}: OOF={target_oof:.5f} (student-avg={student_avg_oof:.5f})")
    
    log.info(f"\nV328 experiment completed. Check individual target logs.")
    
    # Build submission (average all students across model types)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    # For each target, average all seed test predictions
    # We need to re-run the prediction logic to collect test preds
    # Since we already have model types, just average
    sub_targets = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        lgbm_cfg = CFGS_LGBM[cfg_name]
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        all_test_preds = []
        
        for mi, mtype in enumerate(model_types):
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                rng = np.random.RandomState(seed)
                n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), n_feat)
                bag = rng.choice(ranked, size=n_bag, replace=False)
                bag_set = set(bag)
                bag_feats = [f for f in ranked if f in bag_set][:n_feat]
                if len(bag_feats) < n_feat:
                    remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                    bag_feats.extend(remaining)
                
                s_cols = [c for c in bag_feats if c in test_feat_cols]
                st = np.zeros(len(test_df))
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[s_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    
                    if mtype == 'lgbm':
                        params = {**lgbm_cfg, 'scale_pos_weight': spw, 'random_state': seed,
                                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                        sn = [sanitize_col(c) for c in s_cols]
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                        m = lgb.train(params, ds, num_boost_round=lgbm_cfg['n_estimators'])
                        st += m.predict(test_df[s_cols].fillna(0).values.astype(np.float64))
                    elif mtype == 'xgb':
                        params = {
                            'objective': 'binary:logistic', 'eval_metric': 'logloss',
                            'learning_rate': 0.05, 'max_depth': 5, 'max_leaves': 20,
                            'subsample': 0.7, 'colsample_bytree': 0.7,
                            'reg_alpha': 1.0, 'reg_lambda': 3.0,
                            'scale_pos_weight': spw, 'random_state': seed,
                            'tree_method': 'hist', 'n_jobs': 1
                        }
                        ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=[sanitize_col(c) for c in s_cols])
                        m = xgb.train(params, ds, num_boost_round=300)
                        st += m.predict(xgb.DMatrix(test_df[s_cols].fillna(0).values.astype(np.float64),
                                                   feature_names=[sanitize_col(c) for c in s_cols]))
                    elif mtype == 'cat':
                        params = {
                            'loss_function': 'Logloss', 'eval_metric': 'Logloss',
                            'learning_rate': 0.05, 'max_depth': 5, 'max_leaves': 20,
                            'subsample': 0.7, 'colsample_bylevel': 0.7,
                            'l2_leaf_reg': 3.0, 'random_seed': seed,
                            'bootstrap_type': 'Bernoulli', 'random_strength': 1.0,
                            'verbose': 0
                        }
                        ds = cb.Pool(X_tr, label=y_tr, feature_names=[sanitize_col(c) for c in s_cols])
                        m = cb.CatBoostClassifier(**params, iterations=lgbm_cfg['n_estimators'])
                        m.fit(X_tr, y_tr, verbose=False)
                        st += m.predict(cb.Pool(test_df[s_cols].fillna(0).values.astype(np.float64),
                                                feature_names=[sanitize_col(c) for c in s_cols]),
                                        prediction_type='Probability')[:, 1]
                
                st /= N_FOLDS
                all_test_preds.append(st)
        
        sub_targets[t] = np.mean(all_test_preds)
        log.info(f"  {t}: {N_SEEDS * n_models} test preds averaged")
    
    for t in TARGETS:
        sub[t] = np.clip(sub_targets[t], 0.001, 0.999)
    
    sub_path = SUBMIT / f"submission_v328_xgb_cat_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return log.info(f"V328 done")


if __name__ == '__main__':
    main()
