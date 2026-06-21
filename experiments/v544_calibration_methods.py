#!/usr/bin/env python3
"""
V544 — Ensemble Calibration: Temperature Scaling + Isotonic Regression

Hypothesis: The OOF-to-LB gap comes from miscalibrated predictions.
V537/V541/V542 use simple min-max scaling which may not be optimal.
Try:
1. Temperature scaling (learnable T)
2. Isotonic regression (non-parametric calibration)
3. Plackett-Luce blend (multiple calibrated preds)

Compare: min-max vs temperature vs isotonic on V541 base.
"""
import sys, gc, logging, json, re, time, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb, xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
          'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
          'wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean',
          'wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
          'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean',
          'wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
          'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}
SEED = 42
N_FOLDS = 5
N_SEEDS = 13
V308_GAPS = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}

# V540 configs
V540_TOP = {
    'Q1':  ('s_strong', 'heavy_lgb', 600),
    'Q2':  ('q_narrow', 'heavy_lgb', 800),
    'Q3':  ('heavy_reg', 'light_lgb', 500),
    'S1':  ('s_strong', 'heavy_lgb', 500),
    'S2':  ('light_reg', 'heavy_lgb', 500),
    'S3':  ('s_strong', 'heavy_lgb', 1000),
    'S4':  ('s_strong', 'heavy_lgb', 300),
}

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
    'heavy_reg': {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 15.0, 'min_child_weight': 10},
    'light_reg': {'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_weight': 1},
}

LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'heavy_lgb': {'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 25},
    'light_lgb': {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_samples': 3},
}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def rank_features(df, feat_cols, target, seed=SEED):
    y = df[target].values.astype(np.float64)
    X = df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X; gc.collect()
    return [r[0] for r in ranked]

def train_one_xgb(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, **mp):
    params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
    ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
    ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
    m = xgb.train(params, ds_tr, num_boost_round=n_est)
    return m.predict(ds_va), m.predict(ds_te)

def train_one_lgbm(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, **mp):
    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
             'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_names]
    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
    m = lgb.train(params, ds_tr, num_boost_round=n_est)
    return m.predict(X_va), m.predict(X_test)


def calibrate_temperature(train_raw, train_labels, test_raw, T_range=np.logspace(-2, 2, 50)):
    """Find optimal temperature T that minimizes log_loss on train."""
    best_T = 1.0
    best_ll = float('inf')
    for T in T_range:
        logits = np.log(np.clip(train_raw, 0.001, 0.999) / (1 - np.clip(train_raw, 0.001, 0.999)))
        calibrated = 1 / (1 + np.exp(-logits / T))
        calibrated = np.clip(calibrated, 0.001, 0.999)
        ll = log_loss(train_labels, calibrated)
        if ll < best_ll:
            best_ll = ll
            best_T = T
    
    logits_test = np.log(np.clip(test_raw, 0.001, 0.999) / (1 - np.clip(test_raw, 0.001, 0.999)))
    calibrated_test = 1 / (1 + np.exp(-logits_test / best_T))
    calibrated_test = np.clip(calibrated_test, 0.001, 0.999)
    return calibrated_test, best_T, best_ll


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V544: Calibration Methods (Temperature, Isotonic, Min-Max)")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m_val, s_val = np.mean(tv), np.std(tv, ddof=0)
        if s_val < 1e-8: s_val = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m_val) / s_val
        test_df[zc] = (ev - m_val) / s_val
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # ================================================================
    # Run all targets with V540 configs (same as V541/V542)
    # ================================================================
    all_data = {}
    for target in TARGETS:
        xgb_cfg, lgbm_cfg, n_est = V540_TOP[target]
        bc = {'Q1': 3, 'Q2': 10, 'Q3': 7, 'S1': 3, 'S2': 7, 'S3': 23, 'S4': 20}
        n_feat = bc[target]
        
        log.info(f'\n{target}: {xgb_cfg}+{lgbm_cfg}, n_feat={n_feat}, n_est={n_est}')
        
        ranked = rank_features(train_df, get_feature_cols(train_df), target)
        sel_cols = ranked[:n_feat]
        test_cols = [c for c in sel_cols if c in get_feature_cols(test_df)]
        if len(test_cols) != len(sel_cols): sel_cols = test_cols
        
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        n_train = len(train_df)
        n_test = len(test_df)
        
        xgb_mp = XGB_CFGS[xgb_cfg]
        lgbm_mp = LGBM_CFGS[lgbm_cfg]
        
        xgb_oofs = []; lgbm_oofs = []; xgb_tests = []; lgbm_tests = []
        train_preds_for_meta = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 11
            oof_xgb = np.zeros(n_train)
            oof_lgbm = np.zeros(n_train)
            test_xgb = np.zeros(n_test)
            test_lgbm = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                pvxgb, ttxgb = train_one_xgb(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **xgb_mp)
                pvlgbm, ttlgbm = train_one_lgbm(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, n_est, **lgbm_mp)
                oof_xgb[va_idx] = pvxgb; oof_lgbm[va_idx] = pvlgbm
                test_xgb += ttxgb; test_lgbm += ttlgbm
            
            oof_xgb = np.clip(oof_xgb, 0.001, 0.999)
            oof_lgbm = np.clip(oof_lgbm, 0.001, 0.999)
            test_xgb /= N_FOLDS; test_lgbm /= N_FOLDS
            xgb_oofs.append(oof_xgb); lgbm_oofs.append(oof_lgbm)
            xgb_tests.append(test_xgb); lgbm_tests.append(test_lgbm)
            
            # For meta training, use OOF predictions per fold
            train_pred_avg = (oof_xgb + oof_lgbm) / 2
            train_preds_for_meta.append(train_pred_avg)
        
        all_data[target] = {
            'oofs': xgb_oofs + lgbm_oofs,
            'tests': xgb_tests + lgbm_tests,
            'train_preds': train_preds_for_meta,  # N_SEEDS arrays of length N_train
            'y': y,
        }
    
    # ================================================================
    # Compare calibration methods
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V544: Calibration Method Comparison')
    log.info('=' * 70)
    
    cal_methods = {
        'minmax': 'min_max',
        'temperature': 'temperature_scaling',
        'isotonic': 'isotonic_regression',
    }
    
    all_results = {}
    
    for method_name, method_type in cal_methods.items():
        target_gaps = {}
        
        for target in TARGETS:
            oofs_2d = np.column_stack(all_data[target]['oofs'])
            y = all_data[target]['y']
            train_preds = all_data[target]['train_preds']
            
            n_seeds = oofs_2d.shape[1]
            avg_student = np.mean([log_loss(y, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
            
            # Per-seed calibration
            train_calibrated_all = np.zeros((N_SEEDS, len(y)))
            test_calibrated_all = np.zeros((N_SEEDS, len(test_df)))
            
            for si in range(N_SEEDS):
                train_raw = train_preds[si]
                test_raw = np.mean([all_data[target]['tests'][si], all_data[target]['tests'][si + N_SEEDS]], axis=0) if si < N_SEEDS else train_raw
                
                if method_type == 'min_max':
                    # Use global min-max from all train preds
                    pass  # handled below
                
                elif method_type == 'temperature':
                    train_cal = calibrate_temperature(train_raw, y, test_raw)
                    test_calibrated_all[si] = train_cal[0]
                
                elif method_type == 'isotonic':
                    iso = IsotonicRegression(y_out=np.clip(y, 0, 1), out_of_bounds='clip')
                    iso.fit(train_raw, y)
                    train_calibrated_all[si] = iso.predict(train_raw)
                    test_calibrated_all[si] = iso.predict(test_raw)
            
            if method_type == 'min_max':
                # Global min-max from average of all seeds
                train_avg_all = np.mean(train_preds, axis=0)
                test_avg_all = np.mean([np.mean([all_data[target]['tests'][si], all_data[target]['tests'][si + N_SEEDS]], axis=0) 
                                       for si in range(N_SEEDS)], axis=0)
                
                # Learn min-max on OOF
                oof_avg_all = np.mean(oofs_2d, axis=1)
                oof_std_all = np.std(oofs_2d, axis=1)
                X_train = np.column_stack([oof_avg_all, oof_std_all])
                
                meta = Ridge(alpha=V540_TOP[target][0] if target == 'S3' else 0.01)  # placeholder
                # Actually use V542 optimal alpha
                from collections import OrderedDict
                v542_alphas = {'Q1': 0.0001, 'Q2': 10.0, 'Q3': 0.0001, 'S1': 0.0001, 'S2': 0.03, 'S3': 10.0, 'S4': 0.0001}
                alpha = v542_alphas[target]
                meta = Ridge(alpha=alpha)
                meta.fit(X_train, y)
                train_pred = meta.predict(X_train)
                test_pred = meta.predict(np.column_stack([test_avg_all, np.std([np.mean([all_data[target]['tests'][si], all_data[target]['tests'][si + N_SEEDS]], axis=0) for si in range(N_SEEDS)], axis=0)])
                
                pmin, pmax = train_pred.min(), train_pred.max()
                if pmax - pmin < 1e-10:
                    test_proba = np.ones(len(test_pred)) * 0.5
                else:
                    test_proba = (test_pred - pmin) / (pmax - pmin)
                test_proba = np.clip(test_proba, 0.001, 0.999)
                
                # For gap: use train_proba
                if pmax - pmin < 1e-10:
                    train_proba = np.ones(len(train_pred)) * 0.5
                else:
                    train_proba = (train_pred - pmin) / (pmax - pmin)
                train_proba = np.clip(train_proba, 0.001, 0.999)
                meta_ll = log_loss(y, train_proba)
                target_gaps[target] = avg_student - meta_ll
                
            elif method_type == 'temperature':
                # For gap computation, calibrate train OOF preds
                oof_avg_all = np.mean(oofs_2d, axis=1)
                oof_std_all = np.std(oofs_2d, axis=1)
                X_train = np.column_stack([oof_avg_all, oof_std_all])
                
                v542_alphas = {'Q1': 0.0001, 'Q2': 10.0, 'Q3': 0.0001, 'S1': 0.0001, 'S2': 0.03, 'S3': 10.0, 'S4': 0.0001}
                alpha = v542_alphas[target]
                
                meta = Ridge(alpha=alpha)
                meta.fit(X_train, y)
                train_pred = meta.predict(X_train)
                test_pred = meta.predict(np.column_stack([test_avg_all, np.std([np.mean([all_data[target]['tests'][si], all_data[target]['tests'][si + N_SEEDS]], axis=0) for si in range(N_SEEDS)], axis=0)])
                
                # Temperature scale the meta predictions (treat as probs)
                t_cal, T_val, t_ll = calibrate_temperature(
                    np.clip(train_pred, 0.001, 0.999), y,
                    np.clip(test_pred, 0.001, 0.999)
                )
                meta_ll = t_ll
                target_gaps[target] = avg_student - meta_ll
                
            elif method_type == 'isotonic':
                oof_avg_all = np.mean(oofs_2d, axis=1)
                oof_std_all = np.std(oofs_2d, axis=1)
                X_train = np.column_stack([oof_avg_all, oof_std_all])
                
                v542_alphas = {'Q1': 0.0001, 'Q2': 10.0, 'Q3': 0.0001, 'S1': 0.0001, 'S2': 0.03, 'S3': 10.0, 'S4': 0.0001}
                alpha = v542_alphas[target]
                
                meta = Ridge(alpha=alpha)
                meta.fit(X_train, y)
                train_pred = meta.predict(X_train)
                test_pred = meta.predict(np.column_stack([test_avg_all, np.std([np.mean([all_data[target]['tests'][si], all_data[target]['tests'][si + N_SEEDS]], axis=0) for si in range(N_SEEDS)], axis=0)])
                
                # Isotonic regression on meta predictions
                iso = IsotonicRegression(y_out=np.clip(y, 0, 1), out_of_bounds='clip')
                iso.fit(np.clip(train_pred, 0.001, 0.999), y)
                train_proba = np.clip(iso.predict(train_pred), 0.001, 0.999)
                test_proba = np.clip(iso.predict(test_pred), 0.001, 0.999)
                meta_ll = log_loss(y, train_proba)
                target_gaps[target] = avg_student - meta_ll
        
        avg_gap = sum(target_gaps.values()) / 7
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < V308_GAPS[t])
        improvement = (-0.03016) - avg_gap
        
        all_results[method_name] = {
            'avg_gap': avg_gap, 'vs308': vs308, 'improvement': improvement,
            'target_gaps': target_gaps
        }
        
        log.info(f'\n  {method_name}: avg_gap={avg_gap:+.5f}, improvement=+{improvement:+.5f}, vs308={vs308}/7')
        for t in TARGETS:
            vs = "✅" if target_gaps[t] < V308_GAPS[t] else "❌"
            log.info(f'    {t}: {target_gaps[t]:+.5f} {vs}')
    
    # ================================================================
    # Final comparison
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('V544 Final Summary')
    log.info('=' * 70)
    
    log.info(f'  {"Method":30s} {"avg_gap":>10} {"vs308":>8} {"Δ V537":>10}')
    log.info('  ' + '-' * 62)
    for name, v in sorted(all_results.items(), key=lambda x: x[1]['avg_gap']):
        log.info(f'  {name:30s} {v["avg_gap"]:>+10.5f} {v["vs308"]:>8d} {v["improvement"]:>+10.5f}')
    
    best_name = min(all_results, key=lambda x: all_results[x]['avg_gap'])
    best = all_results[best_name]
    log.info(f'\n🏆 Best: {best_name} (avg_gap={best["avg_gap"]:+.5f})')
    
    # Save
    result = {
        'version': 'V544',
        'hypothesis': 'calibration_methods_comparison',
        'results': {k: {'avg_gap': v['avg_gap'], 'vs308': v['vs308'], 'improvement': v['improvement']} 
                    for k, v in all_results.items()},
        'best': best_name,
        'avg_gap': best['avg_gap'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v544_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')

if __name__ == '__main__':
    main()
