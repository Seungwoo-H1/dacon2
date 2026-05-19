"""
V277 — V127 Baseline Reproduction & Improvement Loop
Goal: Match V127 (LB 0.64763) → Improve to LB ~0.5

Phase 1: Reproduce V127 baseline with 5-fold GroupKFold × multi-seed
Phase 2: Isotonic calibration (Δ=-0.073 from V262)
Phase 3: Top-K feature selection
Phase 4: Cross-model ensemble (LGBM + XGB + CatBoost)
Phase 5: Submission
"""
import os, sys, gc, re, json, warnings, time, copy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'

for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}

def sanitize_name(c):
    return re.sub(r'[^a-zA-Z0-9_]', '_', c)

def get_feat_cols(df):
    return [c for c in df.columns 
            if c not in META_COLS | set(TARGETS)
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

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

SEEDS = [42, 7, 999, 777, 123]

def train_one_model(X_tr, y_tr, X_val, y_val, params, model_type='lgbm'):
    """Train a single model and return prediction on val set."""
    if model_type == 'lgbm':
        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set) if len(X_val) > 0 else None
        model = lgb.train(params, train_set, num_boost_round=params['n_estimators'],
                         valid_sets=[val_set] if val_set else None,
                         callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        return model.predict(X_val)
    elif model_type == 'xgb':
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val) if y_val is not None else None
        model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'],
                         evals=[(dval, 'val')] if dval is not None else None,
                         early_stopping_rounds=20, verbose_eval=False)
        return model.predict(dval)
    elif model_type == 'cat':
        model = cb.CatBoostClassifier(**params, silent=True)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                 early_stopping_rounds=20, use_best_model=True)
        return model.predict_proba(X_val)[:, 1]

def train_on_full(X, y, params, model_type='lgbm'):
    """Train on all data and return full model."""
    if model_type == 'lgbm':
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_set, num_boost_round=params['n_estimators'],
                         callbacks=[lgb.log_evaluation(0)])
        return model
    elif model_type == 'xgb':
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
        return model
    elif model_type == 'cat':
        model = cb.CatBoostClassifier(**params, silent=True)
        model.fit(X, y)
        return model

def oof_for_seed(feat_s, feat_cols_s, y, group, seed, cfg_name, model_type='lgbm', top_k=None):
    """Generate OOF predictions for one seed+config+model."""
    if top_k:
        X_all = feat_s[top_k].fillna(0).values.astype(np.float64)
    else:
        X_all = feat_s[feat_cols_s].fillna(0).values.astype(np.float64)
    
    cfg = CFGS[cfg_name]
    spw = max((y == 0).sum() / max((y == 1).sum(), 1), 0.1)
    
    if model_type == 'lgbm':
        params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed, 'verbose': -1, 'n_jobs': 1}
    elif model_type == 'xgb':
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'booster': 'gbtree', 'learning_rate': cfg['learning_rate'],
            'max_depth': cfg['max_depth'], 'num_leaves': cfg['num_leaves'],
            'subsample': cfg['subsample'], 'colsample_bytree': cfg['colsample_bytree'],
            'reg_alpha': cfg['reg_alpha'], 'reg_lambda': cfg['reg_lambda'],
            'min_child_weight': cfg['min_child_samples'],
            'scale_pos_weight': spw, 'random_state': seed, 'n_estimators': cfg['n_estimators'],
            'tree_method': 'hist'
        }
    else:  # cat
        params = {
            'loss_function': 'Logloss', 'eval_metric': 'Logloss',
            'learning_rate': cfg['learning_rate'], 'max_depth': cfg['max_depth'],
            'max_depth': cfg['max_depth'], 'subsample': cfg['subsample'],
            'colsample_bylevel': cfg['colsample_bytree'],
            'l2_leaf_reg': cfg['reg_lambda'], 'random_seed': seed,
            'one_hot_max_size': 2,
            'iterations': cfg['n_estimators']
        }
    
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(feat_s))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y, group)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        pred = train_one_model(X_tr, y_tr, X_val, y_val, params, model_type)
        oof[val_idx] = pred
    
    return oof


def main():
    t0 = time.time()
    print("=" * 70)
    print("V277 — V127 Baseline Reproduction & Improvement Loop")
    print("=" * 70)
    
    # ── Load data ───────────────────────────────────────────
    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f"Loaded features: {feat.shape}")
    
    feat_cols = get_feat_cols(feat)
    feat_s = feat.copy()
    feat_s.columns = [sanitize_name(c) for c in feat_s.columns]
    feat_cols_s = [sanitize_name(c) for c in feat_cols]
    
    y_train = {t: feat_s[t].values for t in TARGETS}
    group = feat_s['subject_id']
    
    # ── Phase 1: V127 Baseline (LGBM only, multi-seed) ──────
    print("\n" + "=" * 70)
    print("PHASE 1: V127 BASELINE (LGBM × multi-seed)")
    print("=" * 70)
    
    oof_base = {t: np.zeros(len(feat_s)) for t in TARGETS}
    exp_log = {'phase': 'v277', 'n_features': len(feat_cols)}
    
    for t in TARGETS:
        sw = V53_SWEEP[t]
        cfg_name = sw['cfg']
        n_feat_target = sw['n_feat']
        y = y_train[t]
        
        seed_preds = []
        for seed in SEEDS:
            oof = oof_for_seed(feat_s, feat_cols_s, y, group, seed, cfg_name, 'lgbm')
            seed_preds.append(oof)
        
        oof_base[t] = np.mean(seed_preds, axis=0)
        ll = log_loss(y, np.clip(oof_base[t], 0.001, 0.999), labels=[0,1])
        exp_log[f'target_{t}'] = {'cfg': cfg_name, 'oof': round(ll, 5)}
        print(f"  {t}: cfg={cfg_name}, seeds={len(SEEDS)}, OOF={ll:.5f}")
    
    lls = [log_loss(y_train[t], np.clip(oof_base[t], 0.001, 0.999), labels=[0,1]) for t in TARGETS]
    avg_oof = np.mean(lls)
    exp_log['avg_oof'] = round(avg_oof, 5)
    print(f"\n  BASELINE AVG OOF: {avg_oof:.5f}")
    
    # ── Phase 2: Isotonic Calibration ───────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: ISOTONIC CALIBRATION")
    print("=" * 70)
    
    oof_cal = {}
    for t in TARGETS:
        y = y_train[t]
        oof = np.clip(oof_base[t].copy(), 0.001, 0.999)
        
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        oof_cal[t] = iso.fit_transform(oof, y)
        
        ll_before = log_loss(y, np.clip(oof, 0.001, 0.999), labels=[0,1])
        ll_after = log_loss(y, oof_cal[t], labels=[0,1])
        print(f"  {t}: {ll_before:.5f} → {ll_after:.5f} (Δ={ll_after-ll_before:+.5f})")
    
    avg_cal = np.mean([log_loss(y_train[t], np.clip(oof_cal[t], 0.001, 0.999), labels=[0,1]) for t in TARGETS])
    exp_log['avg_oof_calibrated'] = round(avg_cal, 5)
    print(f"\n  CALIBRATED AVG OOF: {avg_cal:.5f}")
    
    # ── Phase 3: Top-K Feature Selection ────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: TOP-K FEATURE SELECTION")
    print("=" * 70)
    
    top_k_configs = {t: 100 for t in TARGETS}  # start with 100 features per target
    
    selected = {}
    for t in TARGETS:
        y = y_train[t]
        X = feat_s[feat_cols_s].fillna(0).values.astype(np.float64)
        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 0.1)
        params = {**CFGS['deep'], 'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1, 'n_jobs': 1}
        
        train_set = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_set, num_boost_round=100)
        imp = model.feature_importance(importance_type='gain')
        
        ranked = sorted(zip(feat_cols_s, imp), key=lambda x: -x[1])
        top_k = top_k_configs[t]
        top_k = [r[0] for r in ranked[:top_k]]
        selected[t] = top_k
        
        print(f"  {t}: {top_k}/{len(feat_cols_s)} features selected")
    
    # ── Phase 4: Cross-model Ensemble ───────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: CROSS-MODEL ENSEMBLE (LGBM + XGB + CatBoost)")
    print("=" * 70)
    
    oof_cm = {t: np.zeros(len(feat_s)) for t in TARGETS}
    
    model_types = ['lgbm', 'xgb', 'cat']
    for t in TARGETS:
        y = y_train[t]
        top_k = selected[t]
        
        model_oofs = {}
        for mtype in model_types:
            seed_preds = []
            for cfg_name in CFGS:
                for seed in SEEDS:
                    oof = oof_for_seed(feat_s, feat_cols_s, y, group, seed, cfg_name, mtype, top_k)
                    seed_preds.append(oof)
            model_oofs[mtype] = np.mean(seed_preds, axis=0)
            ll = log_loss(y, np.clip(model_oofs[mtype], 0.001, 0.999), labels=[0,1])
            print(f"  {t}/{mtype}: OOF={ll:.5f}")
        
        # Simple average across models
        oof_cm[t] = np.mean(list(model_oofs.values()), axis=0)
        ll = log_loss(y, np.clip(oof_cm[t], 0.001, 0.999), labels=[0,1])
        print(f"  {t}: ENSEMBLE OOF={ll:.5f}")
    
    avg_cm = np.mean([log_loss(y_train[t], np.clip(oof_cm[t], 0.001, 0.999), labels=[0,1]) for t in TARGETS])
    exp_log['avg_oof_cross_model'] = round(avg_cm, 5)
    print(f"\n  CROSS-MODEL AVG OOF: {avg_cm:.5f}")
    
    # ── Phase 5: Calibration on Cross-model ─────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: ISOTONIC CALIBRATION (Cross-model)")
    print("=" * 70)
    
    oof_cm_cal = {}
    for t in TARGETS:
        y = y_train[t]
        oof = np.clip(oof_cm[t].copy(), 0.001, 0.999)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        oof_cm_cal[t] = iso.fit_transform(oof, y)
        ll = log_loss(y, oof_cm_cal[t], labels=[0,1])
        print(f"  {t}: calibrated OOF={ll:.5f}")
    
    avg_cm_cal = np.mean([log_loss(y_train[t], oof_cm_cal[t], labels=[0,1]) for t in TARGETS])
    exp_log['avg_oof_cross_model_cal'] = round(avg_cm_cal, 5)
    print(f"\n  CALIBRATED CROSS-MODEL AVG OOF: {avg_cm_cal:.5f}")
    
    # ── Phase 6: Final Submission ───────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: GENERATE SUBMISSION")
    print("=" * 70)
    
    feat_test = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
    feat_test_s = feat_test.copy()
    feat_test_s.columns = [sanitize_name(c) for c in feat_test_s.columns]
    
    predictions = {t: np.zeros(len(feat_test_s)) for t in TARGETS}
    
    for t in TARGETS:
        top_k = selected[t]
        y = y_train[t]
        X = feat_s[top_k].fillna(0).values.astype(np.float64)
        X_test = feat_test_s[top_k].fillna(0).values.astype(np.float64)
        
        test_preds = np.zeros(len(feat_test_s))
        
        for mtype in model_types:
            for cfg_name in CFGS:
                for seed in SEEDS:
                    if mtype == 'lgbm':
                        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 0.1)
                        params = {**CFGS[cfg_name], 'scale_pos_weight': spw, 'random_state': seed, 'verbose': -1, 'n_jobs': 1}
                        model = train_on_full(X, y, params, 'lgbm')
                    elif mtype == 'xgb':
                        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 0.1)
                        params = {
                            'objective': 'binary:logistic', 'eval_metric': 'logloss',
                            'booster': 'gbtree', 'learning_rate': CFGS[cfg_name]['learning_rate'],
                            'max_depth': CFGS[cfg_name]['max_depth'], 'num_leaves': CFGS[cfg_name]['num_leaves'],
                            'subsample': CFGS[cfg_name]['subsample'], 'colsample_bytree': CFGS[cfg_name]['colsample_bytree'],
                            'reg_alpha': CFGS[cfg_name]['reg_alpha'], 'reg_lambda': CFGS[cfg_name]['reg_lambda'],
                            'min_child_weight': CFGS[cfg_name]['min_child_samples'],
                            'scale_pos_weight': spw, 'random_state': seed, 'n_estimators': CFGS[cfg_name]['n_estimators'],
                            'tree_method': 'hist'
                        }
                        model = train_on_full(X, y, params, 'xgb')
                    else:
                        params = {
                            'loss_function': 'Logloss', 'eval_metric': 'Logloss',
                            'learning_rate': CFGS[cfg_name]['learning_rate'], 'max_depth': CFGS[cfg_name]['max_depth'],
                            'max_depth': CFGS[cfg_name]['max_depth'], 'subsample': CFGS[cfg_name]['subsample'],
                            'colsample_bylevel': CFGS[cfg_name]['colsample_bytree'],
                            'l2_leaf_reg': CFGS[cfg_name]['reg_lambda'], 'random_seed': seed,
                            'one_hot_max_size': 2,
                            'iterations': CFGS[cfg_name]['n_estimators']
                        }
                        model = train_on_full(X, y, params, 'cat')
                    
                    # Determine model type and predict accordingly
                    if mtype == 'xgb':
                        import xgboost as xgb
                        pred = model.predict(xgb.DMatrix(X_test))
                    elif mtype == 'cat':
                        pred = model.predict_proba(X_test)[:, 1]
                    else:  # lgbm
                        pred = model.predict(X_test)
                    test_preds += pred
        
        test_preds /= (len(model_types) * len(CFGS) * len(SEEDS))
        predictions[t] = np.clip(test_preds, 0, 1)
        print(f"  {t}: test mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    
    # Create submission
    submit = pd.DataFrame({
        'subject_id': feat_test_s['subject_id'],
        'sleep_date': feat_test_s['sleep_date'],
        'lifelog_date': feat_test_s['lifelog_date'],
    })
    for t in TARGETS:
        submit[t] = predictions[t]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f'submission_v277_{timestamp}.csv'
    submit.to_csv(sub_path, index=False)
    print(f"\n  Submission saved: {sub_path}")
    print(f"  Shape: {submit.shape}")
    
    exp_log['submission'] = str(sub_path)
    exp_log['time_total'] = round(time.time() - t0, 1)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = EXPERIMENTS / f'v277_{ts}.json'
    with open(log_path, 'w') as f:
        json.dump(exp_log, f, indent=2, default=str)
    print(f"  Experiment log: {log_path}")
    
    print(f"\n{'=' * 70}")
    print(f"V277 COMPLETE — Time: {time.time()-t0:.1f}s")
    print(f"{'=' * 70}")

if __name__ == '__main__':
    main()
