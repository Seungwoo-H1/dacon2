"""
V302 — Final V301 with 100 seeds + Calibration + Submission
Uses optimal K per target from V301
"""
import re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS.mkdir(exist_ok=True)
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']

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

def get_feat_cols(feat):
    META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    return [c for c in feat.columns 
            if c not in META_COLS | set(TARGETS) 
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def importance_select(feat, y, feat_cols, n_top):
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 200,
              'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
              'min_child_samples': 15, 'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1, 'n_jobs': 1}
    train_set = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_set, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked[:n_top]]

def train_cal_100(feat, y, group, top_k, cfg, n_seeds):
    X_all = feat[top_k].fillna(0).values.astype(np.float64)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(feat))
    for seed in range(n_seeds):
        fold_preds = np.zeros(len(feat))
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y, group)):
            X_tr, X_val = X_all[tr_idx], X_all[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed, 'verbose': -1, 'n_jobs': 1}
            patience = max(10, cfg['min_child_samples'])
            train_set = lgb.Dataset(X_tr, label=y_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            model = lgb.train(params, train_set, num_boost_round=cfg['n_estimators'],
                             valid_sets=[val_set],
                             callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
            pred = model.predict(X_val)
            fold_preds[val_idx] = pred
        oof += fold_preds / n_seeds
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    oof_cal = iso.fit_transform(np.clip(oof, 0.001, 0.999), y)
    ll = log_loss(y, oof_cal, labels=[0,1])
    return oof_cal, ll

def train_on_full_for_test(feat_train, y, top_k, feat_test, cfg, n_seeds):
    X_train = feat_train[top_k].fillna(0).values.astype(np.float64)
    X_test = feat_test[top_k].fillna(0).values.astype(np.float64)
    test_preds = np.zeros(len(feat_test))
    for seed in range(n_seeds):
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed, 'verbose': -1, 'n_jobs': 1}
        patience = max(10, cfg['min_child_samples'])
        train_set = lgb.Dataset(X_train, label=y)
        model = lgb.train(params, train_set, num_boost_round=cfg['n_estimators'], callbacks=[lgb.log_evaluation(0)])
        test_preds += model.predict(X_test) / n_seeds
    return test_preds

def main():
    t0 = time.time()
    print("=" * 70)
    print("V302 — Final V301 (100 seeds + cal + submit)")
    print("=" * 70)
    
    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    feat_test = pd.read_parquet(DATA / 'test_features_clean_v60.parquet')
    feat_cols = get_feat_cols(feat)
    
    # Optimal K from V301
    OPT_K = {'Q1': 5, 'Q2': 15, 'Q3': 18, 'S1': 15, 'S2': 18, 'S3': 12, 'S4': 15}
    
    print("Computing top-K features per target...")
    top_k_sets = {}
    for t in TARGETS:
        y = feat[t].values
        k = OPT_K[t]
        top_k_sets[t] = importance_select(feat, y, feat_cols, k)
        print(f"  {t}: K={k}")
    
    # Train with 100 seeds per target
    print("\nTraining (100 seeds per target)...")
    oof_cal = {}
    lls = {}
    for t in TARGETS:
        y = feat[t].values
        sw = V53_SWEEP[t]
        cfg = CFGS[sw['cfg']]
        oof, ll = train_cal_100(feat, y, feat['subject_id'], top_k_sets[t], cfg, 100)
        oof_cal[t] = oof
        lls[t] = ll
        print(f"  {t}: OOF={ll:.5f}")
    
    avg = np.mean(list(lls.values()))
    print(f"\nAVG OOF: {avg:.5f}")
    print(f"V301: 0.58678")
    print(f"Δ: {avg - 0.58678:+.5f}")
    
    # Generate submission
    print("\nGenerating submission (100 seeds per target)...")
    predictions = {}
    for t in TARGETS:
        y = feat[t].values
        sw = V53_SWEEP[t]
        cfg = CFGS[sw['cfg']]
        pred = train_on_full_for_test(feat, y, top_k_sets[t], feat_test, cfg, 100)
        predictions[t] = np.clip(pred, 0, 1)
        print(f"  {t}: mean={pred.mean():.4f}, std={pred.std():.4f}")
    
    submit = pd.DataFrame({
        'subject_id': feat_test['subject_id'],
        'sleep_date': feat_test['sleep_date'],
        'lifelog_date': feat_test['lifelog_date'],
    })
    for t in TARGETS:
        submit[t] = predictions[t]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_path = SUBMIT / f'submission_v302_{timestamp}.csv'
    submit.to_csv(sub_path, index=False)
    print(f"\nSubmission: {sub_path}")
    print(f"Shape: {submit.shape}")
    
    exp_log = {
        'v302': True,
        'avg_oof': round(avg, 5),
        'per_target_oof': {t: round(lls[t], 5) for t in TARGETS},
        'opt_k': OPT_K,
        'submission': str(sub_path),
        'time': round(time.time() - t0, 1),
    }
    
    ts2 = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(EXPERIMENTS / f'v302_{ts2}.json', 'w') as f:
        json.dump(exp_log, f, indent=2, default=str)
    print(f"Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
