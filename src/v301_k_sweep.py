"""
V301 — Fine-tune K for feature selection, try different K per target
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
EXPERIMENTS.mkdir(exist_ok=True)

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

def train_cal(feat, y, group, feat_cols, cfg, n_seeds, top_k):
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
    # Calibrate
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    oof_cal = iso.fit_transform(np.clip(oof, 0.001, 0.999), y)
    ll = log_loss(y, oof_cal, labels=[0,1])
    return oof, ll

def main():
    t0 = time.time()
    print("=" * 70)
    print("V301 — K Sweep per Target + Cal")
    print("=" * 70)
    
    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    feat_cols = get_feat_cols(feat)
    group = feat['subject_id']
    
    results = []
    
    for t in TARGETS:
        y = feat[t].values
        sw = V53_SWEEP[t]
        cfg = CFGS[sw['cfg']]
        
        print(f"\n--- {t} ---")
        best_ll = 999
        best_k = None
        best_top_k = None
        
        # Sweep K from 5 to 80
        for k in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 80, None]:
            if k is None:
                top_k = feat_cols
            else:
                top_k = importance_select(feat, y, feat_cols, k)
            _, ll = train_cal(feat, y, group, top_k, cfg, 30, top_k)
            label = k if k else 'all'
            print(f"  K={label:>4}: OOF={ll:.5f}", end="")
            if ll < best_ll:
                best_ll = ll
                best_k = k
                best_top_k = top_k
                print(" ← BEST")
            else:
                print()
        
        # Re-run best k with 50 seeds
        if best_k is None:
            top_k_final = feat_cols
        else:
            top_k_final = importance_select(feat, y, feat_cols, best_k)
        
        _, ll_final = train_cal(feat, y, group, top_k_final, cfg, 50, top_k_final)
        results.append({
            'target': t,
            'best_k': best_k,
            'best_oof': round(ll_final, 5),
            'features': top_k_final
        })
        print(f"  >>> FINAL: K={best_k}, OOF={ll_final:.5f} (50 seeds)")
    
    avg = np.mean([r['best_oof'] for r in results])
    print(f"\nAVG OOF: {avg:.5f}")
    
    # Compare with V300 D
    print(f"V300 D: 0.60137")
    print(f"Improvement: {avg - 0.60137:+.5f}")
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(EXPERIMENTS / f'v301_{ts}.json', 'w') as f:
        json.dump({'results': results, 'avg': round(avg, 5)}, f, indent=2, default=str)
    print(f"Saved: experiments/v301_{ts}.json")
    print(f"Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
