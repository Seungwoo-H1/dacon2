"""
V300 — Realistic Improvement Loop from V127 baseline (OOF ~0.66)
Goal: Push toward OOF 0.60 by:
1. Isotonic calibration (proven Δ=-0.07)
2. Per-target feature selection (importance-based)
3. Aggressive ensemble (100 seeds × 4 configs)
4. Calibration fusion
"""
import os, re, json, warnings, time
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

def train_and_predict(feat, y, group, feat_cols, cfg, n_seeds, top_k=None, calibrate=False):
    """Train with n_seeds, return (oof_preds, oof_ll, calibrated_oof, cal_ll)."""
    if top_k:
        X_all = feat[top_k].fillna(0).values.astype(np.float64)
    else:
        X_all = feat[feat_cols].fillna(0).values.astype(np.float64)
    
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
    
    ll = log_loss(y, np.clip(oof, 0.001, 0.999), labels=[0,1])
    
    if calibrate:
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        oof_cal = iso.fit_transform(np.clip(oof, 0.001, 0.999), y)
        ll_cal = log_loss(y, oof_cal, labels=[0,1])
        return oof, ll, oof_cal, ll_cal
    return oof, ll, None, None

def feature_selection_importance(feat, y, feat_cols, n_top=100):
    """Select top-K features by LGBM importance."""
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

def main():
    t0 = time.time()
    print("=" * 70)
    print("V300 — Improvement Loop from V127 Baseline")
    print("=" * 70)
    
    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f'Loaded: {feat.shape}')
    
    feat_cols = get_feat_cols(feat)
    print(f'Feature columns: {len(feat_cols)}')
    
    exp_log = {'v300': {}, 'iterations': []}
    
    # ── Experiment A: Baseline (all features, 50 seeds, no cal) ──
    print("\n=== EXP A: Baseline (all features, 50 seeds) ===")
    oof_A = {}
    lls_A = {}
    for t in TARGETS:
        sw = V53_SWEEP[t]
        y = feat[t].values
        oof, ll, _, _ = train_and_predict(feat, y, feat['subject_id'], feat_cols, CFGS[sw['cfg']], 50)
        oof_A[t] = oof
        lls_A[t] = ll
        print(f"  {t}: OOF={ll:.5f}")
    avg_A = np.mean(list(lls_A.values()))
    exp_log['exp_A_baseline'] = {'avg_oof': round(avg_A, 5)}
    print(f"  AVG OOF: {avg_A:.5f}")
    
    # ── Experiment B: + Isotonic Calibration ──
    print("\n=== EXP B: + Isotonic Calibration ===")
    oof_B = {}
    lls_B = {}
    for t in TARGETS:
        y = feat[t].values
        _, _, oof_cal, ll_cal = train_and_predict(feat, y, feat['subject_id'], feat_cols, CFGS[V53_SWEEP[t]['cfg']], 50, calibrate=True)
        oof_B[t] = oof_cal
        lls_B[t] = ll_cal
        delta = ll_cal - lls_A[t]
        print(f"  {t}: {lls_A[t]:.5f} → {ll_cal:.5f} (Δ={delta:+.5f})")
    avg_B = np.mean(list(lls_B.values()))
    exp_log['exp_B_calibration'] = {'avg_oof': round(avg_B, 5)}
    print(f"  AVG OOF: {avg_B:.5f} (Δ={avg_B-avg_A:+.5f})")
    
    # ── Experiment C: Top-K Feature Selection per target ──
    print("\n=== EXP C: Top-K Feature Selection + 50 seeds ===")
    oof_C = {}
    lls_C = {}
    top_k_sets = {}
    for t in TARGETS:
        y = feat[t].values
        sw = V53_SWEEP[t]
        # Try 50, 80, 100, all features — pick best
        best_ll = 999
        best_k = None
        best_oof = None
        for k in [20, 30, 50, 80, 100, None]:
            if k is None:
                top_k = feat_cols
            else:
                top_k = feature_selection_importance(feat, y, feat_cols, k)
            oof, ll, _, _ = train_and_predict(feat, y, feat['subject_id'], top_k, CFGS[sw['cfg']], 10)
            if ll < best_ll:
                best_ll = ll
                best_k = k
                best_oof = oof
        
        # Now run 50 seeds with best k
        if best_k is None:
            top_k_final = feat_cols
        else:
            top_k_final = feature_selection_importance(feat, y, feat_cols, best_k)
        top_k_sets[t] = top_k_final
        
        oof, ll, _, _ = train_and_predict(feat, y, feat['subject_id'], top_k_final, CFGS[sw['cfg']], 50)
        oof_C[t] = oof
        lls_C[t] = ll
        print(f"  {t}: best_k={best_k}, OOF={ll:.5f}")
    
    avg_C = np.mean(list(lls_C.values()))
    exp_log['exp_C_feature_select'] = {'avg_oof': round(avg_C, 5)}
    print(f"  AVG OOF: {avg_C:.5f} (vs B: {avg_C-avg_B:+.5f})")
    
    # ── Experiment D: Top-K + Isotonic Calibration ──
    print("\n=== EXP D: Top-K + Isotonic Calibration ===")
    oof_D = {}
    lls_D = {}
    for t in TARGETS:
        y = feat[t].values
        _, _, oof_cal, ll_cal = train_and_predict(feat, y, feat['subject_id'], top_k_sets[t], CFGS[V53_SWEEP[t]['cfg']], 50, calibrate=True)
        oof_D[t] = oof_cal
        lls_D[t] = ll_cal
        delta = ll_cal - lls_C[t]
        print(f"  {t}: {lls_C[t]:.5f} → {ll_cal:.5f} (Δ={delta:+.5f})")
    avg_D = np.mean(list(lls_D.values()))
    exp_log['exp_D_feature_select_cal'] = {'avg_oof': round(avg_D, 5)}
    print(f"  AVG OOF: {avg_D:.5f} (vs C: {avg_D-avg_C:+.5f})")
    
    # ── Experiment E: Aggressive — 100 seeds, multiple configs per target ──
    print("\n=== EXP E: 100 seeds + multi-config ensemble ===")
    oof_E = {t: np.zeros(len(feat)) for t in TARGETS}
    lls_E = {}
    
    # Use best config per target: try all 4 configs with 25 seeds each, average
    for t in TARGETS:
        y = feat[t].values
        sw = V53_SWEEP[t]
        # Ensemble: use all 4 configs × 25 seeds
        for cfg_name in CFGS:
            cfg = CFGS[cfg_name]
            oof_tmp, ll_tmp, _, _ = train_and_predict(feat, y, feat['subject_id'], 
                                                       top_k_sets[t], cfg, 25, calibrate=False)
            oof_E[t] += oof_tmp / 4  # average 4 configs
        
        ll = log_loss(y, np.clip(oof_E[t], 0.001, 0.999), labels=[0,1])
        lls_E[t] = ll
        print(f"  {t}: OOF={ll:.5f}")
    
    avg_E = np.mean(list(lls_E.values()))
    exp_log['exp_E_multi_config'] = {'avg_oof': round(avg_E, 5)}
    print(f"  AVG OOF: {avg_E:.5f} (vs D: {avg_E-avg_D:+.5f})")
    
    # ── Experiment F: Multi-config + Calibration ──
    print("\n=== EXP F: Multi-config + Isotonic Calibration ===")
    oof_F = {}
    lls_F = {}
    for t in TARGETS:
        y = feat[t].values
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        oof_cal = iso.fit_transform(np.clip(oof_E[t], 0.001, 0.999), y)
        oof_F[t] = oof_cal
        ll = log_loss(y, oof_cal, labels=[0,1])
        lls_F[t] = ll
        print(f"  {t}: calibrated OOF={ll:.5f}")
    
    avg_F = np.mean(list(lls_F.values()))
    exp_log['exp_F_final'] = {'avg_oof': round(avg_F, 5)}
    print(f"  AVG OOF: {avg_F:.5f}")
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = {
        'A_baseline': avg_A,
        'B_calibration': avg_B,
        'C_feature_select': avg_C,
        'D_feature_select_cal': avg_D,
        'E_multi_config': avg_E,
        'F_multi_config_cal': avg_F,
    }
    for k, v in summary.items():
        print(f"  {k}: {v:.5f}")
    print(f"\n  Best: {min(summary, key=summary.get)} = {min(summary.values()):.5f}")
    print(f"  Target V127: 0.53731 (likely leaked)")
    print(f"  Realistic target: <0.60")
    
    exp_log['summary'] = summary
    exp_log['top_k_sets'] = {t: str(top_k_sets[t][:10]) + f'...({len(top_k_sets[t])} total)' for t in TARGETS}
    exp_log['time'] = round(time.time() - t0, 1)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(EXPERIMENTS / f'v300_{ts}.json', 'w') as f:
        json.dump(exp_log, f, indent=2, default=str)
    print(f"\n  Log: experiments/v300_{ts}.json")
    print(f"  Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
