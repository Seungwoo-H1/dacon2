# LGBM V10 OOF + FT-Transformer V2 Ensemble
# Combine LGBM predictions (V10-like) with FT-Transformer V2 predictions
# Goal: Beat LGBM V10 Cal OOF 0.6038

import os, sys, json, warnings, numpy as np, time, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path
import importlib.util

BASE_DIR = Path("/home/mwoo423/.openclaw/workspace/dl_project")
src_dir = BASE_DIR / "src"
sys.path.insert(0, str(src_dir))

spec = importlib.util.spec_from_file_location("prepare", src_dir / "00_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

load_data = prepare.load_data
extract_meta = prepare.extract_meta
prepare_for_dl = prepare.prepare_for_dl

# ============================================================
# 1. Load FT-Transformer V2 OOF predictions
# ============================================================
print("Loading FT-Transformer V2 OOF predictions...")
ft_v2_preds = {}
for t in ['Q1','Q2','Q3','S1','S2','S3','S4']:
    ft_v2_preds[t] = np.load(f"{BASE_DIR}/results/ft_v2_gpu/{t}_oof.npy")
print(f"  FT-Transformer V2: 7 targets loaded, shape={ft_v2_preds['Q1'].shape}")

# ============================================================
# 2. Train LGBM V10-like models and get OOF predictions
# ============================================================
print("\nTraining LGBM V10-like models...")

df = load_data()
meta_info, df = extract_meta(df)
prepared = prepare_for_dl(df, meta_info)
X = prepared["X"]
groups = prepared["X_subjects"]
targets = meta_info["target_cols"]
gkf = GroupKFold(n_splits=5)

lgbm_oof = {}
lgbm_models = {}

for t in targets:
    y = prepared["y"][t]
    oof = np.zeros(len(X))
    models = []
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
            'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
            'random_state': 42, 'n_jobs': -1,
        }
        
        # Scale_pos_weight for imbalance
        n_pos = max((y_tr==1).sum(), 1)
        n_neg = y_tr.shape[0] - n_pos
        spw = n_neg / n_pos
        params['scale_pos_weight'] = spw
        
        tr_ds = lgb.Dataset(X_tr, y_tr, feature_name=[f'f{i}' for i in range(X.shape[1])], params={'verbose': '-1'})
        va_ds = lgb.Dataset(X_va, y_va, reference=tr_ds, params={'verbose': '-1'})
        
        m = lgb.train(params, tr_ds, valid_sets=[va_ds], num_boost_round=500,
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        
        oof[va_idx] = m.predict(X_va)
        models.append(m)
    
    # Mean matching calibration
    train_preds = np.mean([m.predict(X) for m in models])
    oof_mean = oof.mean()
    calibrated = np.clip(oof + (train_preds - oof_mean), 1e-4, 1-1e-4)
    
    loss = log_loss(y, calibrated)
    auc = roc_auc_score(y, calibrated) if len(np.unique(y)) > 1 else 0.5
    lgbm_oof[t] = calibrated
    lgbm_models[t] = models
    
    print(f"  {t}: LGBM AUC={auc:.4f}, Loss={loss:.6f}")

# ============================================================
# 3. Ensemble: optimize weights via grid search
# ============================================================
print(f"\n{'='*60}")
print("Ensemble Optimization (Grid Search on blend weights)")
print(f"{'='*60}")

best_blend = {}
best_avg_loss = 999

# Try various weight combinations
for w_ft in np.arange(0.0, 0.6, 0.05):
    w_lgb = 1.0 - w_ft
    blend_loss = 0
    
    for t in targets:
        blended = w_lgb * lgbm_oof[t] + w_ft * ft_v2_preds[t]
        blended = np.clip(blended, 1e-4, 1-1e-4)
        blend_loss += log_loss(prepared["y"][t], blended)
    
    avg_loss = blend_loss / len(targets)
    
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss
        best_blend = {'w_ft': w_ft, 'w_lgb': w_lgb}

print(f"\nBest weights: FT={best_blend['w_ft']:.2f}, LGBM={best_blend['w_lgb']:.2f}")
print(f"Best avg log_loss: {best_avg_loss:.6f}")
print(f"LGBM V10 baseline: 0.6038")

# ============================================================
# 4. Evaluate best ensemble
# ============================================================
print(f"\n{'='*60}")
print("FINAL ENSEMBLE RESULTS")
print(f"{'='*60}")
print(f"{'Target':<10} {'LGBM':>8} {'FT-V2':>8} {'Ensemble':>10} {'Δ vs V10':>10}")
print(f"{'-'*48}")

ensemble_oof = {}
ensemble_aucs = {}

for t in targets:
    blended = best_blend['w_lgb'] * lgbm_oof[t] + best_blend['w_ft'] * ft_v2_preds[t]
    blended = np.clip(blended, 1e-4, 1-1e-4)
    ensemble_oof[t] = blended
    
    ensemble_auc = roc_auc_score(prepared["y"][t], blended)
    ensemble_aucs[t] = ensemble_auc
    
    lgbm_auc = roc_auc_score(prepared["y"][t], lgbm_oof[t])
    ft_auc = roc_auc_score(prepared["y"][t], ft_v2_preds[t])
    delta = ensemble_auc - lgbm_auc
    
    print(f"{t:<10} {lgbm_auc:>8.4f} {ft_auc:>8.4f} {ensemble_auc:>10.4f} {delta:>+10.4f}")

avg_ensemble_auc = np.mean(list(ensemble_aucs.values()))
avg_lgbm_auc = np.mean([roc_auc_score(prepared["y"][t], lgbm_oof[t]) for t in targets])
avg_ft_auc = np.mean([roc_auc_score(prepared["y"][t], ft_v2_preds[t]) for t in targets])

print(f"{'-'*48}")
print(f"{'AVG':<10} {avg_lgbm_auc:>8.4f} {avg_ft_auc:>8.4f} {avg_ensemble_auc:>10.4f} {avg_ensemble_auc - avg_lgbm_auc:>+10.4f}")
print(f"\nLGBM V10 Baseline: 0.6038")
print(f"Improvement: {avg_ensemble_auc - 0.6038:+.4f}")

# ============================================================
# 5. Per-target: pick best of LGBM vs FT-V2 vs Ensemble
# ============================================================
print(f"\n{'='*60}")
print("BEST PER TARGET (LGBM / FT-V2 / ENSEMBLE)")
print(f"{'='*60}")
for t in targets:
    lgbm_a = roc_auc_score(prepared["y"][t], lgbm_oof[t])
    ft_a = roc_auc_score(prepared["y"][t], ft_v2_preds[t])
    ens_a = roc_auc_score(prepared["y"][t], ensemble_oof[t])
    best = max([('LGBM', lgbm_a), ('FT-V2', ft_a), ('Ensemble', ens_a)], key=lambda x: x[1])
    print(f"  {t}: {best[0]:<10}={best[1]:.4f} (LGBM={lgbm_a:.4f}, FT={ft_a:.4f}, Ens={ens_a:.4f})")

# ============================================================
# 6. Save
# ============================================================
save_dir = BASE_DIR / "results" / "ensemble_lgbm_ftv2"
save_dir.mkdir(parents=True, exist_ok=True)

for t in targets:
    np.save(save_dir / f"{t}_ensemble_oof.npy", ensemble_oof[t])
    np.save(save_dir / f"{t}_lgbm_oof.npy", lgbm_oof[t])

with open(save_dir / "config.json", "w") as f:
    json.dump({
        'best_blend_weights': best_blend,
        'avg_ensemble_auc': avg_ensemble_auc,
        'avg_lgbm_auc': avg_lgbm_auc,
        'avg_ft_auc': avg_ft_auc,
        'targets': {t: {'lgbm_auc': float(roc_auc_score(prepared["y"][t], lgbm_oof[t])),
                        'ft_auc': float(roc_auc_score(prepared["y"][t], ft_v2_preds[t])),
                        'ensemble_auc': float(ensemble_aucs[t]),
                        'delta_vs_v10': float(ensemble_aucs[t] - 0.6038)} for t in targets},
    }, f, indent=2)

with open(save_dir / "results.txt", "w") as f:
    f.write(f"Best blend: FT={best_blend['w_ft']:.2f}, LGBM={best_blend['w_lgb']:.2f}\n")
    f.write(f"AVG LGBM AUC: {avg_lgbm_auc:.4f}\n")
    f.write(f"AVG FT-V2 AUC: {avg_ft_auc:.4f}\n")
    f.write(f"AVG Ensemble AUC: {avg_ensemble_auc:.4f}\n")
    f.write(f"Improvement over V10 (0.6038): {avg_ensemble_auc - 0.6038:+.4f}\n")
    for t in targets:
        f.write(f"{t}: AUC={ensemble_aucs[t]:.4f}, Δ={ensemble_aucs[t]-0.6038:+.4f}\n")

print(f"\nSaved to {save_dir}")
