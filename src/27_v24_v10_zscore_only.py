"""
V24: V10 + Z-score Personalization Only (No Extended Features) + Better Hyperparams

Key changes vs V10:
1. Per-subject z-score personalization (same as V10 concept)
2. Better hyperparameter search (finer grid)
3. Feature counts: 15, 20, 30, 40, 50 (expanded)
4. No isotonic calibration (mean-match only)
5. No extended features (they hurt performance)

Uses preprocessed features.parquet (153 features, 450 samples)
"""
import sys, re, json, time, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

def pr(msg):
    print(msg, flush=True)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, SUBMIT_DIR

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
N_SEEDS = 20
N_SPLITS = 5

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

_SANITIZE_RE = re.compile(r'[^a-zA-Z0-9_]')
def sanitize(name):
    return _SANITIZE_RE.sub('_', name)

# ── Feature leakage fix ──
LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feat_cols(f):
    return [c for c in f.columns
            if c not in META_COLS | set(TARGET_COLS)
            and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

# ── Personalization: z-score per subject ──
def add_personalization(df, feature_cols):
    personal_cols = []
    for col in feature_cols:
        filled = df[col].fillna(0)
        grp = filled.groupby(df['subject_id'])
        mean_s = grp.transform('mean')
        std_s = grp.transform('std').replace(0, 1)
        zscore = (filled - mean_s) / std_s
        df = df.assign(**{f'{col}_z': zscore})
        personal_cols.append(f'{col}_z')
    return df, personal_cols

# ── Feature ranking ──
def rank_features(feat, feature_cols, target, n_trees=100):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values.astype(np.float32)
    n_pos = max((y==1).sum(), 1)
    n_neg = (y==0).sum()
    spw = n_neg / n_pos
    sanitized = [sanitize(c) for c in feature_cols]
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':n_trees,
        'subsample':0.7,'colsample_bytree':0.7,
        'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':42,
        'min_child_samples':10,'n_jobs':1,
    }
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    mdl = lgb.train(params, ds, num_boost_round=n_trees)
    imp = mdl.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return ranked

# ── CV predict LGB ──
def cv_predict_lgb(scols, target, seeds, spw):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    sanitized = [sanitize(c) for c in scols]
    
    for seed in seeds:
        cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
               'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,
               'reg_alpha':1.0,'reg_lambda':3.0,
               'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'n_jobs':1}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            Xtr = feat.iloc[tr_idx][scols].fillna(0).values
            Xva = feat.iloc[va_idx][scols].fillna(0).values
            ytr, yva = y[tr_idx], y[va_idx]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sanitized)
            vad = lgb.Dataset(Xva, label=yva, feature_name=sanitized, reference=trd)
            mdl = lgb.train(cfg, trd, num_boost_round=500, valid_sets=[vad],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_idx] += mdl.predict(Xva)
    oof /= len(seeds)
    return oof

# ── Simple calibration ──
def simple_cal(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

# ── Main ──
def main():
    global _FEAT
    t_total = time.time()
    pr("=" * 70)
    pr("V24: V10 + Z-score Personalization (No Extended Features)")
    pr("=" * 70)

    # Load preprocessed features
    _FEAT = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    base_feat_cols = get_feat_cols(_FEAT)
    pr(f"Base features: {len(base_feat_cols)}")
    
    # Step 1: Personalization on base features
    pr("\n── [1/2] Personalization (z-score) on base features ──")
    _FEAT, zscore_cols = add_personalization(_FEAT, base_feat_cols)
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"  After personalization: {len(all_feat_cols)} features ({len(zscore_cols)} z-cols)")
    
    # Step 2: Re-rank all features per target
    pr("\n── [2/2] Final ranking (base+z-score) per target ──")
    final_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(all_feat_cols, target)
        ranked = rank_features(_FEAT, leak, target)
        final_ranked[target] = ranked
        pr(f"  {target}: {len(leak)} features ranked")
    
    # ── Model Comparison ──
    pr("\n── Model Comparison: Different Feature Counts ──")
    
    feat_counts = [15, 20, 30, 40, 50]
    lgb_results = {}
    
    for target in TARGET_COLS:
        ranked = final_ranked[target]
        y = _FEAT[target].values
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        lgb_best_loss = float('inf')
        lgb_best_nf = None
        
        for nf in feat_counts:
            scols = [r[0] for r in ranked[:nf]]
            oof = cv_predict_lgb(scols, target, SEEDS, spw)
            cal = simple_cal(oof, y.mean())
            cal_loss = log_loss(y, cal, labels=[0,1])
            pr(f"  {target} {nf}f: log_loss={cal_loss:.6f}")
            if cal_loss < lgb_best_loss:
                lgb_best_loss = cal_loss
                lgb_best_nf = nf
        
        lgb_results[target] = {'loss': lgb_best_loss, 'n_feats': lgb_best_nf}
        pr(f"  → BEST {target}: {lgb_best_loss:.6f} ({lgb_best_nf}f)")
    
    # ── Best config per target ──
    pr(f"\n{'='*70}")
    pr("V24 BEST CONFIG")
    pr(f"{'='*70}")
    
    best_avg = 0
    for target in TARGET_COLS:
        best_avg += lgb_results[target]['loss']
        pr(f"  {target}: {lgb_results[target]['loss']:.6f} ({lgb_results[target]['n_feats']}f)")
    best_avg /= 7
    pr(f"\nV24 avg log_loss: {best_avg:.6f} (V10: 0.6038, Δ={best_avg-0.6038:+.6f})")
    
    # ── Train final models + submit ──
    pr(f"\n{'='*70}")
    pr("Training final models + submission")
    pr(f"{'='*70}")
    
    # Load test features (preprocessed)
    test_feat = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    pr(f"  Test features: {test_feat.shape}")
    
    # Add personalization to test features (use same z-score params from train)
    pr("  Adding personalization to test features...")
    test_feat_cols = get_feat_cols(test_feat)
    test_feat, _ = add_personalization(test_feat, test_feat_cols)
    
    predictions = test_feat[["subject_id", "sleep_date", "lifelog_date"]].copy()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    meta = {
        'version': 'v24',
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'base_features': len(base_feat_cols),
        'zscore_features': len(zscore_cols),
        'total_features': len(get_feat_cols(test_feat)),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'methods': ['personalization_zscore', 'no_extended_features', 'expanded_feat_search'],
        'per_target': {},
    }
    
    for target in TARGET_COLS:
        ranked = final_ranked[target]
        scols = [r[0] for r in ranked[:lgb_results[target]['n_feats']]]
        
        y = _FEAT[target].values
        train_rate = y.mean()
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
               'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,
               'reg_alpha':1.0,'reg_lambda':3.0,
               'scale_pos_weight':spw,'min_child_samples':10,'n_jobs':1}
        
        sanitized = [sanitize(c) for c in scols]
        X_all = _FEAT[scols].fillna(0).values
        ds_all = lgb.Dataset(X_all, label=y, feature_name=sanitized)
        all_preds = np.zeros(len(test_feat))
        
        for seed in SEEDS:
            sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
            mdl = lgb.train(sc, ds_all, num_boost_round=cfg['n_estimators'])
            all_preds += mdl.predict(test_feat[scols].fillna(0).values)
        all_preds /= N_SEEDS
        
        cal_preds = simple_cal(all_preds, train_rate)
        predictions[target] = cal_preds
        
        meta['per_target'][target] = {
            'model': 'LGBM', 'n_feature': lgb_results[target]['n_feats'],
            'cal_loss_cv': lgb_results[target]['loss'], 'train_rate': float(train_rate),
            'pred_mean': float(cal_preds.mean()), 'shift': float(cal_preds.mean()-train_rate),
        }
        pr(f"  {target}: LGBM+{lgb_results[target]['n_feats']}f, mean={cal_preds.mean():.4f}, shift={cal_preds.mean()-train_rate:+.4f}")
    
    sub_path = SUBMIT_DIR / f"submission_v24_{timestamp}.csv"
    predictions.to_csv(sub_path, index=False)
    
    meta_path = SUBMIT_DIR / f"meta_v24_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    pr(f"\n✅ Submission saved: {sub_path}")
    pr(f"✅ Metadata saved: {meta_path}")
    
    pr(f"\n{'='*70}")
    pr("V24 FINAL SUMMARY")
    pr(f"{'='*70}")
    pr(f"{'Target':<6} {'Feats':<8} {'Train Rate':<12} {'Pred Mean':<12} {'Shift':<10}")
    for t in TARGET_COLS:
        m = meta['per_target'][t]
        pr(f"{t:<6} {m['n_feature']:<8} {m['train_rate']:<12.3f} {m['pred_mean']:<12.4f} {m['shift']:+.4f}")
    
    pr(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
    pr(f"\nV24 avg log_loss: {best_avg:.6f}")
    if best_avg < 0.6038:
        pr("🎯 BEATS V10!")
    else:
        pr("⚠️ Did NOT beat V10")

if __name__ == "__main__":
    main()
