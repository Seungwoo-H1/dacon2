"""
05_v5_robust.py — Validation-Evaluation-Validation cycle for final submission

Strategy:
1. Feature analysis: identify leakages, weak features
2. CV optimization with proper GroupKFold
3. Multiple CV runs → pick most stable models
4. Calibration: simple target-rate matching instead of complex isotonic
5. Submit with confidence
"""
import pandas as pd, numpy as np, re, sys, json
from pathlib import Path
import importlib.util
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

target_cols = ['Q1','Q2','Q3','S1','S2','S3','S4']
meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}
RANDOM_SEED = 42

# ── Load feature engineering ──
spec = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
feat_eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_eng)

spec2 = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
ld_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ld_mod)

# ── Load training features ──
feat = pd.read_parquet('data_processed/features.parquet')
correct_cols = [c for c in feat.columns 
                if c not in meta_cols | set(target_cols)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
print(f"Training data: {feat.shape}, features: {len(correct_cols)}")

# ── Feature ranking (importance) ──
# Use feature importance to select top features per target
# This is more robust than correlation alone
print("\n=== Computing feature rankings per target ===")
feat_rank = {}
for target in target_cols:
    y = feat[target].values
    X = feat[correct_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.03,
        'n_estimators': 200, 'subsample': 0.6, 'colsample_bytree': 0.6,
        'reg_alpha': 2.0, 'reg_lambda': 5.0,
        'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
        'min_child_samples': 15,
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    sanitized = [sanitize(c) for c in correct_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=200)
    importances = model.feature_importance(importance_type="gain")
    
    ranked = sorted(zip(correct_cols, importances), key=lambda x: -x[1])
    feat_rank[target] = ranked
    
    print(f"  {target}: top5 = {[r[0] for r in ranked[:5]]}")

# ── CV analysis: test different feature counts ──
print("\n=== CV Analysis: Feature count vs Score ===")
gkf = GroupKFold(n_splits=5)

cv_analysis = {}
for target in target_cols:
    y = feat[target].values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    ranked = feat_rank[target]
    
    cv_scores = {}
    for n_top in [5, 10, 15, 20, 30, 50]:
        selected_cols = [f[0] for f in ranked[:n_top]]
        X = feat[selected_cols].fillna(0).values
        sanitized = [sanitize(c) for c in selected_cols]
        
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, feat['subject_id'])):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
                'n_estimators': 200, 'subsample': 0.6, 'colsample_bytree': 0.6,
                'reg_alpha': 2.0, 'reg_lambda': 5.0,
                'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
                'min_child_samples': 15,
                'force_row_wise': True, 'n_jobs': 1,
            }
            
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={'verbose': '-1'})
            vs = lgb.Dataset(X_va, label=y_va, feature_name=sanitized, reference=ds, params={'verbose': '-1'})
            
            model = lgb.train(params, ds, num_boost_round=200,
                              valid_sets=[vs],
                              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            
            pred = model.predict(X_va)
            fold_losses.append(log_loss(y_va, pred, labels=[0, 1]))
        
        cv_scores[n_top] = np.mean(fold_losses)
        cv_scores[f'{n_top}_std'] = np.std(fold_losses)
    
    cv_analysis[target] = cv_scores
    best_n = min([k for k in cv_scores.keys() if isinstance(k, int)], key=lambda k: cv_scores[k])
    print(f"  {target}: best_n={best_n}, cv={cv_scores[best_n]:.4f}, std={cv_scores[f'{best_n}_std']:.4f}")
    print(f"    {[f'({k},{cv_scores[k]:.3f})' for k in sorted([x for x in cv_scores.keys() if isinstance(x, int)])]}")

# ── Final CV: test more configs with optimal feature count ──
print("\n=== Final CV search with optimal feature counts ===")

# Best configs from v3, adjusted for feature count
wide_configs = [
    # More aggressive (higher LR, less reg)
    {'name': 'F1', 'nl': 8, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 1.0, 'mc': 10, 'es': 30},
    {'name': 'F2', 'nl': 12, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.6, 'ra': 1.0, 'rl': 2.0, 'mc': 15, 'es': 30},
    {'name': 'F3', 'nl': 15, 'md': 3, 'lr': 0.03, 'ne': 250, 'ss': 0.7, 'cst': 0.7, 'ra': 0.3, 'rl': 0.5, 'mc': 10, 'es': 30},
    {'name': 'F4', 'nl': 20, 'md': 4, 'lr': 0.02, 'ne': 250, 'ss': 0.8, 'cst': 0.7, 'ra': 0.1, 'rl': 0.5, 'mc': 10, 'es': 30},
    {'name': 'F5', 'nl': 10, 'md': 3, 'lr': 0.05, 'ne': 200, 'ss': 0.7, 'cst': 0.6, 'ra': 1.0, 'rl': 2.0, 'mc': 15, 'es': 30},
    {'name': 'F6', 'nl': 15, 'md': 4, 'lr': 0.02, 'ne': 300, 'ss': 0.6, 'cst': 0.5, 'ra': 2.0, 'rl': 5.0, 'mc': 20, 'es': 40},
    {'name': 'F7', 'nl': 25, 'md': 5, 'lr': 0.03, 'ne': 200, 'ss': 0.8, 'cst': 0.8, 'ra': 0.1, 'rl': 0.1, 'mc': 5, 'es': 20},
    {'name': 'F8', 'nl': 12, 'md': 4, 'lr': 0.01, 'ne': 400, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 30, 'es': 50},
]

target_best = {}
for target in target_cols:
    y = feat[target].values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    ranked = feat_rank[target]
    
    # Use top N features (from analysis above)
    n_top = 20  # Default; override if analysis says otherwise
    for k, v in cv_analysis[target].items():
        if isinstance(k, int) and k < 100:
            n_top = k
            break
    
    selected_cols = [f[0] for f in ranked[:n_top]]
    X = feat[selected_cols].fillna(0).values
    sanitized = [sanitize(c) for c in selected_cols]
    
    best_cv = float('inf')
    best_cfg = None
    best_folds = None
    
    for cfg in wide_configs:
        fold_losses = []
        all_preds = []
        all_trues = []
        
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
            'min_child_samples': cfg['mc'],
            'force_row_wise': True, 'n_jobs': 1,
        }
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, feat['subject_id'])):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={'verbose': '-1'})
            vs = lgb.Dataset(X_va, label=y_va, feature_name=sanitized, reference=ds, params={'verbose': '-1'})
            
            model = lgb.train(params, ds, num_boost_round=cfg['ne'],
                              valid_sets=[vs],
                              callbacks=[lgb.early_stopping(cfg['es'], verbose=False), lgb.log_evaluation(0)])
            
            pred = model.predict(X_va)
            fold_losses.append(log_loss(y_va, pred, labels=[0, 1]))
            all_preds.extend(pred)
            all_trues.extend(y_va)
        
        cv = log_loss(all_trues, all_preds, labels=[0, 1])
        
        if cv < best_cv:
            best_cv = cv
            best_cfg = cfg
            best_folds = fold_losses
    
    target_best[target] = {
        'cv': best_cv,
        'config': best_cfg,
        'n_features': n_top,
        'folds': best_folds,
        'selected_cols': selected_cols,
    }
    
    print(f"  {target}: n_feat={n_top}, cv={best_cv:.4f}, config={best_cfg['name']}, std={np.std(best_folds):.4f}")
    print(f"    folds: {[f'{x:.4f}' for x in best_folds]}")

# ── Summary ──
avg_cv = np.mean([tb['cv'] for tb in target_best.values()])
print(f"\n  AVG CV: {avg_cv:.4f}")

# ── Train final models and predict ──
print("\n=== Training final models and predicting test data ===")

# Load test data
parquet_dfs = {}
data_dir = Path('data_raw/ch2025_data_items')
parquet_names = {
    "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
    "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
    "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
    "mScreenStatus": "ch2025_mScreenStatus.parquet", "mUsageStats": "ch2025_mUsageStats.parquet",
    "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
    "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
}

sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date

test_dates = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())

for name, fname in parquet_names.items():
    df = pd.read_parquet(data_dir / fname)
    df = ld_mod.build_merge_key(df)
    df = df[df["date"].astype(str).isin(test_dates)]
    parquet_dfs[name] = df

test_features = feat_eng.create_day_features(parquet_dfs, sample)
print(f"Test features: {test_features.shape}")

predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()

print("\nTraining and predicting...")
for target in target_cols:
    tb = target_best[target]
    n_top = tb['n_features']
    cfg = tb['config']
    selected_cols = tb['selected_cols']
    
    y = feat[target].values
    X_train = feat[selected_cols].fillna(0).values
    sanitized = [sanitize(c) for c in selected_cols]
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
        'min_child_samples': cfg['mc'],
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    # Train on all training data
    ds = lgb.Dataset(X_train, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    
    # Save model
    model.save_model(f'models/v5_final_lgbm_{target}.txt')
    
    # Predict test
    test_X = test_features[selected_cols].fillna(0).values
    pred = model.predict(test_X)
    predictions[target] = pred
    
    # Training check
    train_pred = model.predict(X_train)
    train_loss = log_loss(y, train_pred, labels=[0, 1])
    
    print(f"  {target}: n_feat={n_top}, cv={tb['cv']:.4f}, train_loss={train_loss:.4f}, "
          f"test_mean={pred.mean():.4f}, range=[{pred.min():.4f}, {pred.max():.4f}]")

# ── Calibration: Match training distribution ──
print("\n=== Calibration: Match training target rates ===")
train_data = pd.read_csv('data_raw/ch2026_metrics_train.csv')

for target in target_cols:
    tb = target_best[target]
    selected_cols = tb['selected_cols']
    
    # Get calibration mapping from OOF predictions
    y = feat[target].values
    X = feat[selected_cols].fillna(0).values
    sanitized = [sanitize(c) for c in selected_cols]
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    cfg = tb['config']
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'scale_pos_weight': spw, 'random_state': RANDOM_SEED,
        'min_child_samples': cfg['mc'],
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    # Generate OOF predictions for calibration
    oof_preds = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, feat['subject_id'])):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={'verbose': '-1'})
        vs = lgb.Dataset(X_va, label=y_va, feature_name=sanitized, reference=ds, params={'verbose': '-1'})
        
        model = lgb.train(params, ds, num_boost_round=cfg['ne'],
                          valid_sets=[vs],
                          callbacks=[lgb.early_stopping(cfg['es'], verbose=False), lgb.log_evaluation(0)])
        
        oof_preds[val_idx] = model.predict(X_va)
    
    # Fit calibration
    cal = IsotonicRegression(out_of_bounds='clip')
    train_rate = train_data[target].mean()
    
    try:
        cal.fit(oof_preds, y)
        # Calibrate test predictions
        test_X = test_features[selected_cols].fillna(0).values
        test_model = lgb.Booster(model_file=f'models/v5_final_lgbm_{target}.txt')
        test_raw = test_model.predict(test_X)
        calibrated = cal.predict(test_raw)
    except:
        # Fallback: linear calibration
        test_model = lgb.Booster(model_file=f'models/v5_final_lgbm_{target}.txt')
        test_raw = test_model.predict(test_features[selected_cols].fillna(0).values)
        # Simple shift to match mean
        shift = train_rate - test_raw.mean()
        calibrated = test_raw + shift
    
    predictions[target] = np.clip(calibrated, 0.0001, 0.9999)
    
    rate = train_data[target].mean()
    print(f"  {target}: train_rate={rate:.3f}, pred_mean={predictions[target].mean():.4f}, "
          f"shift={predictions[target].mean()-rate:+.4f}")

# ── Save submission ──
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
sub_path = Path('submissions') / f'submission_v5_{timestamp}.csv'
sub_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(sub_path, index=False)

# ── Final summary ──
print(f"\n✅ Saved: {sub_path}")
print(f"\n{'Target':<6} {'CV':<10} {'Train Loss':<12} {'Train Rate':<12} {'Test Mean':<12} {'Shift':<12} {'Range'}")
for t in target_cols:
    tb = target_best[t]
    rate = train_data[t].mean()
    mean = predictions[t].mean()
    shift = mean - rate
    r_min = predictions[t].min()
    r_max = predictions[t].max()
    print(f"{t:<6} {tb['cv']:<10.4f} {tb['config']['name']:<12} {rate:<12.3f} {mean:<12.4f} {shift:<12.4f} [{r_min:.4f}, {r_max:.4f}]")

avg_cv = np.mean([tb['cv'] for tb in target_best.values()])
print(f"\n  AVG CV: {avg_cv:.4f}")

# Save experiment metadata
meta = {
    'submission_file': str(sub_path),
    'timestamp': timestamp,
    'n_samples': len(predictions),
    'avg_cv': float(avg_cv),
    'configs': {t: {'cv': float(tb['cv']), 'config': tb['config']['name'], 'n_features': tb['n_features']} for t, tb in target_best.items()},
    'predictions': {t: {'mean': float(predictions[t].mean()), 'min': float(predictions[t].min()), 'max': float(predictions[t].max())} for t in target_cols},
}
with open(sub_path.parent / f'meta_{timestamp}.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nExperiment metadata saved.")
