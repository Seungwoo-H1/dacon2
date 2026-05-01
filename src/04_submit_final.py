"""
04_submit_final.py — Final submission pipeline
Train per-target models on top-N features for each target, then predict test.
"""
import pandas as pd, numpy as np, re, sys, json
from pathlib import Path
import importlib.util
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# ── Load data pipeline ──
spec = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
feat_eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_eng)

spec2 = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
ld_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ld_mod)

target_cols = ['Q1','Q2','Q3','S1','S2','S3','S4']
meta_cols = {"subject_id", "lifelog_date", "sleep_date", "date"}

# Load training features
feat = pd.read_parquet('data_processed/features.parquet')
correct_cols = [c for c in feat.columns 
                if c not in meta_cols | set(target_cols)
                and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
print(f"Training features: {feat.shape}, {len(correct_cols)} numeric cols")

# ── Get top features per target ──
# From v3 analysis: top 20 features consistently best
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
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': 15,
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    sanitized = [sanitize(c) for c in correct_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=200)
    importances = model.feature_importance(importance_type="gain")
    
    ranked = sorted(zip(correct_cols, importances), key=lambda x: -x[1])
    feat_rank[target] = ranked
    
    print(f"{target} top 5: {[f[0] for f in ranked[:5]]}")

# ── Load test data ──
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

# ── Train models per target with optimal config from v3 ──
# Then predict test directly (no model loading)
# v3 best configs:
# Q1: W8(nl=31,md=5,lr=0.05,ne=200,ss=0.8,cst=0.7,ra=0.5,rl=1.0,mc=10)
# Q2: W9(nl=15,md=3,lr=0.05,ne=200,ss=0.7,cst=0.7,ra=0.1,rl=0.5,mc=10)
# Q3: W10(nl=10,md=3,lr=0.02,ne=300,ss=0.6,cst=0.5,ra=1.0,rl=2.0,mc=15)
# S1: W9(nl=15,md=3,lr=0.05,ne=200,ss=0.7,cst=0.7,ra=0.1,rl=0.5,mc=10)
# S2: W8(nl=31,md=5,lr=0.05,ne=200,ss=0.8,cst=0.7,ra=0.5,rl=1.0,mc=10)
# S3: W4(nl=12,md=3,lr=0.02,ne=300,ss=0.5,cst=0.5,ra=10,rl=20,mc=25)
# S4: W8(nl=31,md=5,lr=0.05,ne=200,ss=0.8,cst=0.7,ra=0.5,rl=1.0,mc=10)

best_configs = {
    'Q1':  {'nl':31,'md':5,'lr':0.05,'ne':200,'ss':0.8,'cst':0.7,'ra':0.5,'rl':1.0,'mc':10},
    'Q2':  {'nl':15,'md':3,'lr':0.05,'ne':200,'ss':0.7,'cst':0.7,'ra':0.1,'rl':0.5,'mc':10},
    'Q3':  {'nl':10,'md':3,'lr':0.02,'ne':300,'ss':0.6,'cst':0.5,'ra':1.0,'rl':2.0,'mc':15},
    'S1':  {'nl':15,'md':3,'lr':0.05,'ne':200,'ss':0.7,'cst':0.7,'ra':0.1,'rl':0.5,'mc':10},
    'S2':  {'nl':31,'md':5,'lr':0.05,'ne':200,'ss':0.8,'cst':0.7,'ra':0.5,'rl':1.0,'mc':10},
    'S3':  {'nl':12,'md':3,'lr':0.02,'ne':300,'ss':0.5,'cst':0.5,'ra':10,'rl':20,'mc':25},
    'S4':  {'nl':31,'md':5,'lr':0.05,'ne':200,'ss':0.8,'cst':0.7,'ra':0.5,'rl':1.0,'mc':10},
}

predictions = test_features[['subject_id', 'sleep_date', 'lifelog_date']].copy()
models = {}

print("\nTraining models on test-relevant features and predicting...")
for target in target_cols:
    # Select top 20 features for this target
    top20 = [f[0] for f in feat_rank[target][:20]]
    sanitized = [sanitize(c) for c in top20]
    
    # Training data
    train_X = feat[top20].fillna(0).values
    y = feat[target].values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    cfg = best_configs[target]
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': cfg['mc'],
        'force_row_wise': True, 'n_jobs': 1,
    }
    
    ds = lgb.Dataset(train_X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    
    # Save model
    model.save_model(f'models/final_lgbm_{target}.txt')
    
    # Predict test
    test_X = test_features[top20].fillna(0).values
    pred = model.predict(test_X)
    predictions[target] = pred
    
    # Training check
    train_pred = model.predict(train_X)
    train_loss = log_loss(y, train_pred, labels=[0, 1])
    
    models[target] = {'model': model, 'top20': top20, 'train_loss': train_loss, 'cfg': cfg}
    
    print(f"  {target}: train_loss={train_loss:.4f}, test_mean={pred.mean():.4f}, range=[{pred.min():.4f}, {pred.max():.4f}]")

# ── Calibration ──
print("\nCalibrating predictions...")
train_data = pd.read_csv('data_raw/ch2026_metrics_train.csv')

for target in target_cols:
    top20 = models[target]['top20']
    model = models[target]['model']
    
    # Get training predictions with this model
    train_X = feat[top20].fillna(0).values
    train_y = feat[target].values
    train_pred = model.predict(train_X)
    
    # Test predictions
    test_X = test_features[top20].fillna(0).values
    test_pred = model.predict(test_X)
    
    # Isotonic regression calibration
    cal = IsotonicRegression(out_of_bounds='clip')
    try:
        cal.fit(train_pred, train_y)
        calibrated = cal.predict(test_pred)
    except:
        calibrated = test_pred
    
    # Clip to [0,1] to ensure valid probabilities
    calibrated = np.clip(calibrated, 0.0001, 0.9999)
    
    predictions[target] = calibrated
    
    rate = train_data[target].mean()
    print(f"  {target}: train_rate={rate:.3f}, pred_mean={calibrated.mean():.4f}, shift={calibrated.mean()-rate:+.4f}")

# ── Save submission ──
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
sub_path = Path('submissions') / f'submission_final_{timestamp}.csv'
sub_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(sub_path, index=False)

# ── Final summary ──
print(f"\n✅ Saved: {sub_path}")
print(f"\n{'Target':<6} {'Train Rate':<12} {'Test Mean':<12} {'Shift':<12} {'Range'}")
for t in target_cols:
    rate = train_data[t].mean()
    mean = predictions[t].mean()
    shift = mean - rate
    r_min = predictions[t].min()
    r_max = predictions[t].max()
    print(f"{t:<6} {rate:<12.3f} {mean:<12.4f} {shift:<12.4f} [{r_min:.4f}, {r_max:.4f}]")

# Save experiment metadata
meta = {
    'submission_file': str(sub_path),
    'timestamp': timestamp,
    'n_samples': len(predictions),
    'features': 'top20 per target',
    'configs': {t: v['cfg'] for t, v in models.items()},
    'train_losses': {t: float(v['train_loss']) for t, v in models.items()},
    'predictions': {t: {'mean': float(predictions[t].mean()), 'min': float(predictions[t].min()), 'max': float(predictions[t].max())} for t in target_cols},
}
with open(sub_path.parent / f'meta_{timestamp}.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nExperiment metadata saved.")
