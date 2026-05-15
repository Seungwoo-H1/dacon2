"""
ETRI Lifelog Dataset 2024 Baseline Model
Based on arXiv:2508.03698 (Understanding Human Daily Experience Through Continuous Sensing)

Paper's best baseline: All+DW (64 features) → Q1=0.570, Q2=0.569, Q3=0.597, S3=0.594

Feature engineering:
  - wPedo: mean and total step per 7 time zones (14 features)
  - wHr: mean HR per time zone + proportion of high HR (>100) (14 features)
  - wLight: mean and std of log10(light) per time zone (14 features)
  - mLight: mean and std of log10(light) per time zone (14 features)
  - mUsageStats: System/Social/Hobby daily usage in minutes (3 features)
  - Demographics: gender, age_40_plus, employed, BMI (4 features)
  - Day of week (1 feature)
  Total: 64 features

Trains LightGBM (default params) per target with 5-fold stratified CV.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os

# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_raw", "ch2025_data_items")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "data_raw", "ch2026_metrics_train.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TIME_ZONES = [(0,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24)]

def get_tz(hour):
    for i, (s, e) in enumerate(TIME_ZONES):
        if s <= hour < e:
            return i
    return -1

# Demographics from paper Table I
DEMO = {
    'id01': {'gender':1, 'age_40_plus':1, 'employed':1, 'bmi':23.7},
    'id02': {'gender':0, 'age_40_plus':1, 'employed':1, 'bmi':22.7},
    'id03': {'gender':1, 'age_40_plus':0, 'employed':0, 'bmi':29.1},
    'id04': {'gender':1, 'age_40_plus':1, 'employed':0, 'bmi':19.6},
    'id05': {'gender':1, 'age_40_plus':1, 'employed':1, 'bmi':26.3},
    'id06': {'gender':0, 'age_40_plus':0, 'employed':1, 'bmi':27.1},
    'id07': {'gender':1, 'age_40_plus':1, 'employed':1, 'bmi':26.3},
    'id08': {'gender':0, 'age_40_plus':0, 'employed':1, 'bmi':29.3},
    'id09': {'gender':0, 'age_40_plus':0, 'employed':1, 'bmi':25.4},
    'id10': {'gender':1, 'age_40_plus':0, 'employed':1, 'bmi':26.0},
}

# ============================================================
# Load data
# ============================================================

print("Loading data...")
metrics = pd.read_csv(METRICS_FILE)
metrics['lifelog_date'] = pd.to_datetime(metrics['lifelog_date']).dt.normalize()

# Load all sensor data (datetime index)
def load_parquet(fname):
    return pd.read_parquet(os.path.join(DATA_DIR, fname))

wPedo = load_parquet("ch2025_wPedo.parquet")
wHr = load_parquet("ch2025_wHr.parquet")
wLight = load_parquet("ch2025_wLight.parquet")
mLight = load_parquet("ch2025_mLight.parquet")
mUsage = load_parquet("ch2025_mUsageStats.parquet")

print(f"wPedo: {wPedo.shape}, wHr: {wHr.shape}, wLight: {wLight.shape}")
print(f"mLight: {mLight.shape}, mUsage: {mUsage.shape}")

# ============================================================
# Add date & hour columns
# ============================================================

for df in [wPedo, wHr, wLight, mLight, mUsage]:
    df['date'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['tz'] = df['hour'].apply(get_tz)

# ============================================================
# Feature extraction per (subject, date)
# ============================================================

def agg_daily_sensor(df, sensor_name, key_col, extra_fn=None):
    """
    Aggregate a sensor DataFrame by (subject_id, date).
    Returns a dict: (subject, date) -> {feature_name: value}
    
    key_col: column for tz-specific agg ('step' for wPedo, 'heart_rate' for wHr, 'w_light'/'m_light' for light)
    extra_fn: optional function to compute additional features (e.g. usage categories)
    """
    result = {}
    for subj, grp in df.groupby('subject_id'):
        for date, day in grp.groupby('date'):
            key = (subj, date)
            if key not in result:
                result[key] = {}
            
            day_tz = day.dropna(subset=['tz'])
            
            for tz_idx, (tz_start, tz_end) in enumerate(TIME_ZONES):
                tz_data = day_tz[day_tz['tz'] == tz_idx]
                
                if sensor_name == "wPedo":
                    result[key][f"wp_mean_tz{tz_idx}"] = tz_data['step'].mean() if len(tz_data) > 0 else 0
                    result[key][f"wp_sum_tz{tz_idx}"] = tz_data['step'].sum() if len(tz_data) > 0 else 0
                elif sensor_name == "wHr":
                    # heart_rate is a numpy array per row
                    if len(tz_data) > 0 and 'heart_rate' in tz_data.columns:
                        all_hr = np.concatenate(tz_data['heart_rate'].values)
                        mean_hr = float(np.mean(all_hr))
                        high_hr_prop = float(np.mean(all_hr > 100))
                    else:
                        mean_hr = 0
                        high_hr_prop = 0
                    result[key][f"whr_mean_tz{tz_idx}"] = mean_hr
                    result[key][f"whr_high_tz{tz_idx}"] = high_hr_prop
                elif sensor_name in ("wLight", "mLight"):
                    light_col = 'w_light' if sensor_name == "wLight" else 'm_light'
                    prefix = sensor_name[:2]  # 'wl' or 'ml'
                    if len(tz_data) > 0:
                        lights = np.log10(tz_data[light_col].values + 1)
                        result[key][f"{prefix}_mean_tz{tz_idx}"] = float(np.mean(lights))
                        result[key][f"{prefix}_std_tz{tz_idx}"] = float(np.std(lights)) if len(lights) > 1 else 0
                    else:
                        result[key][f"{prefix}_mean_tz{tz_idx}"] = 0
                        result[key][f"{prefix}_std_tz{tz_idx}"] = 0
    
    if extra_fn:
        for key in result:
            extras = extra_fn(df, key)
            result[key].update(extras)
    
    return result

print("Extracting wPedo features...")
feats_wpedo = agg_daily_sensor(wPedo, "wPedo", "step")

print("Extracting wHr features...")
feats_whr = agg_daily_sensor(wHr, "wHr", "heart_rate")

print("Extracting wLight features...")
feats_wlight = agg_daily_sensor(wLight, "wLight", "w_light")

print("Extracting mLight features...")
feats_mlight = agg_daily_sensor(mLight, "mLight", "m_light")

# Usage stats: categorize apps into System/Social/Hobby
def extract_usage_features(df, key):
    subj, date = key
    day = df[(df['subject_id'] == subj) & (df['date'] == date)]
    if len(day) == 0:
        return {"mu_system": 0, "mu_social": 0, "mu_hobby": 0}
    
    total_times = []
    app_names = []
    for row in day['m_usage_stats']:
        if isinstance(row, list):
            for item in row:
                if isinstance(item, dict) and 'app_name' in item and 'total_time' in item:
                    total_times.append(item['total_time'])
                    app_names.append(item['app_name'])
    
    # Category rules (based on Korean app categories)
    system_keywords = ['설정', '시스템', 'phone', '통화', 'messages', 'mms', 'samsung', 'xiaomi']
    social_keywords = ['naver', 'kakao', 'kakaoent', '카카오톡', '인스타그램', 'instagram', 'facebook', 'twitter', '트위터', '블랙']
    hobby_keywords = ['netflix', 'youtube', '유튜브', 'music', 'spotty', '스포티', 'game', '게임', '캐시워크', 'webtoon', '웹툰']
    
    system_min = 0
    social_min = 0
    hobby_min = 0
    
    for name, t in zip(app_names, total_times):
        nl = name.lower()
        if any(k in nl for k in system_keywords):
            system_min += t
        elif any(k in nl for k in social_keywords):
            social_min += t
        elif any(k in nl for k in hobby_keywords):
            hobby_min += t
        else:
            # Default to system (background apps)
            system_min += t
    
    return {"mu_system": system_min / 60, "mu_social": social_min / 60, "mu_hobby": hobby_min / 60}

print("Extracting mUsage features...")
feats_usage = agg_daily_sensor(mUsage, "mUsage", "m_usage_stats", extra_fn=extract_usage_features)

# ============================================================
# Merge all features into one matrix
# ============================================================

print("Merging features...")

# Get all keys (subject, date) combinations
all_keys = set()
for feats in [feats_wpedo, feats_whr, feats_wlight, feats_mlight, feats_usage]:
    all_keys.update(feats.keys())

print(f"Total (subject, date) keys: {len(all_keys)}")

# Build combined feature dict
combined = {}
for key in all_keys:
    combined[key] = {}
    for feats in [feats_wpedo, feats_whr, feats_wlight, feats_mlight, feats_usage]:
        if key in feats:
            combined[key].update(feats[key])
    
    # Demographics
    subj = key[0]
    if subj in DEMO:
        combined[key].update(DEMO[subj])
    
    # Day of week
    combined[key]['dow'] = key[1].dayofweek

# Convert to DataFrame
feat_df = pd.DataFrame.from_dict(combined, orient='index')
feat_df.index = pd.MultiIndex.from_tuples(feat_df.index, names=['subject_id', 'date'])
feat_df = feat_df.reset_index()

print(f"Feature matrix: {feat_df.shape}")

# Fill missing values with 0
feat_df = feat_df.fillna(0)

# ============================================================
# Train models for each target
# ============================================================

feature_cols = feat_df.columns.drop(['subject_id', 'date'])
print(f"Number of features: {len(feature_cols)}")

target_cols = ['Q1', 'Q2', 'Q3', 'S3']

for target in target_cols:
    print(f"\n{'='*60}")
    print(f"Training for target: {target}")
    print(f"{'='*60}")
    
    # Merge with metrics
    metrics_copy = metrics.copy()
    merged = feat_df.merge(
        metrics_copy[['subject_id', 'lifelog_date', target]],
        left_on=['subject_id', 'date'],
        right_on=['subject_id', 'lifelog_date'],
        how='inner'
    )
    
    X = merged[feature_cols]
    y = merged[target]
    
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Remove zero-variance features
    var_cols = X.columns[X.var() > 0]
    X = X[var_cols]
    print(f"Features after var filter: {X.shape[1]}")
    
    # 5-fold stratified CV
    y_bin = y.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(y))
    fold_f1s = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_bin)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_bin.iloc[tr_idx], y_bin.iloc[val_idx]
        
        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns))
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=list(X_val.columns), reference=train_set)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
    
    oof_bin = (oof_preds >= 0.5).astype(int)
    macro_f1 = f1_score(y_bin, oof_bin, average='macro')
    print(f"OOF Macro F1 (threshold 0.5): {macro_f1:.4f}")
    
    # Also try optimizing threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        bin_preds = (oof_preds >= thresh).astype(int)
        f = f1_score(y_bin, bin_preds, average='macro')
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    print(f"Best threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")
    
    # Train final model on all data
    final_set = lgb.Dataset(X, label=y_bin, feature_name=list(X.columns))
    final_model = lgb.train(params, final_set, num_boost_round=1000)
    final_model.save_model(os.path.join(MODELS_DIR, f"etri_baseline_{target}.txt"))
    print(f"Model saved: models/etri_baseline_{target}.txt")

print("\n=== All targets done ===")
