"""
External Data Collection & Analysis Pipeline

Collects multiple external datasets, measures:
1. Domain similarity to training data
2. Data quality metrics
3. Feature correlation with our targets
4. Transferability scores

Then runs automated experiments for each dataset/combination.
"""

import os, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

ROOT = Path(__file__).resolve().parent.parent.parent  # Go up from external_data_research/src/
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SRC = ROOT / "src"

os.makedirs(EXPERIMENTS, exist_ok=True)
os.makedirs(EXTERNAL, exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ============================================================
# EXTERNAL DATASET DEFINITIONS
# ============================================================

EXTERNAL_DATASETS = {
    'sleep_health_lifestyle': {
        'path': EXTERNAL / 'sleep_health_lifestyle.csv',
        'name': 'Sleep Health & Lifestyle (Kaggle)',
        'description': '400 synthetic records with sleep, activity, stress, BMI data',
        'source': 'kaggle',
        'rows': 400,
        'type': 'lifestyle',
        'columns': ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                   'Physical Activity Level', 'Stress Level', 'BMI Category',
                   'Blood Pressure Systolic', 'Heart Rate', 'Daily Steps', 'Sleep Disorder'],
    },
    'sleep_health_100k': {
        'path': EXTERNAL / 'sleep_health_100k.csv',
        'name': 'Sleep Health & Daily Performance (Kaggle 100k)',
        'description': '100,000 synthetic patient records with sleep + cognitive performance',
        'source': 'kaggle',
        'rows': None,  # To be downloaded
        'type': 'lifestyle',
    },
    'sleep_lifestyle_1000': {
        'path': EXTERNAL / 'sleep_lifestyle_1000.csv',
        'name': 'Sleep & Lifestyle Health (Kaggle 1000)',
        'description': '1000 records with caffeine, alcohol, smoking, exercise data',
        'source': 'kaggle',
        'rows': None,
        'type': 'lifestyle',
    },
    'external_date_features': {
        'path': None,
        'name': 'External Date Features',
        'description': 'Holiday, season, daylight, school term features',
        'source': 'computed',
        'rows': 183,
        'type': 'temporal',
        'already_loaded': True,
    },
}


# ============================================================
# PHASE 1: Domain Similarity Analysis
# ============================================================

def domain_similarity_analysis(internal_feat, internal_y, external_df, ext_features, target_name):
    """
    Measure how similar external data is to internal training data.
    
    Metrics:
    1. Feature distribution distance (KS test, Hellinger)
    2. Adversarial validation AUC (train vs external classifier)
    3. Target-proxy correlation (if external has target-like columns)
    4. Embedding similarity (if applicable)
    """
    results = {}
    
    # Find overlapping numerical columns
    internal_num = internal_feat.select_dtypes(include=[np.number])
    external_num = external_df.select_dtypes(include=[np.number])
    
    common_cols = list(set(internal_num.columns) & set(external_num.columns))
    common_cols = [c for c in common_cols if c not in META_COLS]
    
    if not common_cols:
        results['overlapping_features'] = 0
        results['avg_ks_stat'] = None
        results['avg_ks_pval'] = None
        return results
    
    # 1. Feature distribution distance
    ks_stats = []
    ks_pvals = []
    for col in common_cols:
        x = internal_num[col].dropna()
        y = external_df[col].dropna()
        if len(x) > 2 and len(y) > 2:
            ks_stat, ks_pval = stats.ks_2samp(x, y)
            ks_stats.append(ks_stat)
            ks_pvals.append(ks_pval)
    
    results['overlapping_features'] = len(common_cols)
    results['common_cols'] = common_cols
    if ks_stats:
        results['avg_ks_stat'] = float(np.mean(ks_stats))
        results['median_ks_stat'] = float(np.median(ks_stats))
        results['min_ks_stat'] = float(np.min(ks_stats))
        results['avg_ks_pval'] = float(np.mean(ks_pvals))
        # Low p-value = different distributions (good for transfer!)
        results['pct_significant'] = float(np.mean([p < 0.05 for p in ks_pvals]))
    
    # 2. Adversarial validation
    # Can we distinguish train vs external?
    if len(common_cols) >= 2:
        try:
            X_adv = pd.concat([
                internal_feat[common_cols].fillna(0).head(450),
                external_df[common_cols].fillna(0).head(400)
            ], axis=0)
            y_adv = np.concatenate([
                np.zeros(450),
                np.ones(min(400, len(external_df)))
            ])
            
            if len(X_adv) > 20:
                X_adv = X_adv.fillna(0).replace([np.inf, -np.inf], 0)
                
                rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
                rf.fit(X_adv, y_adv)
                pred_proba = rf.predict_proba(X_adv)[:, 1]
                adv_auc = roc_auc_score(y_adv, pred_proba)
                
                results['adversarial_auc'] = float(adv_auc)
                # AUC ~0.5 = same distribution (good for transfer)
                # AUC >0.7 = different distributions (domain gap)
                results['domain_gap'] = abs(adv_auc - 0.5) * 2
        except Exception:
            results['adversarial_auc'] = None
    
    # 3. Target-proxy correlation
    # Check if external features correlate with our targets (when projected)
    proxy_corr = {}
    for col in common_cols:
        if col in internal_feat.columns:
            x_int = internal_feat[col].dropna()
            for t in TARGETS:
                if t in internal_feat.columns:
                    corr, pval = stats.pointbiserialr(x_int, internal_feat[t].dropna())
                    proxy_corr[f'{col}_vs_{t}'] = float(corr)
    
    if proxy_corr:
        results['target_proxy_correlations'] = proxy_corr
        # Find the strongest correlations
        abs_corr = {k: abs(v) for k, v in proxy_corr.items()}
        top = sorted(abs_corr.items(), key=lambda x: -x[1])[:5]
        results['top_proxy_correlations'] = [(k, v) for k, v in top]
    
    return results


def measure_data_quality(external_df, dataset_name):
    """
    Measure quality of external dataset.
    """
    results = {}
    results['dataset'] = dataset_name
    results['total_rows'] = len(external_df)
    results['total_cols'] = len(external_df.columns)
    
    # Missing data
    missing = external_df.isnull().sum()
    results['total_missing'] = int(missing.sum())
    results['missing_pct'] = float(missing.sum() / (len(external_df) * len(external_df.columns)) * 100)
    results['cols_with_missing'] = int((missing > 0).sum())
    
    # Duplicate ratio
    dup_ratio = external_df.duplicated().sum() / len(external_df)
    results['duplicate_ratio'] = float(dup_ratio)
    
    # Numeric features stats
    num_cols = external_df.select_dtypes(include=[np.number])
    results['numeric_cols'] = len(num_cols.columns)
    
    if len(num_cols.columns) > 0:
        for col in num_cols.columns:
            col_data = num_cols[col].dropna()
            if len(col_data) > 1:
                kurt, skew = stats.kurtosistest(col_data)[:2]
                results[f'{col}_skew'] = float(skew) if not np.isnan(skew) else None
                results[f'{col}_kurtosis'] = float(kurt) if not np.isnan(kurt) else None
                
                # Outlier detection (IQR method)
                q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((col_data < q1 - 3*iqr) | (col_data > q3 + 3*iqr)).sum()
                    results[f'{col}_outlier_pct'] = float(outliers / len(col_data) * 100)
    
    # Categorical features info
    cat_cols = external_df.select_dtypes(exclude=[np.number])
    results['categorical_cols'] = len(cat_cols.columns)
    for col in cat_cols.columns:
        unique = cat_cols[col].nunique()
        results[f'{col}_unique'] = unique
        if unique <= 10:
            results[f'{col}_top_categories'] = cat_cols[col].value_counts().head(3).to_dict()
    
    return results


def estimate_transferability(internal_feat, external_df, ext_features, ext_features_info, dataset_name):
    """
    Estimate how useful external data is for our task.
    
    Strategy: Use external data to learn feature-target relationships,
    then apply to internal data via feature importance guidance.
    """
    results = {}
    results['dataset'] = dataset_name
    
    # Map external columns to potential internal feature equivalents
    # External: Age, Sleep Duration, Quality of Sleep, Physical Activity, 
    #           Stress Level, BMI, Heart Rate, Daily Steps
    # Internal: various aggregated metrics from wearables
    
    col_mapping = {
        'Age': ['Age'],
        'Sleep Duration': ['s_sleep_duration_est'],  # approximate
        'Quality of Sleep': ['Q1'],  # This IS a target
        'Physical Activity Level': ['mActivity_m_activity_mean', 'wPedo_pedo_step_mean'],
        'Stress Level': ['Q3'],  # This IS a target proxy
        'Heart Rate': ['wHr_hr_mean'],
        'Daily Steps': ['wPedo_pedo_step_mean'],
        'Blood Pressure Systolic': [],  # No direct match
        'BMI Category': [],  # No direct match
        'BMI_Category_Code': [],  # No direct match
        'Sleep Disorder': ['Q1'],  # Proxy
        'Gender': ['Gender'],  # Could be useful
    }
    
    # Measure feature importance of shared concepts
    internal_num = internal_feat.select_dtypes(include=[np.number])
    
    for ext_col, internal_candidates in col_mapping.items():
        if ext_col in external_df.columns and internal_candidates:
            # Check correlations in external data
            if ext_col in external_df.select_dtypes(include=[np.number]).columns:
                ext_vals = external_df[ext_col].dropna()
                results[f'ext_{ext_col}_mean'] = float(ext_vals.mean())
                results[f'ext_{ext_col}_std'] = float(ext_vals.std())
                
                # Check if internal candidates have similar distributions
                for ic in internal_candidates:
                    if ic in internal_num.columns:
                        int_vals = internal_num[ic].dropna()
                        if len(int_vals) > 2 and len(ext_vals) > 2:
                            corr, _ = stats.pearsonr(ext_vals.head(min(len(int_vals), len(ext_vals))),
                                                    int_vals.head(min(len(int_vals), len(ext_vals))))
                            results[f'ext_{ext_col}_vs_int_{ic}_corr'] = float(corr)
    
    return results


# ============================================================
# PHASE 2: Feature Engineering from External Data
# ============================================================

def extract_external_signal(internal_feat, external_df, target_name):
    """
    Extract signal from external data for a specific target.
    
    Methods:
    1. Feature importance from external data
    2. Conditional probability tables
    3. Decision tree rules
    4. Gradient boosting proxy
    """
    results = {}
    
    # Identify target-like columns in external data
    target_map = {
        'Q1': 'Quality of Sleep',
        'Q2': 'Stress Level',  # proxy
        'Q3': 'Stress Level',
        'S1': 'Sleep Duration',
        'S2': 'Sleep Duration',  # proxy (efficiency)
        'S3': 'Sleep Duration',  # proxy (delay)
        'S4': 'Sleep Disorder',  # proxy (awakening)
    }
    
    if target_name not in target_map:
        return results
    
    proxy_col = target_map[target_name]
    
    if proxy_col not in external_df.columns:
        # Try alternative mapping
        alt_map = {
            'Q1': 'Sleep Disorder',
            'Q2': 'Physical Activity Level',  # fatigue proxy
            'Q3': 'Sleep Quality',
            'S1': 'Sleep Duration',
            'S2': 'Sleep Duration',
            'S3': 'Sleep Duration',
            'S4': 'Sleep Disorder',
        }
        proxy_col = alt_map.get(target_name, None)
    
    if proxy_col is None or proxy_col not in external_df.columns:
        results['no_proxy_column'] = True
        return results
    
    results['proxy_column'] = proxy_col
    
    # Measure feature importance of other columns in predicting the proxy
    ext_num = external_df.select_dtypes(include=[np.number])
    if proxy_col in ext_num.columns and len(ext_num) > 50:
        try:
            y_proxy = ext_num[proxy_col].values
            X_ext = ext_num.drop(columns=[proxy_col]).fillna(0)
            
            # Random forest importance
            rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
            rf.fit(X_ext, y_proxy)
            
            importances = dict(zip(X_ext.columns, rf.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
            results['top_external_features'] = top_features
            results['feature_importances'] = importances
            
            # Calculate information gain per feature
            total_imp = sum(importances.values())
            if total_imp > 0:
                results['normalized_importances'] = {k: v/total_imp for k, v in importances.items()}
        except Exception as e:
            results['feature_importance_error'] = str(e)
    
    return results


# ============================================================
# PHASE 3: Pseudo-Label Generation
# ============================================================

def generate_pseudo_labels(internal_feat, external_df, feature_cols, target_name):
    """
    Generate pseudo-labels for external data based on internal model,
    then use to augment training.
    """
    results = {}
    
    try:
        # Train a simple model on internal data
        X_train = internal_feat[feature_cols].fillna(0)
        y_train = internal_feat[target_name].values
        
        # Use only numerical columns
        X_train = X_train.select_dtypes(include=[np.number])
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_train, y_train)
        
        # Get predictions on external data
        # Map external features to internal feature space
        common_cols = [c for c in X_train.columns if c in external_df.columns]
        
        if len(common_cols) > 0:
            X_ext = external_df[common_cols].fillna(0)
            pseudo_labels = rf.predict(X_ext)
            pseudo_probs = rf.predict_proba(X_ext)[:, 1]
            
            results['pseudo_label_dist'] = {
                '0': int((pseudo_labels == 0).sum()),
                '1': int((pseudo_labels == 1).sum()),
            }
            results['avg_confidence'] = float(np.max(rf.predict_proba(X_ext), axis=1).mean())
            results['high_conf_count'] = int((np.max(rf.predict_proba(X_ext), axis=1) > 0.7).sum())
        else:
            results['no_common_features'] = True
    except Exception as e:
        results['error'] = str(e)
    
    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("EXTERNAL DATA RESEARCH PIPELINE")
    print("=" * 70)
    
    # Load internal data
    print("\n[1] Loading internal features...")
    feat = pd.read_parquet(DATA / "features_clean_v60.parquet")
    feat['sleep_date'] = feat['sleep_date'].astype(str)
    feat['lifelog_date'] = feat['lifelog_date'].astype(str)
    
    feature_cols = [c for c in feat.columns 
                   if c not in META_COLS | set(TARGETS) 
                   and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    print(f"    Internal features: {len(feature_cols)} columns, {len(feat)} samples")
    
    # Load external data
    print("\n[2] Loading external datasets...")
    ext_data = {}
    
    # Sleep Health & Lifestyle
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        ext_data['sleep_health_lifestyle'] = pd.read_csv(shl_path)
        print(f"    - sleep_health_lifestyle: {ext_data['sleep_health_lifestyle'].shape}")
    
    # External date features
    ext_date_path = DATA / 'external_data.parquet'
    if ext_date_path.exists():
        ext_data['external_date_features'] = pd.read_parquet(ext_date_path)
        print(f"    - external_date_features: {ext_data['external_date_features'].shape}")
    
    # Generate additional synthetic external datasets
    print("\n[3] Generating synthetic external datasets for research...")
    
    # Synthetic dataset 1: Extended lifestyle with additional features
    np.random.seed(42)
    n_syn = 2000
    syn1 = pd.DataFrame({
        'Age': np.random.randint(18, 70, n_syn),
        'Sleep Duration': np.clip(np.random.normal(7.0, 1.2, n_syn), 2, 12),
        'Sleep Quality': np.clip(np.random.beta(2, 2, n_syn) * 10, 1, 10),
        'Physical Activity': np.clip(np.random.normal(120, 40, n_syn), 0, 300),
        'Stress Level': np.clip(np.random.beta(2, 3, n_syn) * 10, 1, 10),
        'BMI': np.clip(np.random.normal(25, 5, n_syn), 15, 50),
        'Heart Rate': np.clip(np.random.normal(72, 10, n_syn), 50, 120),
        'Daily Steps': np.clip(np.random.normal(6000, 3000, n_syn), 0, 20000),
        'Sleep Disorder': np.random.choice([0, 1], n_syn, p=[0.85, 0.15]),
        'Caffeine Intake': np.random.choice([0, 1, 2, 3], n_syn, p=[0.3, 0.3, 0.25, 0.15]),
        'Alcohol Units': np.random.choice([0, 1, 2, 3, 4], n_syn, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'Exercise Min/week': np.clip(np.random.normal(150, 60, n_syn), 0, 500),
        'Screen Time Hours': np.clip(np.random.normal(6, 2, n_syn), 1, 16),
    })
    ext_data['synthetic_lifestyle_extended'] = syn1
    print(f"    - synthetic_lifestyle_extended: {syn1.shape}")
    
    # Synthetic dataset 2: Stress/HRV focused
    n_syn2 = 1500
    syn2 = pd.DataFrame({
        'Age': np.random.randint(18, 65, n_syn2),
        'Resting HR': np.clip(np.random.normal(68, 8, n_syn2), 50, 100),
        'HRV_mean': np.clip(np.random.normal(55, 15, n_syn2), 20, 120),
        'EDA_mean': np.clip(np.random.normal(5, 2, n_syn2), 1, 15),
        'Accelerometry_mean': np.clip(np.random.normal(0.05, 0.02, n_syn2), 0.01, 0.2),
        'Body Temp': np.clip(np.random.normal(36.8, 0.3, n_syn2), 35.5, 38.0),
        'Stress_Score': np.clip(np.random.beta(2, 2, n_syn2) * 10, 0, 10),
        'Mood_Score': np.clip(np.random.beta(3, 2, n_syn2) * 10, 0, 10),
        'Sleep_Efficiency': np.clip(np.random.normal(85, 10, n_syn2), 50, 100),
        'Wake_After_Sleep': np.clip(np.random.exponential(30, n_syn2), 0, 120),
    })
    ext_data['synthetic_stress_hrv'] = syn2
    print(f"    - synthetic_stress_hrv: {syn2.shape}")
    
    # ============================================================
    # ANALYSIS PHASE
    # ============================================================
    
    print("\n[4] Running domain similarity analysis...")
    domain_results = {}
    quality_results = {}
    transfer_results = {}
    feature_signal = {}
    
    for ext_name, ext_df in ext_data.items():
        print(f"\n    Analyzing: {ext_name}")
        
        # Domain similarity
        dom = domain_similarity_analysis(feat, feat[TARGETS], ext_df, 
                                        ext_df.select_dtypes(include=[np.number]),
                                        TARGETS[0])
        domain_results[ext_name] = dom
        
        # Data quality
        quality = measure_data_quality(ext_df, ext_name)
        quality_results[ext_name] = quality
        
        # Transferability
        trans = estimate_transferability(feat, ext_df, 
                                        ext_df.select_dtypes(include=[np.number]),
                                        ext_df, ext_name)
        transfer_results[ext_name] = trans
        
        # Feature signal extraction
        for t in TARGETS[:3]:  # Just Q1, Q2, Q3 for speed
            signal = extract_external_signal(feat, ext_df, t)
            if signal:
                feature_signal[f'{ext_name}_{t}'] = signal
        
        ks = dom.get('avg_ks_stat'); auc = dom.get('adversarial_auc'); miss = quality.get('missing_pct', 0)
        ks_str = f'{ks:.4f}' if ks is not None else 'N/A'
        auc_str = f'{auc:.4f}' if auc is not None else 'N/A'
        print(f'      Domain KS: {ks_str}, AUC: {auc_str}, Quality: {miss:.1f}% missing')
    
    # ============================================================
    # SYNTHETIC DATA EXPERIMENTS
    # ============================================================
    
    print("\n[5] Running pseudo-label experiments...")
    pseudo_results = {}
    
    for t in TARGETS:
        for ext_name in ['synthetic_lifestyle_extended', 'synthetic_stress_hrv']:
            ext_df = ext_data[ext_name]
            pseudo = generate_pseudo_labels(feat, ext_df, feature_cols, t)
            pseudo_results[f'{ext_name}_{t}'] = pseudo
            if 'error' in pseudo:
                print(f"    {ext_name}/{t}: ERROR - {pseudo['error']}")
            else:
                dist = pseudo.get('pseudo_label_dist', {})
                conf = pseudo.get('avg_confidence', 0)
                print(f"    {ext_name}/{t}: dist={dist}, conf={conf:.3f}")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    print("\n[6] Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    all_results = {
        'timestamp': timestamp,
        'domain_similarity': {},
        'data_quality': {},
        'transferability': {},
        'feature_signal': {},
        'pseudo_labels': {},
    }
    
    for k, v in domain_results.items():
        all_results['domain_similarity'][k] = {
            kk: vv for kk, vv in v.items() 
            if kk not in ('common_cols', 'top_proxy_correlations', 'feature_importances', 'normalized_importances')
            and not isinstance(vv, (dict, list))
        }
    
    all_results['data_quality'] = quality_results
    all_results['transferability'] = transfer_results
    all_results['feature_signal'] = feature_signal
    all_results['pseudo_labels'] = pseudo_results
    
    log_path = EXPERIMENTS / f'external_data_analysis_{timestamp}.json'
    with open(log_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {log_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ext_name in ext_data:
        dom = domain_results.get(ext_name, {})
        qual = quality_results.get(ext_name, {})
        print(f"\n{ext_name}:")
        print(f"  KS Stat: {dom.get('avg_ks_stat', 'N/A'):.4f}")
        print(f"  AUC: {dom.get('adversarial_auc', 'N/A'):.4f}")
        print(f"  Missing: {qual.get('missing_pct', 'N/A'):.1f}%")
        print(f"  Duplicates: {qual.get('duplicate_ratio', 'N/A'):.4f}")
    
    return all_results


if __name__ == '__main__':
    main()
