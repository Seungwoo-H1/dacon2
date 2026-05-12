"""
V127-Based External Data Integration Pipeline

Phase: V127 Fixed + External Data Automatic Exploration
Goal: Break V127 ceiling through external data representation expansion

Key hypothesis: External data provides additional signal for sleep/health
relationships that can improve LB generalization when properly integrated.

Pipeline stages:
1. Domain similarity analysis (KS, adversarial validation AUC)
2. Data quality assessment
3. External → Internal feature mapping
4. Pseudo-label generation + filtering
5. Feature engineering from external data
6. Model training with external features
7. Ensemble combination
8. Automated iteration

External datasets:
A: sleep_health_lifestyle (400 rows, Kaggle)
B: external_date_features (183 rows, computed)  
C: synthetic_lifestyle_extended (2000 rows, generated)
D: synthetic_stress_hrv (1500 rows, generated)

Combinations: A, B, C, D, A+B, A+C, A+D, B+C, B+D, C+D,
              A+B+C, A+B+D, A+C+D, B+C+D, A+B+C+D
"""

import os, sys, gc, re, json, warnings, time, copy, itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Fix path resolution
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SUBMIT = ROOT / "submissions"

os.makedirs(EXPERIMENTS, exist_ok=True)
os.makedirs(EXTERNAL, exist_ok=True)
os.makedirs(SUBMIT, exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}


# ============================================================
# V127 CONFIGURATIONS
# ============================================================

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety', 'n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide':   {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep':   {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

# Leaky features per target (from V127 code)
LEAK_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min',
    'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_mean', 'wPedo_pedo_walking_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
}

SEEDS = [42, 7, 999, 777]


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    personal_cols = []
    df = df.copy()
    all_stats = {}
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        if not for_test:
            all_stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        if fit_stats is not None and col in fit_stats:
            subj_mean = fit_stats[col]['mean']
            subj_std = fit_stats[col]['std']
        else:
            subj_mean = df[f'{col}_subj_mean']
            subj_std = df[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols, all_stats


def add_pairwise_interactions(feat, top_features):
    feat = feat.copy()
    added = []
    for i in range(min(len(top_features), 10)):
        for j in range(i+1, min(len(top_features), 10)):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns:
                continue
            col_prod = f'{f1}_x_{f2}'
            feat[col_prod] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(col_prod)
            if feat[f1].std() > 0 and feat[f2].std() > 0:
                col_ratio = f'{f1}_div_{f2}'
                feat[col_ratio] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
                added.append(col_ratio)
    for f in top_features[:5]:
        if f in feat.columns:
            col_sq = f'{f}_sq'
            feat[col_sq] = feat[f].fillna(0) ** 2
            added.append(col_sq)
    return feat, added


def add_transformed_features(feat, top_features):
    feat = feat.copy()
    added = []
    for f in top_features[:15]:
        if f not in feat.columns:
            continue
        vals = feat[f].fillna(0).values
        vals_abs = np.abs(vals) + 1e-8
        feat[f'{f}_log'] = np.sign(vals) * np.log1p(vals_abs)
        added.append(f'{f}_log')
        feat[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(vals_abs)
        added.append(f'{f}_sqrt')
        feat[f'{f}_abs'] = np.abs(vals)
        added.append(f'{f}_abs')
    return feat, added


def rank_features_importance(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]


def train_cv_model(feat, feat_tst, cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    test_preds = np.zeros((len(feat_tst), len(seeds)))
    sn = [sanitize(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64) if feat_tst is not None else None
    for si, seed in enumerate(seeds):
        cfg_full = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'random_state': seed, 'scale_pos_weight': spw,
        }
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_full, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            if X_test is not None:
                test_preds[:, si] = m.predict(X_test)
            del ds, vd, m
            gc.collect()
    return oof, test_preds


# ============================================================
# EXTERNAL DATA LOADING
# ============================================================

def load_external_data():
    """Load all available external datasets."""
    ext_data = {}
    
    # Dataset A: Sleep Health & Lifestyle (Kaggle)
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        ext_data['A_sleep_health'] = {
            'name': 'Sleep Health & Lifestyle',
            'path': str(shl_path),
            'df': pd.read_csv(shl_path),
            'n': len(pd.read_csv(shl_path)),
            'type': 'lifestyle',
            'target_mapping': {
                'Quality of Sleep': 'Q1',
                'Stress Level': 'Q3',
                'Sleep Duration': 'S1',
                'Physical Activity Level': 'Q2',
                'Sleep Disorder': 'S4',
            }
        }
    
    # Dataset B: External date features
    date_path = DATA / 'external_data.parquet'
    if date_path.exists():
        ext_data['B_date_features'] = {
            'name': 'Date/Temperature Features',
            'path': str(date_path),
            'df': pd.read_parquet(date_path),
            'n': 183,
            'type': 'temporal',
            'target_mapping': {}
        }
    
    # Dataset C: Synthetic extended lifestyle (generated)
    ext_data['C_synthetic_extended'] = {
        'name': 'Synthetic Lifestyle Extended',
        'df': generate_synthetic_lifestyle(2000),
        'n': 2000,
        'type': 'synthetic_lifestyle',
        'target_mapping': {
            'Quality of Sleep': 'Q1',
            'Stress Level': 'Q3',
            'Sleep Duration': 'S1',
            'Physical Activity Level': 'Q2',
            'Sleep Disorder': 'S4',
            'Caffeine Intake': 'Q3',
            'Alcohol Units': 'Q2',
        }
    }
    
    # Dataset D: Synthetic stress/HRV
    ext_data['D_synthetic_stress_hrv'] = {
        'name': 'Synthetic Stress/HRV',
        'df': generate_synthetic_stress_hrv(1500),
        'n': 1500,
        'type': 'synthetic_stress',
        'target_mapping': {
            'Stress_Score': 'Q3',
            'HRV_mean': 'Q2',
            'Sleep_Efficiency': 'S2',
            'Resting HR': 'Q2',
            'Mood_Score': 'Q1',
            'Wake_After_Sleep': 'S4',
        }
    }
    
    return ext_data


def generate_synthetic_lifestyle(n=2000):
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(18, 70, n),
        'Sleep Duration': np.clip(np.random.normal(7.0, 1.2, n), 2, 12),
        'Quality of Sleep': np.clip(np.random.beta(2, 2, n) * 10, 1, 10),
        'Physical Activity Level': np.clip(np.random.normal(150, 50, n), 0, 300),
        'Stress Level': np.clip(np.random.beta(2, 3, n) * 10, 1, 10),
        'BMI': np.clip(np.random.normal(25, 5, n), 15, 50),
        'Heart Rate': np.clip(np.random.normal(72, 10, n), 50, 120),
        'Daily Steps': np.clip(np.random.normal(6000, 3000, n), 0, 20000),
        'Sleep Disorder': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'Caffeine Intake': np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15]),
        'Alcohol Units': np.random.choice([0, 1, 2, 3, 4], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'Exercise Min/week': np.clip(np.random.normal(150, 60, n), 0, 500),
        'Screen Time Hours': np.clip(np.random.normal(6, 2, n), 1, 16),
    })


def generate_synthetic_stress_hrv(n=1500):
    np.random.seed(77)
    return pd.DataFrame({
        'Resting HR': np.clip(np.random.normal(68, 8, n), 50, 100),
        'HRV_mean': np.clip(np.random.normal(55, 15, n), 20, 120),
        'EDA_mean': np.clip(np.random.normal(5, 2, n), 1, 15),
        'Accelerometry_mean': np.clip(np.random.normal(0.05, 0.02, n), 0.01, 0.2),
        'Body Temp': np.clip(np.random.normal(36.8, 0.3, n), 35.5, 38.0),
        'Stress_Score': np.clip(np.random.beta(2, 2, n) * 10, 0, 10),
        'Mood_Score': np.clip(np.random.beta(3, 2, n) * 10, 0, 10),
        'Sleep_Efficiency': np.clip(np.random.normal(85, 10, n), 50, 100),
        'Wake_After_Sleep': np.clip(np.random.exponential(30, n), 0, 120),
        'Activity Intensity': np.clip(np.random.normal(50, 20, n), 0, 100),
    })


# ============================================================
# DOMAIN SIMILARITY ANALYSIS
# ============================================================

def analyze_domain_similarity(feat, ext_df, name):
    """Measure domain similarity between internal and external data."""
    results = {}
    results['dataset'] = name
    
    # Get numerical columns in both
    internal_num = feat.select_dtypes(include=[np.number]).copy()
    external_num = ext_df.select_dtypes(include=[np.number])
    
    # For external data, compute per-subject aggregates to match internal schema
    # Since external data is not per-subject per-date, we'll compare distribution-level
    
    common_cols = list(set(internal_num.columns) & set(external_num.columns))
    common_cols = [c for c in common_cols if c not in META]
    results['common_features'] = len(common_cols)
    
    if not common_cols:
        results['ks_avg'] = 1.0  # No overlap = max distance
        results['ks_median'] = 1.0
        results['domain_gap'] = 1.0
        return results
    
    # KS test for each common feature
    ks_stats = []
    for col in common_cols:
        x = internal_num[col].dropna()
        y = external_num[col].dropna()
        if len(x) > 10 and len(y) > 10:
            try:
                ks_stat, _ = stats.ks_2samp(x.sample(min(100, len(x))), 
                                           y.sample(min(100, len(y))))
                ks_stats.append(ks_stat)
            except:
                pass
    
    results['ks_avg'] = float(np.mean(ks_stats)) if ks_stats else 1.0
    results['ks_median'] = float(np.median(ks_stats)) if ks_stats else 1.0
    results['ks_min'] = float(np.min(ks_stats)) if ks_stats else 1.0
    results['ks_max'] = float(np.max(ks_stats)) if ks_stats else 1.0
    
    # Domain gap score (0=same, 1=completely different)
    results['domain_gap'] = min(1.0, results['ks_avg'])
    
    # Adversarial validation (can we distinguish train vs external?)
    if len(common_cols) >= 2 and len(internal_num) > 100 and len(external_num) > 100:
        try:
            X_adv = pd.concat([
                internal_num[common_cols[:min(20, len(common_cols))]].fillna(0).head(200),
                external_num[common_cols[:min(20, len(common_cols))]].fillna(0).head(200)
            ], axis=0)
            y_adv = np.concatenate([np.zeros(200), np.ones(200)])
            X_adv = X_adv.replace([np.inf, -np.inf], 0).fillna(0)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            rf.fit(X_adv, y_adv)
            pred = rf.predict_proba(X_adv)[:, 1]
            auc = stats.roc_auc_score(y_adv, pred)
            results['adversarial_auc'] = float(auc)
            results['adversarial_gap'] = abs(auc - 0.5) * 2
        except:
            results['adversarial_auc'] = None
    
    # Quality metrics
    results['missing_pct'] = float(ext_df.isnull().sum().sum() / (len(ext_df) * len(ext_df.columns)) * 100)
    results['duplicate_ratio'] = float(ext_df.duplicated().sum() / len(ext_df))
    
    return results


# ============================================================
# FEATURE MAPPING & SIGNAL EXTRACTION
# ============================================================

def get_target_signal_strength(feat, target, ext_df, target_mapping):
    """Measure how well external features correlate with a target."""
    results = {}
    results['target'] = target
    
    # Find which external column maps to this target
    proxy_ext = None
    proxy_int = None
    for ext_col, int_col in target_mapping.items():
        if int_col == target and ext_col in ext_df.columns:
            proxy_ext = ext_col
            proxy_int = int_col
            break
    
    if proxy_ext is None:
        results['no_proxy'] = True
        return results
    
    results['proxy_ext_col'] = proxy_ext
    results['proxy_int_col'] = proxy_int
    
    # Compute correlation between external proxy and internal target
    # This is a rough proxy for signal strength
    if proxy_ext in ext_df.select_dtypes(include=[np.number]).columns:
        ext_vals = ext_df[proxy_ext].dropna()
        results['proxy_ext_mean'] = float(ext_vals.mean())
        results['proxy_ext_std'] = float(ext_vals.std())
    
    if proxy_int in feat.columns:
        int_vals = feat[proxy_int].dropna()
        results['proxy_int_mean'] = float(int_vals.mean())
        results['proxy_int_std'] = float(int_vals.std())
    
    return results


# ============================================================
# PHASE 1: Load data and run domain similarity analysis
# ============================================================

def run_domain_analysis(feat, ext_data):
    """Run domain similarity analysis on all external datasets."""
    print("\n" + "="*60)
    print("PHASE 1: Domain Similarity Analysis")
    print("="*60)
    
    all_domain_results = {}
    all_target_signals = {}
    
    for ext_key, ext_info in ext_data.items():
        ext_df = ext_info['df']
        name = ext_info['name']
        
        print(f"\n[{ext_key}] {name} (n={ext_info['n']})")
        
        # Domain similarity
        dom = analyze_domain_similarity(feat, ext_df, name)
        all_domain_results[ext_key] = dom
        print(f"  Domain gap: {dom['domain_gap']:.4f}, KS avg: {dom['ks_avg']:.4f}")
        print(f"  Common features: {dom['common_features']}")
        
        # Target signal strength for each target
        target_map = ext_info.get('target_mapping', {})
        for t in TARGETS:
            sig = get_target_signal_strength(feat, t, ext_df, target_map)
            all_target_signals[f'{ext_key}_{t}'] = sig
            if 'no_proxy' not in sig:
                print(f"  Target {t} proxy: {sig.get('proxy_ext_col', 'N/A')} "
                      f"(ext_mean={sig.get('proxy_ext_mean', 'N/A'):.2f})")
    
    return all_domain_results, all_target_signals


# ============================================================
# PHASE 2: Feature Engineering from External Data
# ============================================================

def create_external_features(feat, feat_test, ext_data, domain_results, target_signals):
    """
    Create features from external data and merge with internal features.
    
    Since external data doesn't have subject/date alignment with internal data,
    we compute summary statistics and use them as global features.
    """
    print("\n" + "="*60)
    print("PHASE 2: Feature Engineering from External Data")
    print("="*60)
    
    results = {}
    
    for ext_key, ext_info in ext_data.items():
        ext_df = ext_info['df']
        name = ext_info['name']
        target_map = ext_info.get('target_mapping', {})
        dom = domain_results.get(ext_key, {})
        
        print(f"\n[{ext_key}] {name}:")
        
        # Create external feature columns (summary stats per target-relevant feature)
        ext_features = {}
        
        # For each target, find the proxy column and compute stats
        for t in TARGETS:
            sig = target_signals.get(f'{ext_key}_{t}', {})
            if sig.get('no_proxy', True):
                continue
            
            proxy_col = sig.get('proxy_ext_col')
            if proxy_col and proxy_col in ext_df.columns:
                ext_col = f'ext_{ext_key}_{proxy_col}_mean'
                ext_features[ext_col] = float(ext_df[proxy_col].mean())
                
                ext_col_std = f'ext_{ext_key}_{proxy_col}_std'
                ext_features[ext_col_std] = float(ext_df[proxy_col].std())
                
                ext_col_skew = f'ext_{ext_key}_{proxy_col}_skew'
                if ext_df[proxy_col].skew() != 0:
                    ext_features[ext_col_skew] = float(ext_df[proxy_col].skew())
                else:
                    ext_features[ext_col_skew] = 0.0
        
        results[ext_key] = ext_features
        print(f"  Created {len(ext_features)} external features")
    
    return results


# ============================================================
# PHASE 3: Pseudo-Label Generation
# ============================================================

def generate_pseudo_labels(feat, target, feature_cols):
    """Generate pseudo-labels using external features as additional signal."""
    try:
        X = feat[feature_cols].fillna(0)
        y = feat[target].values
        
        # Filter to numerical columns only
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            return None, None
        
        # Train a simple RF to identify important features
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        # Feature importances
        imp = dict(zip(X.columns, rf.feature_importances_))
        top = sorted(imp.items(), key=lambda x: -x[1])[:10]
        
        return {t: v for t, v in top}, X.columns.tolist()
    except:
        return None, None


# ============================================================
# PHASE 4: V127 Baseline (with external features)
# ============================================================

def run_v127_experiment(feat, feat_test, ext_features, ext_key, cfg_name, n_feat, strategy='base'):
    """
    Run V127-style experiment with optional external features.
    
    Returns: (oof, test_preds, avg_logloss)
    """
    if ext_key and ext_key in ext_features:
        ext_feat_df = pd.DataFrame([{k: v for k, v in ext_features[ext_key].items()}])
        print(f"  [INFO] Using external features from {ext_key}")
        print(f"  [INFO] Features: {list(ext_features[ext_key].keys())}")
    
    return None  # Placeholder - will implement in phase 5


# ============================================================
# PHASE 5: Experiment with external features
# ============================================================

def run_external_experiment(feat, feat_test, ext_key, ext_df, domain_results, target_signals):
    """
    Run a single experiment combining internal + external data.
    
    Strategy: Add external summary features to the feature set.
    Since external data is not per-subject, these become global features.
    """
    t_start = time.time()
    
    print(f"\n  [{ext_key}] Running experiment...")
    
    # Load external features as additional columns
    ext_info = ext_data.get(ext_key, {})
    ext_df = ext_info.get('df')
    if ext_df is None:
        return None
    
    # Create external feature columns
    ext_cols = []
    target_map = ext_info.get('target_mapping', {})
    dom = domain_results.get(ext_key, {})
    
    # Only use external features if domain gap is reasonable (not too different)
    if dom.get('domain_gap', 0) > 0.8:
        print(f"    Domain gap too high ({dom['domain_gap']:.4f}), skipping")
        return None
    
    # Create external feature columns (per-target proxy stats)
    for t in TARGETS:
        sig = target_signals.get(f'{ext_key}_{t}', {})
        if sig.get('no_proxy', True):
            continue
        
        proxy_col = sig.get('proxy_ext_col')
        if proxy_col and proxy_col in ext_df.columns:
            ext_cols.append(f'ext_{proxy_col}_mean')
            ext_cols.append(f'ext_{proxy_col}_std')
    
    print(f"    External proxy features: {ext_cols}")
    
    return None  # Placeholder


# ============================================================
# MAIN
# ============================================================

# Load external data
ext_data = load_external_data()
print(f"Loaded {len(ext_data)} external datasets: {list(ext_data.keys())}")

# Load internal features
feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")
for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

# Get base feature columns
feat_cols_raw = get_feature_cols(feat)
print(f"Base features: {len(feat_cols_raw)}")

# Personalization
feat, zscore_cols, fit_stats = add_personalization(feat, feat_cols_raw)
feat_test_z, _, _ = add_personalization(feat_test, feat_cols_raw, fit_stats=fit_stats, for_test=True)
all_cols = feat_cols_raw + zscore_cols
print(f"After personalization: train={feat.shape}, test={feat_test_z.shape}")

# Run domain analysis
all_domain_results, all_target_signals = run_domain_analysis(feat, ext_data)

# Run feature engineering from external data
ext_features = create_external_features(feat, feat_test, ext_data, 
                                       all_domain_results, all_target_signals)

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for ext_key, dom in all_domain_results.items():
    print(f"\n{ext_key}:")
    print(f"  Domain gap: {dom['domain_gap']:.4f}")
    print(f"  KS avg: {dom['ks_avg']:.4f}")
    print(f"  Common features: {dom['common_features']}")
    print(f"  Missing: {dom.get('missing_pct', 0):.1f}%")
    print(f"  Features created: {len(ext_features.get(ext_key, {}))}")
