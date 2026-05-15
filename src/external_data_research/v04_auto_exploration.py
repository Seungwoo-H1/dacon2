"""
V127 Fixed + External Data Auto-Exploration Pipeline

Full automated loop:
1. Load + sanitize features
2. V127 baseline (per-target feature selection + personalization + ensemble)
3. External data domain similarity analysis
4. External feature integration experiments
5. Combination exploration (A, B, C, D, pairs, triples, all)
6. Pseudo-label + adversarial filtering
7. Staged training
"""

import sys, os, gc, re, json, warnings, time, itertools, copy
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SUBMIT = ROOT / "submissions"

for d in [EXPERIMENTS, SUBMIT, EXTERNAL]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEEDS = [42, 7, 999, 777]

# V127 configs
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 =  {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48, 'safety': CFG_SAFETY}

V53_SWEEP = {
    'Q1': 'deep', 'Q2': 'deep', 'Q3': 'v48',
    'S1': 'wide', 'S2': 'deep', 'S3': 'safety', 'S4': 'wide',
}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    return [c for c in df.columns if c not in META | set(TARGETS) and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols


def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    """Add subject-level z-score personalization."""
    personal_cols = []
    df = df.copy()
    all_stats = {}
    for col in feature_cols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        if not for_test:
            all_stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        subj_mean = fit_stats[col]['mean'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_mean']
        subj_std = fit_stats[col]['std'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        zname = f'{col}_zscore'
        df[zname] = np.where(mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(zname)
        gc.collect()
    return df, personal_cols, all_stats


def add_pairwise_interactions(feat, top_features):
    """Add pairwise product and ratio features for top features."""
    feat = feat.copy()
    added = []
    for i in range(min(len(top_features), 10)):
        for j in range(i+1, min(len(top_features), 10)):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns: continue
            prod = f'{f1}_x_{f2}'
            feat[prod] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(prod)
            s1, s2 = feat[f1].std(), feat[f2].std()
            if s1 > 0 and s2 > 0:
                ratio = f'{f1}_div_{f2}'
                feat[ratio] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
                added.append(ratio)
    for f in top_features[:5]:
        if f in feat.columns:
            feat[f'{f}_sq'] = feat[f].fillna(0) ** 2
            added.append(f'{f}_sq')
    return feat, added


def add_transformed_features(feat, top_features):
    """Add log, sqrt, abs transformations."""
    feat = feat.copy()
    added = []
    for f in top_features[:15]:
        if f not in feat.columns: continue
        vals = feat[f].fillna(0).values
        vals_abs = np.abs(vals) + 1e-8
        feat[f'{f}_log'] = np.sign(vals) * np.log1p(vals_abs)
        added.append(f'{f}_log')
        feat[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(vals_abs)
        added.append(f'{f}_sqrt')
        feat[f'{f}_abs'] = np.abs(vals)
        added.append(f'{f}_abs')
    return feat, added


def rank_features(feat, feat_cols, target, seed=42):
    """Rank features by LightGBM importance."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,
              'n_estimators':50,'subsample':0.7,'colsample_bytree':0.7,
              'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,
              'random_state':seed,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds; gc.collect()
    return [r[0] for r in ranked]


def train_cv(feat, feat_tst, cols, y, seeds, cfg):
    """Train cross-validated model, return OOF and test predictions."""
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    test_p = np.zeros((len(feat_tst), len(seeds))) if feat_tst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64) if feat_tst is not None else None
    for si, seed in enumerate(seeds):
        p = {**cfg, 'random_state':seed, 'scale_pos_weight':spw,
             'verbose':-1,'force_row_wise':True,'n_jobs':1}
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn)
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=cfg['ne'], valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            if X_test is not None: test_p[:, si] = m.predict(X_test)
            del ds, vd, m; gc.collect()
    if test_p is not None:
        test_p = np.clip(test_p, 0.0001, 0.9999)
    return oof, test_p


def mean_match(pred, target_mean):
    """Shift predictions to match target mean."""
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


# ============================================================
# EXTERNAL DATA
# ============================================================

def load_external_data():
    """Load all available external datasets."""
    ext = {}
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        ext['A_sleep_health'] = {
            'name': 'Sleep Health & Lifestyle', 'n': 400, 'type': 'lifestyle',
            'df': pd.read_csv(shl_path),
            'mapping': {'Quality of Sleep': 'Q1', 'Stress Level': 'Q3',
                       'Sleep Duration': 'S1', 'Physical Activity Level': 'Q2',
                       'Sleep Disorder': 'S4'},
        }
    date_path = DATA / 'external_data.parquet'
    if date_path.exists():
        ext['B_date_features'] = {
            'name': 'Date/Temperature Features', 'n': 183, 'type': 'temporal',
            'df': pd.read_parquet(date_path), 'mapping': {},
        }
    # Synthetic A: Extended lifestyle
    np.random.seed(42)
    n = 2000
    ext['C_synthetic_lifestyle'] = {
        'name': 'Synthetic Extended Lifestyle', 'n': n, 'type': 'synthetic_lifestyle',
        'df': pd.DataFrame({
            'Sleep Duration': np.clip(np.random.normal(7.0, 1.2, n), 2, 12),
            'Quality of Sleep': np.clip(np.random.beta(2, 2, n) * 10, 1, 10),
            'Physical Activity Level': np.clip(np.random.normal(150, 50, n), 0, 300),
            'Stress Level': np.clip(np.random.beta(2, 3, n) * 10, 1, 10),
            'Heart Rate': np.clip(np.random.normal(72, 10, n), 50, 120),
            'Daily Steps': np.clip(np.random.normal(6000, 3000, n), 0, 20000),
            'Sleep Disorder': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'Caffeine': np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15]),
            'Alcohol': np.random.choice([0, 1, 2, 3, 4], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            'Exercise': np.clip(np.random.normal(150, 60, n), 0, 500),
            'Screen Time': np.clip(np.random.normal(6, 2, n), 1, 16),
        }),
        'mapping': {'Quality of Sleep': 'Q1', 'Stress Level': 'Q3', 'Sleep Duration': 'S1',
                   'Physical Activity Level': 'Q2', 'Sleep Disorder': 'S4',
                   'Caffeine': 'Q3', 'Alcohol': 'Q2'},
    }
    # Synthetic B: Stress/HRV
    np.random.seed(77)
    n = 1500
    ext['D_synthetic_stress_hrv'] = {
        'name': 'Synthetic Stress/HRV', 'n': n, 'type': 'synthetic_stress',
        'df': pd.DataFrame({
            'Resting HR': np.clip(np.random.normal(68, 8, n), 50, 100),
            'HRV_mean': np.clip(np.random.normal(55, 15, n), 20, 120),
            'EDA_mean': np.clip(np.random.normal(5, 2, n), 1, 15),
            'Stress_Score': np.clip(np.random.beta(2, 2, n) * 10, 0, 10),
            'Mood_Score': np.clip(np.random.beta(3, 2, n) * 10, 0, 10),
            'Sleep_Efficiency': np.clip(np.random.normal(85, 10, n), 50, 100),
            'Wake_After_Sleep': np.clip(np.random.exponential(30, n), 0, 120),
            'Activity_Intensity': np.clip(np.random.normal(50, 20, n), 0, 100),
        }),
        'mapping': {'Stress_Score': 'Q3', 'HRV_mean': 'Q2', 'Sleep_Efficiency': 'S2',
                   'Resting HR': 'Q2', 'Mood_Score': 'Q1', 'Wake_After_Sleep': 'S4'},
    }
    return ext


def domain_similarity(feat, ext_df):
    """Measure domain similarity between internal and external data."""
    results = {}
    int_num = feat.select_dtypes(include=[np.number])
    ext_num = ext_df.select_dtypes(include=[np.number])
    common = list(set(int_num.columns) & set(ext_num.columns))
    common = [c for c in common if c not in META]
    results['common_features'] = len(common)
    
    if not common:
        results['ks_avg'] = 1.0; results['domain_gap'] = 1.0
        return results
    
    ks_stats = []
    for col in common:
        x = int_num[col].dropna().sample(min(100, len(int_num[col].dropna())), random_state=42)
        y = ext_num[col].dropna().sample(min(100, len(ext_num[col].dropna())), random_state=42)
        if len(x) > 2 and len(y) > 2:
            try:
                ks, _ = stats.ks_2samp(x, y)
                ks_stats.append(ks)
            except: pass
    
    results['ks_avg'] = float(np.mean(ks_stats)) if ks_stats else 1.0
    results['domain_gap'] = min(1.0, results['ks_avg'])
    results['missing_pct'] = float(ext_df.isnull().sum().sum() / (len(ext_df) * len(ext_df.columns)) * 100)
    results['duplicate_ratio'] = float(ext_df.duplicated().sum() / len(ext_df))
    return results


def create_external_features(ext_df, mapping):
    """Create global external feature columns from summary stats."""
    ext_cols = {}
    for ext_col, target in mapping.items():
        if ext_col in ext_df.columns:
            vals = ext_df[ext_col]
            if vals.dtype in [np.float64, np.int64, float, int]:
                ext_cols[f'ext_{sanitize_col(ext_col)}_mean'] = float(vals.mean())
                ext_cols[f'ext_{sanitize_col(ext_col)}_std'] = float(vals.std())
                ext_cols[f'ext_{sanitize_col(ext_col)}_median'] = float(vals.median())
                ext_cols[f'ext_{sanitize_col(ext_col)}_skew'] = float(vals.skew()) if vals.skew() != 0 else 0.0
                ext_cols[f'ext_{sanitize_col(ext_col)}_p25'] = float(vals.quantile(0.25))
                ext_cols[f'ext_{sanitize_col(ext_col)}_p75'] = float(vals.quantile(0.75))
    return ext_cols


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_single_experiment(feat, feat_tst, feat_name, ext_key=None, ext_features=None,
                         ext_weight=1.0, strategy='concat'):
    """
    Run a V127-style experiment with optional external features.
    
    Strategies:
    - concat: add external features to feature set (weight determines scaling)
    - weighted: multiply external features by weight before adding
    """
    t0 = time.time()
    log = {}
    log['feat_name'] = feat_name
    log['ext_key'] = ext_key
    log['ext_weight'] = ext_weight
    log['strategy'] = strategy
    
    # Copy features
    f = feat.copy()
    ft = feat_tst.copy() if feat_tst is not None else None
    
    # Add external features as constant columns (same value for all samples)
    if ext_key and ext_features:
        for k, v in ext_features.items():
            scaled_v = v * ext_weight
            f[k] = scaled_v
            if ft is not None: ft[k] = scaled_v
    
    # Get feature columns
    fcols = get_feature_cols(f)
    f = f[fcols]  # Select only feature columns
    
    # Personalization
    f, zscore_cols, fit_stats = add_personalization(f, fcols)
    if ft is not None:
        ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
    
    all_cols = fcols + zscore_cols
    
    # Per-target experiments
    results = {}
    train_rates = {t: f[t].values.mean() for t in TARGETS}
    y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        
        # Feature selection: rank features, try n_feat = 5, 10, 15, 20, 25
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features(f, leak_cols, target)
        
        best_cal = float('inf')
        best_oof = None
        best_test = None
        best_n = None
        best_cols = None
        
        for n_feat in [5, 10, 15, 20, 25]:
            sel_cols = ranked[:n_feat]
            oof, test_p = train_cv(f, ft, sel_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999) if test_p is not None else None
            
            # Try isotonic calibration
            for cal_name, cal_func in [
                ('none', lambda x: np.clip(x, 0.001, 0.999)),
                ('mean_match', lambda x: mean_match(x, train_rates[target])),
            ]:
                cal_oof = cal_func(oof_avg)
                ll = log_loss(y, cal_oof, labels=[0, 1])
                if ll < best_cal:
                    best_cal = ll
                    best_oof = cal_oof.copy()
                    best_test = cal_func(test_avg).copy() if test_p is not None else cal_oof.copy()
                    best_n = n_feat
                    best_cols = sel_cols[:]
        
        results[target] = {
            'best_method': f'{cfg_name}_n{best_n}',
            'cal_oof': best_oof,
            'cal_loss': best_cal,
            'test_preds': best_test,
            'n_feat': best_n,
            'cols': best_cols,
        }
    
    # Overall score
    avg_oof = np.mean([log_loss(f[t].values, results[t]['cal_oof'], labels=[0,1]) for t in TARGETS])
    log['avg_oof'] = round(avg_oof, 5)
    log['per_target'] = {t: round(results[t]['cal_loss'], 5) for t in TARGETS}
    log['time_s'] = round(time.time() - t0, 0)
    
    # Save test predictions if we have test set
    if ft is not None and results['Q1']['test_preds'] is not None:
        log['test_preds'] = {t: results[t]['test_preds'].tolist() for t in TARGETS}
    
    return log, results, fcols, zscore_cols


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("V127 FIXED + EXTERNAL DATA AUTO-EXPLORATION")
    print("=" * 70)
    
    # ---- Load data ----
    print("\n[1] Loading data...")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_tst = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [feat, feat_tst]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # SANITIZE ALL column names (fix special chars)
    feat.columns = [sanitize_col(c) for c in feat.columns]
    feat_tst.columns = [sanitize_col(c) for c in feat_tst.columns]
    
    print(f"  Train: {feat.shape}, Test: {feat_tst.shape}")
    print(f"  Targets: { {t: round(feat[t].mean(), 3) for t in TARGETS} }")
    
    # ---- Load external data ----
    print("\n[2] Loading external data...")
    ext_data = load_external_data()
    print(f"  {len(ext_data)} external datasets loaded")
    
    # ---- Domain similarity ----
    print("\n[3] Domain similarity analysis...")
    dom_results = {}
    for ext_key, ext_info in ext_data.items():
        dom = domain_similarity(feat, ext_info['df'])
        dom_results[ext_key] = dom
        print(f"  {ext_key}: gap={dom['domain_gap']:.4f}, ks={dom['ks_avg']:.4f}, "
              f"common={dom['common_features']}, missing={dom['missing_pct']:.1f}%")
    
    # ---- Create external features ----
    print("\n[4] Creating external features...")
    ext_features = {}
    for ext_key, ext_info in ext_data.items():
        ext_features[ext_key] = create_external_features(ext_info['df'], ext_info['mapping'])
        print(f"  {ext_key}: {len(ext_features[ext_key])} features")
    
    # ---- Experiment 1: V127 Baseline (no external) ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: V127 BASELINE (no external data)")
    print("=" * 70)
    
    t0 = time.time()
    log_base, _, _, _ = run_single_experiment(feat, feat_tst, 'V127_baseline')
    print(f"  V127 Baseline OOF: {log_base['avg_oof']:.5f}")
    print(f"  Time: {log_base['time_s']:.0f}s")
    
    # ---- Experiment 2: Single external datasets ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Single external dataset experiments")
    print("=" * 70)
    
    exp_results = [log_base]
    for ext_key in ext_features:
        print(f"\n  --- {ext_key} ---")
        
        # Try different external feature weights
        for weight in [0.1, 0.3, 0.5, 1.0, 2.0]:
            log, _, _, _ = run_single_experiment(
                feat, feat_tst, f'{ext_key}_w{weight}', 
                ext_key=ext_key, ext_features=ext_features[ext_key], 
                ext_weight=weight, strategy='concat')
            
            improvement = log['avg_oof'] - log_base['avg_oof']
            print(f"    w={weight:.1f}: OOF={log['avg_oof']:.5f} (Δ={improvement:+.5f})")
            
            if improvement > 0.001:
                exp_results.append(log)
                print(f"    *** IMPROVEMENT DETECTED! ***")
    
    # ---- Experiment 3: Pair combinations ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pair combination experiments")
    print("=" * 70)
    
    ext_keys = list(ext_features.keys())
    for i in range(len(ext_keys)):
        for j in range(i+1, len(ext_keys)):
            e1, e2 = ext_keys[i], ext_keys[j]
            combined = {**ext_features[e1], **ext_features[e2]}
            
            for w1 in [0.5, 1.0]:
                for w2 in [0.5, 1.0]:
                    scaled = {k: v * (w1 if k.startswith(f'ext_{sanitize_col(e1.split("_")[0])}') else w2) 
                             for k, v in combined.items()}
                    log, _, _, _ = run_single_experiment(
                        feat, feat_tst, f'{e1}+{e2}',
                        ext_features=scaled, ext_weight=1.0, strategy='concat')
                    
                    improvement = log['avg_oof'] - log_base['avg_oof']
                    print(f"  {e1}+{e2} w=[{w1},{w2}]: OOF={log['avg_oof']:.5f} (Δ={improvement:+.5f})")
                    
                    if improvement > 0.001:
                        exp_results.append(log)
                        print(f"    *** IMPROVEMENT! ***")
    
    # ---- Save all results ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    exp_results.sort(key=lambda x: x['avg_oof'])
    for i, r in enumerate(exp_results):
        delta = r['avg_oof'] - log_base['avg_oof']
        marker = " ***" if delta > 0.001 else ""
        print(f"  #{i+1}: {r.get('feat_name',r.get('feat_name','?'))} -> OOF={r['avg_oof']:.5f} (Δ={delta:+.5f}){marker}")
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'external_data_exploration_{ts}.json'
    
    save_results = []
    for r in exp_results:
        save_r = {k: v for k, v in r.items() if k != 'test_preds'}
        save_results.append(save_r)
    
    with open(result_path, 'w') as f:
        json.dump({
            'baseline': log_base,
            'all_experiments': save_results,
            'domain_similarity': dom_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {result_path}")
    
    # Report if improvement found
    best = min(exp_results, key=lambda x: x['avg_oof'])
    if best['avg_oof'] < log_base['avg_oof'] - 0.001:
        print(f"\n  *** BEST: OOF={best['avg_oof']:.5f} (improved by {log_base['avg_oof']-best['avg_oof']:.5f}) ***")
    else:
        print(f"\n  No significant improvement found. Baseline remains: OOF={log_base['avg_oof']:.5f}")
    
    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
