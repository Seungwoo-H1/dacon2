"""
V127 External Data V2 - External data as GLOBAL FEATURES

Key insight: external datasets (Sleep Health Kaggle, WESAD, etc.) have COMPLETELY
DIFFERENT schemas from internal data. They share NO column names.
This means external data cannot be merged row-by-row.

Instead, we use external data to derive GLOBAL statistics that serve as
contextual features for ALL samples. This is analogous to using population-level
statistics to inform individual predictions.

Approach:
1. Compute summary statistics from each external dataset
2. Add them as constant columns (same value for all rows)
3. Train with these global features alongside internal features
4. Since external data has DIFFERENT domain, the global features act as
   prior knowledge about normal ranges for sleep/health metrics
"""

import sys, os, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
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
    """Get feature columns, excluding metadata, targets, and personalization intermediate cols."""
    exclude = META | set(TARGETS) | {'subject_id'} | {
        f'_subj_mean', f'_subj_std'  # placeholder for suffix matching
    }
    return [c for c in df.columns 
            if c not in exclude 
            and not c.endswith('_subj_mean') 
            and not c.endswith('_subj_std') 
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols


def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    """Add subject-level z-score personalization. Drops intermediate _subj_mean/_subj_std cols."""
    personal_cols = []
    df = df.copy()
    all_stats = {}
    subj_cols_added = []  # Track columns to drop after use
    for col in feature_cols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        subj_cols_added.extend([f'{col}_subj_mean', f'{col}_subj_std'])
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
    # Drop intermediate subj_mean/subj_std columns
    drop_cols = [c for c in subj_cols_added if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, personal_cols, all_stats


def rank_features(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,
              'n_estimators':50,'subsample':0.7,'colsample_bytree':0.7,
              'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,
              'random_state':seed,'min_child_samples':10,'force_row_wise':True,
              'n_jobs':1,'feature_pre_filter':False,'deterministic':True}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds; gc.collect()
    return [r[0] for r in ranked]


def cfg_to_params(cfg_short, seed, spw):
    """Convert short-form cfg (nl/md/lr/ne/ss/cb/ra/rl/mc) to full LGBM params."""
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'force_row_wise': True,
        'feature_pre_filter': False,
        'num_leaves': int(cfg_short['nl']),
        'max_depth': int(cfg_short['md']),
        'learning_rate': float(cfg_short['lr']),
        'n_estimators': int(cfg_short['ne']),
        'subsample': float(cfg_short['ss']),
        'colsample_bytree': float(cfg_short['cb']),
        'reg_alpha': float(cfg_short['ra']),
        'reg_lambda': float(cfg_short['rl']),
        'min_child_samples': max(1, int(cfg_short['mc'])),
        'scale_pos_weight': spw,
        'random_state': seed,
        'n_jobs': 1,
    }


def train_cv(feat, feat_tst, cols, y, seeds, cfg):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    test_p = np.zeros((len(feat_tst), len(seeds))) if feat_tst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64) if feat_tst is not None else None
    n_rounds = int(cfg['ne'])
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn)
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=n_rounds, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            if X_test is not None: test_p[:, si] = m.predict(X_test)
            del ds, vd, m; gc.collect()
    if test_p is not None:
        test_p = np.clip(test_p, 0.0001, 0.9999)
    return oof, test_p


def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


# ============================================================
# EXTERNAL DATA FEATURE GENERATION
# ============================================================

def create_global_features(feat, ext_name, ext_df):
    """
    Create global statistical features from external data.
    
    Strategy: Use external data to compute NORMAL RANGE statistics
    for health metrics, then express internal features as DEVIATIONS
    from these external normals.
    
    This transforms external data from "unmergeable" into
    "informative prior" features.
    """
    new_cols = {}
    num_cols = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in num_cols:
        vals = ext_df[col].dropna()
        if len(vals) < 10:
            continue
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_mean'] = float(vals.mean())
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_std'] = float(vals.std())
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_median'] = float(vals.median())
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_p25'] = float(vals.quantile(0.25))
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_p75'] = float(vals.quantile(0.75))
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_min'] = float(vals.min())
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_max'] = float(vals.max())
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_skew'] = float(vals.skew()) if vals.skew() != 0 else 0.0
        new_cols[f'ext_{ext_name}_{sanitize_col(col)}_kurt'] = float(vals.kurtosis()) if vals.kurtosis() != 0 else 0.0
    
    # Cross-feature ratios from external data (these capture domain relationships)
    if len(num_cols) >= 2:
        # Ratio of key metrics
        for i in range(min(len(num_cols), 8)):
            for j in range(i+1, min(len(num_cols), 8)):
                c1, c2 = num_cols[i], num_cols[j]
                v1, v2 = ext_df[c1].dropna(), ext_df[c2].dropna()
                if len(v1) >= 20 and len(v2) >= 20:
                    common_idx = v1.index.intersection(v2.index)
                    if len(common_idx) >= 20:
                        try:
                            r = v1[common_idx] / (v2[common_idx] + 1e-8)
                            new_cols[f'ext_{ext_name}_{sanitize_col(c1)}_div_{sanitize_col(c2)}_mean'] = float(r.mean())
                            new_cols[f'ext_{ext_name}_{sanitize_col(c1)}_div_{sanitize_col(c2)}_std'] = float(r.std())
                        except: pass
    
    # Correlation between features in external data (capturing domain structure)
    if len(num_cols) >= 3:
        corr_matrix = ext_df[num_cols[:8]].corr()
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    c1, c2 = num_cols[i], num_cols[j]
                    new_cols[f'ext_{ext_name}_corr_{sanitize_col(c1)}_{sanitize_col(c2)}'] = float(corr_val)
    
    return new_cols


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(feat, feat_tst, exp_id, external_features=None, weight=1.0):
    """Run a V127-style experiment with optional external features."""
    t0 = time.time()
    
    # Copy features
    f = feat.copy()
    ft = feat_tst.copy()
    
    # Add external features as constant columns
    if external_features:
        for k, v in external_features.items():
            f[k] = v * weight
            ft[k] = v * weight
    
    # Get feature columns (exclude targets and metadata)
    fcols = get_feature_cols(f)
    
    # Personalization
    f, zscore_cols, fit_stats = add_personalization(f, fcols)
    ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
    
    all_cols = fcols + zscore_cols
    
    # Remove external columns that are constant (all same value)
    # LGBM can still use them but they won't help split
    non_const = []
    for c in all_cols:
        if f[c].std() > 0:
            non_const.append(c)
    
    # Per-target experiments
    results = {}
    train_rates = {t: f[t].values.mean() for t in TARGETS}
    y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        
        # Feature selection
        leak_cols = remove_leak(non_const, target)
        ranked = rank_features(f, leak_cols, target)
        
        best_cal = float('inf')
        best_oof = None
        best_test = None
        best_n = None
        
        for n_feat in [5, 10, 15, 20, 25]:
            sel_cols = ranked[:n_feat]
            oof, test_p = train_cv(f, ft, sel_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            for cal_name, cal_func in [
                ('none', lambda x: np.clip(x, 0.001, 0.999)),
                ('mean_match', lambda x: mean_match(x, train_rates[target])),
            ]:
                cal_oof = cal_func(oof_avg)
                ll = log_loss(y, cal_oof, labels=[0, 1])
                if ll < best_cal:
                    best_cal = ll
                    best_oof = cal_oof.copy()
                    best_test = cal_func(test_avg).copy()
                    best_n = n_feat
        
        results[target] = {
            'best_method': f'{cfg_name}_n{best_n}',
            'cal_oof': best_oof,
            'cal_loss': best_cal,
            'test_preds': best_test,
            'n_feat': best_n,
        }
    
    avg_oof = np.mean([log_loss(f[t].values, results[t]['cal_oof'], labels=[0,1]) for t in TARGETS])
    
    log = {
        'exp_id': exp_id,
        'avg_oof': round(avg_oof, 5),
        'per_target': {t: round(results[t]['cal_loss'], 5) for t in TARGETS},
        'time_s': round(time.time() - t0, 0),
    }
    
    return log, results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("V127 EXTERNAL DATA V2 - GLOBAL FEATURES APPROACH")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_tst = pd.read_parquet(DATA / "test_features.parquet")
    for df in [feat, feat_tst]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    feat.columns = [sanitize_col(c) for c in feat.columns]
    feat_tst.columns = [sanitize_col(c) for c in feat_tst.columns]
    print(f"  Train: {feat.shape}, Test: {feat_tst.shape}")
    
    # External data
    print("\n[2] Loading external data...")
    external_datasets = {}
    
    # A: Sleep Health & Lifestyle
    if (EXTERNAL / 'sleep_health_lifestyle.csv').exists():
        df = pd.read_csv(EXTERNAL / 'sleep_health_lifestyle.csv')
        # Select numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        external_datasets['A_sleep_health'] = df[num_cols]
        print(f"  A: {df.shape}, numeric cols: {num_cols[:10]}")
    
    # B: Date features
    if (DATA / 'external_data.parquet').exists():
        df = pd.read_parquet(DATA / 'external_data.parquet')
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        external_datasets['B_date'] = df[num_cols]
        print(f"  B: {df.shape}, numeric cols: {num_cols}")
    
    # C: Synthetic extended lifestyle
    np.random.seed(42)
    n = 2000
    ext_c = pd.DataFrame({
        'Sleep Duration': np.clip(np.random.normal(7.0, 1.2, n), 2, 12),
        'Quality of Sleep': np.clip(np.random.beta(2, 2, n) * 10, 1, 10),
        'Physical Activity': np.clip(np.random.normal(150, 50, n), 0, 300),
        'Stress Level': np.clip(np.random.beta(2, 3, n) * 10, 1, 10),
        'Heart Rate': np.clip(np.random.normal(72, 10, n), 50, 120),
        'Daily Steps': np.clip(np.random.normal(6000, 3000, n), 0, 20000),
        'Sleep Disorder': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'Caffeine': np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.25, 0.15]),
        'Alcohol': np.random.choice([0, 1, 2, 3, 4], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'Exercise': np.clip(np.random.normal(150, 60, n), 0, 500),
        'Screen Time': np.clip(np.random.normal(6, 2, n), 1, 16),
        'BMI': np.clip(np.random.normal(25, 5, n), 15, 50),
    })
    external_datasets['C_synthetic_extended'] = ext_c
    print(f"  C: {ext_c.shape}")
    
    # D: Synthetic stress/HRV
    np.random.seed(77)
    n = 1500
    ext_d = pd.DataFrame({
        'Resting HR': np.clip(np.random.normal(68, 8, n), 50, 100),
        'HRV_mean': np.clip(np.random.normal(55, 15, n), 20, 120),
        'EDA_mean': np.clip(np.random.normal(5, 2, n), 1, 15),
        'Stress_Score': np.clip(np.random.beta(2, 2, n) * 10, 0, 10),
        'Mood_Score': np.clip(np.random.beta(3, 2, n) * 10, 0, 10),
        'Sleep_Efficiency': np.clip(np.random.normal(85, 10, n), 50, 100),
        'Wake_After_Sleep': np.clip(np.random.exponential(30, n), 0, 120),
        'Activity_Intensity': np.clip(np.random.normal(50, 20, n), 0, 100),
    })
    external_datasets['D_synthetic_stress'] = ext_d
    print(f"  D: {ext_d.shape}")
    
    # Create global features for each external dataset
    print("\n[3] Creating global features...")
    global_features = {}
    for ext_name, ext_df in external_datasets.items():
        gf = create_global_features(feat, ext_name, ext_df)
        global_features[ext_name] = gf
        print(f"  {ext_name}: {len(gf)} features")
        # Show a few
        for k in list(gf.keys())[:3]:
            print(f"    {k}: {gf[k]:.4f}")
    
    # Run experiments
    print("\n" + "=" * 70)
    print("EXPERIMENTS")
    print("=" * 70)
    
    all_results = []
    
    # Experiment 1: Baseline (no external)
    print("\n--- Baseline (V127, no external) ---")
    log_base, _ = run_experiment(feat, feat_tst, 'baseline')
    print(f"  OOF: {log_base['avg_oof']:.5f}")
    all_results.append(log_base)
    
    # Experiment 2: Single external datasets
    for ext_name in global_features:
        for weight in [0.1, 0.3, 0.5, 1.0]:
            print(f"\n--- {ext_name} w={weight} ---")
            log, _ = run_experiment(feat, feat_tst, f'{ext_name}_w{weight}',
                                   external_features=global_features[ext_name], weight=weight)
            delta = log['avg_oof'] - log_base['avg_oof']
            print(f"  OOF: {log['avg_oof']:.5f} (Δ={delta:+.5f})")
            all_results.append(log)
    
    # Experiment 3: Pair combinations
    ext_names = list(global_features.keys())
    for i in range(len(ext_names)):
        for j in range(i+1, len(ext_names)):
            e1, e2 = ext_names[i], ext_names[j]
            combined = {**global_features[e1], **global_features[e2]}
            print(f"\n--- {e1}+{e2} ---")
            log, _ = run_experiment(feat, feat_tst, f'{e1}+{e2}',
                                   external_features=combined)
            delta = log['avg_oof'] - log_base['avg_oof']
            print(f"  OOF: {log['avg_oof']:.5f} (Δ={delta:+.5f})")
            all_results.append(log)
    
    # Experiment 4: All combined
    all_ext = {}
    for gf in global_features.values():
        all_ext.update(gf)
    print(f"\n--- All external combined ---")
    log, _ = run_experiment(feat, feat_tst, 'all_external',
                           external_features=all_ext)
    delta = log['avg_oof'] - log_base['avg_oof']
    print(f"  OOF: {log['avg_oof']:.5f} (Δ={delta:+.5f})")
    all_results.append(log)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by OOF)")
    print("=" * 70)
    all_results.sort(key=lambda x: x['avg_oof'])
    for r in all_results:
        delta = r['avg_oof'] - log_base['avg_oof']
        marker = " ***" if delta < -0.001 else ""
        print(f"  {r['exp_id']:30s} OOF={r['avg_oof']:.5f} (Δ={delta:+.5f}){marker}")
    
    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'external_v2_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump({
            'baseline': log_base,
            'all_experiments': all_results,
            'global_features': {k: len(v) for k, v in global_features.items()},
        }, f, indent=2, default=str)
    print(f"\n  Saved: {result_path}")


if __name__ == '__main__':
    main()
