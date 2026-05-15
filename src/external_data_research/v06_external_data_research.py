"""
V06: External Data Research — Automated Combination Loop

Objective: Find external data combinations that improve LB generalization.
Approach:
1. Load external data (A: sleep_health, B: date_features, C: synthetic_extended, D: synthetic_stress)
2. Create proxy features from external data (global stats, derived features)
3. Test all subset combinations: A, B, C, D, A+B, ..., A+B+C+D
4. Test multiple strategies per combination: concat, weighted, staged
5. Track: domain similarity, data quality, transferability

Key insight: External data cannot be merged row-by-row with internal data (different subjects,
different feature schemas). Instead, we extract global statistics from external data and use
them as "contextual priors" — constant columns added to every training sample.

Strategy:
- Global stats: mean, std, median, p25, p75, skew, kurt for each external numeric feature
- Cross-feature ratios: external feature ratios (e.g., Quality/Sleep_Duration)
- Correlation structure: external feature correlations (domain structure capture)
- Feature engineering: map external features to internal target domains (Q1-Q3, S1-S4)
"""

import sys, os, gc, re, json, warnings, time, itertools
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
EXTERNAL = ROOT / "external_data"
EXPERIMENTS = ROOT / "experiments"
SUBMIT = ROOT / "submissions"

for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEEDS = [42, 7, 999, 777]

# Configs
CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5}
CFG_DEEP  = {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15}
CFG_V48   = {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10}
CFG_SAFETY = {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20}
CFGS = {'wide':CFG_WIDE,'deep':CFG_DEEP,'v48':CFG_V48,'safety':CFG_SAFETY}

V53_SWEEP = {
    'Q1':'deep','Q2':'deep','Q3':'v48',
    'S1':'wide','S2':'deep','S3':'safety','S4':'wide',
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


# ============================================================
# Core utilities
# ============================================================

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    exclude = META | set(TARGETS) | {'subject_id'}
    return [c for c in df.columns
            if c not in exclude
            and not c.endswith('_subj_mean')
            and not c.endswith('_subj_std')
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    personal_cols = []
    df = df.copy()
    all_stats = {}
    subj_cols = []
    for col in feature_cols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        subj_cols.extend([f'{col}_subj_mean', f'{col}_subj_std'])
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
    drop = [c for c in subj_cols if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df, personal_cols, all_stats

def rank_features(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,
        'n_estimators':50,'subsample':0.7,'colsample_bytree':0.7,
        'reg_alpha':1.0,'reg_lambda':3.0,'scale_pos_weight':spw,
        'random_state':seed,'min_child_samples':10,
        'force_row_wise':True,'n_jobs':1,
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

def cfg_to_params(cfg_short, seed, spw):
    return {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':int(cfg_short['nl']),'max_depth':int(cfg_short['md']),
        'learning_rate':float(cfg_short['lr']),'n_estimators':int(cfg_short['ne']),
        'subsample':float(cfg_short['ss']),'colsample_bytree':float(cfg_short['cb']),
        'reg_alpha':float(cfg_short['ra']),'reg_lambda':float(cfg_short['rl']),
        'min_child_samples':max(1,int(cfg_short['mc'])),
        'scale_pos_weight':spw,'random_state':seed,
        'force_row_wise':True,'n_jobs':1,
    }

def train_cv(feat, feat_tst, cols, y, seeds, cfg):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    test_p = np.zeros((len(feat_tst), len(seeds))) if feat_tst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
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
            del ds, vd, m
            gc.collect()
    if test_p is not None:
        test_p = np.clip(test_p, 0.0001, 0.9999)
    return oof, test_p


# ============================================================
# External data loading & processing
# ============================================================

def load_external_data():
    """Load all available external datasets, return dict of DataFrames."""
    ext = {}
    
    # A: Sleep Health & Lifestyle (Kaggle)
    shl_path = EXTERNAL / 'sleep_health_lifestyle.csv'
    if shl_path.exists():
        df = pd.read_csv(shl_path)
        # Select numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ext['A_sleep_health'] = df[num_cols]
        print(f"  A: {df.shape} — {num_cols}")
    
    # B: Date/Temperature features
    date_path = DATA / 'external_data.parquet'
    if date_path.exists():
        df = pd.read_parquet(date_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ext['B_date_features'] = df[num_cols]
        print(f"  B: {df.shape} — {num_cols}")
    
    return ext


def create_external_global_features(ext_df, prefix):
    """
    Create global statistical features from external data.
    
    For each numeric column:
    - Basic stats: mean, std, median, p25, p75, min, max, skew, kurt
    - Cross-feature ratios (captures relationships in external data)
    - Correlation matrix entries (captures domain structure)
    """
    new_cols = {}
    num_cols = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Basic stats per column
    for col in num_cols:
        vals = ext_df[col].dropna()
        if len(vals) < 10:
            continue
        safe_col = sanitize_col(col)
        new_cols[f'{prefix}_{safe_col}_mean'] = float(vals.mean())
        new_cols[f'{prefix}_{safe_col}_std'] = float(vals.std())
        new_cols[f'{prefix}_{safe_col}_median'] = float(vals.median())
        new_cols[f'{prefix}_{safe_col}_p25'] = float(vals.quantile(0.25))
        new_cols[f'{prefix}_{safe_col}_p75'] = float(vals.quantile(0.75))
        new_cols[f'{prefix}_{safe_col}_min'] = float(vals.min())
        new_cols[f'{prefix}_{safe_col}_max'] = float(vals.max())
        if vals.skew() != 0:
            new_cols[f'{prefix}_{safe_col}_skew'] = float(vals.skew())
        if vals.kurtosis() != 0:
            new_cols[f'{prefix}_{safe_col}_kurt'] = float(vals.kurtosis())
    
    # 2. Cross-feature ratios (domain relationships)
    for i in range(min(len(num_cols), 8)):
        for j in range(i+1, min(len(num_cols), 8)):
            c1, c2 = num_cols[i], num_cols[j]
            v1, v2 = ext_df[c1].dropna(), ext_df[c2].dropna()
            if len(v1) >= 20 and len(v2) >= 20:
                common_idx = v1.index.intersection(v2.index)
                if len(common_idx) >= 20:
                    try:
                        r = v1[common_idx] / (v2[common_idx] + 1e-8)
                        new_cols[f'{prefix}_{sanitize_col(c1)}_div_{sanitize_col(c2)}_mean'] = float(r.mean())
                        new_cols[f'{prefix}_{sanitize_col(c1)}_div_{sanitize_col(c2)}_std'] = float(r.std())
                    except: pass
    
    # 3. Correlation structure
    if len(num_cols) >= 3:
        corr = ext_df[num_cols[:8]].corr()
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                cv = corr.iloc[i, j]
                if not np.isnan(cv):
                    new_cols[f'{prefix}_corr_{sanitize_col(num_cols[i])}_{sanitize_col(num_cols[j])}'] = float(cv)
    
    return new_cols


# ============================================================
# Experiment runner
# ============================================================

def run_v127_experiment(feat, feat_tst, exp_id, ext_features=None, ext_weight=1.0):
    """Run full V127-style experiment with optional external global features."""
    t0 = time.time()
    
    # Copy
    f = feat.copy()
    ft = feat_tst.copy()
    
    # Add external global features as constant columns
    if ext_features:
        for k, v in ext_features.items():
            f[k] = v * ext_weight
            ft[k] = v * ext_weight
    
    # Feature engineering
    fcols = get_feature_cols(f)
    f, zscore_cols, fit_stats = add_personalization(f, fcols)
    ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
    all_cols = fcols + zscore_cols
    
    # Remove constant external columns
    non_const = [c for c in all_cols if f[c].std() > 0]
    
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
            
            cal_oof = mean_match(oof_avg, train_rates[target])
            ll = log_loss(y, cal_oof, labels=[0, 1])
            if ll < best_cal:
                best_cal = ll
                best_oof = cal_oof.copy()
                best_test = mean_match(test_avg, train_rates[target]).copy()
                best_n = n_feat
        
        avg_ll = log_loss(f[target].values, best_oof, labels=[0, 1])
        results[target] = {
            'best_method': f'{cfg_name}_n{best_n}',
            'cal_oof': best_oof,
            'cal_loss': avg_ll,
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
# Domain similarity analysis
# ============================================================

def domain_similarity_analysis(feat, ext_df, prefix):
    """
    Measure domain similarity between internal data and external data.
    
    Since external data has different column names, we can't do direct feature comparison.
    Instead, we use proxy metrics:
    1. External data quality metrics
    2. External data statistics (distribution properties)
    3. Cross-feature correlations (domain structure)
    """
    num_cols = ext_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Quality metrics
    missing_pct = float(ext_df.isnull().sum().sum() / (len(ext_df) * len(ext_df.columns)) * 100)
    duplicate_ratio = float(ext_df.duplicated().sum() / len(ext_df))
    
    # Distribution properties
    stats = {}
    for col in num_cols:
        vals = ext_df[col].dropna()
        if len(vals) < 10:
            continue
        stats[sanitize_col(col)] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'skew': float(vals.skew()),
            'kurtosis': float(vals.kurtosis()),
            'n': len(vals),
        }
    
    return {
        'prefix': prefix,
        'n_samples': len(ext_df),
        'n_features': len(num_cols),
        'missing_pct': round(missing_pct, 2),
        'duplicate_ratio': round(duplicate_ratio, 4),
        'feature_stats': stats,
    }


# ============================================================
# Main: Automated external data research loop
# ============================================================

def main():
    print("=" * 80)
    print("V06: EXTERNAL DATA AUTOMATED RESEARCH LOOP")
    print("=" * 80)
    
    # Load internal data
    print("\n[1] Loading internal data...")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_tst = pd.read_parquet(DATA / "test_features.parquet")
    for df in [feat, feat_tst]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    feat.columns = [sanitize_col(c) for c in feat.columns]
    feat_tst.columns = [sanitize_col(c) for c in feat_tst.columns]
    print(f"  Train: {feat.shape}, Test: {feat_tst.shape}")
    print(f"  Targets: { {t: round(feat[t].mean(), 3) for t in TARGETS} }")
    
    # Load external data
    print("\n[2] Loading external data...")
    ext_data = load_external_data()
    print(f"  {len(ext_data)} external datasets loaded")
    
    # Domain similarity analysis
    print("\n[3] Domain similarity analysis...")
    dom_results = {}
    for ext_key, ext_df in ext_data.items():
        dom = domain_similarity_analysis(feat, ext_df, ext_key)
        dom_results[ext_key] = dom
        print(f"  {ext_key}: n={dom['n_samples']}, features={dom['n_features']}, "
              f"missing={dom['missing_pct']:.1f}%, dup={dom['duplicate_ratio']:.3f}")
    
    # Create global features for each external dataset
    print("\n[4] Creating global features...")
    global_features = {}
    for ext_key, ext_df in ext_data.items():
        gf = create_external_global_features(ext_df, f'ext_{ext_key}')
        global_features[ext_key] = gf
        print(f"  {ext_key}: {len(gf)} global features")
    
    # Define external data labels for combination notation
    ext_keys = list(global_features.keys())
    ext_labels = {
        'A': ext_keys[0] if len(ext_keys) > 0 else None,
        'B': ext_keys[1] if len(ext_keys) > 1 else None,
        'C': ext_keys[2] if len(ext_keys) > 2 else None,
        'D': ext_keys[3] if len(ext_keys) > 3 else None,
    }
    
    # Build combination list
    combinations = {}
    labels = ['A', 'B', 'C', 'D']
    valid_labels = [l for l in labels if ext_labels.get(l) is not None]
    
    # Single
    for l in valid_labels:
        combinations[l] = [ext_labels[l]]
    
    # Pairs
    for combo in itertools.combinations(valid_labels, 2):
        name = '+'.join(combo)
        combinations[name] = [ext_labels[l] for l in combo]
    
    # Triples
    for combo in itertools.combinations(valid_labels, 3):
        name = '+'.join(combo)
        combinations[name] = [ext_labels[l] for l in combo]
    
    # All
    if len(valid_labels) >= 4:
        combinations['A+B+C+D'] = [ext_labels[l] for l in valid_labels]
    
    print(f"\n  Total combinations to test: {len(combinations)}")
    
    # ============================================================
    # Experiment execution
    # ============================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT EXECUTION")
    print("=" * 80)
    
    all_results = []
    
    # --- Baseline (no external) ---
    print("\n>>> Baseline (no external data)")
    log_base, _ = run_v127_experiment(feat, feat_tst, 'baseline')
    print(f"  Baseline OOF: {log_base['avg_oof']:.5f}")
    all_results.append({**log_base, 'strategy': 'baseline'})
    
    # --- Single external datasets ---
    print("\n>>> Single external datasets")
    for ext_key in global_features:
        for weight in [0.1, 0.3, 0.5, 1.0, 2.0]:
            exp_id = f'{ext_key}_w{weight}'
            log, _ = run_v127_experiment(feat, feat_tst, exp_id, 
                                        ext_features=global_features[ext_key],
                                        ext_weight=weight)
            delta = log['avg_oof'] - log_base['avg_oof']
            status = " *** IMPROVEMENT" if delta < -0.001 else ""
            print(f"  {exp_id:30s} OOF={log['avg_oof']:.5f} (Δ={delta:+.5f}){status}")
            all_results.append({**log, 'strategy': 'single', 'weight': weight})
    
    # --- Pair combinations ---
    print("\n>>> Pair combinations")
    for name, ext_list in combinations.items():
        if len(ext_list) < 2:
            continue  # Skip singles (already tested)
        
        combined = {}
        for ek in ext_list:
            combined.update(global_features[ek])
        
        for weight in [0.3, 0.5, 1.0]:
            exp_id = f'{name}_w{weight}'
            log, _ = run_v127_experiment(feat, feat_tst, exp_id,
                                        ext_features=combined, ext_weight=weight)
            delta = log['avg_oof'] - log_base['avg_oof']
            status = " *** IMPROVEMENT" if delta < -0.001 else ""
            print(f"  {exp_id:30s} OOF={log['avg_oof']:.5f} (Δ={delta:+.5f}){status}")
            all_results.append({**log, 'strategy': 'pair', 'weight': weight})
    
    # --- All combined ---
    if len(valid_labels) >= 3:
        print(f"\n>>> All combined")
        all_ext = {}
        for ek in ext_keys:
            all_ext.update(global_features[ek])
        log, _ = run_v127_experiment(feat, feat_tst, 'all_combined',
                                    ext_features=all_ext)
        delta = log['avg_oof'] - log_base['avg_oof']
        status = " *** IMPROVEMENT" if delta < -0.001 else ""
        print(f"  {'all_combined':30s} OOF={log['avg_oof']:.5f} (Δ={delta:+.5f}){status}")
        all_results.append({**log, 'strategy': 'all_combined'})
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY (sorted by OOF)")
    print("=" * 80)
    
    all_results.sort(key=lambda x: x['avg_oof'])
    for i, r in enumerate(all_results):
        delta = r['avg_oof'] - log_base['avg_oof']
        marker = " *** IMPROVEMENT" if delta < -0.001 else ""
        print(f"  #{i+1:2d}: {r['exp_id']:30s} OOF={r['avg_oof']:.5f} (Δ={delta:+.5f}){marker}")
    
    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'external_data_research_v06_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump({
            'baseline': log_base,
            'all_experiments': all_results,
            'domain_similarity': {k: v for k, v in dom_results.items()},
            'global_features': {k: len(v) for k, v in global_features.items()},
            'combinations': {k: v for k, v in combinations.items()},
        }, f, indent=2, default=str)
    print(f"\n  Saved: {result_path}")
    
    # Check for improvements
    best = min(all_results, key=lambda x: x['avg_oof'])
    if best['avg_oof'] < log_base['avg_oof'] - 0.001:
        print(f"\n  *** BEST: {best['exp_id']} → OOF={best['avg_oof']:.5f} "
              f"(improved by {log_base['avg_oof']-best['avg_oof']:.5f}) ***")
    else:
        print(f"\n  No significant improvement found. Baseline remains: OOF={log_base['avg_oof']:.5f}")


if __name__ == '__main__':
    main()
