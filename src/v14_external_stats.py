"""
V14: External data global statistics as supplementary features.

Since external data rows can't be joined to our subjects, we compute
global statistics from external datasets and add them as per-subject
"deviation from external mean" features.

External datasets used:
- sleep_study_1000.csv (1000 rows, unique features: SleepEfficiency, 
  REM%, Deep%, Light%, Awakenings, Caffeine, Alcohol, Bedtime, etc.)

Approach:
1. Compute global stats from each external dataset
2. For each subject in our train data, compute their equivalent aggregated
   features (where possible from lifelog data) or just add the global stats
3. Evaluate via GroupKFold OOF with V127 configs

Author: 집가헤응
Date: 2026-05-12
"""

import sys, gc, logging, json, time, re, os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data_processed"
DATA_EXT = ROOT / "external_data"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# V53 configs (same as V127)
V53_CONFIGS = {
    'Q1': {'cfg': 'deep', 'n_feat': 20},
    'Q2': {'cfg': 'deep', 'n_feat': 15},
    'Q3': {'cfg': 'v48', 'n_feat': 8},
    'S1': {'cfg': 'wide', 'n_feat': 20},
    'S2': {'cfg': 'deep', 'n_feat': 20},
    'S3': {'cfg': 'safety', 'n_feat': 20},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide':  {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep':  {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':   {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety':{'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

N_SEEDS = 4  # per-target seeds
SEEDS = [42, 123, 456, 789]


def sanitize_name(n):
    """Sanitize column name for LightGBM compatibility."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def load_external_stats():
    """Load external datasets and compute global statistics."""
    ext_stats = {}
    
    # sleep_study_1000 - the most diverse external dataset
    fpath = DATA_EXT / "sleep_lifestyle_1000_kaggle_extracted" / "sleep_study_1000.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'min': df[col].min(),
                'max': df[col].max(),
            }
        ext_stats['sleep_study_1000'] = stats
        
        # Bedtime-derived features
        if 'Bedtime' in df.columns:
            bts = pd.to_datetime(df['Bedtime'], errors='coerce')
            if bts.notna().any():
                ext_stats['sleep_study_1000']['_bedtime_hour'] = {
                    'mean': bts.dt.hour.mean(),
                    'std': bts.dt.hour.std(),
                    'median': bts.dt.hour.median(),
                }
                ext_stats['sleep_study_1000']['bedtime_is_night'] = {
                    'mean': ((bts.dt.hour >= 20) | (bts.dt.hour < 6)).mean()
                }
    
    # sleep_health_1.csv
    fpath = DATA_EXT / "sleep_health_1.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Person ID', 'BMI_Category_Code']:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                }
        ext_stats['sleep_health_1'] = stats
    
    # sleep_health_2.csv
    fpath = DATA_EXT / "sleep_health_2.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Person ID', 'BMI_Category_Code']:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                }
        ext_stats['sleep_health_2'] = stats
    
    # sleep_health_lifestyle.csv
    fpath = DATA_EXT / "sleep_health_lifestyle.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['Person ID', 'BMI_Category_Code']:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                }
        ext_stats['sleep_health_lifestyle'] = stats
    
    return ext_stats


def build_v14_features(df, ext_stats):
    """
    Build V14 feature set:
    - All existing V11 personalized features (from parquet)
    - Plus external-derived global stats
    """
    feature_cols = [c for c in df.columns if c not in {
        'subject_id', 'lifelog_date', 'sleep_date', 'date'
    } | set(TARGETS)]
    
    # Filter to numeric
    num_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    # Add external global stats as constant features per subject
    # These capture population-level baselines
    # Sanitize external feature names to avoid LightGBM special char issues
    for ds_name in list(ext_stats.keys()):
        sanitized_ds = sanitize_name(ds_name)
        ext_stats[sanitized_ds] = ext_stats.pop(ds_name)
        for feat_name in list(ext_stats[sanitized_ds].keys()):
            sanitized_feat = sanitize_name(feat_name)
            ext_stats[sanitized_ds][sanitized_feat] = ext_stats[sanitized_ds].pop(feat_name)
    
    ext_features = []
    for ds_name, stats in ext_stats.items():
        for feat_name, desc in stats.items():
            if 'mean' in desc:
                ext_features.append(f'ext_{ds_name}_{feat_name}_mean')
                ext_features.append(f'ext_{ds_name}_{feat_name}_std')
            if 'median' in desc:
                ext_features.append(f'ext_{ds_name}_{feat_name}_median')
    
    # Create constant feature columns
    for ds_name, stats in ext_stats.items():
        for feat_name, desc in stats.items():
            col_mean = f'ext_{ds_name}_{feat_name}_mean'
            col_std = f'ext_{ds_name}_{feat_name}_std'
            col_med = f'ext_{ds_name}_{feat_name}_median'
            
            if col_mean not in df.columns:
                df[col_mean] = desc.get('mean', 0.0) if isinstance(desc, dict) else 0.0
            if col_std not in df.columns:
                df[col_std] = desc.get('std', 0.0) if isinstance(desc, dict) else 0.0
            if col_med not in df.columns:
                df[col_med] = desc.get('median', 0.0) if isinstance(desc, dict) else 0.0
    
    return df


def get_feature_cols(df):
    META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    feature_cols = [c for c in df.columns
                    if c not in META | set(TARGETS)
                    and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    return feature_cols


def add_personalization(df, feature_cols):
    """Add subject-level zscore features."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = df.merge(agg_df, on='subject_id', how='left')
        for fc in feature_cols:
            ms, ss = f'{fc}_subj_mean', f'{fc}_subj_std'
            if ms in df.columns and ss in df.columns:
                zcol = f'{fc}_zscore'
                df[zcol] = (df[fc] - df[ms]) / (df[ss] + 1e-8)
                zscore_cols.append(zcol)
    return df, zscore_cols


def select_features_per_target(df, target, cfg_info):
    """Select top features per target using correlation + importance.
    Sanitizes all feature names for LightGBM compatibility.
    """
    cfg_name = cfg_info['cfg']
    cfg = CFGS[cfg_name]
    feature_cols = get_feature_cols(df)
    
    # Remove known leakage
    LEAK = {
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
    
    available = [c for c in feature_cols if c not in LEAK]
    
    # Correlation-based pre-filter (on original names)
    target_col = target
    correlations = {}
    for col in available:
        r = df[[col, target_col]].corr().iloc[0, 1] if df[col].nunique() > 1 else 0.0
        correlations[col] = abs(r)
    
    # Top 200 by correlation
    top_cols_orig = sorted(correlations, key=correlations.get, reverse=True)[:200]
    
    # Sanitize column names for LightGBM
    col_map = {}  # sanitized -> original
    top_cols_sanitized = []
    for orig in top_cols_orig:
        san = sanitize_name(orig)
        # Handle duplicates from different original names
        base, counter = san, 0
        while base in col_map:
            counter += 1
            base = f"{san}_{counter}"
        col_map[base] = orig
        top_cols_sanitized.append(base)
    
    # Build sanitized dataframe
    X_san = df[top_cols_orig].fillna(0).copy()
    X_san.columns = top_cols_sanitized
    
    y = df[target_col].values
    
    trn_idx = y != -1
    if trn_idx.sum() < 10:
        return [col_map[c] for c in top_cols_sanitized[:min(len(top_cols_sanitized), cfg['n_feat'])]]
    
    dtrain = lgb.Dataset(X_san.iloc[trn_idx], label=y[trn_idx])
    
    model = lgb.LGBMClassifier(**{
        'num_leaves': cfg['nl'],
        'max_depth': cfg['md'],
        'learning_rate': cfg['lr'],
        'n_estimators': min(cfg['ne'], 100),  # Quick importance
        'subsample': cfg['ss'],
        'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'],
        'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
        'n_jobs': -1,
        'verbose': -1,
    })
    model.fit(X_san.iloc[trn_idx], y[trn_idx])
    
    importances = model.feature_importances_
    feat_imp = sorted(zip(top_cols_sanitized, importances), key=lambda x: x[1], reverse=True)
    
    selected_san = [f[0] for f in feat_imp[:cfg_info['n_feat']]]
    # Return original names
    selected = [col_map[s] for s in selected_san]
    return selected


def train_v14():
    """Main V14 experiment: load data, add external stats, OOF evaluation."""
    start_time = time.time()
    
    # Load external stats
    log.info("Loading external data statistics...")
    ext_stats = load_external_stats()
    for name, stats in ext_stats.items():
        log.info(f"  {name}: {len(stats)} feature groups")
    
    # Save external stats for reference
    ext_stats_path = ROOT / "experiments" / f"v14_external_stats_{int(time.time())}.json"
    # Convert to JSON-serializable
    ext_stats_serializable = {}
    for name, stats in ext_stats.items():
        ext_stats_serializable[name] = {}
        for feat, desc in stats.items():
            ext_stats_serializable[name][feat] = {k: float(v) if isinstance(v, (np.floating, float, np.integer, int)) else str(v) for k, v in desc.items()}
    with open(ext_stats_path, 'w') as f:
        json.dump(ext_stats_serializable, f, indent=2)
    log.info(f"Saved external stats to {ext_stats_path}")
    
    # Load processed features
    log.info("Loading processed features...")
    parquet_path = DATA_PROC / "features_v11_personalized.parquet"
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded shape: {df.shape}")
    
    # Build V14 features
    log.info("Building V14 features with external stats...")
    df = build_v14_features(df, ext_stats)
    log.info(f"After adding ext stats: {df.shape}")
    
    # Add personalization
    log.info("Adding personalization (z-score)...")
    feature_cols = get_feature_cols(df)
    df, zscore_cols = add_personalization(df, feature_cols)
    log.info(f"Final features: {len(get_feature_cols(df))}, zscore: {len(zscore_cols)}")
    
    # Save enriched features
    enriched_path = DATA_PROC / "features_v14_enriched.parquet"
    df.to_parquet(enriched_path, index=False)
    log.info(f"Saved enriched features to {enriched_path}")
    
    # GroupKFold OOF evaluation
    log.info("Starting GroupKFold OOF evaluation...")
    gkf = GroupKFold(n_splits=5)
    oof_preds = {t: np.full(len(df), np.nan) for t in TARGETS}
    all_features_used = {}
    target_results = {}
    
    for target in TARGETS:
        target_start = time.time()
        log.info(f"\n=== Target: {target} ===")
        
        # Select features per target
        cfg_info = V53_CONFIGS[target]
        cfg = CFGS[cfg_info['cfg']]
        selected_feats = select_features_per_target(df, target, cfg_info)
        all_features_used[target] = selected_feats
        log.info(f"Selected {len(selected_feats)} features for {target} (cfg={cfg_info['cfg']})")
        
        # Check if any external features were selected
        ext_selected = [f for f in selected_feats if f.startswith('ext_')]
        log.info(f"  External features selected: {len(ext_selected)}")
        if ext_selected:
            log.info(f"  External feat names: {ext_selected[:10]}")
        
        # Train with multiple seeds
        target_oofs = []
        for seed in SEEDS:
            oof_fold = np.full(len(df), np.nan)
            X = df[selected_feats].fillna(0).values
            y = df[target].values
            
            for fold_i, (trn_idx, val_idx) in enumerate(gkf.split(X, y, df['subject_id'])):
                tr_y = y[trn_idx]
                valid_y = y[val_idx]
                
                dtrain = lgb.Dataset(X[trn_idx], label=tr_y)
                dvalid = lgb.Dataset(X[val_idx], label=valid_y, reference=dtrain)
                
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'verbose': -1,
                    'seed': seed,
                    'num_leaves': cfg['nl'],
                    'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'],
                    'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'],
                    'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'],
                    'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                    'is_unbalance': True,
                }
                
                model = lgb.train(params, dtrain, valid_sets=[dvalid],
                                 callbacks=[lgb.early_stopping(int(cfg['rl']), verbose=False),
                                           lgb.log_evaluation(0)])
                
                val_preds = model.predict(X[val_idx])
                oof_fold[val_idx] = val_preds
            
            # Compute OOF logloss
            valid_mask = ~np.isnan(oof_fold)
            if valid_mask.sum() > 0 and y[valid_mask].sum() > 0:
                from scipy.special import kl_div
                y_true = y[valid_mask]
                y_pred = np.clip(oof_fold[valid_mask], 1e-7, 1-1e-7)
                ll = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                oof_preds[target] = oof_fold
                target_oofs.append(ll)
                log.info(f"  Seed {seed}: OOF={ll:.6f}")
            
            del oof_fold
            gc.collect()
        
        avg_oof = np.mean(target_oofs) if target_oofs else np.nan
        target_results[target] = {
            'cfg': cfg_info['cfg'],
            'n_feat': len(selected_feats),
            'n_ext_feat': len(ext_selected),
            'oof_seeds': [round(x, 6) for x in target_oofs],
            'avg_oof': round(avg_oof, 6),
            'all_features': selected_feats,
        }
        log.info(f"  AVG OOF: {avg_oof:.6f}")
    
    # Summary
    all_avg_oofs = [r['avg_oof'] for r in target_results.values() if r['avg_oof'] is not None]
    overall_avg = np.mean(all_avg_oofs) if all_avg_oofs else np.nan
    
    elapsed = time.time() - start_time
    
    results = {
        'version': 'v14_external_stats',
        'timestamp': int(time.time()),
        'elapsed_seconds': round(elapsed, 1),
        'n_features_total': len(get_feature_cols(df)),
        'avg_oof': round(overall_avg, 6),
        'target_results': target_results,
    }
    
    # Save results
    result_path = ROOT / "experiments" / f"v14_{int(time.time())}.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n=== RESULTS ===")
    log.info(f"Version: v14_external_stats")
    log.info(f"Overall AVG OOF: {overall_avg:.6f}")
    log.info(f"Elapsed: {elapsed:.1f}s")
    for t, r in target_results.items():
        log.info(f"  {t}: OOF={r['avg_oof']:.6f} (n_feat={r['n_feat']}, ext={r['n_ext_feat']})")
    log.info(f"Results saved to {result_path}")
    
    return results


if __name__ == "__main__":
    results = train_v14()
