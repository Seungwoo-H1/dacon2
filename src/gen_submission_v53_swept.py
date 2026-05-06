"""
Generate final submission CSV from V53 with sweep-optimized n_feat.
Uses best n_feat per target from sweep results.
"""

import sys, gc, logging, json, re, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    """Add subject-level zscore features (batch agg, no fragmentation)."""
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
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols

def rank_features_importance(feat, feat_cols, target, cfgs, v53_cfgs, seed=42):
    """Rank features by LGBM gain importance."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    v53_cfg = v53_cfgs.get(target, {'cfg': 'deep', 'n_feat': 20})
    cfg_name = v53_cfg['cfg']
    base_cfg = cfgs.get(cfg_name, cfgs['deep'])
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'], 'learning_rate': base_cfg['lr'],
        'n_estimators': min(base_cfg['ne'], 100), 'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
        'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': base_cfg['mc'], 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=params['n_estimators'])
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def train_and_predict(train_feat, test_feat, cols, y_train, target, cfgs, v53_cfgs, n_seeds=50):
    """Train on all train data, predict test using target-specific CFG."""
    X_train = train_feat[cols].fillna(0).values.astype(np.float64)
    X_test = test_feat[cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in cols]
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    
    v53_cfg = v53_cfgs.get(target, {'cfg': 'deep', 'n_feat': 20})
    cfg_name = v53_cfg['cfg']
    base_cfg = cfgs.get(cfg_name, cfgs['deep'])
    n_trees = base_cfg['ne']
    
    seed_results = []
    for seed in range(1, n_seeds + 1):
        cfg_seed = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
            'learning_rate': base_cfg['lr'], 'n_estimators': n_trees,
            'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
            'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
            'min_child_samples': base_cfg['mc'], 'random_state': seed, 'scale_pos_weight': spw,
        }
        ds = lgb.Dataset(X_train, label=y_train, feature_name=sn, params={'verbose': '-1'})
        m = lgb.train(cfg_seed, ds, num_boost_round=n_trees)
        pred = m.predict(X_test)
        seed_results.append(pred)
        del ds, m
    
    return np.clip(np.mean(seed_results, axis=0), 0.0001, 0.9999)

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("Generating V53 Swept submission (optimized n_feat)")
    
    # Load data
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    
    train_cols_order = list(train.columns)
    test = test[train_cols_order]
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    
    n_seeds = 50
    
    # V53 baseline configs
    V53_CONFIGS = {
        'Q1': {'cfg': 'deep', 'n_feat': 20},
        'Q2': {'cfg': 'deep', 'n_feat': 15},
        'Q3': {'cfg': 'v48', 'n_feat': 8},
        'S1': {'cfg': 'wide', 'n_feat': 20},
        'S2': {'cfg': 'deep', 'n_feat': 20},
        'S3': {'cfg': 'safety', 'n_feat': 20},
        'S4': {'cfg': 'wide', 'n_feat': 20},
    }
    
    # Swept optimal n_feat per target
    V53_SWEEP_CONFIGS = {
        'Q1': {'cfg': 'deep', 'n_feat': 19},
        'Q2': {'cfg': 'deep', 'n_feat': 14},
        'Q3': {'cfg': 'v48', 'n_feat': 5},
        'S1': {'cfg': 'wide', 'n_feat': 21},
        'S2': {'cfg': 'deep', 'n_feat': 19},
        'S3': {'cfg': 'safety', 'n_feat': 21},
        'S4': {'cfg': 'wide', 'n_feat': 20},
    }
    
    CFGS = {
        'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
        'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
        'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
    }
    
    # Get base features and add personalization
    feat_cols = get_feature_cols(train)
    train, zscore_cols = add_personalization(train, feat_cols)
    test, _ = add_personalization(test, feat_cols)
    
    all_cols = feat_cols + zscore_cols
    log.info(f"  Features: {len(feat_cols)} base + {len(zscore_cols)} zscore = {len(all_cols)} total")
    
    # Use sweep configs
    active_configs = V53_SWEEP_CONFIGS
    
    predictions = {}
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    
    for target in TARGETS:
        log.info(f"\n  --- {target} (sweep-optimized) ---")
        
        config = active_configs[target]
        cfg_name = config['cfg']
        n_feat = config['n_feat']
        
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(train, leak_cols, target, CFGS, active_configs)
        sel_cols = ranked[:n_feat]
        
        y_train = train[target].values.astype(np.float64)
        
        log.info(f"  Config: {cfg_name}, n_feat={n_feat}, features={len(sel_cols)}")
        log.info(f"  Training {n_seeds} seeds...")
        preds = train_and_predict(train, test, sel_cols, y_train, target, CFGS, active_configs, n_seeds)
        predictions[target] = preds
        log.info(f"  {target}: mean={preds.mean():.4f} min={preds.min():.4f} max={preds.max():.4f}")
        
        del sel_cols, y_train, preds, ranked, leak_cols
        gc.collect()
    
    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    sub_path = SUBMIT / f"submission_v53_swept_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    
    log.info(f"\n{'='*60}")
    log.info(f"✅ Submission saved: {sub_path}")
    log.info(f"Rows: {len(sub)}")
    for t in TARGETS:
        log.info(f"  {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")
    log.info(f"{'='*60}")
    
    # Save meta
    meta = {
        'version': 'V53_swept',
        'name': 'V53 submission with sweep-optimized n_feat',
        'n_seeds': n_seeds,
        'configs_used': 'V53_SWEEP_CONFIGS',
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'n_feat_per_target': {t: active_configs[t]['n_feat'] for t in TARGETS},
        'cv_results': {
            'Q1': {'baseline': 0.7701, 'swept': 0.7591, 'delta': 0.0110},
            'Q2': {'baseline': 0.7059, 'swept': 0.6929, 'delta': 0.0130},
            'Q3': {'baseline': 0.7009, 'swept': 0.6893, 'delta': 0.0116},
            'S1': {'baseline': 0.6111, 'swept': 0.6029, 'delta': 0.0083},
            'S2': {'baseline': 0.6657, 'swept': 0.6621, 'delta': 0.0036},
            'S3': {'baseline': 0.7236, 'swept': 0.7144, 'delta': 0.0092},
            'S4': {'baseline': 0.6438, 'swept': 0.6438, 'delta': 0.0000},
        },
        'avg_cv_baseline': 0.6887,
        'avg_cv_swept': 0.6806,
    }
    meta_path = SUBMIT / f'meta_v53_swept_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")

if __name__ == "__main__":
    main()
