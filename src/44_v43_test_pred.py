"""
V43 Test Prediction — train OOF 기반으로 test prediction

train에서 best config + features 선택 결과를 test에 적용
"""

import sys, re, gc, time, warnings, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    return np.clip(p + (r.mean() - p.mean()), 0.0001, 0.9999)

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

CONFIGS = {
    'C1': {'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    'C2': {'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C3': {'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C4': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C5': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    'C6': {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
}

def add_personalization(df, feature_cols):
    for col in feature_cols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col] - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        gc.collect()
    return df

def rank_features(feat, feat_cols, target, seed=42):
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

# Leakage
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
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V43 Test Prediction")
    log.info("=" * 70)
    
    # Load train features
    log.info("\n--- Load train features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Train: {feat.shape}")
    
    feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS)
                 and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    feat = add_personalization(feat, feat_cols)
    zscore_cols = [f'{c}_zscore' for c in feat_cols]
    all_cols = feat_cols + zscore_cols
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Train rates: {train_rate}")
    
    # Load test
    log.info("\n--- Load test features ---")
    test = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"  Test: {test.shape}")
    
    test_feat_cols = [c for c in test.columns if c not in META | set(TARGET_COLS)
                      and test[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    test = add_personalization(test, test_feat_cols)
    
    # Load V43 config
    meta_path = DATA_PROCESSED / "v43_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            v43_meta = json.load(f)
        log.info(f"  V43 meta loaded: {meta_path}")
        results = v43_meta['results']
    else:
        log.warning("  No v43_meta.json found — using V10 configs as fallback")
        # Default: assume same config as V10
        results = {t: {'config': 'C4', 'n_feat': 20} for t in TARGET_COLS}
    
    test_preds = {t: np.zeros(len(test)) for t in TARGET_COLS}
    
    for target in TARGET_COLS:
        r = results[target]
        best_cfg_name = r['config']
        best_n = r['n_feat']
        cfg = CONFIGS[best_cfg_name]
        
        # Remove leakage
        leak_cols = remove_leak(all_cols, target)
        
        # Feature ranking on train
        ranked = rank_features(feat, leak_cols, target)
        sel = ranked[:best_n]
        
        cfg_full = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
        }
        spw = max(((feat[target].values == 0).sum()) / max((feat[target].values == 1).sum(), 1), 0.1)
        
        sn = [sanitize(c) for c in sel]
        all_preds = np.zeros(len(test))
        
        for seed in SEEDS:
            cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
            X_all = feat[sel].fillna(0).values.astype(np.float64)
            y_all = feat[target].values.astype(np.float64)
            X_test = test[sel].fillna(0).values.astype(np.float64)
            ds_all = lgb.Dataset(X_all, label=y_all, feature_name=sn, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg['ne'])
            all_preds += m.predict(X_test)
            del m, ds_all
            gc.collect()
        
        all_preds /= len(SEEDS)
        cal_preds = mm(all_preds, train_rate[target])
        test_preds[target] = cal_preds
        log.info(f"  {target}: {best_cfg_name} n={best_n}, test_mean={cal_preds.mean():.4f}, train_rate={train_rate[target]:.3f}")
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_df = pd.DataFrame({
        'subject_id': test['subject_id'].values,
        'sleep_date': test['sleep_date'].values,
        'lifelog_date': test['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        sub_df[target] = test_preds[target]
    
    sub_path = SUBMIT_DIR / f'submission_v43_{timestamp}.csv'
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")
    log.info(f"Total: {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
