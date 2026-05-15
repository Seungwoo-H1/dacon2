"""
V46 Test Prediction — Lightweight version (feature limit to avoid OOM)

Uses fewer features (base + date only, NO personalization z-scores)
to stay within RAM limits for test prediction.
"""

import sys, re, gc, time, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

SEEDS = [42, 456, 2048, 8001, 14000]  # 5 seeds

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

def add_date_features(df):
    date_col = pd.to_datetime(df['sleep_date'])
    df = df.copy()
    df['dayofweek'] = date_col.dt.dayofweek
    df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
    df['month'] = date_col.dt.month
    df['is_monday'] = (date_col.dt.dayofweek == 0).astype(int)
    df['is_friday'] = (date_col.dt.dayofweek == 4).astype(int)
    df['dayofyear'] = date_col.dt.dayofyear
    df['is_q1'] = date_col.dt.month.isin([6,7,8]).astype(int)
    df['is_q2'] = date_col.dt.month.isin([3,4,5,9,10,11,12,1,2]).astype(int)
    return df

def simple_mm(p, r):
    shift = r - p.mean()
    return np.clip(p + shift, 0.0001, 0.9999)

def rank_features(feat, feat_cols, target, seed=42, n_trees=50):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': n_trees, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    
    sn = [re.sub(r'[^a-zA-Z0-9_]','_',c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=n_trees)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

V46_CONFIGS = {
    'Q1':  {'config': 'C2', 'n_feat': 10, 'w_cat': 0.70},
    'Q2':  {'config': 'C5', 'n_feat': 30, 'w_cat': 0.70},
    'Q3':  {'config': 'C5', 'n_feat': 20, 'w_cat': 0.70},
    'S1':  {'config': 'C5', 'n_feat': 30, 'w_cat': 0.70},
    'S2':  {'config': 'C5', 'n_feat': 20, 'w_cat': 0.70},
    'S3':  {'config': 'C4', 'n_feat': 20, 'w_cat': 0.70},
    'S4':  {'config': 'C5', 'n_feat': 20, 'w_cat': 0.70},
}

LGBM_CFG = {
    'C2': {'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C4': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C5': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
}

CB_PARAMS = {
    'loss_function': 'Logloss', 'verbose': 0,
    'learning_rate': 0.03, 'depth': 3,
    'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5,
}

# ── Main ──
def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V46 Test Prediction (lightweight — base features only)")
    log.info("=" * 70)
    
    # Load train features
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"  Train: {feat.shape}, Test: {test.shape}")
    
    # Add date features only (NO personalization to save RAM)
    log.info("\n--- Date features ---")
    feat = add_date_features(feat)
    test = add_date_features(test)
    date_cols = ['dayofweek', 'is_weekend', 'month', 'is_monday', 'is_friday',
                 'dayofyear', 'is_q1', 'is_q2']
    
    # Base feature columns (no z-score)
    feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS) | set(date_cols)
                 and pd.api.types.is_numeric_dtype(feat[c])]
    log.info(f"  Base feature cols: {len(feat_cols)}")
    
    all_feat_cols = feat_cols + date_cols
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    
    # Feature ranking and test prediction
    predictions = pd.DataFrame()
    test_meta = {}
    
    for target in TARGET_COLS:
        tgt_t = time.time()
        cfg = V46_CONFIGS[target]
        cfg_name = cfg['config']
        n_feat = cfg['n_feat']
        w_cat = cfg['w_cat']
        w_lgb = 1.0 - w_cat
        
        leak_free = remove_leak(all_feat_cols, target)
        ranked = rank_features(feat, leak_free, target)
        sel = ranked[:n_feat]
        
        sn = [re.sub(r'[^a-zA-Z0-9_]','_',c) for c in sel]
        log.info(f"\n  {target}: Config={cfg_name} n={n_feat} sel={len(sel)} w_CB={w_cat:.2f}")
        
        y_train = feat[target].values
        X_train = feat[sel].fillna(0).values.astype(np.float64)
        X_test = test[sel].fillna(0).values.astype(np.float64)
        
        # LGBM
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': LGBM_CFG[cfg_name]['nl'],
            'max_depth': LGBM_CFG[cfg_name]['md'],
            'learning_rate': LGBM_CFG[cfg_name]['lr'],
            'n_estimators': LGBM_CFG[cfg_name]['ne'],
            'subsample': LGBM_CFG[cfg_name]['ss'],
            'colsample_bytree': LGBM_CFG[cfg_name]['cb'],
            'reg_alpha': LGBM_CFG[cfg_name]['ra'],
            'reg_lambda': LGBM_CFG[cfg_name]['rl'],
            'min_child_samples': LGBM_CFG[cfg_name]['mc'],
        }
        spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
        lgb_params['scale_pos_weight'] = spw
        
        lgb_preds = np.zeros(len(test))
        for s, seed in enumerate(SEEDS):
            ds = lgb.Dataset(X_train, label=y_train, feature_name=sn, params={'verbose': '-1'})
            params = {**lgb_params, 'random_state': seed}
            m = lgb.train(params, ds, num_boost_round=LGBM_CFG[cfg_name]['ne'])
            lgb_preds += m.predict(X_test)
            del m, ds
            gc.collect()
        lgb_preds /= len(SEEDS)
        
        # CatBoost
        cat_preds = np.zeros(len(test))
        for s, seed in enumerate(SEEDS):
            X_cb = feat[sel].fillna(0).values.astype(np.float32)
            Xt_cb = test[sel].fillna(0).values.astype(np.float32)
            m = cb.CatBoostClassifier(
                **{**CB_PARAMS, 'random_seed': seed, 'num_boost_round': LGBM_CFG[cfg_name]['ne']}
            )
            m.fit(X_cb, y_train, use_best_model=False)
            cat_preds += m.predict_proba(Xt_cb)[:, 1]
            del m
            gc.collect()
        cat_preds /= len(SEEDS)
        
        ens_preds = w_lgb * lgb_preds + w_cat * cat_preds
        cal_preds = simple_mm(ens_preds, train_rate[target])
        
        predictions[target] = cal_preds
        test_meta[target] = {
            'config': cfg_name, 'n_feat': n_feat, 'w_cat': w_cat,
            'pred_mean': float(cal_preds.mean()),
            'pred_min': float(cal_preds.min()),
            'pred_max': float(cal_preds.max()),
        }
        log.info(f"    {target}: test_mean={cal_preds.mean():.4f} | Time: {time.time()-tgt_t:.0f}s")
    
    # Save submission
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    predictions['subject_id'] = test['subject_id'].values
    predictions['sleep_date'] = test['sleep_date'].values
    predictions['lifelog_date'] = test['lifelog_date'].values
    predictions = predictions[['subject_id', 'sleep_date', 'lifelog_date'] + TARGET_COLS]
    
    sub_path = SUBMIT_DIR / f'submission_v46_{timestamp}.csv'
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")
    
    test_meta['version'] = 'v46'
    test_meta['submission_file'] = str(sub_path)
    test_meta['avg_cal_training'] = 0.2612  # from OOF
    test_meta['note'] = 'Base features only (no personalization z-scores) due to RAM limits'
    
    meta_path = SUBMIT_DIR / f'meta_v46_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(test_meta, f, indent=2, default=str)
    log.info(f"  Meta saved: {meta_path}")
    
    log.info(f"\n{'='*70}")
    log.info("V46 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    for t in TARGET_COLS:
        m = test_meta[t]
        log.info(f"  {t}: config={m['config']} n={m['n_feat']} test_mean={m['pred_mean']:.4f}")
    log.info(f"  V46 Avg Cal (training OOF): 0.2612")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: -0.3426")
    log.info(f"  Total: {time.time()-t_start:.0f}s")
    
    return predictions


if __name__ == "__main__":
    main()
