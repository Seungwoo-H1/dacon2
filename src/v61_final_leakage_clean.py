"""
V61 - Final CatBoost submission with leakage-clean features.
Best model: V60 leakage-clean + CatBoost (avg CV 0.5830, delta +0.0976 vs V53)
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb

warnings.filterwarnings('ignore')
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

NIGHTTIME_LEAK = {
    'mScreenStatus_hour_night', 'mACStatus_hour_night', 
    'mScreenStatus_hour_morning', 'wLight_w_light_sum',
    'mACStatus_charging_sum', 'mACStatus_charging_max',
}
SLEEP_DIRECT_LEAK = {
    'mGps_gps_avg_speed_max', 'mGps_gps_count_mean',
    'mActivity_m_activity_sum', 'mActivity_m_activity_max',
    'mActivity_m_activity_min',
}

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S and c not in NIGHTTIME_LEAK and c not in SLEEP_DIRECT_LEAK]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q and c not in NIGHTTIME_LEAK]
    return cols

def get_feature_cols(df):
    # Only numeric columns (exclude object/string columns like ambience cat)
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def add_personalization(df, feature_cols):
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

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("V61 - Final CatBoost + Leakage-clean Submission")
    log.info("=" * 60)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    
    # Clean parquet already has zscore features, just get all feature cols
    feat_cols = get_feature_cols(train)
    # Verify zscore already present
    zscore_cols = [c for c in feat_cols if '_zscore' in c]
    log.info(f"  Z-score features already in clean parquet: {len(zscore_cols)}")
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)} total (base {len(feat_cols)-len(zscore_cols)} + zscore {len(zscore_cols)})")
    all_cols = feat_cols
    
    n_seeds = 30
    
    V60_CONFIGS = {
        'Q1': {'n_feat': 19},
        'Q2': {'n_feat': 14},
        'Q3': {'n_feat': 5},
        'S1': {'n_feat': 21},
        'S2': {'n_feat': 19},
        'S3': {'n_feat': 21},
        'S4': {'n_feat': 20},
    }
    
    predictions = {}
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    
    for target in TARGETS:
        log.info(f"\n  --- {target} (CatBoost + leakage-clean) ---")
        y = train[target].values.astype(np.float64)
        
        n_feat = V60_CONFIGS[target]['n_feat']
        final_cols = remove_leak(all_cols, target)
        
        # Feature ranking
        X_all = train[final_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1,
        }
        sn = [sanitize(c) for c in final_cols]
        ds = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds, num_boost_round=100)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(final_cols, imp), key=lambda x: -x[1])
        sel_cols = [ranked[i][0] for i in range(min(n_feat, len(ranked)))]
        del model_rank, ds, X_all
        gc.collect()
        
        X_df = train[sel_cols].fillna(0)
        X_all_arr = X_df.values.astype(np.float64)
        X_test_arr = test[sel_cols].fillna(0).values.astype(np.float64)
        
        # CatBoost on full data
        test_preds = []
        n_splits = 3
        seeds_per_fold = n_seeds // n_splits
        for fold in range(n_splits):
            fold_preds = []
            for s in range(seeds_per_fold):
                seed = fold * seeds_per_fold + s + 1
                model = cb.CatBoostClassifier(
                    iterations=1000, learning_rate=0.03, depth=6,
                    loss_function='Logloss', eval_metric='Logloss',
                    random_seed=seed, verbose=0, task_type='CPU',
                    bagging_temperature=0.5, l2_leaf_reg=3.0, random_strength=1.0,
                )
                model.fit(X_all_arr, y, verbose=0)
                fold_preds.append(model.predict_proba(np.where(np.isnan(X_test_arr), 0, X_test_arr))[:, 1])
            test_preds.append(np.mean(fold_preds, axis=0))
        
        predictions[target] = np.clip(np.mean(test_preds, axis=0), 0.0001, 0.9999)
        log.info(f"  {target}: mean={predictions[target].mean():.4f} min={predictions[target].min():.4f} max={predictions[target].max():.4f}")
        
        del sel_cols, y, X_all_arr, X_df, test_preds
        gc.collect()
    
    # Build submission
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    
    sub_path = SUBMIT / f"submission_v61_leakage_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    
    log.info(f"\n{'='*60}")
    log.info(f"✅ Submission saved: {sub_path}")
    log.info(f"Rows: {len(sub)}")
    for t in TARGETS:
        log.info(f"  {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")
    
    # Meta
    meta = {
        'version': 'V61_leakage_clean_final',
        'name': 'CatBoost + leakage-clean features (nighttime + sleep-direct removed)',
        'model': 'CatBoostClassifier',
        'n_seeds': n_seeds,
        'n_splits': 3,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'cv_results': {
            'Q1': {'v60': 0.6309, 'v53': 0.7591, 'delta': 0.1282},
            'Q2': {'v60': 0.5912, 'v53': 0.6929, 'delta': 0.1017},
            'Q3': {'v60': 0.5926, 'v53': 0.6893, 'delta': 0.0967},
            'S1': {'v60': 0.5538, 'v53': 0.6029, 'delta': 0.0491},
            'S2': {'v60': 0.5630, 'v53': 0.6621, 'delta': 0.0991},
            'S3': {'v60': 0.5284, 'v53': 0.7144, 'delta': 0.1860},
            'S4': {'v60': 0.6214, 'v53': 0.6438, 'delta': 0.0224},
        },
        'avg_cv_v60': 0.5830,
        'avg_cv_v53': 0.6806,
        'avg_delta': 0.0976,
        'v53_leaderboard': 0.65358,
        'removed_features': list(NIGHTTIME_LEAK | SLEEP_DIRECT_LEAK),
        'notes': 'Leakage removal: nighttime features (hour_night, hour_morning, light_sum, charging_sum/max) + sleep-direct features (gps speed/count, activity sum/max/min)',
    }
    meta_path = SUBMIT / f'meta_v61_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")

if __name__ == "__main__":
    main()
