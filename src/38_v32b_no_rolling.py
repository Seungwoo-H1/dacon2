"""
V32-B — Personalization ONLY, no rolling
Test whether personalization alone (without rolling) is better
"""

import sys, re, json, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR, PARQUET_FILES, DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SPLITS = 10

CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std','mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std','mWifi_wifi_bssid_count_max',
]
LEAKAGE_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def mm(p, r): return np.clip(p+(r-p.mean()), 0.0001, 0.9999)

def build_features():
    """Build features: base + personalization, NO rolling."""
    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    
    merged = None
    for name, fname in PARQUET_FILES.items():
        df = pd.read_parquet(DATA_DIR / fname)
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
        df.columns = [f'{name}_{c}' if c not in ('subject_id','timestamp','date','hour') else c for c in df.columns]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=['subject_id', 'date'], how='outer')
    
    labels_date = labels.copy()
    labels_date['date'] = labels_date['lifelog_date'].dt.date
    merged = merged.merge(labels_date[['subject_id', 'date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']],
                          on=['subject_id', 'date'], how='left')
    
    log.info(f"  Merged: {merged.shape}")
    
    remove = set(CONSTANT_COLS + COLLINEAR_DROP) | META_COLS | set(TARGET_COLS)
    num_cols = [c for c in merged.columns if c not in remove and merged[c].dtype in [np.float64, np.int64, float, int, bool]]
    log.info(f"  Base cols: {len(num_cols)}")
    
    leak = LEAKAGE_S | LEAKAGE_Q
    avail = [c for c in num_cols if c not in leak]
    log.info(f"  After leak removal: {len(avail)}")
    
    if 'wHr_hr_mean' in avail:
        mask = (merged['wHr_hr_mean'] < 20) | (merged['wHr_hr_mean'] > 180)
        merged.loc[mask, 'wHr_hr_mean'] = np.nan
    
    # Personalization (z-score) — NO rolling
    personal_cols = []
    for col in avail:
        col_filled = merged[col].fillna(0)
        subj_stats = col_filled.groupby(merged['subject_id']).agg(['mean', 'std']).reset_index()
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        merged = merged.merge(subj_stats, on='subject_id', how='left')
        mask_std_zero = (merged[f'{col}_subj_std'] == 0)
        mask_null = merged[col].isnull()
        merged[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null, 0.0,
            (merged[col].fillna(0) - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std'])
        personal_cols.append(f'{col}_zscore')
    
    log.info(f"  After personalization: {len(num_cols) + len(personal_cols)} total")
    
    return merged, num_cols + personal_cols

def rank_features(feat, feature_cols, target, seed=42):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,
              'force_row_wise':True,'n_jobs':-1}
    sn = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type="gain")
    return sorted(zip(feature_cols, imp), key=lambda x: -x[1])

LGB_BASE = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1,
}
LGB_CONFIGS = [
    {'name':'C2','nl':10,'md':3,'lr':0.03,'ne':300,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C4','nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C5','nl':20,'md':5,'lr':0.02,'ne':300,'ss':0.7,'cst':0.7,'ra':0.5,'rl':2.0,'mc':8},
]

def lgb_cv(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]
    for si, s in enumerate(seeds):
        cfg = {**LGB_BASE, 'random_state': s}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            y_tr, y_va = y[tr], y[va]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train({**cfg, 'scale_pos_weight': spw}, ds, num_boost_round=cfg['n_estimators'],
                         valid_sets=[vd], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
    return oof

def main():
    log.info("V32-B: Personalization ONLY (no rolling)")
    feat, all_feature_cols = build_features()
    
    all_oof = {}; all_cal = {}; all_best = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        best_cv = float('inf'); best_cfg = None; best_n = None; best_cols = None
        
        for cfg in LGB_CONFIGS:
            for n_feat in [10, 20, 30]:
                ranked = rank_features(feat, all_feature_cols, target, seed=42)
                sel_cols = [c for c, _ in ranked[:n_feat]]
                oof = lgb_cv(feat, sel_cols, target, SEEDS)
                cv = log_loss(feat[target], oof.mean(axis=1), labels=[0,1])
                if cv < best_cv:
                    best_cv = cv; best_cfg = cfg; best_n = n_feat; best_cols = sel_cols
        
        oof = lgb_cv(feat, best_cols, target, SEEDS)
        cal = mm(oof.mean(axis=1), feat[target].values)
        all_oof[target] = oof.mean(axis=1)
        all_cal[target] = cal
        all_best[target] = {'config': best_cfg['name'], 'n': best_n, 'cv': best_cv}
        log.info(f"  ** BEST: {best_cfg['name']} n={best_n}, CV={best_cv:.4f}, Cal={log_loss(feat[target],cal,labels=[0,1]):.4f}")
    
    log.info(f"\n{'='*50}")
    for t in TARGET_COLS:
        log.info(f"  {t}: OOF={log_loss(feat[t],all_oof[t],labels=[0,1]):.4f} Cal={log_loss(feat[t],all_cal[t],labels=[0,1]):.4f}")
    avg = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"  V32-B Avg Cal OOF: {avg:.4f}")

if __name__ == "__main__":
    main()
