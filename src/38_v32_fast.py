"""
V32 Fast — V10 Strengths + LOSO CV + Rolling Mean

Optimized for speed: 10 folds × 10 seeds × 3 configs × 2 n_feat = 600 models per target
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
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

# ── Fast config ──
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]  # 10 seeds
N_SPLITS = 10  # LOSO
N_TOP = 20

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
LEAKAGE_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def mm(p, r): return np.clip(p+(r-p.mean()), 0.0001, 0.9999)

# ── Feature engineering ──
def build_features():
    """Load parquet, merge, apply constant/collinear removal, add personalization + rolling."""
    from config import PARQUET_FILES, DATA_DIR
    
    dfs = {}
    for name in PARQUET_FILES:
        path = DATA_DIR / PARQUET_FILES[name]
        df = pd.read_parquet(path)
        dfs[name] = df
    
    # Merge all
    df = dfs[PARQUET_FILES.get("mACStatus", "ch2025_mACStatus.parquet")]
    for name in list(PARQUET_FILES.keys())[1:]:
        col_key = list(PARQUET_FILES.keys())[list(PARQUET_FILES.keys()).index(name)]
        other = dfs[col_key]
        # Merge on subject_id + date
        if 'timestamp' in other.columns:
            other['date'] = other['timestamp'].dt.date
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
        if 'date' not in other.columns:
            continue
        df = df.merge(other, on=['subject_id', 'date'], how='outer', suffixes=('', f'_{name}'))
    
    # Load labels
    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    
    # Merge labels
    df = df.merge(labels, on=['subject_id', 'lifelog_date'], how='left')
    
    # Convert date to date type for merge
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    if 'lifelog_date' in df.columns:
        df['lifelog_date_date'] = pd.to_datetime(df['lifelog_date']).dt.date
    df = df.merge(
        labels[['subject_id', 'lifelog_date']].assign(
            lifelog_date_date=pd.to_datetime(labels['lifelog_date']).dt.date
        ).drop_duplicates('lifelog_date_date'),
        left_on=['subject_id', 'lifelog_date_date'],
        right_on=['subject_id', 'lifelog_date_date'],
        how='left'
    )
    
    log.info(f"  Merged shape: {df.shape}")
    return df

def add_features_simple():
    """Build features using the existing pipeline."""
    from config import PARQUET_FILES, DATA_DIR
    
    # Load labels
    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    
    # Load each parquet and merge
    merged = None
    for name, fname in PARQUET_FILES.items():
        path = DATA_DIR / fname
        df = pd.read_parquet(path)
        
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
        
        # Add subject prefix to columns
        df.columns = [f'{name}_{c}' if c != 'subject_id' and c != 'timestamp' and c != 'date' and c != 'hour' else c 
                      for c in df.columns]
        
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=['subject_id', 'date'], how='outer')
    
    # Merge labels
    labels_date = labels.copy()
    labels_date['date'] = labels_date['lifelog_date'].dt.date
    merged = merged.merge(labels_date[['subject_id', 'date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']], 
                          on=['subject_id', 'date'], how='left')
    
    log.info(f"  Merged shape: {merged.shape}")
    
    # Remove constant columns
    remove = set(CONSTANT_COLS + COLLINEAR_DROP)
    remove |= META_COLS | set(TARGET_COLS)
    
    # Get numeric columns
    num_cols = [c for c in merged.columns if c not in remove and merged[c].dtype in [np.float64, np.int64, float, int, bool]]
    log.info(f"  Base numeric cols: {len(num_cols)}")
    
    # Remove leakage
    leak_cols = LEAKAGE_S | LEAKAGE_Q
    avail = [c for c in num_cols if c not in leak_cols]
    log.info(f"  After leakage removal: {len(avail)}")
    
    # Handle wHr outliers
    if 'wHr_hr_mean' in avail:
        mask = (merged['wHr_hr_mean'] < 20) | (merged['wHr_hr_mean'] > 180)
        merged.loc[mask, 'wHr_hr_mean'] = np.nan
    
    # Add personalization (z-score)
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
            (merged[col].fillna(0) - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
    
    log.info(f"  After personalization: {len(num_cols) + len(personal_cols)} total")
    
    # Add rolling mean (3d, 7d)
    rolling_cols = []
    merged = merged.sort_values(['subject_id', 'date']).reset_index(drop=True)
    for col in avail:
        grp = merged.groupby('subject_id')[col]
        for w in [3, 7]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            merged[f'{col}_rm{w}'] = rm.values
            rolling_cols.append(f'{col}_rm{w}')
    
    log.info(f"  After rolling mean: {len(num_cols) + len(personal_cols) + len(rolling_cols)} total")
    
    # Build feature matrix
    all_feature_cols = num_cols + personal_cols + rolling_cols
    
    return merged, all_feature_cols

# ── Feature ranking ──
def rank_features(feat, feature_cols, target, seed=42):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    spw = ((y == 0).sum()) / n_pos
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
    }
    sn = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return ranked

# ── Hyper configs ──
LGB_BASE = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'num_leaves': 15, 'max_depth': 4,
    'learning_rate': 0.03, 'n_estimators': 500,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 3.0,
    'min_child_samples': 10,
    'force_row_wise': True, 'n_jobs': -1, 'verbose': -1,
}

LGB_CONFIGS = [
    {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
]

# ── CV ──
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
            
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds, params={'verbose': '-1'})
            
            cfg_spw = {**cfg, 'scale_pos_weight': spw}
            m = lgb.train(cfg_spw, ds, num_boost_round=cfg['n_estimators'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
    
    return oof

def main():
    log.info("=" * 70)
    log.info("V32 Fast — V10 Strengths + LOSO CV + Rolling Mean + Personalization")
    log.info("=" * 70)
    
    # Build features
    log.info("\n--- Building features ---")
    feat, all_feature_cols = add_features_simple()
    
    log.info(f"\nTotal feature pool: {len(all_feature_cols)}")
    
    # Per-target tuning
    log.info(f"\n=== Per-target tuning (LOSO CV, {N_SPLITS} folds, {len(SEEDS)} seeds) ===")
    
    all_oof = {}
    all_cal = {}
    all_best = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} ---")
        train_rate = feat[target].mean()
        log.info(f"  Train rate: {train_rate:.3f}")
        
        best_cv = float('inf')
        best_config = None
        best_n = None
        best_cols = None
        
        for cfg_idx, cfg in enumerate(LGB_CONFIGS):
            for n_feat in [10, 20]:
                # Feature ranking
                ranked = rank_features(feat, all_feature_cols, target, seed=42)
                sel_cols = [c for c, _ in ranked[:n_feat]]
                
                # CV
                oof = lgb_cv(feat, sel_cols, target, SEEDS)
                oof_avg = oof.mean(axis=1)
                cv_loss = log_loss(feat[target], oof_avg, labels=[0, 1])
                
                log.info(f"  cfg={cfg['name']:3s} n={n_feat:2d}: cv={cv_loss:.4f}")
                
                if cv_loss < best_cv:
                    best_cv = cv_loss
                    best_config = cfg
                    best_n = n_feat
                    best_cols = sel_cols
        
        # Build final OOF with best config + all seeds
        oof = lgb_cv(feat, best_cols, target, SEEDS)
        oof_avg = oof.mean(axis=1)
        cal_oof = mm(oof_avg, feat[target].values)
        
        all_oof[target] = oof_avg
        all_cal[target] = cal_oof
        all_best[target] = {'config': best_config, 'n': best_n, 'cols': best_cols, 'cv': best_cv}
        
        cal_loss = log_loss(feat[target], cal_oof, labels=[0, 1])
        log.info(f"  ** BEST: cfg={best_config['name']} n={best_n}, CV={best_cv:.4f}, Cal={cal_loss:.4f}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V32 FAST SUMMARY")
    log.info(f"{'='*70}")
    
    for target in TARGET_COLS:
        oof_l = log_loss(feat[target], all_oof[target], labels=[0, 1])
        cal_l = log_loss(feat[target], all_cal[target], labels=[0, 1])
        tr = feat[target].mean()
        log.info(f"  {target}: OOF={oof_l:.4f} Cal={cal_l:.4f} Rate={tr:.3f}")
    
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  V32 Avg Cal OOF: {avg_cal:.4f}")
    
    return all_cal

if __name__ == "__main__":
    main()
