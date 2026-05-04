"""
V33 — V10 Strengths + LOSO CV + Rolling + Personalization

Uses the exact same pipeline as V10 (02_feature_engineering.py) to avoid
memory issues and merge-key problems.
10 seeds × 3 configs × 3 n_feat = 90 models/target (fast).
"""

import sys, re, json, warnings, logging, importlib.util
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

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SPLITS = 10  # LOSO

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

# ── Load features ──
def load_features():
    """Load features using 02_feature_engineering pipeline."""
    # Load parquet data
    spec_ld = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    ld = importlib.util.module_from_spec(spec_ld)
    spec_ld.loader.exec_module(ld)
    
    # Load labels
    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    
    # Build parquet_dfs the V10 way
    data_dir = DATA_RAW / "ch2025_data_items"
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet", "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }
    
    parquet_dfs = {}
    train_dates = set(labels['lifelog_date'].dt.date.astype(str).unique())
    for name, fname in parquet_names.items():
        df = pd.read_parquet(data_dir / fname)
        df = ld.build_merge_key(df)
        df = df[df['date'].astype(str).isin(train_dates)]
        parquet_dfs[name] = df
    
    # Build features via existing pipeline
    spec_fe = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    fe = importlib.util.module_from_spec(spec_fe)
    spec_fe.loader.exec_module(fe)
    
    feat = fe.create_day_features(parquet_dfs, labels)
    log.info(f"  Features loaded: {feat.shape}")
    
    return feat

# ── Feature additions ──
def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_rolling(df, cols, windows=[3, 7]):
    df = df.copy().sort_values(['subject_id', 'date'])
    new_cols = []
    for col in cols:
        grp = df.groupby('subject_id')[col]
        for w in windows:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_rm{w}'] = rm.values
            new_cols.append(f'{col}_rm{w}')
    return df, new_cols

def add_personalization(df, feature_cols, stats=None):
    df = df.copy()
    personal_cols = []
    computed_stats = {} if stats is None else stats
    for col in feature_cols:
        if stats is None:
            col_filled = df[col].fillna(0)
            subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
            subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
            subj_stats = subj_stats.reset_index()
            df = df.merge(subj_stats, on='subject_id', how='left')
            computed_stats[col] = {
                'mean': df[f'{col}_subj_mean'].mean(),
                'std': max(df[f'{col}_subj_std'].max(), 1e-6),
            }
        else:
            df[f'{col}_subj_mean'] = stats[col]['mean']
            df[f'{col}_subj_std'] = stats[col]['std']
        mask_std_zero = (df[f'{col}_subj_std'] == 0)
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(mask_std_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std'])
        personal_cols.append(f'{col}_zscore')
    return df, personal_cols, computed_stats

# ── Feature ranking ──
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

# ── Hyper params ──
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
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train({**cfg, 'scale_pos_weight': spw}, ds, num_boost_round=cfg['n_estimators'],
                         valid_sets=[vd], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
    return oof

def main():
    log.info("=" * 70)
    log.info("V33 — V10 Strengths + LOSO CV + Rolling + Personalization")
    log.info("=" * 70)
    
    feat = load_features()
    
    # Remove leakage from feature list
    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols if '_rm' not in c and '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    log.info(f"  Base features (from pipeline): {len(raw_base)}")
    
    # Remove constant/collinear from base
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    log.info(f"  After constant/collinear removal: {len(raw_base)}")
    
    # Add rolling mean
    feat, rolling_cols = add_rolling(feat, raw_base)
    log.info(f"  Rolling mean added: {len(rolling_cols)}")
    
    # Add personalization
    all_after_rolling = get_feature_cols(feat)
    base_plus_rolling = [c for c in all_after_rolling if '_subj_mean' not in c and '_zscore' not in c]
    feat, personal_cols, stats = add_personalization(feat, base_plus_rolling)
    log.info(f"  Personalization added: {len(personal_cols)}")
    
    # Total feature pool
    total_cols = get_feature_cols(feat)
    total_cols = [c for c in total_cols if c not in META_COLS and c not in set(TARGET_COLS)]
    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in total_cols if c not in leak_all]
    log.info(f"  Total feature pool: {len(total_cols)}, available (no leak): {len(available)}")
    
    # Per-target tuning
    log.info(f"\n=== Per-target tuning (LOSO {N_SPLITS}-fold, {len(SEEDS)} seeds) ===")
    
    all_oof = {}; all_cal = {}; all_best = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        best_cv = float('inf'); best_cfg = None; best_n = None; best_cols = None
        
        for cfg in LGB_CONFIGS:
            for n_feat in [10, 20, 30]:
                ranked = rank_features(feat, available, target, seed=42)
                sel_cols = [c for c, _ in ranked[:n_feat]]
                oof = lgb_cv(feat, sel_cols, target, SEEDS)
                cv = log_loss(feat[target], oof.mean(axis=1), labels=[0,1])
                log.info(f"  cfg={cfg['name']} n={n_feat}: cv={cv:.4f}")
                if cv < best_cv:
                    best_cv = cv; best_cfg = cfg; best_n = n_feat; best_cols = sel_cols
        
        oof = lgb_cv(feat, best_cols, target, SEEDS)
        cal = mm(oof.mean(axis=1), feat[target].values)
        all_oof[target] = oof.mean(axis=1)
        all_cal[target] = cal
        all_best[target] = {'config': best_cfg['name'], 'n': best_n, 'cv': best_cv}
        cal_loss = log_loss(feat[target], cal, labels=[0,1])
        log.info(f"  ** BEST: {best_cfg['name']} n={best_n}, CV={best_cv:.4f}, Cal={cal_loss:.4f}")
    
    # Summary
    log.info(f"\n{'='*50}")
    log.info("V33 SUMMARY")
    for t in TARGET_COLS:
        oof_l = log_loss(feat[t], all_oof[t], labels=[0,1])
        cal_l = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"  {t}: OOF={oof_l:.4f} Cal={cal_l:.4f}")
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  V33 Avg Cal OOF: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal OOF: 0.6038")
    log.info(f"  Δ: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")
    log.info(f"  V8 Submission: 0.6537")

if __name__ == "__main__":
    main()
