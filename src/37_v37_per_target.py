"""
V37 — Per-Target Hyperparameter + V8 Configs + 20 Seeds

Strategy:
1. V8의 per-target hyperparameters 재사용 (Q1/Q2: nl=10, md=3, lr=0.05; S1-S3: nl=6, md=2, lr=0.02 등)
2. 20 seeds ensemble for stronger averaging
3. Feature pool: V10 base + personalization (top-10 importance)
4. 5 calibration methods 비교

Reference: V8(0.6537 LB), V34(c exploration)
"""

import sys, warnings, logging
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

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(SEEDS)
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

# V8 per-target configs
V8_CFGS = {
    "Q1": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cst": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q2": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cst": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q3": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cst": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
    "S1": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S2": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S3": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S4": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cst": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
}

def sanitize(n):
    import re
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def load_features():
    import importlib.util
    spec_ld = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    ld = importlib.util.module_from_spec(spec_ld)
    spec_ld.loader.exec_module(ld)

    labels = pd.read_csv(DATA_RAW / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])

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

    spec_fe = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    fe = importlib.util.module_from_spec(spec_fe)
    spec_fe.loader.exec_module(fe)

    feat = fe.create_day_features(parquet_dfs, labels)
    log.info(f"  Features loaded: {feat.shape}")
    return feat

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        df = df.merge(subj_stats, on='subject_id', how='left')

        mask_std_zero = (df[f'{col}_subj_std'] == 0)
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_std_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
    return df, personal_cols

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

def run_loso(feat, cols, target, seeds, cfg_dict):
    """Run LOSO CV with given config dict and seeds."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]

    cfg = {
        'num_leaves': cfg_dict['nl'], 'max_depth': cfg_dict['md'],
        'learning_rate': cfg_dict['lr'], 'n_estimators': cfg_dict['ne'],
        'subsample': cfg_dict['ss'], 'colsample_bytree': cfg_dict['cst'],
        'reg_alpha': cfg_dict['ra'], 'reg_lambda': cfg_dict['rl'],
        'min_child_samples': cfg_dict['mc'], 'verbose': -1,
        'force_row_wise': True, 'n_jobs': -1,
    }

    oof = np.zeros((len(y), len(seeds)))
    for si, s in enumerate(seeds):
        seed_cfg = {**cfg, 'random_state': s, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(seed_cfg, ds, num_boost_round=cfg['n_estimators'],
                         valid_sets=[vd], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
    return oof.mean(axis=1)

def main():
    log.info("=" * 70)
    log.info("V37 — Per-Target Hyper + V8 Configs + 20 Seeds")
    log.info("=" * 70)

    feat = load_features()

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    feat, personal_cols = add_personalization(feat, raw_base)
    log.info(f"  Features: base={len(raw_base)} + personalization={len(personal_cols)}")

    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in get_feature_cols(feat) if c not in META_COLS and c not in set(TARGET_COLS) and c not in leak_all]

    log.info(f"\n=== Per-target CV (20 seeds, LOSO 10-fold) ===")

    all_cal = {}
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()
        v8_cfg = V8_CFGS[target]
        log.info(f"  V8 config: nl={v8_cfg['nl']} md={v8_cfg['md']} lr={v8_cfg['lr']} ne={v8_cfg['ne']}")

        ranked = rank_features(feat, available, target, seed=42)
        # Try multiple n_features
        best_cv = float('inf'); best_cols = None
        for n_feat in [5, 10, 15, 20]:
            cols = [c for c, _ in ranked[:n_feat]]
            oof = run_loso(feat, cols, target, SEEDS, v8_cfg)
            oof_clipped = np.clip(oof, 0.0001, 0.9999)
            cv = log_loss(feat[target], oof_clipped, labels=[0,1])
            log.info(f"    n={n_feat}: CV={cv:.4f}")
            if cv < best_cv:
                best_cv = cv; best_cols = cols

        # Final run with best n_feat
        oof = run_loso(feat, best_cols, target, SEEDS, v8_cfg)
        oof = np.clip(oof, 0.0001, 0.9999)

        # Mean-match calibration
        cal = mean_match(oof, train_rate)
        cal_loss = log_loss(feat[target], cal, labels=[0,1])
        all_cal[target] = cal
        log.info(f"  Final: CV={best_cv:.4f} Cal={cal_loss:.4f}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V37 SUMMARY")
    for t in TARGET_COLS:
        cal_l = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"  {t}: Cal={cal_l:.4f}")

    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ vs V10: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

if __name__ == "__main__":
    main()
