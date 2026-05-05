"""
V35 — XGBoost GPU + LightGBM Hybrid Blend

Strategy:
1. XGBoost (GPU)와 LightGBM 모두 훈련
2. per-target로 최적 blend weight 탐색 (LOSO CV)
3. Feature pool: V10 base + personalization (no rolling)
4. 10 seeds × 3 algos (XGB-1, XGB-2, LGB) = 30 models/target

Reference: V10(0.6038 avg cal), FT-Transformer V2(0.5847 avg AUC)
"""

import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import xgboost as xgb
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

SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for memory
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

def sanitize(n):
    import re
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

# ── Load features ──
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

# ── XGB GPU ──
def train_xgb(X_tr, y_tr, X_va, params, n_rounds, seed):
    model = xgb.XGBClassifier(**params, random_state=seed, verbosity=0, n_jobs=-1,
                               tree_method='hist')
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict_proba(X_va)[:, 1]

# ── CV ──
def run_loso_cv(feat, cols, target, seeds, algo='lgbm', target_cfg=None):
    """Run LOSO CV and return oof predictions per seed."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)

    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]

    for si, s in enumerate(seeds):
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            y_tr, y_va = y[tr], y[va]

            if algo == 'lgbm':
                cfg = {**{
                    'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
                    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                    'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
                    'random_state': s, 'scale_pos_weight': spw,
                }}
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose':'-1'})
                vd = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds, params={'verbose':'-1'})
                m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof[va, si] = m.predict(X_va)

            elif algo == 'xgb':
                params = {
                    'objective':'binary:logistic','max_depth':4,'learning_rate':0.03,
                    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                    'min_child_weight':10,'tree_method':'hist','n_estimators':500,
                    'scale_pos_weight': spw, 'early_stopping_rounds': 50,
                    'random_state': s, 'verbosity': 0, 'n_jobs': -1,
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                oof[va, si] = model.predict_proba(X_va)[:, 1]

    return oof

def main():
    log.info("=" * 70)
    log.info("V35 — XGBoost GPU + LightGBM Hybrid Blend")
    log.info("=" * 70)

    feat = load_features()

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c
                and '_rm' not in c]
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    log.info(f"  Base features: {len(raw_base)}")

    feat, personal_cols = add_personalization(feat, raw_base)
    log.info(f"  Personalization added: {len(personal_cols)}")

    total_cols = get_feature_cols(feat)
    total_cols = [c for c in total_cols if c not in META_COLS and c not in set(TARGET_COLS)]
    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in total_cols if c not in leak_all]
    log.info(f"  Available features: {len(available)}")

    # Per-target tuning
    log.info(f"\n=== Per-target tuning (LOSO 10-fold, {len(SEEDS)} seeds, 3 algos) ===")

    all_oof = {}
    all_cal = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()

        ranked = rank_features(feat, available, target, seed=42)
        sel_cols = [c for c, _ in ranked[:20]]
        log.info(f"  Selected top-20 features")

        # Train LGBM
        oof_lgb = run_loso_cv(feat, sel_cols, target, SEEDS, algo='lgbm')
        oof_lgb_avg = np.clip(oof_lgb.mean(axis=1), 0.0001, 0.9999)
        # Train XGB
        oof_xgb = run_loso_cv(feat, sel_cols, target, SEEDS, algo='xgb')
        oof_xgb_avg = np.clip(oof_xgb.mean(axis=1), 0.0001, 0.9999)

        # Find best blend weight
        best_w = 0.5; best_loss = float('inf')
        for w in np.arange(0.0, 1.05, 0.05):
            blend = w * oof_xgb_avg + (1-w) * oof_lgb_avg
            loss = log_loss(feat[target], blend, labels=[0,1])
            if loss < best_loss:
                best_loss = loss; best_w = w

        cal_blend = mean_match(best_w * oof_xgb_avg + (1-best_w) * oof_lgb_avg, train_rate)
        cal_loss = log_loss(feat[target], cal_blend, labels=[0,1])

        all_oof[target] = best_w * oof_xgb_avg + (1-best_w) * oof_lgb_avg
        all_cal[target] = cal_blend

        oof_lgb_loss = log_loss(feat[target], oof_lgb_avg, labels=[0,1])
        oof_xgb_loss = log_loss(feat[target], oof_xgb_avg, labels=[0,1])
        log.info(f"  LGBM: {oof_lgb_loss:.4f}  XGB: {oof_xgb_loss:.4f}  Blend(w_xgb={best_w:.2f}): {best_loss:.4f}  Cal: {cal_loss:.4f}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V35 SUMMARY")
    log.info(f"{'='*70}")
    for t in TARGET_COLS:
        oof_l = log_loss(feat[t], all_oof[t], labels=[0,1])
        cal_l = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"  {t}: OOF={oof_l:.4f} Cal={cal_l:.4f}")

    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    avg_oof = np.mean([log_loss(feat[t], all_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  Avg OOF:  {avg_oof:.4f}")
    log.info(f"  Avg Cal:  {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ vs V10: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

if __name__ == "__main__":
    main()
