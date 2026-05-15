"""
V39 — Feature Interaction + Polynomial Features + LGBM

Strategy:
1. Key feature interactions (step × hr, screen × usage, gps × activity)
2. Polynomial features (degree 2) for top-10 features
3. LGBM with expanded feature set
4. 10 seeds × LOSO CV
5. Compare: base vs interactions vs poly

Reference: V10(0.6038), V33(rolling+personalization)
"""

import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
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

def add_interactions(feat, base_cols, top_n=10):
    """Add pairwise interactions of top features."""
    ranked = []
    for t in TARGET_COLS[:1]:  # Use Q1 for ranking
        r = rank_features(feat, base_cols, t, seed=42)
        ranked.append(r)
    # Use top 10 from Q1 (most consistent)
    top_cols = [c for c, _ in ranked[0][:top_n]]
    log.info(f"  Top {top_n} interaction features: {top_cols}")

    feat = feat.copy()
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            name = f'{top_cols[i]}_x_{top_cols[j]}'
            feat[name] = (feat[top_cols[i]].fillna(0) * feat[top_cols[j]].fillna(0))
    return feat, [c for c in feat.columns if '_x_' in c]

def add_poly(feat, base_cols, top_n=10, degree=2):
    """Add polynomial features of top features."""
    ranked = []
    for t in TARGET_COLS[:1]:
        r = rank_features(feat, base_cols, t, seed=42)
        ranked.append(r)
    top_cols = [c for c, _ in ranked[0][:top_n]]

    feat = feat.copy()
    X_top = feat[top_cols].fillna(0).values
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_top)
    # Skip original (first degree-1 features) and bias
    poly_cols = poly.get_feature_names_out(top_cols)
    poly_feature_names = [f'poly_{c}' for c in poly_cols[1:]]  # skip x0, x1, ... originals
    for i, name in enumerate(poly_feature_names):
        feat[name] = X_poly[:, i+1]
    return feat, poly_feature_names

def run_loso_lgb(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]

    oof = np.zeros((len(y), len(seeds)))
    for si, s in enumerate(seeds):
        cfg = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
            'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
            'random_state': s, 'scale_pos_weight': spw,
        }
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
    return oof.mean(axis=1)

def main():
    log.info("=" * 70)
    log.info("V39 — Feature Interaction + Polynomial + LGBM")
    log.info("=" * 70)

    feat = load_features()

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    feat, personal_cols = add_personalization(feat, raw_base)

    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in get_feature_cols(feat) if c not in META_COLS and c not in set(TARGET_COLS) and c not in leak_all]
    log.info(f"  Base features: {len(raw_base)} + personalization={len(personal_cols)}")
    log.info(f"  Available: {len(available)}")

    # Feature sets to compare
    ranked = rank_features(feat, available, TARGET_COLS[0], seed=42)
    feat_inter = feat.copy()
    feat_poly = feat.copy()

    log.info(f"\n=== Feature expansion ===")
    feat_inter, inter_cols = add_interactions(feat_inter, available, top_n=10)
    feat_poly, poly_cols = add_poly(feat_poly, available, top_n=10, degree=2)
    log.info(f"  Interactions added: {len(inter_cols)}")
    log.info(f"  Polynomial added: {len(poly_cols)}")

    # For each target: try base, +interactions, +poly
    log.info(f"\n=== Per-target tuning ===")

    all_results = {}
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()

        # Select top-20 from available for base
        sel_base = [c for c, _ in ranked[:20]]

        # Option 1: Base only
        oof_base = run_loso_lgb(feat, sel_base, target, SEEDS)
        cal_base = mean_match(oof_base, train_rate)
        cv_base = log_loss(feat[target], cal_base, labels=[0,1])
        log.info(f"  Base (top-20): {cv_base:.4f}")

        # Option 2: Base + interactions
        sel_inter = sel_base + [c for c in inter_cols if c in feat_inter.columns]
        oof_inter = run_loso_lgb(feat_inter, sel_inter, target, SEEDS)
        cal_inter = mean_match(oof_inter, train_rate)
        cv_inter = log_loss(feat[target], cal_inter, labels=[0,1])
        log.info(f"  +Interactions ({len(inter_cols)}): {cv_inter:.4f}")

        # Option 3: Base + poly
        sel_poly = sel_base + poly_cols
        oof_poly = run_loso_lgb(feat_poly, sel_poly, target, SEEDS)
        cal_poly = mean_match(oof_poly, train_rate)
        cv_poly = log_loss(feat[target], cal_poly, labels=[0,1])
        log.info(f"  +Poly({len(poly_cols)}): {cv_poly:.4f}")

        # Pick best
        best_cv = min(cv_base, cv_inter, cv_poly)
        best_name = ['base', '+inter', '+poly'][[cv_base, cv_inter, cv_poly].index(best_cv)]
        best_cal = [cal_base, cal_inter, cal_poly][[cv_base, cv_inter, cv_poly].index(best_cv)]
        all_results[target] = {'best': best_name, 'cv': best_cv, 'cal': best_cal}
        log.info(f"  ✅ Best: {best_name} ({best_cv:.4f})")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V39 SUMMARY")
    for t in TARGET_COLS:
        r = all_results[t]
        cal_l = log_loss(feat[t], r['cal'], labels=[0,1])
        log.info(f"  {t}: best={r['best']} Cal={cal_l:.4f}")

    avg_cal = np.mean([log_loss(feat[t], all_results[t]['cal'], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ vs V10: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

if __name__ == "__main__":
    main()
