"""
V36 — Multi-Model Stacking (V10 LGBM + V32 personalization + LOSO CV)

Strategy:
1. Level-1 features: multiple LGBM variants + XGB per fold
2. Level-2: simple logistic regression meta-learner
3. Feature pool: V10 base + personalization + rolling (3d, 7d)
4. 5 seeds × 5 base models = 25 models/target

Reference: V10(0.6038), V32(rolling + personalization)
"""

import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

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

# ── Base model trainers ──
def get_base_models(cols, target, seeds, feat_ref=None):
    """Return list of (name, train_fn) for base models."""
    y = feat_ref[target].values
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    sn = [sanitize(c) for c in cols]

    def train_v1(seed):
        """V10-style LGBM."""
        params = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
            'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
            'random_state': seed, 'scale_pos_weight': spw,
        }
        return [('lgbm_v10', lambda tr, va: train_lgb(params, tr, va, sn, feat_ref, target, cols)) for seed in seeds]

    def train_v2(seed):
        """Aggressive LGBM (deep, strong regularization)."""
        params = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':300,
            'subsample':0.6,'colsample_bytree':0.5,'reg_alpha':2.0,'reg_lambda':5.0,
            'min_child_samples':20,'force_row_wise':True,'n_jobs':-1,
            'random_state': seed, 'scale_pos_weight': spw,
        }
        return [('lgbm_v2', lambda tr, va: train_lgb(params, tr, va, sn, feat_ref, target, cols)) for seed in seeds]

    def train_xgb1(seed):
        """XGB medium depth."""
        params = {
            'objective':'binary:logistic','max_depth':4,'learning_rate':0.03,
            'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_weight':10,'tree_method':'hist','n_estimators':500,
            'scale_pos_weight': spw,
        }
        return [('xgb_1', lambda tr, va: train_xgb(params, tr, va, feat_ref, target, cols)) for seed in seeds]

    def train_xgb2(seed):
        """XGB light."""
        params = {
            'objective':'binary:logistic','max_depth':3,'learning_rate':0.02,
            'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.5,'reg_lambda':1.0,
            'min_child_weight':15,'tree_method':'hist','n_estimators':300,
            'scale_pos_weight': spw,
        }
        return [('xgb_2', lambda tr, va: train_xgb(params, tr, va, feat_ref, target, cols)) for seed in seeds]

    return train_v1(0), train_v2(0), train_xgb1(0), train_xgb2(0)

def train_lgb(params, tr, va, sn, feat_ref, target, cols):
    y = feat_ref[target].values
    X_tr = feat_ref.iloc[tr][cols].fillna(0).values
    X_va = feat_ref.iloc[va][cols].fillna(0).values
    ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
    vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
    m = lgb.train(params, ds, num_boost_round=params['n_estimators'], valid_sets=[vd],
                 callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m.predict(X_va)

def train_xgb(params, tr, va, feat_ref, target, cols):
    X_tr = feat_ref.iloc[tr][cols].fillna(0).values
    X_va = feat_ref.iloc[va][cols].fillna(0).values
    y_tr, y_va = feat_ref[target].values[tr], feat_ref[target].values[va]
    model = xgb.XGBClassifier(**params, verbosity=0, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict_proba(X_va)[:, 1]

# ── Main ──
def main():
    log.info("=" * 70)
    log.info("V36 — Multi-Model Stacking")
    log.info("=" * 70)

    feat = load_features()

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c
                and '_rm' not in c]
    raw_base = [c for c in raw_base if c not in set(CONSTANT_COLS + COLLINEAR_DROP)]
    feat, personal_cols = add_personalization(feat, raw_base)
    feat, rolling_cols = add_rolling(feat, raw_base)
    log.info(f"  Features: base={len(raw_base)} + personalization={len(personal_cols)} + rolling={len(rolling_cols)}")

    total_cols = get_feature_cols(feat)
    total_cols = [c for c in total_cols if c not in META_COLS and c not in set(TARGET_COLS)]
    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in total_cols if c not in leak_all]
    log.info(f"  Available features: {len(available)}")

    ranked = rank_features(feat, available, TARGET_COLS[0], seed=42)
    sel_cols = [c for c, _ in ranked[:30]]
    log.info(f"  Selected top-30 features")

    y = None
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_stacking = None

    log.info(f"\n=== Stacking CV ({len(SEEDS)} seeds × 4 models) ===")

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()
        y = feat[target].values

        n_seeds = 5  # Use first 5 seeds for stacking (save time)
        seeds_used = SEEDS[:n_seeds]
        n_models = 4  # 4 model variants

        n_samples = len(y)
        # oof for each model: shape (n_samples, n_seeds)
        oof_lgb1 = np.zeros((n_samples, n_seeds))
        oof_lgb2 = np.zeros((n_samples, n_seeds))
        oof_xgb1 = np.zeros((n_samples, n_seeds))
        oof_xgb2 = np.zeros((n_samples, n_seeds))

        for si, s in enumerate(seeds_used):
            log.info(f"  seed {s}...")
            fold_idx = 0
            for tr, va in gkf.split(feat, y, feat['subject_id']):
                X_tr = feat.iloc[tr][sel_cols].fillna(0).values
                X_va = feat.iloc[va][sel_cols].fillna(0).values

                oof_lgb1[va, si] += train_lgb({
                    'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
                    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                    'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
                    'random_state': s, 'scale_pos_weight': ((y==0).sum())/max((y==1).sum(),1)
                }, tr, va, [sanitize(c) for c in sel_cols], feat, target, sel_cols)

                oof_lgb2[va, si] += train_lgb({
                    'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':300,
                    'subsample':0.6,'colsample_bytree':0.5,'reg_alpha':2.0,'reg_lambda':5.0,
                    'min_child_samples':20,'force_row_wise':True,'n_jobs':-1,
                    'random_state': s, 'scale_pos_weight': ((y==0).sum())/max((y==1).sum(),1)
                }, tr, va, [sanitize(c) for c in sel_cols], feat, target, sel_cols)

                oof_xgb1[va, si] += train_xgb({
                    'objective':'binary:logistic','max_depth':4,'learning_rate':0.03,
                    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                    'min_child_weight':10,'tree_method':'hist','n_estimators':500,
                    'scale_pos_weight': ((y==0).sum())/max((y==1).sum(),1),
                }, tr, va, feat, target, sel_cols)

                oof_xgb2[va, si] += train_xgb({
                    'objective':'binary:logistic','max_depth':3,'learning_rate':0.02,
                    'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.5,'reg_lambda':1.0,
                    'min_child_weight':15,'tree_method':'hist','n_estimators':300,
                    'scale_pos_weight': ((y==0).sum())/max((y==1).sum(),1),
                }, tr, va, feat, target, sel_cols)

                fold_idx += 1

        # Average over seeds
        meta_features = np.column_stack([
            oof_lgb1.mean(axis=1), oof_lgb2.mean(axis=1),
            oof_xgb1.mean(axis=1), oof_xgb2.mean(axis=1)
        ])

        # Train meta-learner via LOSO
        meta_oof = np.zeros(n_samples)
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            meta_tr = meta_features[tr]
            meta_va = meta_features[va]
            meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            meta_model.fit(meta_tr, y[tr])
            meta_oof[va] = meta_model.predict_proba(meta_va)[:, 1]

        cal = mean_match(meta_oof, train_rate)
        cal_loss = log_loss(y, cal, labels=[0,1])

        # Also check simple average
        simple_blend = np.clip(np.mean([oof_lgb1, oof_lgb2, oof_xgb1, oof_xgb2], axis=0).mean(axis=1), 0.0001, 0.9999)
        simple_cal = mean_match(simple_blend, train_rate)
        simple_loss = log_loss(y, simple_cal, labels=[0,1])

        best_method = 'stacking' if cal_loss <= simple_loss else 'simple_avg'
        best_loss = min(cal_loss, simple_loss)

        log.info(f"  Stacking: {cal_loss:.4f}  |  Simple avg: {simple_loss:.4f}  |  Best: {best_method} ({best_loss:.4f})")

        if best_method == 'stacking':
            oof_stacking = meta_oof
        else:
            oof_stacking = simple_blend

        if oof_stacking is None:
            oof_stacking = np.zeros(n_samples)

    # Final summary
    log.info(f"\n{'='*70}")
    log.info("V36 SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"  Stacking approach complete (see per-target logs above)")
    log.info(f"  V10 Avg Cal: 0.6038")

if __name__ == "__main__":
    main()
