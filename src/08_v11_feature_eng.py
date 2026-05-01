"""
08_v11_feature_eng.py - V11: Enhanced feature engineering on top of V10
"""
import sys
import re
import json
import warnings
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, "src")
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 5
LGB_CONSERVATIVE = {
    "objective": "binary", "metric": "binary_logloss",
    "num_leaves": 15, "max_depth": 4,
    "learning_rate": 0.03, "n_estimators": 500,
    "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 1.0, "reg_lambda": 3.0,
    "min_child_samples": 10,
    "force_row_wise": True, "n_jobs": -1,
    "verbose": -1,
}

LEAKAGE_FEATURES_S = {
    "wLight_w_light_mean", "wLight_w_light_std", "wLight_w_light_min", "wLight_w_light_max", "wLight_w_light_count",
    "wHr_hr_mean", "wHr_hr_std", "wHr_hr_min", "wHr_hr_max", "wHr_hr_median", "wHr_hr_count",
    "wPedo_pedo_step_mean", "wPedo_pedo_step_sum",
    "wPedo_pedo_step_frequency_mean", "wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean", "wPedo_pedo_running_step_sum",
    "wPedo_pedo_walking_step_mean", "wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean", "wPedo_pedo_distance_sum",
    "wPedo_pedo_speed_mean", "wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean", "wPedo_pedo_burned_calories_sum",
}
LEAKAGE_FEATURES_Q = {
    "wHr_hr_mean", "wHr_hr_std", "wHr_hr_min", "wHr_hr_max", "wHr_hr_median", "wHr_hr_count",
}

def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def get_feature_cols(feat):
    return [c for c in feat.columns
            if c not in META_COLS | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def remove_leakage_features(feature_cols, target):
    if target.startswith("S"):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_S]
    elif target.startswith("Q"):
        return [c for c in feature_cols if c not in LEAKAGE_FEATURES_Q]
    return feature_cols

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        subj_stats = col_filled.groupby(df["subject_id"]).agg(["mean", "std"])
        subj_stats.columns = [f"{col}_subj_mean", f"{col}_subj_std"]
        subj_stats = subj_stats.reset_index()
        merged = df.merge(subj_stats, on="subject_id", how="left")
        mask_std_zero = merged[f"{col}_subj_std"] == 0
        mask_null = df[col].isnull()
        merged[f"{col}_zscore"] = np.where(
            mask_std_zero | mask_null, 0.0,
            (merged[col] - merged[f"{col}_subj_mean"]) / merged[f"{col}_subj_std"]
        )
        personal_cols.append(f"{col}_zscore")
        df = merged
    return df, personal_cols

# V11 NEW: Temporal features (rolling windows across days per subject)
def add_temporal_features(feat):
    """Add rolling window features across days per subject."""
    feat = feat.copy()
    # Sort by subject and date
    feat = feat.sort_values(["subject_id", "date"]).reset_index(drop=True)
    
    new_cols = []
    numeric_cols = [c for c in feat.columns if c not in META_COLS | set(TARGET_COLS) 
                    and feat[c].dtype in [np.float64, np.int64, float, int]]
    
    for col in numeric_cols:
        # Rolling 3-day mean
        for sid in feat["subject_id"].unique():
            mask = feat["subject_id"] == sid
            indices = feat[mask].index
            vals = feat.loc[indices, col].values
            if len(vals) >= 3:
                for i in range(2, len(vals)):
                    feat.iloc[indices[i], feat.columns.get_loc(col)] = vals[i]
            
        # 3-day rolling mean
        vals = feat[col].fillna(0)
        for sid in feat["subject_id"].unique():
            mask = feat["subject_id"] == sid
            indices = feat[mask].index.tolist()
            svals = [vals.iloc[i] for i in indices]
            for i in range(1, len(svals)):
                start = max(0, i - 2)
                svals[i] = np.mean(svals[start:i+1])
            for j, idx in enumerate(indices):
                feat.iloc[idx, feat.columns.get_loc(col)] = svals[j]
        
        new_cols.append(f"{col}_rolling3m")
    
    return feat, new_cols

# V11 NEW: Date-based features
def add_date_features(feat):
    """Add day_of_week, is_weekend, month, quarter features."""
    feat = feat.copy()
    dates = pd.to_datetime(feat["date"], errors="coerce")
    feat["day_of_week"] = dates.dt.dayofweek
    feat["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
    feat["month"] = dates.dt.month
    feat["quarter"] = dates.dt.quarter
    feat["day_of_month"] = dates.dt.day
    return feat

# V11 NEW: Cross-sensor interaction features
def add_interaction_features(feat):
    """Add cross-sensor interaction features."""
    feat = feat.copy()
    interactions = []
    
    # Screen usage × Activity
    if "mScreenStatus_m_screen_use_mean" in feat.columns and "mActivity_m_activity_mean" in feat.columns:
        feat["interaction_screen_activity"] = feat["mScreenStatus_m_screen_use_mean"] * feat["mActivity_m_activity_mean"]
        interactions.append("interaction_screen_activity")
    
    # WiFi RSSI × BLE RSSI
    if "mWifi_wifi_avg_rssi_mean" in feat.columns and "mBle_ble_avg_rssi_mean" in feat.columns:
        feat["interaction_wifi_ble_rssi"] = feat["mWifi_wifi_avg_rssi_mean"] * feat["mBle_ble_avg_rssi_mean"]
        interactions.append("interaction_wifi_ble_rssi")
    
    # Step count × Screen use (activity vs sedentary)
    if "wPedo_pedo_step_mean" in feat.columns and "mScreenStatus_m_screen_use_mean" in feat.columns:
        feat["interaction_step_screen"] = feat["wPedo_pedo_step_mean"] * feat["mScreenStatus_m_screen_use_mean"]
        interactions.append("interaction_step_screen")
    
    # Heart rate × Activity
    if "wHr_hr_mean" in feat.columns and "mActivity_m_activity_mean" in feat.columns:
        feat["interaction_hr_activity"] = feat["wHr_hr_mean"] * feat["mActivity_m_activity_mean"]
        interactions.append("interaction_hr_activity")
    
    # GPS speed × Activity
    if "mGps_gps_avg_speed_mean" in feat.columns and "mActivity_m_activity_mean" in feat.columns:
        feat["interaction_gps_activity"] = feat["mGps_gps_avg_speed_mean"] * feat["mActivity_m_activity_mean"]
        interactions.append("interaction_gps_activity")
    
    # Charging × Screen use
    if "mACStatus_m_charging_mean" in feat.columns and "mScreenStatus_m_screen_use_mean" in feat.columns:
        feat["interaction_charging_screen"] = feat["mACStatus_m_charging_mean"] * feat["mScreenStatus_m_screen_use_mean"]
        interactions.append("interaction_charging_screen")
    
    # Light × Screen use
    if "mLight_m_light_mean" in feat.columns and "mScreenStatus_m_screen_use_mean" in feat.columns:
        feat["interaction_light_screen"] = feat["mLight_m_light_mean"] * feat["mScreenStatus_m_screen_use_mean"]
        interactions.append("interaction_light_screen")
    
    # Ambience indoor ratio (total ambience - outdoor)
    ambience_cols = [c for c in feat.columns if c.startswith("mAmbience_ambience_") and c.endswith("_sum")]
    if ambience_cols:
        total_ambience = feat[ambience_cols].sum(axis=1)
        feat["total_ambience"] = total_ambience
        interactions.append("total_ambience")
    
    # Step ratio (running / total steps)
    if "wPedo_pedo_running_step_mean" in feat.columns and "wPedo_pedo_step_mean" in feat.columns:
        feat["step_running_ratio"] = feat["wPedo_pedo_running_step_mean"] / (feat["wPedo_pedo_step_mean"] + 1e-10)
        interactions.append("step_running_ratio")
    
    # Distance per step
    if "wPedo_pedo_distance_mean" in feat.columns and "wPedo_pedo_step_mean" in feat.columns:
        feat["distance_per_step"] = feat["wPedo_pedo_distance_mean"] / (feat["wPedo_pedo_step_mean"] + 1e-10)
        interactions.append("distance_per_step")
    
    return feat, interactions

# V11 NEW: Advanced statistics
def add_advanced_stats(feat):
    """Add percentile, skewness, kurtosis features."""
    feat = feat.copy()
    stats_cols = []
    
    numeric_cols = [c for c in feat.columns if c not in META_COLS | set(TARGET_COLS)
                    and feat[c].dtype in [np.float64, np.int64, float, int]
                    and feat[c].notna().sum() > 10]
    
    for col in numeric_cols:
        vals = feat[col].values
        if np.all(np.isnan(vals)):
            continue
        
        feat[f"{col}_p25"] = np.percentile(vals, 25)
        feat[f"{col}_p50"] = np.percentile(vals, 50)
        feat[f"{col}_p75"] = np.percentile(vals, 75)
        feat[f"{col}_range"] = np.nanmax(vals) - np.nanmin(vals)
        
        if feat[col].std() > 0:
            feat[f"{col}_skewness"] = feat[col].skew()
            feat[f"{col}_kurtosis"] = feat[col].kurtosis()
        
        stats_cols.extend([f"{col}_p25", f"{col}_p50", f"{col}_p75", f"{col}_range"])
        if feat[col].std() > 0:
            stats_cols.extend([f"{col}_skewness", f"{col}_kurtosis"])
    
    return feat, stats_cols

def rank_features(feat, feature_cols, target, random_seed=42):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos
    
    params = {
        "objective": "binary", "metric": "binary_logloss", "verbose": -1,
        "num_leaves": 15, "max_depth": 4, "learning_rate": 0.03,
        "n_estimators": 100, "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 1.0, "reg_lambda": 3.0,
        "scale_pos_weight": spw, "random_state": random_seed,
        "min_child_samples": 10,
        "force_row_wise": True, "n_jobs": -1,
    }
    sanitized = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={"verbose": "-1"})
    model = lgb.train(params, ds, num_boost_round=100)
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    return ranked

LGB_CONFS = [
    {"name": "C1", "nl": 8, "md": 3, "lr": 0.02, "ne": 200, "ss": 0.6, "cst": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
    {"name": "C2", "nl": 10, "md": 3, "lr": 0.03, "ne": 300, "ss": 0.7, "cst": 0.7, "ra": 1.0, "rl": 3.0, "mc": 10},
    {"name": "C3", "nl": 12, "md": 4, "lr": 0.03, "ne": 200, "ss": 0.7, "cst": 0.7, "ra": 1.0, "rl": 3.0, "mc": 10},
    {"name": "C4", "nl": 15, "md": 4, "lr": 0.03, "ne": 500, "ss": 0.7, "cst": 0.7, "ra": 1.0, "rl": 3.0, "mc": 10},
    {"name": "C5", "nl": 20, "md": 5, "lr": 0.02, "ne": 300, "ss": 0.7, "cst": 0.7, "ra": 0.5, "rl": 2.0, "mc": 8},
    {"name": "C6", "nl": 6, "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cst": 0.5, "ra": 5.0, "rl": 10.0, "mc": 20},
]

def lgb_cv_predict(feat, selected_cols, target, seeds, spw):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    all_fold_losses = {i: [] for i in range(N_SPLITS)}
    sanitized = [sanitize(c) for c in selected_cols]
    
    for seed_i, seed in enumerate(seeds):
        cfg = {**LGB_CONSERVATIVE, "random_state": seed}
        for fold, (train_idx, val_idx) in enumerate(gkf.split(feat, y, feat["subject_id"])):
            X_tr = feat.iloc[train_idx][selected_cols].fillna(0).values
            X_va = feat.iloc[val_idx][selected_cols].fillna(0).values
            y_tr, y_va = y[train_idx], y[val_idx]
            train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitized, params={"verbose": "-1"})
            val_set = lgb.Dataset(X_va, label=y_va, feature_name=sanitized, reference=train_set, params={"verbose": "-1"})
            params = {**cfg, "scale_pos_weight": spw}
            model = lgb.train(params, train_set, num_boost_round=cfg["n_estimators"],
                            valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pred = model.predict(X_va)
            oof_full[val_idx, seed_i] = pred
            fold_loss = log_loss(y_va, pred, labels=[0, 1])
            all_fold_losses[fold].append(fold_loss)
    
    oof_avg = oof_full.mean(axis=1)
    fold_avg_losses = [np.mean(all_fold_losses[i]) for i in range(N_SPLITS)]
    cv_loss = log_loss(y, oof_avg, labels=[0, 1])
    cv_std = np.std(fold_avg_losses)
    return oof_avg, oof_full, cv_loss, cv_std, fold_avg_losses

def simple_mean_match(pred, target_rate):
    shift = target_rate - pred.mean()
    calibrated = pred + shift
    calibrated = np.clip(calibrated, 0.0001, 0.9999)
    return calibrated

def tune_target(feat, feature_cols, target):
    best_config = None
    best_cv = float("inf")
    best_oof = None
    best_selected_cols = None
    ranked = rank_features(feat, feature_cols, target)
    
    for n_feat in [10, 20, 30, 40]:
        if n_feat > len(ranked):
            continue
        selected_cols = [r[0] for r in ranked[:n_feat]]
        y = feat[target].values
        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos
        
        for cfg in LGB_CONFS:
            test_cfg = {**LGB_CONSERVATIVE,
                       "num_leaves": cfg["nl"], "max_depth": cfg["md"],
                       "learning_rate": cfg["lr"], "n_estimators": cfg["ne"],
                       "subsample": cfg["ss"], "colsample_bytree": cfg["cst"],
                       "reg_alpha": cfg["ra"], "reg_lambda": cfg["rl"],
                       "min_child_samples": cfg["mc"],}
            oof_avg, oof_full, cv_loss, cv_std, fold_losses = lgb_cv_predict(feat, selected_cols, target, RANDOM_SEEDS, spw)
            train_rate = y.mean()
            pred_mean_shift = abs(oof_avg.mean() - train_rate)
            score = cv_loss + 0.5 * cv_std + 0.1 * pred_mean_shift
            if score < best_cv:
                best_cv = score
                best_config = {**cfg, "_n_feats": n_feat}
                best_oof = oof_avg
                best_selected_cols = selected_cols
    
    return best_config, best_selected_cols, best_oof

def main():
    log.info("=" * 70)
    log.info("V11: Enhanced Feature Engineering Pipeline")
    log.info("=" * 70)
    
    # Step 1: Load features
    log.info("\n=== Step 1: Load features ===")
    feat = pd.read_parquet("data_processed/features.parquet")
    log.info(f"Loaded features: {feat.shape}")
    
    # Step 2: V11 Feature Engineering
    log.info("\n=== Step 2: V11 Feature Engineering ===")
    
    # 2a: Date features
    feat = add_date_features(feat)
    log.info(f"  Added date features: day_of_week, is_weekend, month, quarter, day_of_month")
    
    # 2b: Interaction features
    feat, interactions = add_interaction_features(feat)
    log.info(f"  Added {len(interactions)} interaction features: {interactions}")
    
    # 2c: Advanced statistics (percentile, skewness, kurtosis, range)
    feat, stats_cols = add_advanced_stats(feat)
    log.info(f"  Added advanced stats features (percentile, skewness, kurtosis, range)")
    
    # 2d: Personalization
    feature_cols = get_feature_cols(feat)
    feat, personal_cols = add_personalization(feat, feature_cols)
    log.info(f"  Personalization columns: {len(personal_cols)}")
    
    log.info(f"Total features after V11 enhancements: {len(feat.columns) - len(META_COLS) - len(TARGET_COLS)}")
    
    # Step 3: Per-target model tuning with OOF
    log.info("\n=== Step 3: Per-target model tuning ===")
    
    train_rate = {}
    for target in TARGET_COLS:
        y = feat[target].values
        train_rate[target] = float(y.mean())
    
    all_best_configs = {}
    all_best_cols = {}
    all_oof = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (train_rate={train_rate[target]:.3f}) ---")
        leak_free_cols = remove_leakage_features(feature_cols + personal_cols, target)
        log.info(f"  Leakage-free cols: {len(leak_free_cols)}")
        
        ranked = rank_features(feat, leak_free_cols, target)
        log.info(f"  Top 10: {[r[0] for r in ranked[:10]]}")
        
        best_config, best_cols, best_oof = tune_target(feat, leak_free_cols, target)
        all_best_configs[target] = best_config
        all_best_cols[target] = best_cols
        all_oof[target] = best_oof
        
        y = feat[target].values
        oof_cv = log_loss(y, best_oof, labels=[0, 1])
        log.info(f"  Best config: {best_config}")
        log.info(f"  OOF CV loss: {oof_cv:.4f}")
        log.info(f"  Selected {len(best_cols)} features")
    
    # Summary
    log.info("\n=== OOF CV Scores ===")
    log.info(f"{'Target':<6} {'OOF Loss':<12} {'OOF Mean':<12} {'Train Rate':<12}")
    for target in TARGET_COLS:
        y = feat[target].values
        oof_loss = log_loss(y, all_oof[target], labels=[0, 1])
        log.info(f"{target:<6} {oof_loss:<12.4f} {all_oof[target].mean():<12.4f} {train_rate[target]:<12.3f}")
    
    avg_oof = np.mean([log_loss(feat[t], all_oof[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"  Average raw OOF loss: {avg_oof:.4f}")
    
    # Calibration
    log.info("\n=== Calibration ===")
    calibrated_oof = {}
    for target in TARGET_COLS:
        cal = simple_mean_match(all_oof[target], train_rate[target])
        calibrated_oof[target] = cal
        cal_loss = log_loss(feat[target], cal, labels=[0, 1])
        log.info(f"  {target}: cal_loss={cal_loss:.4f}, mean={cal.mean():.4f}")
    
    avg_cal = np.mean([log_loss(feat[t], calibrated_oof[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"  Average calibrated OOF loss: {avg_cal:.4f}")
    log.info(f"  V10 avg cal OOF: 0.6038")
    log.info(f"  V11 vs V10 diff: {avg_cal - 0.6038:+.4f}")
    
    # Step 4: Generate submission
    log.info("\n=== Step 4: Generate submission ===")
    
    spec = importlib.util.spec_from_file_location("02_feature_engineering", Path("src/02_feature_engineering.py"))
    feat_eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feat_eng)
    spec2 = importlib.util.spec_from_file_location("01_load_data", Path("src/01_load_data.py"))
    ld_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ld_mod)
    
    parquet_dfs = {}
    data_dir = Path("data_raw/ch2025_data_items")
    parquet_names = {
        "mACStatus": "ch2025_mACStatus.parquet", "mActivity": "ch2025_mActivity.parquet",
        "mAmbience": "ch2025_mAmbience.parquet", "mBle": "ch2025_mBle.parquet",
        "mGps": "ch2025_mGps.parquet", "mLight": "ch2025_mLight.parquet",
        "mScreenStatus": "ch2025_mScreenStatus.parquet", "mUsageStats": "ch2025_mUsageStats.parquet",
        "mWifi": "ch2025_mWifi.parquet", "wHr": "ch2025_wHr.parquet",
        "wLight": "ch2025_wLight.parquet", "wPedo": "ch2025_wPedo.parquet",
    }
    
    sample = pd.read_csv("data_raw/ch2026_submission_sample.csv")
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
    sample["sleep_date"] = pd.to_datetime(sample["sleep_date"]).dt.date
    
    test_dates = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())
    
    for name, fname in parquet_names.items():
        df = pd.read_parquet(data_dir / fname)
        df = ld_mod.build_merge_key(df)
        df = df[df["date"].astype(str).isin(test_dates)]
        parquet_dfs[name] = df
    
    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    
    # Add V11 date features to test
    test_features = add_date_features(test_features)
    
    # Add V11 interaction features to test
    test_features, _ = add_interaction_features(test_features)
    
    # Add V11 advanced stats to test
    test_features, _ = add_advanced_stats(test_features)
    
    # Add personalization to test
    test_feat_cols = get_feature_cols(test_features)
    test_features, _ = add_personalization(test_features, test_feat_cols)
    
    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()
    
    for target in TARGET_COLS:
        log.info(f"\n  Training final models for {target}...")
        selected_cols = all_best_cols[target]
        
        y_all = feat[target].values
        X_all = feat[selected_cols].fillna(0).values
        test_X = test_features[selected_cols].fillna(0).values
        sanitized = [sanitize(c) for c in selected_cols]
        
        cfg = all_best_configs[target]
        n_pos = max((y_all == 1).sum(), 1)
        n_neg = (y_all == 0).sum()
        spw = n_neg / n_pos
        
        lgb_params = {**LGB_CONSERVATIVE,
                      "num_leaves": cfg["nl"], "max_depth": cfg["md"],
                      "learning_rate": cfg["lr"], "n_estimators": cfg["ne"],
                      "subsample": cfg["ss"], "colsample_bytree": cfg["cst"],
                      "reg_alpha": cfg["ra"], "reg_lambda": cfg["rl"],
                      "min_child_samples": cfg["mc"], "scale_pos_weight": spw,}
        
        all_preds = np.zeros(len(test_X))
        for seed_i, seed in enumerate(RANDOM_SEEDS):
            seed_params = {**lgb_params, "random_state": seed}
            ds_all = lgb.Dataset(X_all, label=y_all, feature_name=sanitized, params={"verbose": "-1"})
            model = lgb.train(seed_params, ds_all, num_boost_round=cfg["ne"])
            all_preds += model.predict(test_X)
            if (seed_i + 1) % 5 == 0:
                log.info(f"    [{target}] seed {seed_i + 1}/{N_SEEDS} done")
        
        all_preds /= N_SEEDS
        cal_preds = simple_mean_match(all_preds, train_rate[target])
        predictions[target] = cal_preds
        log.info(f"    {target}: mean={cal_preds.mean():.4f}, range=[{cal_preds.min():.4f}, {cal_preds.max():.4f}]")
    
    # Save
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT_DIR / f"submission_v11_{timestamp}.csv"
    predictions.to_csv(sub_path, index=False)
    log.info(f"\nSubmission saved: {sub_path}")
    
    meta = {
        "version": "v11",
        "submission_file": str(sub_path),
        "timestamp": timestamp,
        "n_samples": len(predictions),
        "n_seeds": N_SEEDS,
        "n_splits": N_SPLITS,
        "improvements": [
            "date features (day_of_week, is_weekend, month, quarter)",
            "cross-sensor interaction features",
            "advanced statistics (percentile, skewness, kurtosis, range)",
            "V10 baseline (leakage fix, personalization, mean-matching calibration)",
        ],
        "per_target": {},
    }
    
    for target in TARGET_COLS:
        oof_loss = log_loss(feat[target], all_oof[target], labels=[0, 1])
        cal_loss = log_loss(feat[target], calibrated_oof[target], labels=[0, 1])
        meta["per_target"][target] = {
            "config": all_best_configs[target],
            "n_features": len(all_best_cols[target]),
            "oof_cv_loss": float(oof_loss),
            "cal_oof_loss": float(cal_loss),
            "oof_mean": float(all_oof[target].mean()),
            "cal_mean": float(predictions[target].mean()),
            "train_rate": float(train_rate[target]),
            "pred_min": float(predictions[target].min()),
            "pred_max": float(predictions[target].max()),
        }
        log.info(f"\n  {target}: config={all_best_configs[target]}, features={len(all_best_cols[target])}")
        log.info(f"    OOF_loss={oof_loss:.4f}, cal_OOF={cal_loss:.4f}, test_mean={predictions[target].mean():.4f}")
    
    meta_path = sub_path.parent / f"meta_v11_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Metadata saved: {meta_path}")
    
    log.info(f"\n{'='*70}")
    log.info("V11 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'OOF Loss':<12} {'Cal OOF':<12} {'Test Mean':<12} {'Train Rate':<12}")
    for target in TARGET_COLS:
        oof_loss = log_loss(feat[target], all_oof[target], labels=[0, 1])
        cal_loss = log_loss(feat[target], calibrated_oof[target], labels=[0, 1])
        log.info(f"{target:<6} {oof_loss:<12.4f} {cal_loss:<12.4f} {predictions[target].mean():<12.4f} {train_rate[target]:<12.3f}")
    log.info(f"\nAverage cal OOF: {avg_cal:.4f} (V10: 0.6038, diff: {avg_cal - 0.6038:+.4f})")

if __name__ == "__main__":
    main()
