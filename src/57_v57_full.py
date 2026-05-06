"""
V57 — Full 12-sensor feature engineering + GroupKFold LGBM ensemble
Optimizations:
  - Per-sensor daily aggregation → discard raw immediately
  - GPS 1D ndarray of dicts handled correctly
  - Z-score personalization
  - External data (age, bmi, gender) merged if available
"""
import sys, gc, time, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ─── paths ───────────────────────────────────────────────────────────────
ROOT = Path("/home/mwoo423/projects/dacon2")
DATA_PROCESSED = ROOT / "data_processed"
DATA_RAW = ROOT / "data_raw"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
parquet_dir = DATA_RAW / "ch2025_data_items"


# ─── helper: parse dict-array columns ────────────────────────────────────
def _extract_dict_array(series, keys):
    """
    series: pandas Series of numpy-1d-arrays of dicts
    keys:   list of dict keys to extract, e.g. ['rssi', 'bssid']
    returns: DataFrame with per-row stats for each key
    """
    out = {}
    for k in keys:
        out[f"{k}_mean"] = []
        out[f"{k}_std"] = []
        out[f"{k}_max"] = []
        out[f"{k}_min"] = []
        out[f"{k}_cnt"] = []

    for val in series:
        if not isinstance(val, np.ndarray) or val.ndim != 1 or len(val) == 0:
            for k in keys:
                out[f"{k}_mean"].append(np.nan)
                out[f"{k}_std"].append(0.0)
                out[f"{k}_max"].append(np.nan)
                out[f"{k}_min"].append(np.nan)
                out[f"{k}_cnt"].append(0)
            continue
        # For each key, extract numeric values from all dicts in this row
        for k in keys:
            nums_k = []
            for d in val:
                if isinstance(d, dict) and k in d:
                    try:
                        nums_k.append(float(d[k]))
                    except (ValueError, TypeError):
                        pass
            if nums_k:
                out[f"{k}_mean"].append(np.mean(nums_k))
                out[f"{k}_std"].append(np.std(nums_k) if len(nums_k) > 1 else 0.0)
                out[f"{k}_max"].append(np.max(nums_k))
                out[f"{k}_min"].append(np.min(nums_k))
                out[f"{k}_cnt"].append(len(nums_k))
            else:
                out[f"{k}_mean"].append(np.nan)
                out[f"{k}_std"].append(0.0)
                out[f"{k}_max"].append(np.nan)
                out[f"{k}_min"].append(np.nan)
                out[f"{k}_cnt"].append(0)
    return pd.DataFrame(out)


# ─── helper: parse numeric-array column (e.g. heart_rate) ────────────────
def _extract_numeric_array(series):
    """heart_rate column: 1D array of ints → mean, std, min, max, count"""
    means, stds, mins, maxs, cnts = [], [], [], [], []
    for val in series:
        if isinstance(val, np.ndarray) and val.ndim == 1:
            arr = val.astype(float)
            means.append(arr.mean())
            stds.append(arr.std() if len(arr) > 1 else 0.0)
            mins.append(arr.min())
            maxs.append(arr.max())
            cnts.append(len(arr))
        else:
            means.append(np.nan); stds.append(0.0); mins.append(np.nan)
            maxs.append(np.nan); cnts.append(0)
    return pd.DataFrame({"na_mean": means, "na_std": stds, "na_min": mins,
                         "na_max": maxs, "na_cnt": cnts})


# ─── build features for one subject ──────────────────────────────────────
def build_subject(subj, labels_df):
    subj_labels = labels_df[labels_df["subject_id"] == subj].copy()
    subj_labels["date"] = subj_labels["lifelog_date"].dt.date
    parts = {}  # date-indexed daily tables

    # ── 1. wPedo ──────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_wPedo.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["pedo"] = (
        df.groupby("date")
        .agg(
            step_sum=("step", "sum"), step_mean=("step", "mean"), step_max=("step", "max"),
            step_cnt=("step", "count"), step_freq_mean=("step_frequency", "mean"),
            run_sum=("running_step", "sum"), walk_sum=("walking_step", "sum"),
            dist_sum=("distance", "sum"), dist_max=("distance", "max"),
            spd_mean=("speed", "mean"), spd_max=("speed", "max"),
            cal_sum=("burned_calories", "sum"),
        )
        .reset_index()
    )
    parts["pedo"]["run_ratio"] = parts["pedo"]["run_sum"] / (parts["pedo"]["step_sum"] + 1e-9)
    parts["pedo"]["walk_ratio"] = parts["pedo"]["walk_sum"] / (parts["pedo"]["step_sum"] + 1e-9)
    parts["pedo"]["dist_per_step"] = parts["pedo"]["dist_sum"] / (parts["pedo"]["step_sum"] + 1e-9)
    del df; gc.collect()

    # ── 2. mActivity ──────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mActivity.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["act"] = (
        df.groupby("date")
        .agg(act_mean=("m_activity", "mean"), act_std=("m_activity", "std"),
             act_max=("m_activity", "max"), act_min=("m_activity", "min"),
             act_cnt=("m_activity", "count"))
        .reset_index()
    )
    del df; gc.collect()

    # ── 3. mScreenStatus ──────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mScreenStatus.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["scr"] = (
        df.groupby("date")
        .agg(scr_mean=("m_screen_use", "mean"), scr_std=("m_screen_use", "std"),
             scr_max=("m_screen_use", "max"), scr_min=("m_screen_use", "min"),
             scr_cnt=("m_screen_use", "count"))
        .reset_index()
    )
    del df; gc.collect()

    # ── 4. mLight ─────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mLight.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["lgt"] = (
        df.groupby("date")
        .agg(lgt_mean=("m_light", "mean"), lgt_std=("m_light", "std"),
             lgt_max=("m_light", "max"), lgt_min=("m_light", "min"),
             lgt_cnt=("m_light", "count"))
        .reset_index()
    )
    del df; gc.collect()

    # ── 5. wLight ─────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_wLight.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["wlgt"] = (
        df.groupby("date")
        .agg(wl_mean=("w_light", "mean"), wl_std=("w_light", "std"),
             wl_max=("w_light", "max"), wl_min=("w_light", "min"),
             wl_cnt=("w_light", "count"))
        .reset_index()
    )
    del df; gc.collect()

    # ── 6. wHr (heart-rate numeric arrays) ────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_wHr.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    hr = _extract_numeric_array(df["heart_rate"])
    df = pd.concat([df[["date"]], hr], axis=1)
    parts["hr"] = (
        df.groupby("date")
        .agg(hr_mean=("na_mean", "mean"), hr_std=("na_std", "mean"),
             hr_min=("na_min", "mean"), hr_max=("na_max", "mean"),
             hr_cnt=("na_cnt", "sum"))
        .reset_index()
    )
    # nighttime HR (22:00-06:00)
    hrs = pd.read_parquet(parquet_dir / "ch2025_wHr.parquet")
    hrs = hrs[hrs["subject_id"] == subj]
    hrs["date"] = hrs["timestamp"].dt.date
    hrs["hour"] = hrs["timestamp"].dt.hour
    night = hrs[hrs["hour"].between(22, 24) | hrs["hour"].between(0, 6)]
    if len(night) > 0:
        hr_n = _extract_numeric_array(night["heart_rate"])
        tmp = pd.concat([night[["date"]], hr_n], axis=1)
        night_grp = tmp.groupby("date").agg(
            hr_n_mean=("na_mean", "mean"), hr_n_std=("na_std", "mean"),
            hr_n_cnt=("na_cnt", "sum"),
        ).reset_index()
        parts["hr"] = parts["hr"].merge(night_grp, on="date", how="left")
    del df, hrs, night, hr_n; gc.collect()

    # ── 7. mACStatus ──────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mACStatus.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    parts["chg"] = (
        df.groupby("date")
        .agg(chg_mean=("m_charging", "mean"), chg_std=("m_charging", "std"),
             chg_max=("m_charging", "max"), chg_cnt=("m_charging", "count"))
        .reset_index()
    )
    del df; gc.collect()

    # ── 8. mWifi ──────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mWifi.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    wifi_feat = _extract_dict_array(df["m_wifi"], ["bssid", "rssi"])
    parts["wifi"] = (
        pd.concat([df[["date"]], wifi_feat], axis=1)
        .groupby("date")
        .agg(
            wifi_rssi_mean=("rssi_mean", "mean"), wifi_rssi_std=("rssi_std", "mean"),
            wifi_rssi_max=("rssi_max", "max"), wifi_bssid_cnt_sum=("bssid_cnt", "sum"),
            wifi_bssid_cnt_mean=("bssid_cnt", "mean"), wifi_bssid_cnt_max=("bssid_cnt", "max"),
            wifi_rssi_cnt=("rssi_cnt", "count"),
        )
        .reset_index()
    )
    del df, wifi_feat; gc.collect()

    # ── 9. mBle ───────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mBle.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    ble_feat = _extract_dict_array(df["m_ble"], ["address", "rssi"])
    parts["ble"] = (
        pd.concat([df[["date"]], ble_feat], axis=1)
        .groupby("date")
        .agg(
            ble_rssi_mean=("rssi_mean", "mean"), ble_rssi_std=("rssi_std", "mean"),
            ble_rssi_max=("rssi_max", "max"), ble_addr_cnt_sum=("address_cnt", "sum"),
            ble_addr_cnt_mean=("address_cnt", "mean"), ble_rssi_cnt=("rssi_cnt", "count"),
        )
        .reset_index()
    )
    del df, ble_feat; gc.collect()

    # ── 10. mAmbience ─────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mAmbience.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date
    amb_top1, amb_top2, amb_top3 = [], [], []
    amb_cats = {
        "outside,_urban_or_manmade": [],
        "inside,_domestic_or_personal": [],
        "Music": [], "Nature": [], "Vehicle": [], "Animal": [],
    }
    for val in df["m_ambience"]:
        if isinstance(val, np.ndarray) and len(val) > 0:
            pairs = [(str(x[0]), float(x[1])) for x in val]
            pairs.sort(key=lambda x: x[1], reverse=True)
            amb_top1.append(pairs[0][1]); amb_top2.append(pairs[1][1] if len(pairs) > 1 else 0.0)
            amb_top3.append(pairs[2][1] if len(pairs) > 2 else 0.0)
            total_p = sum(p[1] for p in pairs)
            for cat in amb_cats:
                s = sum(p[1] for p in pairs if p[0] == cat)
                amb_cats[cat].append(s / total_p if total_p > 0 else 0.0)
        else:
            amb_top1.append(0.0); amb_top2.append(0.0); amb_top3.append(0.0)
            for cat in amb_cats:
                amb_cats[cat].append(0.0)
    amb_df = pd.DataFrame({
        "date": df["date"].values, "amb_top1": amb_top1, "amb_top2": amb_top2,
        "amb_top3": amb_top3,
    })
    for cat in amb_cats:
        amb_df[f"amb_{cat}"] = amb_cats[cat]
    # drop category-name cols (non-numeric)
    keep_cats = [c for c in amb_df.columns if c.startswith("amb_") and c not in
                 {"amb_top1", "amb_top2", "amb_top3"}]
    parts["amb"] = amb_df.groupby("date")[["amb_top1", "amb_top2", "amb_top3"] + keep_cats].mean().reset_index()
    del df, amb_df; gc.collect()

    # ── 11. mGps ──────────────────────────────────────────────────────────
    # GPS: each row = 1D array of dicts {altitude, latitude, longitude, speed}
    df = pd.read_parquet(parquet_dir / "ch2025_mGps.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date

    spd_means, spd_maxes, alt_means, alt_maxes, alt_mins = [], [], [], [], []
    for val in df["m_gps"]:
        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 0:
            spds = [float(x.get("speed", 0)) for x in val if isinstance(x, dict)]
            alts = [float(x.get("altitude", 0)) for x in val if isinstance(x, dict)]
            spd_means.append(np.mean(spds) if spds else np.nan)
            spd_maxes.append(max(spds) if spds else np.nan)
            alt_means.append(np.mean(alts) if alts else np.nan)
            alt_maxes.append(max(alts) if alts else np.nan)
            alt_mins.append(min(alts) if alts else np.nan)
        else:
            spd_means.append(np.nan); spd_maxes.append(np.nan)
            alt_means.append(np.nan); alt_maxes.append(np.nan); alt_mins.append(np.nan)

    parts["gps"] = (
        pd.DataFrame({
            "date": df["date"].values,
            "gps_spd_mean": spd_means, "gps_spd_max": spd_maxes,
            "gps_alt_mean": alt_means, "gps_alt_max": alt_maxes, "gps_alt_min": alt_mins,
        })
        .groupby("date")
        .agg(gps_spd_mean=("gps_spd_mean", "mean"), gps_spd_max=("gps_spd_max", "max"),
             gps_alt_mean=("gps_alt_mean", "mean"), gps_alt_max=("gps_alt_max", "max"),
             gps_alt_min=("gps_alt_min", "min"))
        .reset_index()
    )
    parts["gps"]["gps_alt_range"] = parts["gps"]["gps_alt_max"] - parts["gps"]["gps_alt_min"]
    del df, spd_means, spd_maxes, alt_means, alt_maxes, alt_mins; gc.collect()

    # ── 12. mUsageStats ───────────────────────────────────────────────────
    df = pd.read_parquet(parquet_dir / "ch2025_mUsageStats.parquet")
    df = df[df["subject_id"] == subj]
    df["date"] = df["timestamp"].dt.date

    us_cnts, us_times = [], []
    for val in df["m_usage_stats"]:
        if isinstance(val, np.ndarray) and len(val) > 0:
            times = [float(x.get("total_time", 0)) for x in val if isinstance(x, dict)]
            us_cnts.append(len(times))
            us_times.append(np.mean(times) if times else np.nan)
        else:
            us_cnts.append(0); us_times.append(np.nan)

    parts["usage"] = (
        pd.DataFrame({
            "date": df["date"].values, "us_cnt": us_cnts, "us_time_mean": us_times,
        })
        .groupby("date")
        .agg(us_app_cnt=("us_cnt", "count"),
             us_time_mean=("us_time_mean", "mean"),
             us_time_max=("us_time_mean", "max"),
             us_time_cnt=("us_time_mean", "count"))
        .reset_index()
    )
    del df, us_cnts, us_times; gc.collect()

    # ── merge ─────────────────────────────────────────────────────────────
    df = parts["pedo"]
    for k in ["act", "scr", "lgt", "wlgt", "hr", "chg", "wifi", "ble", "amb", "gps", "usage"]:
        df = df.merge(parts[k], on="date", how="left")

    # labels
    df = df.merge(subj_labels[["lifelog_date", "date"] + TARGETS], on="date", how="left")
    del parts; gc.collect()

    # ── temporal ──────────────────────────────────────────────────────────
    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # ── cross-sensor ──────────────────────────────────────────────────────
    if "step_sum" in df.columns and "act_mean" in df.columns:
        df["step_act_ratio"] = df["step_sum"] / (df["act_mean"] * 60 + 1e-9)
    if "scr_mean" in df.columns and "act_mean" in df.columns:
        df["scr_act_ratio"] = df["scr_mean"] / (df["act_mean"] + 1e-9)
    if "lgt_mean" in df.columns and "chg_mean" in df.columns:
        df["light_chg"] = df["lgt_mean"] * df["chg_mean"]
    if "wifi_bssid_cnt_sum" in df.columns and "ble_addr_cnt_sum" in df.columns:
        df["wireless_total"] = df["wifi_bssid_cnt_sum"] + df["ble_addr_cnt_sum"]

    # ── rolling ───────────────────────────────────────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    key_num = [c for c in [
        "step_sum", "hr_mean", "lgt_mean", "wl_mean", "act_mean",
        "scr_mean", "chg_mean", "gps_spd_mean", "us_app_cnt",
    ] if c in df.columns]
    for col in key_num:
        for w in (3, 7, 14):
            df[f"{col}_rm{w}"] = df[col].rolling(w, min_periods=1).mean().values
            df[f"{col}_rs{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0).values
            df[f"{col}_d1"] = df[col].diff(1).fillna(0)
            df[f"{col}_d7"] = df[col].diff(7).fillna(0)
    for col in key_num:
        dow_m = df.groupby("dayofweek")[col].mean().to_dict()
        df[f"{col}_dowdev"] = df["dayofweek"].map(dow_m) - df[col]

    # ── z-score personalization ───────────────────────────────────────────
    meta = {"subject_id", "lifelog_date", "sleep_date", "date",
            "dayofweek", "dayofyear", "is_weekend"}
    num_cols = [c for c in df.columns if c not in meta | set(TARGETS)
                and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]
    num_cols = [c for c in num_cols if df[c].nunique() > 2]
    for c in num_cols:
        m, s = df[c].mean(), df[c].std()
        if s > 0:
            df[f"{c}_z"] = (df[c] - m) / s

    log.info(f"  {subj}: {df.shape[1]} cols, {df.dropna(subset=TARGETS).shape[0]} labeled")
    df["subject_id"] = subj
    return df, num_cols


# ─── external data ───────────────────────────────────────────────────────
def load_external_data():
    """Try loading external demographic/health data."""
    ext_paths = [
        ROOT / "data_raw" / "ch2025_data_subjects.csv",
        ROOT / "data_raw" / "ch2026_subject_info.csv",
        ROOT / "data_raw" / "subject_external.csv",
    ]
    for p in ext_paths:
        if p.exists():
            log.info(f"External data: {p}")
            return pd.read_csv(p)
    log.info("No external data found — continuing without it")
    return None


# ─── main ────────────────────────────────────────────────────────────────
def main():
    start = time.time()

    labels = pd.read_csv(
        DATA_RAW / "ch2026_metrics_train.csv",
        parse_dates=["sleep_date", "lifelog_date"],
    )
    subjects = sorted(labels["subject_id"].unique())
    log.info(f"Labels: {labels.shape}, Subjects: {len(subjects)}")

    # external data
    ext = load_external_data()
    if ext is not None:
        ext["subject_id"] = ext["subject_id"].astype(str)
        ext = ext.groupby("subject_id").first().reset_index()

    all_feats = []
    global_cols = []

    for subj in subjects:
        feat, cols = build_subject(subj, labels)
        all_feats.append(feat)
        if not global_cols:
            global_cols = cols
        del feat; gc.collect()

    combined = pd.concat(all_feats, ignore_index=True)
    log.info(f"\nCombined: {combined.shape}, features={len(global_cols)}")

    # merge external
    if ext is not None:
        combined = combined.merge(ext.drop(columns=["subject_id"], errors="ignore"),
                                  on="subject_id", how="left")
        for c in ext.columns:
            if c not in ("subject_id",) and c in combined.columns:
                if combined[c].nunique() > 2:
                    global_cols.append(c)

    combined.to_parquet(DATA_PROCESSED / "features_v57.parquet")
    log.info("Saved features_v57.parquet")
    gc.collect()

    # ── GroupKFold ────────────────────────────────────────────────────────
    log.info("\n=== GroupKFold (5-fold) ===")
    import lightgbm as lgb

    gkf = GroupKFold(n_splits=5)
    all_oof = {t: np.zeros(len(combined)) for t in TARGETS}
    oof_mask = {t: np.zeros(len(combined), dtype=bool) for t in TARGETS}
    feat_imp = {t: np.zeros(len(global_cols)) for t in TARGETS}

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(combined, groups=combined["subject_id"])
    ):
        log.info(f"\nFold {fold + 1}/5")
        Xtr = combined.iloc[train_idx][global_cols].fillna(0).values
        Xv = combined.iloc[val_idx][global_cols].fillna(0).values
        ytr = combined.iloc[train_idx][TARGETS]

        for tgt in TARGETS:
            mask_tr = ytr[tgt].notna()
            tr_y = ytr[tgt].values[mask_tr]
            if len(tr_y) < 20:
                continue
            tmean = tr_y.mean()

            cfg = dict(
                objective="binary", metric="binary_logloss",
                num_leaves=8, max_depth=3,
                learning_rate=0.02, n_estimators=200,
                subsample=0.6, colsample_bytree=0.6,
                reg_alpha=2.0, reg_lambda=5.0,
                min_child_samples=15, verbose=-1, seed=42,
            )
            mdl = lgb.LGBMClassifier(**cfg)
            mdl.fit(Xtr[mask_tr], tr_y)
            feat_imp[tgt] += mdl.feature_importances_

            # Predict only on labeled val rows
            yval = combined.iloc[val_idx][tgt]
            mask_val = yval.notna()
            if mask_val.sum() > 0:
                op = mdl.predict_proba(Xv[mask_val.values])[:, 1]
                op = np.clip(op + (tmean - op.mean()), 0.0001, 0.9999)
                # Store at actual positions
                val_labels_idx = val_idx[mask_val.values]
                all_oof[tgt][val_labels_idx] = op
                oof_mask[tgt][val_labels_idx] = True

    for t in TARGETS:
        feat_imp[t] /= 5.0

    # ── results ───────────────────────────────────────────────────────────
    log.info("\n=== OOF Scores ===")
    total_loss = 0
    for tgt in TARGETS:
        m = oof_mask[tgt]
        if m.sum() > 0:
            yt = combined.loc[m, tgt].values
            yp = all_oof[tgt][m]
            # Double-check no NaN
            clean = yt.notna() if hasattr(yt, 'notna') else ~np.isnan(yt)
            yt, yp = yt[clean], yp[clean]
            if len(yt) > 0:
                loss = log_loss(yt, yp)
                total_loss += loss
                log.info(f"  {tgt}: logloss={loss:.4f}, n={len(yt)}, pred_mean={yp.mean():.4f}, actual={yt.mean():.4f}")
    avg = total_loss / len(TARGETS)
    log.info(f"\n  AVG OOF log_loss: {avg:.4f}")

    log.info("\n=== Top 10 Features per Target ===")
    for tgt in TARGETS:
        top = np.argsort(feat_imp[tgt])[::-1][:10]
        log.info(f"\n  {tgt}:")
        for i in top:
            if feat_imp[tgt][i] > 0:
                log.info(f"    {global_cols[i]}: {feat_imp[tgt][i]:.0f}")

    log.info(f"\nTime: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
