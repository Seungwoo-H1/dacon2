"""
10_v11_fast.py - V11 Fast: V10 OOF reference + V11 only tuning
Uses V10 meta OOF as baseline, only tunes V11 with new features.
"""
import sys, re, json, warnings, logging, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, "src")
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
RANDOM_SEEDS = [42,123,456,789,1024,1337,2048,3037,4096,5001,6000,7123,8001,9000,10000,11111,12000,13001,14000,15001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 5

LGB_BASE = {
    "objective":"binary", "metric":"binary_logloss", "num_leaves":15, "max_depth":4,
    "learning_rate":0.03, "n_estimators":500, "subsample":0.7, "colsample_bytree":0.7,
    "reg_alpha":1.0, "reg_lambda":3.0, "min_child_samples":10,
    "force_row_wise":True, "n_jobs":-1, "verbose":-1,
}
LGB_CONFS = [
    {"name":"C1","nl":8,"md":3,"lr":0.02,"ne":200,"ss":0.6,"cst":0.6,"ra":2.0,"rl":5.0,"mc":15},
    {"name":"C2","nl":10,"md":3,"lr":0.03,"ne":300,"ss":0.7,"cst":0.7,"ra":1.0,"rl":3.0,"mc":10},
    {"name":"C3","nl":12,"md":4,"lr":0.03,"ne":200,"ss":0.7,"cst":0.7,"ra":1.0,"rl":3.0,"mc":10},
    {"name":"C4","nl":15,"md":4,"lr":0.03,"ne":500,"ss":0.7,"cst":0.7,"ra":1.0,"rl":3.0,"mc":10},
    {"name":"C5","nl":20,"md":5,"lr":0.02,"ne":300,"ss":0.7,"cst":0.7,"ra":0.5,"rl":2.0,"mc":8},
    {"name":"C6","nl":6,"md":2,"lr":0.02,"ne":200,"ss":0.5,"cst":0.5,"ra":5.0,"rl":10.0,"mc":20},
]

LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

def sanitize(n): return re.sub(r"[^a-zA-Z0-9_]", "_", n)

def get_feat_cols(f):
    return [c for c in f.columns if c not in META_COLS | set(TARGET_COLS)
            and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

def remove_leak(cols, t):
    if t.startswith("S"): return [c for c in cols if c not in LEAK_S]
    elif t.startswith("Q"): return [c for c in cols if c not in LEAK_Q]
    return cols

def add_personalization(df, feat_cols):
    df = df.copy(); pcols = []
    for col in feat_cols:
        cf = df[col].fillna(0)
        ss = cf.groupby(df["subject_id"]).agg(["mean","std"]); ss.columns = [f"{col}_subj_mean", f"{col}_subj_std"]
        ss = ss.reset_index(); m = df.merge(ss, on="subject_id", how="left")
        m[f"{col}_zscore"] = np.where((m[f"{col}_subj_std"]==0) | df[col].isnull(), 0.0,
            (m[col]-m[f"{col}_subj_mean"])/m[f"{col}_subj_std"])
        pcols.append(f"{col}_zscore"); df = m
    return df, pcols

def rank_features(feat, fcols, target, seed=42):
    y = feat[target].values; X = feat[fcols].fillna(0).values
    np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
    params = {"objective":"binary","metric":"binary_logloss","verbose":-1,"num_leaves":15,"max_depth":4,
        "learning_rate":0.03,"n_estimators":100,"subsample":0.7,"colsample_bytree":0.7,
        "reg_alpha":1.0,"reg_lambda":3.0,"scale_pos_weight":spw,"random_state":seed,
        "min_child_samples":10,"force_row_wise":True,"n_jobs":-1}
    sn = [sanitize(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={"verbose":"-1"})
    mdl = lgb.train(params, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type="gain")
    return sorted(zip(fcols, imp), key=lambda x: -x[1])

def lgb_cv_predict(feat, scols, target, seeds, spw):
    y = feat[target].values; gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    floss = {i:[] for i in range(N_SPLITS)}
    sn = [sanitize(c) for c in scols]
    for si, seed in enumerate(seeds):
        cfg = {**LGB_BASE, "random_state":seed}
        for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat["subject_id"])):
            Xtr = feat.iloc[ti][scols].fillna(0).values; Xva = feat.iloc[vi][scols].fillna(0).values
            ytr, yva = y[ti], y[vi]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn, params={"verbose":"-1"})
            vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd, params={"verbose":"-1"})
            mdl = lgb.train({**cfg,"scale_pos_weight":spw}, trd, num_boost_round=cfg["n_estimators"],
                valid_sets=[vad], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pred = mdl.predict(Xva); oof[vi, si] = pred
            floss[fold].append(log_loss(yva, pred, labels=[0,1]))
    oa = oof.mean(axis=1)
    fl = [np.mean(floss[i]) for i in range(N_SPLITS)]
    return oa, oof, log_loss(y, oa, labels=[0,1]), np.std(fl), fl

def simple_mean_match(pred, rate):
    s = rate - pred.mean(); c = pred + s; return np.clip(c, 0.0001, 0.9999)

def tune_target(feat, fcols, target, n_feats_range=(10,20,30,40)):
    best_cfg, best_cv, best_oof, best_cols = None, float("inf"), None, None
    ranked = rank_features(feat, fcols, target)
    for nf in n_feats_range:
        if nf > len(ranked): continue
        sc = [r[0] for r in ranked[:nf]]
        y = feat[target].values; np_ = max((y==1).sum(),1); nn = (y==0).sum(); spw = nn/np_
        for cfg in LGB_CONFS:
            tc = {**LGB_BASE, "num_leaves":cfg["nl"],"max_depth":cfg["md"],"learning_rate":cfg["lr"],
                "n_estimators":cfg["ne"],"subsample":cfg["ss"],"colsample_bytree":cfg["cst"],
                "reg_alpha":cfg["ra"],"reg_lambda":cfg["rl"],"min_child_samples":cfg["mc"]}
            oa, of, cl, cs, fl = lgb_cv_predict(feat, sc, target, RANDOM_SEEDS, spw)
            tr = y.mean(); ps = abs(oa.mean() - tr)
            score = cl + 0.5*cs + 0.1*ps
            if score < best_cv:
                best_cv, best_cfg, best_oof, best_cols = score, {**cfg,"_n_feats":nf}, oa, sc
    return best_cfg, best_cols, best_oof

def main():
    log.info("="*70)
    log.info("V11 Fast: V10 baseline reference + V11 tuning")
    log.info("="*70)
    
    # Load V10 meta
    meta = json.load(open("submissions/meta_v10_20260501_170715.json"))
    v10_oof = {t: meta["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_oof.values()))
    log.info(f"V10 avg cal OOF (from meta): {v10_avg:.6f}")
    
    # Load V10 features + add V11 features
    log.info("\n=== Load features + add V11 ===")
    feat = pd.read_parquet("data_processed/features.parquet")
    
    # Date features
    dates = pd.to_datetime(feat["date"], errors="coerce")
    for col, fn in [("day_of_week", lambda d: d.dt.dayofweek),
                    ("is_weekend", lambda d: (d.dt.dayofweek>=5).astype(float)),
                    ("month", lambda d: d.dt.month),
                    ("quarter", lambda d: d.dt.quarter),
                    ("day_of_month", lambda d: d.dt.day)]:
        feat[col] = fn(dates)
    
    # Interaction features
    feat["interaction_screen_activity"] = feat["mScreenStatus_m_screen_use_mean"].fillna(0)*feat["mActivity_m_activity_mean"].fillna(0)
    feat["interaction_wifi_ble_rssi"] = feat["mWifi_wifi_avg_rssi_mean"].fillna(0)*feat["mBle_ble_avg_rssi_mean"].fillna(0)
    feat["interaction_step_screen"] = feat["wPedo_pedo_step_mean"].fillna(0)*feat["mScreenStatus_m_screen_use_mean"].fillna(0)
    feat["interaction_hr_activity"] = feat["wHr_hr_mean"].fillna(0)*feat["mActivity_m_activity_mean"].fillna(0)
    feat["interaction_gps_activity"] = feat["mGps_gps_avg_speed_mean"].fillna(0)*feat["mActivity_m_activity_mean"].fillna(0)
    feat["interaction_charging_screen"] = feat["mACStatus_m_charging_mean"].fillna(0)*feat["mScreenStatus_m_screen_use_mean"].fillna(0)
    feat["interaction_light_screen"] = feat["mLight_m_light_mean"].fillna(0)*feat["mScreenStatus_m_screen_use_mean"].fillna(0)
    amb_cols = [c for c in feat.columns if c.startswith("mAmbience_ambience_") and c.endswith("_sum")]
    feat["total_ambience"] = feat[amb_cols].sum(axis=1)
    feat["step_running_ratio"] = feat["wPedo_pedo_running_step_mean"].fillna(0)/(feat["wPedo_pedo_step_mean"].fillna(0)+1e-10)
    feat["distance_per_step"] = feat["wPedo_pedo_distance_mean"].fillna(0)/(feat["wPedo_pedo_step_mean"].fillna(0)+1e-10)
    
    log.info(f"Feature count: {len(feat.columns) - len(META_COLS) - len(TARGET_COLS)}")
    
    # Personalization
    log.info("\n=== Personalization ===")
    feat_cols = get_feat_cols(feat)
    feat, pcols = add_personalization(feat, feat_cols)
    log.info(f"Personalization cols: {len(pcols)}")
    
    # V11 tuning
    log.info("\n=== V11 tuning ===")
    train_rate = {}
    for t in TARGET_COLS:
        y = feat[t].values; train_rate[t] = float(y.mean())
    
    all_cfg, all_cols, all_oof = {}, {}, {}
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (train_rate={train_rate[target]:.3f}, V10 OOF={v10_oof[target]:.4f}) ---")
        leak = remove_leak(feat_cols + pcols, target)
        ranked = rank_features(feat, leak, target)
        log.info(f"  Top 5: {[r[0] for r in ranked[:5]]}")
        
        best_cfg, best_cols, best_oof = tune_target(feat, leak, target)
        all_cfg[target] = best_cfg; all_cols[target] = best_cols; all_oof[target] = best_oof
        y = feat[target].values
        oof_loss = log_loss(y, best_oof, labels=[0,1])
        log.info(f"  Best: {best_cfg}, OOF={oof_loss:.4f}, feats={len(best_cols)}")
    
    # Compare
    log.info("\n=== V10 vs V11 ===")
    log.info(f"{'Target':<6} {'V10 OOF':<10} {'V11 OOF':<10} {'Diff':<10} {'Status':<10}")
    v11_avg = 0
    for t in TARGET_COLS:
        y = feat[t].values; v11_loss = log_loss(y, all_oof[t], labels=[0,1])
        diff = v11_loss - v10_oof[t]
        v11_avg += v11_loss
        better = "✓ BETTER" if diff < -0.001 else ("✗ same" if abs(diff) <= 0.001 else "✗ worse")
        log.info(f"  {t:<6} {v10_oof[t]:<10.4f} {v11_loss:<10.4f} {diff:+.4f} {better}")
    v11_avg /= 7
    log.info(f"\nV10 avg: {v10_avg:.4f}")
    log.info(f"V11 avg: {v11_avg:.4f}")
    log.info(f"Improvement: {v11_avg-v10_avg:+.4f}")
    
    if v11_avg >= v10_avg - 0.001:
        log.info(f"\n❌ V11 NOT better than V10. Stopping.")
        return
    
    log.info(f"\n✅ V11 BEATS V10! Generating submission...")
    
    # Calibration
    log.info("\n=== Calibration ===")
    cal_oof = {}
    for t in TARGET_COLS:
        cal_oof[t] = simple_mean_match(all_oof[t], train_rate[t])
        cl = log_loss(feat[t], cal_oof[t], labels=[0,1])
        log.info(f"  {t}: cal_OOF={cl:.4f}, mean={cal_oof[t].mean():.4f}")
    avg_cal = np.mean([log_loss(feat[t], cal_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"  Avg cal OOF: {avg_cal:.4f}")
    
    # Generate submission
    log.info("\n=== Generate submission ===")
    spec = importlib.util.spec_from_file_location("02_feature_engineering", Path("src/02_feature_engineering.py"))
    fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
    spec2 = importlib.util.spec_from_file_location("01_load_data", Path("src/01_load_data.py"))
    ld = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ld)
    
    pdfs = {}; dd = Path("data_raw/ch2025_data_items")
    pn = {"mACStatus":"ch2025_mACStatus.parquet","mActivity":"ch2025_mActivity.parquet","mAmbience":"ch2025_mAmbience.parquet",
        "mBle":"ch2025_mBle.parquet","mGps":"ch2025_mGps.parquet","mLight":"ch2025_mLight.parquet",
        "mScreenStatus":"ch2025_mScreenStatus.parquet","mUsageStats":"ch2025_mUsageStats.parquet",
        "mWifi":"ch2025_mWifi.parquet","wHr":"ch2025_wHr.parquet","wLight":"ch2025_wLight.parquet","wPedo":"ch2025_wPedo.parquet"}
    
    sample = pd.read_csv("data_raw/ch2026_submission_sample.csv")
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
    sample["sleep_date"] = pd.to_datetime(sample["sleep_date"]).dt.date
    td = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())
    for n, fn in pn.items():
        df = pd.read_parquet(dd / fn); df = ld.build_merge_key(df)
        pdfs[n] = df[df["date"].astype(str).isin(td)]
    
    tf = fe.create_day_features(pdfs, sample)
    dates_t = pd.to_datetime(tf["date"], errors="coerce")
    tf["day_of_week"] = dates_t.dt.dayofweek
    tf["is_weekend"] = (dates_t.dt.dayofweek >= 5).astype(float)
    tf["month"] = dates_t.dt.month
    tf["quarter"] = dates_t.dt.quarter
    tf["day_of_month"] = dates_t.dt.day
    tf["interaction_screen_activity"] = tf["mScreenStatus_m_screen_use_mean"].fillna(0)*tf["mActivity_m_activity_mean"].fillna(0)
    tf["interaction_wifi_ble_rssi"] = tf["mWifi_wifi_avg_rssi_mean"].fillna(0)*tf["mBle_ble_avg_rssi_mean"].fillna(0)
    tf["interaction_step_screen"] = tf["wPedo_pedo_step_mean"].fillna(0)*tf["mScreenStatus_m_screen_use_mean"].fillna(0)
    tf["interaction_hr_activity"] = tf["wHr_hr_mean"].fillna(0)*tf["mActivity_m_activity_mean"].fillna(0)
    tf["interaction_gps_activity"] = tf["mGps_gps_avg_speed_mean"].fillna(0)*tf["mActivity_m_activity_mean"].fillna(0)
    tf["interaction_charging_screen"] = tf["mACStatus_m_charging_mean"].fillna(0)*tf["mScreenStatus_m_screen_use_mean"].fillna(0)
    tf["interaction_light_screen"] = tf["mLight_m_light_mean"].fillna(0)*tf["mScreenStatus_m_screen_use_mean"].fillna(0)
    amb_t = [c for c in tf.columns if c.startswith("mAmbience_ambience_") and c.endswith("_sum")]
    tf["total_ambience"] = tf[amb_t].sum(axis=1)
    tf["step_running_ratio"] = tf["wPedo_pedo_running_step_mean"].fillna(0)/(tf["wPedo_pedo_step_mean"].fillna(0)+1e-10)
    tf["distance_per_step"] = tf["wPedo_pedo_distance_mean"].fillna(0)/(tf["wPedo_pedo_step_mean"].fillna(0)+1e-10)
    
    tfc = get_feat_cols(tf)
    tf, _ = add_personalization(tf, tfc)
    
    preds = tf[["subject_id","sleep_date","lifelog_date"]].copy()
    for target in TARGET_COLS:
        log.info(f"\n  Training {target}...")
        sc = all_cols[target]; y_a = feat[target].values
        Xa = feat[sc].fillna(0).values; tX = tf[sc].fillna(0).values
        sn = [sanitize(c) for c in sc]
        cfg = all_cfg[target]
        np_ = max((y_a==1).sum(),1); nn = (y_a==0).sum(); spw = nn/np_
        lp = {**LGB_BASE,"num_leaves":cfg["nl"],"max_depth":cfg["md"],"learning_rate":cfg["lr"],
            "n_estimators":cfg["ne"],"subsample":cfg["ss"],"colsample_bytree":cfg["cst"],
            "reg_alpha":cfg["ra"],"reg_lambda":cfg["rl"],"min_child_samples":cfg["mc"],"scale_pos_weight":spw}
        ap = np.zeros(len(tX))
        for si, seed in enumerate(RANDOM_SEEDS):
            sp = {**lp,"random_state":seed}
            da = lgb.Dataset(Xa, label=y_a, feature_name=sn, params={"verbose":"-1"})
            ap += lgb.train(sp, da, num_boost_round=cfg["ne"]).predict(tX)
            if (si+1)%5==0: log.info(f"    {target} seed {si+1}/{N_SEEDS}")
        ap /= N_SEEDS
        cp = simple_mean_match(ap, train_rate[target])
        preds[target] = cp
        log.info(f"    {target}: mean={cp.mean():.4f}, range=[{cp.min():.4f},{cp.max():.4f}]")
    
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sp = SUBMIT_DIR / f"submission_v11_{ts}.csv"
    preds.to_csv(sp, index=False)
    log.info(f"\nSubmission: {sp}")
    
    meta_out = {"version":"v11","submission_file":str(sp),"timestamp":ts,"n_samples":len(preds),"n_seeds":N_SEEDS,
        "n_splits":N_SPLITS,"improvements":["date features","cross-sensor interactions","ratio features"],
        "per_target":{}}
    for t in TARGET_COLS:
        ol = log_loss(feat[t], all_oof[t], labels=[0,1])
        cl = log_loss(feat[t], cal_oof[t], labels=[0,1])
        meta_out["per_target"][t] = {"config":all_cfg[t],"n_features":len(all_cols[t]),
            "oof_cv_loss":float(ol),"cal_oof_loss":float(cl),"oof_mean":float(all_oof[t].mean()),
            "cal_mean":float(preds[t].mean()),"train_rate":float(train_rate[t]),
            "pred_min":float(preds[t].min()),"pred_max":float(preds[t].max())}
    
    mp = SUBMIT_DIR / f"meta_v11_{ts}.json"
    with open(mp,"w") as f: json.dump(meta_out, f, indent=2, default=str)
    log.info(f"Metadata: {mp}")
    
    log.info(f"\n{'='*70} V11 SUMMARY {'='*70}")
    log.info(f"{'Target':<6} {'V10 OOF':<10} {'V11 OOF':<10} {'V11 Cal':<10} {'Test Mean':<10} {'Train':<10}")
    for t in TARGET_COLS:
        ol = log_loss(feat[t], all_oof[t], labels=[0,1])
        cl = log_loss(feat[t], cal_oof[t], labels=[0,1])
        log.info(f"{t:<6} {v10_oof[t]:<10.4f} {ol:<10.4f} {cl:<10.4f} {preds[t].mean():<10.4f} {train_rate[t]:<10.3f}")
    log.info(f"\nV11 avg cal OOF: {avg_cal:.6f} (V10: {v10_avg:.6f}, diff: {avg_cal-v10_avg:+.6f})")

if __name__ == "__main__":
    main()
