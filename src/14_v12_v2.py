"""
14_v12_v2.py — V12: CatBoost ensemble, fixed
"""
import sys, re, json, time, warnings, logging, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

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
N_SEEDS_LGB = len(RANDOM_SEEDS)
RANDOM_SEEDS_CB = [42, 123, 456, 789, 1024]
N_SEEDS_CB = len(RANDOM_SEEDS_CB)
N_SPLITS = 5

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
        ss = cf.groupby(df["subject_id"]).agg(["mean","std"])
        ss.columns = [f"{col}_subj_mean", f"{col}_subj_std"]
        ss = ss.reset_index()
        m = df.merge(ss, on="subject_id", how="left")
        m[f"{col}_zscore"] = np.where((m[f"{col}_subj_std"]==0)|df[col].isnull(), 0.0,
            (m[col]-m[f"{col}_subj_mean"])/m[f"{col}_subj_std"])
        pcols.append(f"{col}_zscore"); df = m
    return df, pcols

def lgb_cv(feat, scols, target, seeds, spw, cfg):
    y = feat[target].values; gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in scols]
    for si, seed in enumerate(seeds):
        sc = {**cfg, "random_state":seed, "scale_pos_weight":spw}
        for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat["subject_id"])):
            Xtr = feat.iloc[ti][scols].fillna(0).values
            Xva = feat.iloc[vi][scols].fillna(0).values
            ytr, yva = y[ti], y[vi]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn, params={"verbose":-1})
            vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd, params={"verbose":-1})
            mdl = lgb.train(sc, trd, num_boost_round=cfg["n_estimators"], valid_sets=[vad],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vi, si] = mdl.predict(Xva)
    return oof.mean(axis=1)

def cb_cv(feat, scols, target, seeds, spw, cb_config):
    """CatBoost CV with separate config dict."""
    y = feat[target].values; gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    for si, seed in enumerate(seeds):
        for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat["subject_id"])):
            Xtr = feat.iloc[ti][scols].fillna(-1).values
            Xva = feat.iloc[vi][scols].fillna(-1).values
            ytr, yva = y[ti], y[vi]
            trd = Pool(Xtr, ytr)
            vad = Pool(Xva, yva)
            mdl = CatBoostClassifier(
                iterations=cb_config["iterations"],
                depth=cb_config["depth"],
                learning_rate=cb_config["learning_rate"],
                l2_leaf_reg=cb_config["l2_leaf_reg"],
                bagging_temperature=cb_config["bagging_temperature"],
                random_strength=cb_config["random_strength"],
                subsample=cb_config["subsample"],
                loss_function="Logloss",
                one_hot_max_size=3,
                random_state=seed,
                scale_pos_weight=spw,
                verbose=0,
            )
            mdl.fit(trd, eval_set=vad, early_stopping_rounds=50)
            oof[vi, si] = mdl.predict(Xva, prediction_type="Probability")[:, 1]
    return oof.mean(axis=1)

def rank_feat(feat, fcols, target):
    y = feat[target].values
    np_ = max((y==1).sum(),1); nn = (y==0).sum(); spw = nn/np_
    params = {"objective":"binary","metric":"binary_logloss","verbose":-1,"num_leaves":15,"max_depth":4,
        "learning_rate":0.03,"n_estimators":100,"subsample":0.7,"colsample_bytree":0.7,
        "reg_alpha":1.0,"reg_lambda":3.0,"scale_pos_weight":spw,"random_state":42,
        "min_child_samples":10,"force_row_wise":True,"n_jobs":-1}
    sn = [sanitize(c) for c in fcols]
    X = feat[fcols].fillna(0).values
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={"verbose":"-1"})
    mdl = lgb.train(params, ds, num_boost_round=100)
    imp = mdl.feature_importance(importance_type="gain")
    return sorted(zip(fcols, imp), key=lambda x: -x[1])

def main():
    t_total = time.time()
    log.info("="*70)
    log.info("V12 v2: CatBoost ensemble")
    log.info("="*70)
    
    # Load V10 meta
    meta_v10 = json.load(open("submissions/meta_v10_20260501_170715.json"))
    v10_cfgs = {t: meta_v10["per_target"][t]["config"] for t in TARGET_COLS}
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    log.info(f"V10 avg cal OOF: {v10_avg:.6f}")
    
    # Load features
    feat = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(feat)
    feat, pcols = add_personalization(feat, feat_cols)
    all_cols = feat_cols + pcols
    train_rate = {t: float(feat[t].mean()) for t in TARGET_COLS}
    log.info(f"Features: {len(feat_cols)} + {len(pcols)} z-score = {len(all_cols)}")
    
    # ===== LGB baseline =====
    log.info("\n=== LGB baseline ===")
    lgb_oof = {}; lgb_scols = {}
    for target in TARGET_COLS:
        tc = v10_cfgs[target]
        cfg = {"objective":"binary","metric":"binary_logloss","verbose":-1,
            "num_leaves":tc["nl"],"max_depth":tc["md"],"learning_rate":tc["lr"],
            "n_estimators":tc["ne"],"subsample":tc["ss"],"colsample_bytree":tc["cst"],
            "reg_alpha":tc["ra"],"reg_lambda":tc["rl"],"min_child_samples":tc["mc"],
            "force_row_wise":True,"n_jobs":-1}
        leak = remove_leak(all_cols, target)
        ranked = rank_feat(feat, leak, target)
        n_feats = tc.get("_n_feats", 10)
        scols = [r[0] for r in ranked[:n_feats]]
        y = feat[target].values; np_ = max((y==1).sum(),1); nn = (y==0).sum(); spw = nn/np_
        t0 = time.time()
        oof = lgb_cv(feat, scols, target, RANDOM_SEEDS, spw, cfg)
        elapsed = time.time() - t0
        oof_loss = log_loss(y, oof, labels=[0,1])
        lgb_oof[target] = oof; lgb_scols[target] = scols
        v10_cal = v10_cal_oof[target]; diff = oof_loss - v10_cal
        better = "✓" if diff < -0.001 else ("~" if abs(diff) <= 0.001 else "✗")
        log.info(f"  {target}: V10={v10_cal:.4f}, LGB={oof_loss:.4f}, diff={diff:+.4f} {better} ({elapsed:.1f}s)")
    lgb_avg = np.mean([log_loss(feat[t], lgb_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"  LGB avg: {lgb_avg:.6f} (V10: {v10_avg:.6f}, diff: {lgb_avg-v10_avg:+.6f})")
    
    # ===== CatBoost configs =====
    CB_CFGS = [
        {"name":"CB1","iterations":300,"depth":5,"learning_rate":0.03,"l2_leaf_reg":3.0,
         "bagging_temperature":0.5,"random_strength":1.0,"subsample":0.8},
        {"name":"CB2","iterations":400,"depth":6,"learning_rate":0.02,"l2_leaf_reg":5.0,
         "bagging_temperature":0.3,"random_strength":1.5,"subsample":0.7},
        {"name":"CB3","iterations":500,"depth":4,"learning_rate":0.01,"l2_leaf_reg":3.0,
         "bagging_temperature":0.5,"random_strength":2.0,"subsample":0.8},
    ]
    
    log.info("\n=== CatBoost CV ===")
    cb_oof = {}; cb_best_cfg = {}
    
    for target in TARGET_COLS:
        log.info(f"\n  {target}:")
        leak = remove_leak(all_cols, target)
        ranked = rank_feat(feat, leak, target)
        n_feats = v10_cfgs[target].get("_n_feats", 10)
        scols = [r[0] for r in ranked[:n_feats]]
        y = feat[target].values; np_ = max((y==1).sum(),1); nn = (y==0).sum(); spw = nn/np_
        
        best_cb_loss = float("inf")
        best_cb_oof = None
        best_cb_cfg = None
        
        for cb_cfg in CB_CFGS:
            t0 = time.time()
            oof = cb_cv(feat, scols, target, RANDOM_SEEDS_CB, spw, cb_cfg)
            elapsed = time.time() - t0
            oof_loss = log_loss(y, oof, labels=[0,1])
            log.info(f"    {cb_cfg['name']}: {oof_loss:.4f} ({elapsed:.1f}s)")
            if oof_loss < best_cb_loss:
                best_cb_loss = oof_loss; best_cb_oof = oof; best_cb_cfg = cb_cfg
        
        cb_oof[target] = best_cb_oof
        cb_best_cfg[target] = best_cb_cfg
        v10_cal = v10_cal_oof[target]
        log.info(f"    Best: {best_cb_cfg['name']}, OOF={best_cb_loss:.4f}, V10={v10_cal:.4f}")
    
    cb_avg = np.mean([log_loss(feat[t], cb_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  CB avg: {cb_avg:.6f}")
    
    # ===== Weighted blend =====
    log.info("\n=== Weighted Blend ===")
    best_blend = float("inf")
    best_w = 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        blended = {}
        for t in TARGET_COLS:
            blended[t] = w * lgb_oof[t] + (1-w) * cb_oof[t]
        blend_loss = np.mean([log_loss(feat[t], blended[t], labels=[0,1]) for t in TARGET_COLS])
        if blend_loss < best_blend:
            best_blend = blend_loss; best_w = w
    
    log.info(f"  Best blend w_LGB={best_w:.1f}, w_CB={1-best_w:.1f}, loss={best_blend:.6f}")
    log.info(f"  LGB-only: {lgb_avg:.6f}, CB-only: {cb_avg:.6f}")
    
    # Pick best
    candidates = {
        "lgb_only": lgb_avg,
        "cb_only": cb_avg,
        "blend": best_blend,
    }
    best_name = min(candidates, key=candidates.get)
    best_avg = candidates[best_name]
    
    log.info(f"\n  Best strategy: {best_name} ({best_avg:.6f})")
    
    if best_avg >= v10_avg - 0.001:
        log.info(f"\n❌ V12 NOT better than V10 ({best_avg:.6f} >= {v10_avg:.6f}). Stopping.")
        return
    
    log.info(f"\n✅ V12 BEATS V10! diff: {best_avg-v10_avg:+.6f}")
    
    # Build final OOF
    if best_name == "lgb_only":
        v12_oof = lgb_oof
    elif best_name == "cb_only":
        v12_oof = cb_oof
    else:
        v12_oof = {}
        for t in TARGET_COLS:
            v12_oof[t] = best_w * lgb_oof[t] + (1-best_w) * cb_oof[t]
    
    # Calibration
    log.info("\n=== Calibration ===")
    cal_oof = {}
    for t in TARGET_COLS:
        shift = train_rate[t] - v12_oof[t].mean()
        cal = np.clip(v12_oof[t] + shift, 0.0001, 0.9999)
        cal_oof[t] = cal
        cl = log_loss(feat[t], cal, labels=[0,1])
        log.info(f"  {t}: cal_OOF={cl:.4f}")
    avg_cal = np.mean([log_loss(feat[t], cal_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"  Avg cal OOF: {avg_cal:.6f}")
    
    # Generate submission
    log.info("\n=== Generate submission ===")
    spec = importlib.util.spec_from_file_location("02_feature_engineering", Path("src/02_feature_engineering.py"))
    fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
    spec2 = importlib.util.spec_from_file_location("01_load_data", Path("src/01_load_data.py"))
    ld = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ld)
    
    pdfs = {}; dd = Path("data_raw/ch2025_data_items")
    pn = {"mACStatus":"ch2025_mACStatus.parquet","mActivity":"ch2025_mActivity.parquet",
        "mAmbience":"ch2025_mAmbience.parquet","mBle":"ch2025_mBle.parquet",
        "mGps":"ch2025_mGps.parquet","mLight":"ch2025_mLight.parquet",
        "mScreenStatus":"ch2025_mScreenStatus.parquet","mUsageStats":"ch2025_mUsageStats.parquet",
        "mWifi":"ch2025_mWifi.parquet","wHr":"ch2025_wHr.parquet",
        "wLight":"ch2025_wLight.parquet","wPedo":"ch2025_wPedo.parquet"}
    sample = pd.read_csv("data_raw/ch2026_submission_sample.csv")
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
    sample["sleep_date"] = pd.to_datetime(sample["sleep_date"]).dt.date
    td = set(sample["sleep_date"].astype(str).tolist() + sample["lifelog_date"].astype(str).tolist())
    for n, fn in pn.items():
        df = pd.read_parquet(dd / fn); df = ld.build_merge_key(df)
        pdfs[n] = df[df["date"].astype(str).isin(td)]
    tf = fe.create_day_features(pdfs, sample)
    tfc = get_feat_cols(tf)
    tf, _ = add_personalization(tf, tfc)
    
    preds = tf[["subject_id","sleep_date","lifelog_date"]].copy()
    
    # Train final models
    for target in TARGET_COLS:
        log.info(f"\n  Training {target} ({best_name})...")
        sc = lgb_scols[target]; y_a = feat[target].values
        Xa = feat[sc].fillna(0).values; tX = tf[sc].fillna(0).values
        sn = [sanitize(c) for c in sc]
        tc = v10_cfgs[target]
        lp = {"objective":"binary","metric":"binary_logloss","verbose":-1,
            "num_leaves":tc["nl"],"max_depth":tc["md"],"learning_rate":tc["lr"],
            "n_estimators":tc["ne"],"subsample":tc["ss"],"colsample_bytree":tc["cst"],
            "reg_alpha":tc["ra"],"reg_lambda":tc["rl"],"min_child_samples":tc["mc"],
            "force_row_wise":True,"n_jobs":-1}
        np_ = max((y_a==1).sum(),1); nn = (y_a==0).sum(); spw = nn/np_
        
        # LGB
        ap = np.zeros(len(tX))
        for si, seed in enumerate(RANDOM_SEEDS):
            sp = {**lp, "random_state":seed, "scale_pos_weight":spw}
            da = lgb.Dataset(Xa, label=y_a, feature_name=sn, params={"verbose":"-1"})
            ap += lgb.train(sp, da, num_boost_round=tc["ne"]).predict(tX)
            if (si+1)%5==0: log.info(f"    LGB seed {si+1}/{N_SEEDS_LGB}")
        ap /= N_SEEDS_LGB
        
        blended = ap  # default LGB
        if best_name == "cb_only":
            # CB only
            cap = np.zeros(len(tX))
            cb_cfg = cb_best_cfg[target]
            for si, seed in enumerate(RANDOM_SEEDS_CB):
                cb_pool = Pool(Xa, y_a)
                mdl = CatBoostClassifier(
                    iterations=cb_cfg["iterations"],depth=cb_cfg["depth"],
                    learning_rate=cb_cfg["learning_rate"],l2_leaf_reg=cb_cfg["l2_leaf_reg"],
                    bagging_temperature=cb_cfg["bagging_temperature"],
                    random_strength=cb_cfg["random_strength"],subsample=cb_cfg["subsample"],
                    loss_function="Logloss",one_hot_max_size=3,
                    random_state=seed,scale_pos_weight=spw,verbose=0)
                mdl.fit(cb_pool)
                cap += mdl.predict(tX, prediction_type="Probability")[:, 1]
            blended = cap
        elif best_name == "blend":
            # CB predictions for blending
            cap = np.zeros(len(tX))
            cb_cfg = cb_best_cfg[target]
            for si, seed in enumerate(RANDOM_SEEDS_CB):
                cb_pool = Pool(Xa, y_a)
                mdl = CatBoostClassifier(
                    iterations=cb_cfg["iterations"],depth=cb_cfg["depth"],
                    learning_rate=cb_cfg["learning_rate"],l2_leaf_reg=cb_cfg["l2_leaf_reg"],
                    bagging_temperature=cb_cfg["bagging_temperature"],
                    random_strength=cb_cfg["random_strength"],subsample=cb_cfg["subsample"],
                    loss_function="Logloss",one_hot_max_size=3,
                    random_state=seed,scale_pos_weight=spw,verbose=0)
                mdl.fit(cb_pool)
                cap += mdl.predict(tX, prediction_type="Probability")[:, 1]
            blended = best_w * ap + (1-best_w) * cap
        
        shift = train_rate[target] - blended.mean()
        cp = np.clip(blended + shift, 0.0001, 0.9999)
        preds[target] = cp
        log.info(f"    {target}: mean={cp.mean():.4f}, range=[{cp.min():.4f},{cp.max():.4f}]")
    
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sp = SUBMIT_DIR / f"submission_v12_{ts}.csv"
    preds.to_csv(sp, index=False)
    log.info(f"\nSubmission: {sp}")
    
    meta_out = {"version":"v12","submission_file":str(sp),"timestamp":ts,"n_samples":len(preds),
        "n_seeds_lgb":N_SEEDS_LGB,"n_seeds_cb":N_SEEDS_CB,"n_splits":N_SPLITS,
        "best_strategy":best_name,"lgb_avg":float(lgb_avg),"cb_avg":float(cb_avg),
        "blend_weight":best_w,"avg_cal":float(avg_cal),
        "improvements":["CatBoost ensemble","weighted blending"],
        "per_target":{}}
    for t in TARGET_COLS:
        ol = log_loss(feat[t], v12_oof[t], labels=[0,1])
        cl = log_loss(feat[t], cal_oof[t], labels=[0,1])
        meta_out["per_target"][t] = {"oof_cv_loss":float(ol),"cal_oof_loss":float(cl),
            "oof_mean":float(v12_oof[t].mean()),"cal_mean":float(preds[t].mean()),
            "train_rate":float(train_rate[t]),"pred_min":float(preds[t].min()),"pred_max":float(preds[t].max())}
    
    mp = SUBMIT_DIR / f"meta_v12_{ts}.json"
    with open(mp,"w") as f: json.dump(meta_out, f, indent=2, default=str)
    log.info(f"Metadata: {mp}")
    
    log.info(f"\n{'='*70} V12 SUMMARY {'='*70}")
    log.info(f"{'Target':<6} {'V10 Cal':<10} {'V12 OOF':<10} {'V12 Cal':<10} {'Diff':<10}")
    for t in TARGET_COLS:
        ol = log_loss(feat[t], v12_oof[t], labels=[0,1])
        cl = log_loss(feat[t], cal_oof[t], labels=[0,1])
        log.info(f"{t:<6} {v10_cal_oof[t]:<10.4f} {ol:<10.4f} {cl:<10.4f} {cl-v10_cal_oof[t]:+-.4f}")
    log.info(f"\nV12 avg cal OOF: {avg_cal:.6f} (V10: {v10_avg:.6f}, diff: {avg_cal-v10_avg:+.6f})")
    log.info(f"Best strategy: {best_name}")
    log.info(f"Total time: {time.time()-t_total:.0f}s")

if __name__ == "__main__":
    main()
