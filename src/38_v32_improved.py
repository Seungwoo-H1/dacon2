"""
V32 — Improved: V10 Strengths + LOSO CV + Selective Rolling

Key design:
1. LOSO CV: GroupKFold(n_splits=10) — each subject once out
2. Selective rolling: mean-only (3d, 7d), no std
3. Personalization: per-subject z-score
4. Per-target hyperparameter tuning via LOSO CV
5. 20-seed ensemble for final submission

Optimization: Use 5 seeds for CV tuning search (fast), 20 seeds for final submission.
This keeps tuning manageable (~400 model trainings/target vs 2,400).
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

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────
sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

# ── Seeds ────────────────────────────────────────────────
# Fewer seeds for CV tuning (fast), 20 for final submission
CV_SEEDS = [42, 123, 456, 789, 1024]
N_CV_SEEDS = len(CV_SEEDS)
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
                6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 10  # LOSO

# ── Column cleanup ───────────────────────────────────────
CONSTANT_COLS = [
    'mACStatus_m_charging_min', 'mACStatus_m_charging_max',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min', 'mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean', 'mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max', 'mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min', 'mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean', 'mBle_ble_device_count_std', 'mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean', 'mWifi_wifi_bssid_count_std', 'mWifi_wifi_bssid_count_max',
]
LEAKAGE_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count'}


def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def get_fc(df):
    return [c for c in df.columns if c not in META_COLS|set(TARGET_COLS)
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

# ── Rolling (mean only) ──────────────────────────────────
def add_rolling(df, cols, windows=[3, 7]):
    df = df.copy().sort_values(['subject_id', 'date'])
    nc = []
    for c in cols:
        g = df.groupby('subject_id')[c]
        for w in windows:
            rm = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{c}_rm{w}'] = rm.values
            nc.append(f'{c}_rm{w}')
    return df, nc

# ── Personalization ──────────────────────────────────────
def add_personalization(df, feature_cols, stats=None):
    df = df.copy()
    pcols = []
    cs = {} if stats is None else stats
    for col in feature_cols:
        if stats is None:
            cf = df[col].fillna(0)
            ss = cf.groupby(df['subject_id']).agg(['mean','std'])
            ss.columns = [f'{col}_sm', f'{col}_ss']
            ss = ss.reset_index()
            df = df.merge(ss, on='subject_id', how='left')
            cs[col] = {'mean': df[f'{col}_sm'].mean(), 'std': float(df[f'{col}_ss'].max())}
        else:
            sm, s = stats[col]['mean'], max(stats[col]['std'], 1e-6)
            df[f'{col}_sm'] = sm
            df[f'{col}_ss'] = s
        m1 = (df[f'{col}_ss'] == 0) | df[col].isnull()
        df[f'{col}_z'] = np.where(m1, 0.0, (df[col].fillna(0) - df[f'{col}_sm']) / df[f'{col}_ss'])
        pcols.append(f'{col}_z')
    return df, pcols, cs

# ── Feature ranking (fast, small pool) ───────────────────
def rank_features(feat, fcols, target):
    y = feat[target].values
    X = feat[fcols].fillna(0).values
    np_ = max((y==1).sum(), 1); nn = (y==0).sum()
    spw = nn / np_
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
              'force_row_wise':True,'n_jobs':-1}
    sn = [sanitize(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    m = lgb.train(params, ds, num_boost_round=100)
    imp = m.feature_importance(importance_type="gain")
    return sorted(zip(fcols, imp), key=lambda x: -x[1])

# ── LGB base params ──────────────────────────────────────
LGB_BASE = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1,
}
LGB_CFGS = [
    {'name':'C1','nl':8,'md':3,'lr':0.02,'ne':200,'ss':0.6,'cst':0.6,'ra':2.0,'rl':5.0,'mc':15},
    {'name':'C2','nl':10,'md':3,'lr':0.03,'ne':300,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C3','nl':12,'md':4,'lr':0.03,'ne':200,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C4','nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C5','nl':20,'md':5,'lr':0.02,'ne':300,'ss':0.7,'cst':0.7,'ra':0.5,'rl':2.0,'mc':8},
    {'name':'C6','nl':6,'md':2,'lr':0.02,'ne':200,'ss':0.5,'cst':0.5,'ra':5.0,'rl':10.0,'mc':20},
]

# ── LOSO CV (use CV_SEEDS for speed) ────────────────────
def lgb_cv(feat, cols, target, seeds=CV_SEEDS):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    spw = ((y==0).sum())/max((y==1).sum(),1)
    sn = [sanitize(c) for c in cols]
    for si, seed in enumerate(seeds):
        cfg = {**LGB_BASE, 'random_state':seed, 'scale_pos_weight':spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            Xtr = feat.iloc[tr][cols].fillna(0).values
            Xva = feat.iloc[va][cols].fillna(0).values
            dtr = lgb.Dataset(Xtr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            dva = lgb.Dataset(Xva, label=y[va], feature_name=sn, reference=dtr, params={'verbose':'-1'})
            m = lgb.train(cfg, dtr, num_boost_round=500, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(Xva)
    return oof.mean(axis=1)

# ── Calibration ──────────────────────────────────────────
def mm(pred, rate): return np.clip(pred + (rate - pred.mean()), 0.0001, 0.9999)

# ── Tuning ───────────────────────────────────────────────
def tune_target(feat, leak_free, target, tr_rate):
    y = feat[target].values
    best_cfg = None; best_cv = float('inf'); best_oof = None; best_cols = None

    # 2-pass ranking to keep pool small
    base_roll = [c for c in leak_free if not c.endswith('_z')]
    r1 = rank_features(feat, base_roll, target)
    top40 = [r[0] for r in r1[:40]]
    z_of_top = [c + '_z' for c in top40 if c + '_z' in leak_free]
    combined = top40 + z_of_top
    r2 = rank_features(feat, combined, target)
    ranked = [r[0] for r in r2]

    for nf in [20, 30]:
        if nf > len(ranked): continue
        sel = ranked[:nf]
        spw = ((y==0).sum())/max((y==1).sum(),1)
        for cfg in LGB_CFGS:
            tc = {**LGB_BASE, 'num_leaves':cfg['nl'],'max_depth':cfg['md'],
                  'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
                  'subsample':cfg['ss'],'colsample_bytree':cfg['cst'],
                  'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'min_child_samples':cfg['mc']}
            oof = lgb_cv(feat, sel, target)
            loss = log_loss(y, oof, labels=[0,1])
            shift = abs(oof.mean() - tr_rate)
            # Use OOF loss as score (lower is better)
            if loss < best_cv:
                best_cv = loss
                best_cfg = {**cfg, '_n_feats': nf}
                best_oof = oof
                best_cols = sel

    return best_cfg, best_cols, best_oof

# ── Main ─────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("V32 — V10 Strengths + LOSO CV(10) + Rolling mean(3d,7d) + Z-score")
    log.info("Fast tuning: 5 seeds for CV search, 20 seeds for submission")
    log.info("=" * 70)

    # 1. Load
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Loaded: {feat.shape}")

    raw = get_fc(feat)
    clean = [c for c in raw if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Clean base: {len(clean)}")

    bad = (feat['wHr_hr_mean']<20)|(feat['wHr_hr_mean']>180)
    feat = feat.copy()
    feat.loc[bad,'wHr_hr_mean']=np.nan; feat.loc[bad,'wHr_hr_std']=np.nan

    # 2. Personalization
    log.info("\n--- Personalization ---")
    feat, pers, pers_stats = add_personalization(feat, clean)
    log.info(f"Personalization: {len(pers)} z-cols")

    # 3. Rolling
    log.info("\n--- Rolling mean (3d, 7d) ---")
    feat, roll = add_rolling(feat, clean)
    log.info(f"Rolling: {len(roll)} cols")

    feat = feat.fillna(0)
    tr = {t: feat[t].mean() for t in TARGET_COLS}
    all_num = get_fc(feat)
    log.info(f"Total pool: {len(all_num)}")

    # 4. Per-target tuning
    log.info("\n=== Per-target Tuning (LOSO CV, 5 seeds) ===")
    all_cfg = {}; all_cols = {}; all_oof = {}; all_cal = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (tr={tr[target]:.3f}) ---")
        leak = LEAKAGE_S if target.startswith('S') else LEAKAGE_Q
        lf = [c for c in all_num if c not in leak]
        log.info(f"  Pool: {len(lf)}")

        cfg, sel, oof = tune_target(feat, lf, target, tr[target])
        all_cfg[target] = cfg; all_cols[target] = sel; all_oof[target] = oof
        cal = mm(oof, tr[target])
        all_cal[target] = cal

        lo = log_loss(feat[target], oof, labels=[0,1])
        clo = log_loss(feat[target], cal, labels=[0,1])
        log.info(f"  Config: {cfg}")
        log.info(f"  Pre-cal OOF: {lo:.4f}, Cal OOF: {clo:.4f}")
        log.info(f"  Features: {len(sel)}")

    # 5. Summary
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n{'='*70}")
    log.info("=== SUMMARY ===")
    log.info(f"{'Target':<6} {'Pre-Cal':<12} {'Cal OOF':<12} {'Train':<8} {'CalMean':<10} {'Shift'}")
    for t in TARGET_COLS:
        lo = log_loss(feat[t], all_oof[t], labels=[0,1])
        clo = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"{t:<6} {lo:<12.4f} {clo:<12.4f} {tr[t]:<8.3f} {all_cal[t].mean():<10.4f} {all_cal[t].mean()-tr[t]:+.4f}")
    log.info(f"\n  Avg Cal OOF: {avg_cal:.4f}")
    log.info(f"  V10 benchmark: 0.6038")
    log.info(f"  Δ: {0.6038-avg_cal:+.4f} ({'✅' if 0.6038-avg_cal>0 else '⚠️'})")

    # 6. Submission
    log.info("\n=== Submission ===")
    spec1 = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    ld = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ld)
    spec2 = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    fe = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fe)

    sample = pd.read_csv('data_raw/ch2026_submission_sample.csv')
    sample['lifelog_date'] = pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date'] = pd.to_datetime(sample['sleep_date']).dt.date
    td = set(sample["sleep_date"].astype(str).tolist()+sample["lifelog_date"].astype(str).tolist())

    pqs = {}
    for n, f in {"mACStatus":"ch2025_mACStatus.parquet","mActivity":"ch2025_mActivity.parquet",
                 "mAmbience":"ch2025_mAmbience.parquet","mBle":"ch2025_mBle.parquet",
                 "mGps":"ch2025_mGps.parquet","mLight":"ch2025_mLight.parquet",
                 "mScreenStatus":"ch2025_mScreenStatus.parquet","mUsageStats":"ch2025_mUsageStats.parquet",
                 "mWifi":"ch2025_mWifi.parquet","wHr":"ch2025_wHr.parquet",
                 "wLight":"ch2025_wLight.parquet","wPedo":"ch2025_wPedo.parquet"}.items():
        p = DATA_RAW/"ch2025_data_items"/f
        if p.exists():
            df = pd.read_parquet(p); df = ld.build_merge_key(df)
            df = df[df["date"].astype(str).isin(td)]; pqs[n]=df

    tf = fe.create_day_features(pqs, sample)
    tc = get_fc(tf)

    bt = (tf['wHr_hr_mean']<20)|(tf['wHr_hr_mean']>180)
    tf = tf.copy(); tf.loc[bt,'wHr_hr_mean']=np.nan; tf.loc[bt,'wHr_hr_std']=np.nan

    # Only apply personalization/rolling for cols that exist in training clean set
    tf_cols = [c for c in tc if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    # Keep only cols that have stats (from training)
    tf_cols = [c for c in tf_cols if c in pers_stats]
    tf, _, _ = add_personalization(tf, tf_cols, stats=pers_stats)
    tf, _ = add_rolling(tf, tf_cols); tf = tf.fillna(0)

    pred = tf[['subject_id','sleep_date','lifelog_date']].copy()

    for target in TARGET_COLS:
        log.info(f"\n  {target}: 20-seed ensemble on full data...")
        sel = all_cols[target]; cfg = all_cfg[target]
        ya = feat[target].values; Xa = feat[sel].fillna(0).values
        Xt = tf[sel].fillna(0).values; sn = [sanitize(c) for c in sel]
        spw = ((ya==0).sum())/max((ya==1).sum(),1)
        lp = {**LGB_BASE,'num_leaves':cfg['nl'],'max_depth':cfg['md'],
              'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
              'subsample':cfg['ss'],'colsample_bytree':cfg['cst'],
              'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
              'min_child_samples':cfg['mc'],'scale_pos_weight':spw}

        ap = np.zeros(len(Xt))
        for si, s in enumerate(RANDOM_SEEDS):
            d = lgb.Dataset(Xa, label=ya, feature_name=sn, params={'verbose':'-1'})
            m = lgb.train({**lp,'random_state':s}, d, num_boost_round=cfg['ne'])
            ap += m.predict(Xt)
            if (si+1)%5==0: log.info(f"    seed {si+1}/{N_SEEDS}")

        ap /= N_SEEDS; cal = mm(ap, tr[target])
        pred[target] = cal
        log.info(f"    mean={cal.mean():.4f}, shift={cal.mean()-tr[target]:+.4f}")

    # Save
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sp = SUBMIT_DIR/f'submission_v32_{ts}.csv'
    pred.to_csv(sp, index=False)
    log.info(f"\n✅ Submission: {sp}")

    meta = {'version':'v32','submission_file':str(sp),'timestamp':ts,
            'n_samples':len(pred),'n_seeds':N_SEEDS,'n_splits':N_SPLITS,
            'cv_type':'LOSO','tuning_seeds':N_CV_SEEDS,
            'features':{'base':len(clean),'personalization':len(pers),'rolling':len(roll),'total':len(all_num)},
            'per_target':{}}
    for t in TARGET_COLS:
        meta['per_target'][t] = {
            'config': all_cfg[t], 'n_features': len(all_cols[t]),
            'cal_oof_loss': float(log_loss(feat[t], all_cal[t], labels=[0,1])),
            'cal_mean': float(pred[t].mean()), 'train_rate': float(tr[t]),
        }
    mp = SUBMIT_DIR/f'meta_v32_{ts}.json'
    with open(mp,'w') as f: json.dump(meta,f,indent=2,default=str)
    log.info(f"Metadata: {mp}")

    # Final
    log.info(f"\n{'='*70}\nV32 FINAL\n{'='*70}")
    log.info(f"{'Target':<6} {'OOF':<12} {'Cal OOF':<12} {'TestMean':<10} {'Train':<8} {'Shift'}")
    for t in TARGET_COLS:
        ool = log_loss(feat[t], all_oof[t], labels=[0,1])
        col = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"{t:<6} {ool:<12.4f} {col:<12.4f} {pred[t].mean():<10.4f} {tr[t]:<8.3f} {pred[t].mean()-tr[t]:+.4f}")
    avg_v32 = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\nV32 Avg Cal OOF: {avg_v32:.4f} (V10: 0.6038, Δ: {0.6038-avg_v32:+.4f})")
    return pred

if __name__ == "__main__":
    main()
