"""
V33 — Rolling(3d,7d) Only — Submission (diverse 50 features, 5 seeds)

Pick 10 diverse base features from 10 groups, then add 3d/7d rolling.
Total ~50 features. Skip all ranking.
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
from config import TARGETS, DATA_PROCESSED, SUBMIT_DIR

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
SEEDS = [42,123,456,789,1024]
N_SEEDS = len(SEEDS)
LGB = {'objective':'binary','metric':'binary_logloss','num_leaves':15,'max_depth':4,
       'learning_rate':0.03,'n_estimators':500,'subsample':0.7,'colsample_bytree':0.7,
       'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def mm(p, r): return np.clip(p+(r-p.mean()),0.0001,0.9999)

CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max','mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std','mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std','mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std','mWifi_wifi_bssid_count_max',
]
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

# 10 diverse base features from 10 different groups
BASE_PICK = [
    'mACStatus_m_charging_mean',      # mACStatus
    'mActivity_m_activity_mean',      # mActivity
    'mAmbience_ambience_speech_sum',  # mAmbience
    'mBle_ble_count_mean',            # mBle
    'mGps_gps_count_mean',            # mGps
    'mLight_m_light_mean',            # mLight
    'mScreenStatus_m_screen_use_mean',# mScreenStatus
    'mUsageStats_usage_total_time_mean', # mUsageStats
    'mWifi_wifi_count_mean',          # mWifi
    'wPedo_pedo_step_mean',           # wPedo (will be dropped for S targets)
]

def add_rolling(df, cols):
    df=df.copy().sort_values(['subject_id','date']); new=[]
    for c in cols:
        g=df.groupby('subject_id')[c]
        for w in [3,7]:
            rm=g.rolling(w,min_periods=1).mean().reset_index(level=0,drop=True)
            rs=g.rolling(w,min_periods=1).std().fillna(0).reset_index(level=0,drop=True)
            df[f'{c}_rm{w}']=rm.values; df[f'{c}_rs{w}']=rs.values
            new.extend([f'{c}_rm{w}',f'{c}_rs{w}'])
    return df,new

def cv_f(feat, cols, target, seeds):
    y=feat[target].values; gkf=GroupKFold(n_splits=5)
    oof=np.zeros((len(y),len(seeds)))
    spw=((y==0).sum())/max((y==1).sum(),1); sn=[sanitize(c) for c in cols]
    X=feat[cols].fillna(0).values
    for si,s in enumerate(seeds):
        cfg={**LGB,'random_state':s,'scale_pos_weight':spw}
        for tr,va in gkf.split(feat,y,feat['subject_id']):
            ds=lgb.Dataset(X[tr],label=y[tr],feature_name=sn,params={'verbose':'-1'})
            vd=lgb.Dataset(X[va],label=y[va],feature_name=sn,reference=ds,params={'verbose':'-1'})
            m=lgb.train(cfg,ds,num_boost_round=500,valid_sets=[vd],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
            oof[va,si]=m.predict(X[va])
    return oof.mean(axis=1)

def main():
    log.info("="*70)
    log.info("V33 — Rolling(3d,7d) Only — 10 Base + Rolling(3d,7d) — Fast")
    log.info("="*70)

    feat=pd.read_parquet(DATA_PROCESSED/"features.parquet")
    raw=[c for c in feat.columns if c not in META_COLS|set(TARGET_COLS) and feat[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

    # Check which base_pick features exist
    existing_bp = [c for c in BASE_PICK if c in raw]
    missing_bp = [c for c in BASE_PICK if c not in raw]
    log.info(f"Base pick: {len(existing_bp)}/{len(BASE_PICK)} found, missing: {missing_bp}")

    feat=feat.copy()
    bad=(feat['wHr_hr_mean']<20)|(feat['wHr_hr_mean']>180)
    feat.loc[bad,'wHr_hr_mean']=np.nan; feat.loc[bad,'wHr_hr_std']=np.nan

    feat_r, r_cols=add_rolling(feat, existing_bp)
    feat_r=feat_r.fillna(0)
    all_cols=existing_bp+r_cols

    train_rate={t:feat_r[t].mean() for t in TARGET_COLS}
    log.info(f"Total features: {len(all_cols)} ({len(existing_bp)} base + {len(r_cols)} rolling)")

    # ── Per-target CV ──
    log.info("\n=== Per-target CV (rolling only, 50 features, 5 seeds) ===")
    all_oof={}; all_losses={}; all_seles={}

    for target in TARGET_COLS:
        log.info(f"--- {target} ---")
        leak=LEAK_S if target.startswith('S') else LEAK_Q
        sel=[c for c in all_cols if c not in leak]
        log.info(f"  Using {len(sel)} features (after leak removal)")

        oof=cv_f(feat_r, sel, target, SEEDS)
        cal=mm(oof, train_rate[target])
        loss=log_loss(feat_r[target],cal,labels=[0,1])
        all_oof[target]=oof; all_losses[target]=loss; all_seles[target]=sel
        log.info(f"  Cal OOF={loss:.4f}, train_rate={train_rate[target]:.3f}")

    avg=np.mean([all_losses[t] for t in TARGET_COLS])
    log.info(f"\nV33 Cal OOF Avg: {avg:.4f} (V10: 0.6038, V25: 0.5778, Δ: {avg-0.6038:+.4f})")

    # ── Submission ──
    log.info("\n=== Generating submission ===")
    spec1=importlib.util.spec_from_file_location("01_load_data",Path('src/01_load_data.py'))
    ld_mod=importlib.util.module_from_spec(spec1); spec1.loader.exec_module(ld_mod)
    spec2=importlib.util.spec_from_file_location("02_feature_engineering",Path('src/02_feature_engineering.py'))
    fe=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fe)

    sample=pd.read_csv('data_raw/ch2026_submission_sample.csv')
    sample['lifelog_date']=pd.to_datetime(sample['lifelog_date']).dt.date
    sample['sleep_date']=pd.to_datetime(sample['sleep_date']).dt.date
    test_dates=set(sample["sleep_date"].astype(str).tolist()+sample["lifelog_date"].astype(str).tolist())

    pq={n:f"ch2025_{n}.parquet" for n in ["mACStatus","mActivity","mAmbience","mBle","mGps","mLight","mScreenStatus","mUsageStats","mWifi","wHr","wLight","wPedo"]}
    pdfs={}
    for n,f in pq.items():
        p=Path("data_raw/ch2025_data_items")/f
        if p.exists():
            df=pd.read_parquet(p); df=ld_mod.build_merge_key(df)
            df=df[df["date"].astype(str).isin(test_dates)]; pdfs[n]=df

    tf=fe.create_day_features(pdfs, sample)

    # Match existing base_pick features in test data
    tcols=[c for c in tf.columns if c not in META_COLS|set(TARGET_COLS) and tf[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
    tf_base_pick=[c for c in tcols if c in existing_bp]
    tf,_=add_rolling(tf, tf_base_pick)
    tf=tf.fillna(0)

    predictions=tf[['subject_id','sleep_date','lifelog_date']].copy()
    for target in TARGET_COLS:
        sel=all_seles[target]
        ya=feat_r[target].values; Xa=feat_r[sel].fillna(0).values
        Xt=tf[sel].fillna(0).values; sn=[sanitize(c) for c in sel]
        spw=((ya==0).sum())/max((ya==1).sum(),1)
        ap=np.zeros(len(Xt))
        for s in SEEDS:
            ds=lgb.Dataset(Xa,label=ya,feature_name=sn,params={'verbose':'-1'})
            m=lgb.train({**LGB,'random_state':s,'scale_pos_weight':spw},ds,num_boost_round=500)
            ap+=m.predict(Xt)
        ap/=N_SEEDS; cal=mm(ap,train_rate[target])
        predictions[target]=cal
        log.info(f"  {target}: mean={cal.mean():.4f}, shift={cal.mean()-train_rate[target]:+.4f}")

    ts=pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sp=SUBMIT_DIR/f'submission_v33_{ts}.csv'
    predictions.to_csv(sp,index=False)
    log.info(f"✅ Saved: {sp}")

    meta={'version':'v33','submission_file':str(sp),'timestamp':ts,'n_samples':len(predictions),
          'n_seeds':N_SEEDS,'n_splits':5,
          'features':{'base':len(existing_bp),'rolling':len(r_cols),'total':len(all_cols)},
          'calibration':'mean-matching+clip','leakage_fix':'wrist night data removed',
          'strategy':'rolling(3d,7d) — 10 diverse base + rolling',
          'cv_avg':float(avg),
          'per_target':{}}
    for t in TARGET_COLS:
        meta['per_target'][t]={'n_features':len(all_seles[t]),'cal_oof_loss':float(all_losses[t]),
            'cal_mean':float(predictions[t].mean()),'train_rate':float(train_rate[t]),
            'pred_min':float(predictions[t].min()),'pred_max':float(predictions[t].max())}
    mp=SUBMIT_DIR/f'meta_v33_{ts}.json'
    with open(mp,'w') as f: json.dump(meta,f,indent=2,default=str)
    log.info(f"Metadata: {mp}")

    log.info(f"\n{'='*70}")
    log.info("V33 FINAL")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sp}")
    log.info(f"{'Target':<6} {'N_Feat':<8} {'Cal OOF':<12} {'Test Mean':<12} {'Train':<8} {'Shift':<8}")
    for t in TARGET_COLS:
        log.info(f"{t:<6} {len(all_seles[t]):<8} {all_losses[t]:<12.4f} {predictions[t].mean():<12.4f} {train_rate[t]:<8.3f} {predictions[t].mean()-train_rate[t]:<+8.4f}")
    log.info(f"  Avg Cal OOF: {avg:.4f} (V10: 0.6038, V25: 0.5778, Δ: {avg-0.6038:+.4f})")

if __name__=="__main__": main()
