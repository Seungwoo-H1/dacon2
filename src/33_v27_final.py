"""
V27 — Rolling(3d,7d) + Seasonal + Missing — Top-30

Based on ablation: rolling alone is best. Add seasonal + missing indicators.
Top-20 or Top-30 only. 5 seeds for speed.
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
SEEDS = [42,123,456,789,1024]
N_SEEDS = len(SEEDS)
LGB = {'objective':'binary','metric':'binary_logloss','num_leaves':15,'max_depth':4,
       'learning_rate':0.03,'n_estimators':500,'subsample':0.7,'colsample_bytree':0.7,
       'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def mm(p, r): return np.clip(p+(r-p.mean()),0.0001,0.9999)

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

def add_seasonal(df):
    df=df.copy(); d=pd.to_datetime(df['date'])
    df['is_weekend']=(d.dt.dayofweek>=5).astype(int)
    df['month']=d.dt.month
    df['doy_sin']=np.sin(2*np.pi*d.dt.dayofyear/365.25)
    df['doy_cos']=np.cos(2*np.pi*d.dt.dayofyear/365.25)
    return df

def rank_f(feat, cols, target):
    y=feat[target].values; X=feat[cols].fillna(0).values
    spw=((y==0).sum())/max((y==1).sum(),1)
    p={**LGB,'num_leaves':15,'max_depth':4,'n_estimators':50,'scale_pos_weight':spw,'random_state':42}
    ds=lgb.Dataset(X,label=y,feature_name=[sanitize(c) for c in cols],params={'verbose':'-1'})
    m=lgb.train(p,ds,num_boost_round=50)
    imp=m.feature_importance(importance_type='gain')
    return sorted(zip(cols,imp),key=lambda x:-x[1])

def cv_f(feat, cols, target):
    y=feat[target].values; gkf=GroupKFold(n_splits=5)
    oof=np.zeros((len(y),N_SEEDS))
    spw=((y==0).sum())/max((y==1).sum(),1); sn=[sanitize(c) for c in cols]
    X=feat[cols].fillna(0).values
    for si,s in enumerate(SEEDS):
        cfg={**LGB,'random_state':s,'scale_pos_weight':spw}
        for tr,va in gkf.split(feat,y,feat['subject_id']):
            ds=lgb.Dataset(X[tr],label=y[tr],feature_name=sn,params={'verbose':'-1'})
            vd=lgb.Dataset(X[va],label=y[va],feature_name=sn,reference=ds,params={'verbose':'-1'})
            m=lgb.train(cfg,ds,num_boost_round=500,valid_sets=[vd],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
            oof[va,si]=m.predict(X[va])
    return oof.mean(axis=1)

def main():
    log.info("="*60)
    log.info("V27 — Rolling(3d,7d)+Seasonal+Missing Top-30")
    log.info("="*60)

    feat=pd.read_parquet(DATA_PROCESSED/"features.parquet")
    raw=[c for c in feat.columns if c not in META_COLS|set(TARGET_COLS) and feat[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
    base=[c for c in raw if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    log.info(f"Base cleaned: {len(base)}")

    feat=feat.copy()
    bad=(feat['wHr_hr_mean']<20)|(feat['wHr_hr_mean']>180)
    feat.loc[bad,'wHr_hr_mean']=np.nan; feat.loc[bad,'wHr_hr_std']=np.nan

    feat_r, r_cols=add_rolling(feat, base)
    feat_s=add_seasonal(feat_r)

    # Missing indicators
    all_num=[c for c in feat_s.columns if c not in META_COLS|set(TARGET_COLS) and feat_s[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
    miss=[c for c in all_num if feat_s[c].isnull().mean()>0.05]
    for c in miss: feat_s[f'{c}_missing']=feat_s[c].isnull().astype(int)
    feat_s=feat_s.fillna(0)
    train_rate={t:feat_s[t].mean() for t in TARGET_COLS}

    all_cols=base+r_cols+['is_weekend','month','doy_sin','doy_cos']+[f'{c}_missing' for c in miss]
    log.info(f"Total features: {len(all_cols)}")

    # ── Per-target CV ──
    log.info("\n=== Per-target CV ===")
    all_oof={}; all_sel={}; best_n_map={}
    for target in TARGET_COLS:
        log.info(f"--- {target} ---")
        leak=LEAK_S if target.startswith('S') else LEAK_Q
        avail=[c for c in all_cols if c not in leak]

        ranked=rank_f(feat_s, avail, target)

        # Test N=20, 30
        best_n=30; best_cv=float('inf'); best_s=None
        for n in [20,30]:
            sel=[r[0] for r in ranked[:n]]
            oof=cv_f(feat_s, sel, target)
            cal=mm(oof, train_rate[target])
            loss=log_loss(feat_s[target],cal,labels=[0,1])
            if loss<best_cv: best_cv=loss; best_n=n; best_s=sel

        oof=cv_f(feat_s, best_s, target)
        cal=mm(oof, train_rate[target])
        loss=log_loss(feat_s[target],cal,labels=[0,1])
        all_oof[target]=oof; all_sel[target]=best_s; best_n_map[target]=best_n
        log.info(f"  N={best_n}, Cal OOF={loss:.4f}, train_rate={train_rate[target]:.3f}")

    avg=np.mean([log_loss(feat_s[t],mm(all_oof[t],train_rate[t]),labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\nV27 Cal OOF Avg: {avg:.4f} (V10: 0.6038, Δ: {avg-0.6038:+.4f})")

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
    tcols=[c for c in tf.columns if c not in META_COLS|set(TARGET_COLS) and tf[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
    tcols=[c for c in tcols if c not in CONSTANT_COLS and c not in COLLINEAR_DROP]
    tf,_=add_rolling(tf, tcols)
    tf=add_seasonal(tf)
    for c in miss:
        if c in tf.columns: tf[f'{c}_missing']=tf[c].isnull().astype(int)
    tf=tf.fillna(0)

    predictions=tf[['subject_id','sleep_date','lifelog_date']].copy()
    for target in TARGET_COLS:
        sel=all_sel[target]
        ya=feat_s[target].values; Xa=feat_s[sel].fillna(0).values
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
    sp=SUBMIT_DIR/f'submission_v27_{ts}.csv'
    predictions.to_csv(sp,index=False)
    log.info(f"✅ Saved: {sp}")

    meta={'version':'v27','submission_file':str(sp),'timestamp':ts,'n_samples':len(predictions),
          'n_seeds':N_SEEDS,'n_splits':5,'features':{'base':len(base),'rolling':len(r_cols),
          'seasonal':4,'missing':len(miss),'total':len(all_cols)},
          'calibration':'mean-matching+clip','leakage_fix':'wrist night data removed',
          'cv_avg':float(avg),'per_target':{}}
    for t in TARGET_COLS:
        co=log_loss(feat_s[t],mm(all_oof[t],train_rate[t]),labels=[0,1])
        meta['per_target'][t]={'n_features':len(all_sel[t]),'cal_oof_loss':float(co),
            'cal_mean':float(predictions[t].mean()),'train_rate':float(train_rate[t]),
            'pred_min':float(predictions[t].min()),'pred_max':float(predictions[t].max())}
    mp=SUBMIT_DIR/f'meta_v27_{ts}.json'
    with open(mp,'w') as f: json.dump(meta,f,indent=2,default=str)
    log.info(f"Metadata: {mp}")

    log.info(f"\n{'='*60}")
    log.info("V27 FINAL")
    log.info(f"{'='*60}")
    log.info(f"Submission: {sp}")
    log.info(f"{'Target':<6} {'Cal OOF':<12} {'Test Mean':<12} {'Train Rate':<10} {'Shift'}")
    for t in TARGET_COLS:
        co=log_loss(feat_s[t],mm(all_oof[t],train_rate[t]),labels=[0,1])
        log.info(f"{t:<6} {co:<12.4f} {predictions[t].mean():<12.4f} {train_rate[t]:<10.3f} {predictions[t].mean()-train_rate[t]:+.4f}")
    log.info(f"  Avg Cal OOF: {avg:.4f} (V10: 0.6038, Δ: {avg-0.6038:+.4f})")

if __name__=="__main__": main()
