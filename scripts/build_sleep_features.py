"""
build_sleep_v3.py — multi-sensor sleep-window detection (Tier-1 idea #1-6).
Per subject-night [lifelog_date 20:00 -> sleep_date 12:00], fuse minute-resolution phone sensors:
  asleep(min) = screen-off (m_screen_use==0) AND still (m_activity==3)
Longest contiguous asleep block (gap tol 10min) = main sleep period.
Features: tst (sleep minutes), onset/wake hour, in_bed (first charge/screen-off), sol (in_bed->onset),
  waso (arousal minutes inside window), se (tst/time_in_bed), n_arousal, light_sleep_env, + per-subject z.
Output data_processed/sleep_v3.parquet for all 700 subject-nights (train+test).
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='data_raw/ch2025_data_items/'
key=pd.concat([
    pd.read_csv('data_raw/ch2026_metrics_train.csv',parse_dates=['sleep_date','lifelog_date']),
    pd.read_csv('data_raw/ch2026_submission_sample.csv',parse_dates=['sleep_date','lifelog_date'])
],ignore_index=True)[['subject_id','sleep_date','lifelog_date']].drop_duplicates().reset_index(drop=True)

def load(fn,col):
    df=pd.read_parquet(D+f'ch2025_{fn}.parquet',columns=['subject_id','timestamp',col])
    df[col]=pd.to_numeric(df[col],errors='coerce'); return df.sort_values('timestamp')
scr=load('mScreenStatus','m_screen_use'); act=load('mActivity','m_activity')
chg=load('mACStatus','m_charging'); wl=load('wLight','w_light')

def longest_run(mask, idx_min, gap=10):
    # mask: bool array on 1-min grid; return (len, start_i, end_i) of longest run allowing gaps<=gap mins
    best=(0,0,0); i=0; n=len(mask)
    while i<n:
        if not mask[i]: i+=1; continue
        j=i; last=i
        while j<n:
            if mask[j]: last=j; j+=1
            elif j-last<=gap: j+=1
            else: break
        runlen=last-i+1
        if runlen>best[0]: best=(runlen,i,last)
        i=j
    return best

rows=[]
gsc=dict(tuple(scr.groupby('subject_id'))); gac=dict(tuple(act.groupby('subject_id')))
gch=dict(tuple(chg.groupby('subject_id'))); gwl=dict(tuple(wl.groupby('subject_id')))
for _,r in key.iterrows():
    sid=r['subject_id']; t0=r['lifelog_date'].replace(hour=20,minute=0); t1=(r['sleep_date']).replace(hour=12,minute=0)
    grid=pd.date_range(t0,t1,freq='1min')
    def series(g):
        if sid not in g: return None
        d=g[sid]; d=d[(d['timestamp']>=t0)&(d['timestamp']<=t1)]
        if len(d)==0: return None
        s=d.set_index('timestamp').iloc[:,-1]
        return s.reindex(grid,method='ffill')
    s_scr=series(gsc); s_act=series(gac); s_chg=series(gch); s_wl=series(gwl)
    rec={'subject_id':sid,'sleep_date':r['sleep_date'],'lifelog_date':r['lifelog_date']}
    if s_scr is None or s_act is None:
        rows.append(rec); continue
    asleep=((s_scr.fillna(1).values==0)&(s_act.fillna(0).values==3))
    L,si,ei=longest_run(asleep,grid,gap=10)
    rec['tst']=float(L)  # minutes
    if L>0:
        rec['onset_h']=grid[si].hour+grid[si].minute/60.0
        rec['wake_h']=grid[ei].hour+grid[ei].minute/60.0
        win=slice(si,ei+1)
        rec['waso']=float((~asleep[win]).sum())     # arousal mins inside sleep block
        # arousals = transitions asleep->awake inside block
        a=asleep[win].astype(int); rec['n_arousal']=float(((a[:-1]==1)&(a[1:]==0)).sum())
        rec['se']=L/max(ei-si+1,1)                    # efficiency proxy
        # in-bed: first charging-on OR first screen-off sustained before onset
        if s_chg is not None:
            ch=(s_chg.fillna(0).values==1)
            ci=np.where(ch[:si+1])[0]
            ib=ci[0] if len(ci)>0 else si
        else: ib=si
        rec['in_bed_h']=grid[ib].hour+grid[ib].minute/60.0
        rec['sol']=max(si-ib,0)                       # onset latency mins
        if s_wl is not None:
            rec['sleep_light']=float(np.nanmean(s_wl.values[win]))
    rows.append(rec)
out=pd.DataFrame(rows)
# per-subject z-scores
feats=[c for c in out.columns if c not in ['subject_id','sleep_date','lifelog_date']]
for c in feats:
    out[c+'_z']=out.groupby('subject_id')[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
out.to_parquet('data_processed/sleep_v3.parquet',index=False)
print('wrote data_processed/sleep_v3.parquet',out.shape)
print('coverage tst non-null:',out['tst'].notna().sum(),'/',len(out))
print(out[['tst','se','sol','waso','onset_h','wake_h']].describe().round(2).to_string())
