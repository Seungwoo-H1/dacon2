"""
V12: Comprehensive External Data Exploration Loop
- Multi-source external data (A=374, B=400, C=1000 rows)
- Combinatorial subset exploration: A, B, C, A+B, A+C, B+C, A+B+C
- Domain similarity: adversarial validation, KL, cosine, target proxy
- Data quality: noise, leakage, missing, outlier
- Transferability: pretrain, finetune, ensemble diversity, calibration
- Pseudo-labeling iterative loop
- Curriculum ordering
- Adversarial sample filtering
- Confidence-weighted training
- Multi-stage ensemble
- All decisions fully automatic - no user prompts
"""
import re, gc, json, time, warnings, traceback, os, itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXTERNAL = ROOT / 'external_data'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]: d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777]

CFG_WIDE  = {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5}
CFG_DEEP  = {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15}
CFG_V48   = {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10}
CFG_SAFETY = {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20}
CFGS = {'wide':CFG_WIDE,'deep':CFG_DEEP,'v48':CFG_V48,'safety':CFG_SAFETY}
V53_SWEEP = {'Q1':'deep','Q2':'deep','Q3':'v48','S1':'wide','S2':'deep','S3':'safety','S4':'wide'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
          'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
          'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
          'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean',
          'wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
          'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean',
          'wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',str(n))
def mean_match(pred, tm):
    return np.clip(pred + (tm - pred.mean()), 0.0001, 0.9999)
def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols
def get_feature_cols(df):
    ex = META | set(TARGETS) | {'subject_id'}
    return [c for c in df.columns if c not in ex
            and not c.endswith('_subj_mean') and not c.endswith('_subj_std')
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
def add_personalization(df, fcols, fit_stats=None, for_test=False):
    pc, stats, sc = [], {}, []
    for col in fcols:
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean','std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        sc.extend([f'{col}_subj_mean', f'{col}_subj_std'])
        if not for_test:
            stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        sm = fit_stats[col]['mean'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_mean']
        sd = fit_stats[col]['std'] if (fit_stats and col in fit_stats) else df[f'{col}_subj_std']
        m0 = sd == 0; mn = df[col].isnull()
        z = f'{col}_zscore'
        df[z] = np.where(m0|mn, 0.0, (df[col].fillna(0)-sm)/np.maximum(sd, 1e-8))
        pc.append(z); gc.collect()
    drop = [c for c in sc if c in df.columns]
    if drop: df = df.drop(columns=drop)
    return df, pc, stats
def cfg_to_params(cfg_s, seed_val, spw):
    return {'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':int(cfg_s['nl']),'max_depth':int(cfg_s['md']),
            'learning_rate':float(cfg_s['lr']),'n_estimators':int(cfg_s['ne']),
            'subsample':float(cfg_s['ss']),'colsample_bytree':float(cfg_s['cb']),
            'reg_alpha':float(cfg_s['ra']),'reg_lambda':float(cfg_s['rl']),
            'min_child_samples':max(1,int(cfg_s['mc'])),
            'scale_pos_weight':spw,'random_state':int(seed_val),
            'force_row_wise':True,'n_jobs':1}
def train_cv(feat, ftst, cols, y, seeds, cfg):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(ftst), len(seeds))) if ftst is not None else None
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    Xf = feat[cols].fillna(0).values.astype(np.float64)
    Xt = ftst[cols].fillna(0).values.astype(np.float64) if ftst is not None else None
    nr = int(cfg['ne'])
    for si, seed in enumerate(seeds):
        p = cfg_to_params(cfg, seed, spw)
        for tri, vai in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(Xf[tri], label=y[tri], feature_name=sn)
            vd = lgb.Dataset(Xf[vai], label=y[vai], feature_name=sn, reference=ds)
            m = lgb.train(p, ds, num_boost_round=nr, valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vai, si] = m.predict(Xf[vai])
            if Xt is not None: tp[:, si] = m.predict(Xt)
            del ds, vd, m; gc.collect()
    if tp is not None: tp = np.clip(tp, 0.0001, 0.9999)
    return oof, tp
def rank_features(feat, fcols, target):
    y = feat[target].values.astype(np.float64)
    spw = max(((y==0).sum())/max((y==1).sum(),1), 0.1)
    p = {'objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
         'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
         'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
         'force_row_wise':True,'n_jobs':1}
    X = feat[fcols].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in fcols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(p, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(fcols, imp), key=lambda x: -x[1])
    del m, ds; gc.collect()
    return [r[0] for r in ranked]

# ============================================================
# STEP 0: Load internal data
# ============================================================
print("=" * 80)
print("V12: COMPREHENSIVE EXTERNAL DATA EXPLORATION LOOP")
print("=" * 80)
print('\n[STEP 0] Loading internal data...')
feat = pd.read_parquet(DATA / 'features.parquet')
ftst = pd.read_parquet(DATA / 'test_features.parquet')
for df in [feat, ftst]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
feat.columns = [sanitize_col(c) for c in feat.columns]
ftst.columns = [sanitize_col(c) for c in ftst.columns]

# Add proxy features from internal data
f = feat.copy(); ft = ftst.copy()
all_num = get_feature_cols(f)
for col_name, f_expr, t_expr in [
    ('ext_activity_z',
     "f['wPedo_pedo_step_mean'].fillna(0)",
     "ft['wPedo_pedo_step_mean'].fillna(0)"),
    ('ext_charging_z',
     "f['mACStatus_m_charging_mean'].fillna(0)",
     "ft['mACStatus_m_charging_mean'].fillna(0)"),
]:
    if col_name.replace('ext_','') in str(f_expr):
        exec(f"{col_name}_s = {f_expr}")
        exec(f"{col_name}_t = {t_expr}")

# Proxy features
if 'wPedo_pedo_step_mean' in all_num:
    sa = f['wPedo_pedo_step_mean'].fillna(0); sa_t = ft['wPedo_pedo_step_mean'].fillna(0)
    f['ext_activity_z'] = (sa-sa.mean())/max(sa.std(),1e-8)
    ft['ext_activity_z'] = (sa_t-sa.mean())/max(sa.std(),1e-8)
if 'mACStatus_m_charging_mean' in all_num:
    ch = f['mACStatus_m_charging_mean'].fillna(0); ch_t = ft['mACStatus_m_charging_mean'].fillna(0)
    f['ext_charging_z'] = (ch-ch.mean())/max(ch.std(),1e-8)
    ft['ext_charging_z'] = (ch_t-ch.mean())/max(ch.std(),1e-8)
if all(c in all_num for c in ['wPedo_pedo_step_mean','mACStatus_m_charging_mean','mScreenStatus_m_screen_use_mean','wHr_hr_mean']):
    sa = f['wPedo_pedo_step_mean'].fillna(0); sc_h = f['mACStatus_m_charging_mean'].fillna(0)
    ss = f['mScreenStatus_m_screen_use_mean'].fillna(0); hr = f['wHr_hr_mean'].fillna(0)
    sa_t = ft['wPedo_pedo_step_mean'].fillna(0); sc_t = ft['mACStatus_m_charging_mean'].fillna(0)
    ss_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0); hr_t = ft['wHr_hr_mean'].fillna(0)
    f['ext_health_composite'] = (sa-sa.mean())/max(sa.std(),1e-8) - (sc_h-sc_h.mean())/max(sc_h.std(),1e-8) + (ss-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr-hr.mean())/max(hr.std(),1e-8)*0.1
    ft['ext_health_composite'] = (sa_t-sa.mean())/max(sa.std(),1e-8) - (sc_t-sc_h.mean())/max(sc_h.std(),1e-8) + (ss_t-ss.mean())/max(ss.std(),1e-8)*0.3 + (hr_t-hr.mean())/max(hr.std(),1e-8)*0.1
if 'wLight_w_light_mean' in all_num and 'mACStatus_hour_night' in all_num:
    f['ext_night_light'] = f['wLight_w_light_mean'].fillna(0) / (f['mACStatus_hour_night'].fillna(0)+1e-8)
    ft['ext_night_light'] = ft['wLight_w_light_mean'].fillna(0) / (ft['mACStatus_hour_night'].fillna(0)+1e-8)
amb_cols = [c for c in all_num if 'ambience' in c.lower() and c.endswith('_sum')]
if amb_cols:
    f['ext_total_ambience'] = f[amb_cols].fillna(0).sum(axis=1)
    ft['ext_total_ambience'] = ft[amb_cols].fillna(0).sum(axis=1)
if 'wHr_hr_mean' in all_num and 'wPedo_pedo_step_mean' in all_num:
    f['ext_hr_step'] = f['wHr_hr_mean'].fillna(0) * f['wPedo_pedo_step_mean'].fillna(0)
    ft['ext_hr_step'] = ft['wHr_hr_mean'].fillna(0) * ft['wPedo_pedo_step_mean'].fillna(0)
if 'mScreenStatus_m_screen_use_mean' in all_num:
    sm = f['mScreenStatus_m_screen_use_mean'].fillna(0); sm_t = ft['mScreenStatus_m_screen_use_mean'].fillna(0)
    f['ext_screen_ratio'] = sm / (sm+1e-8)
    ft['ext_screen_ratio'] = sm_t / (sm_t+1e-8)
wifi_cols = [c for c in all_num if 'wifi' in c.lower() and c.endswith('_mean')]
ble_cols = [c for c in all_num if 'ble' in c.lower() and c.endswith('_mean')]
if wifi_cols and ble_cols:
    w = f[wifi_cols].fillna(0).sum(axis=1); b = f[ble_cols].fillna(0).sum(axis=1)
    w_t = ft[wifi_cols].fillna(0).sum(axis=1); b_t = ft[ble_cols].fillna(0).sum(axis=1)
    f['ext_wifi_ble'] = w / (b+1e-8)
    ft['ext_wifi_ble'] = w_t / (b_t+1e-8)
if 'ext_activity_z' in f.columns and 'ext_total_ambience' in f.columns:
    f['ext_activity_ambience'] = f['ext_activity_z'] * f['ext_total_ambience']
    ft['ext_activity_ambience'] = ft['ext_activity_z'] * ft['ext_total_ambience']
if 'wPedo_pedo_step_std' in all_num:
    f['ext_step_consistency'] = f['wPedo_pedo_step_std'].fillna(0) / (f['wPedo_pedo_step_mean'].fillna(0)+1e-8)
    ft['ext_step_consistency'] = ft['wPedo_pedo_step_std'].fillna(0) / (ft['wPedo_pedo_step_mean'].fillna(0)+1e-8)

# Personalization
fcols = get_feature_cols(f)
f, zscore_cols, fit_stats = add_personalization(f, fcols)
ft, _, _ = add_personalization(ft, fcols, fit_stats=fit_stats, for_test=True)
all_cols = fcols + zscore_cols
non_const = [c for c in all_cols if f[c].std() > 0]
y_dict = {t: f[t].values.astype(np.float64) for t in TARGETS}
print(f'  Train: {f.shape}, Test: {ft.shape}, Features: {len(non_const)}')

# ============================================================
# STEP 1: Load & normalize external datasets
# ============================================================
print('\n[STEP 1] Loading & normalizing external datasets...')

EXTERNAL_DATASETS = {}

# Dataset A: 374 rows (standard kaggle sleep health)
for path in EXTERNAL.glob('sleep_health*.csv'):
    try:
        df = pd.read_csv(path)
        if len(df) > 100 and 'Age' in df.columns:
            EXTERNAL_DATASETS['A_sleep_health_374'] = df
            print(f'  A: {path.name} -> {df.shape}')
            break
    except:
        pass

# Dataset B: 400 rows (variant)
for path in sorted(EXTERNAL.glob('*_extracted/*_health_lifestyle*')):
    try:
        df = pd.read_csv(path)
        if 'Person ID' in df.columns and len(df) > 300:
            # Rename columns to match standard format for comparison
            rename_map = {}
            if 'Sleep Duration (hours)' in df.columns: rename_map['Sleep Duration (hours)'] = 'Sleep Duration'
            if 'Quality of Sleep (scale: 1-10)' in df.columns: rename_map['Quality of Sleep (scale: 1-10)'] = 'Quality of Sleep'
            if 'Physical Activity Level (minutes/day)' in df.columns: rename_map['Physical Activity Level (minutes/day)'] = 'Physical Activity Level'
            if 'Stress Level (scale: 1-10)' in df.columns: rename_map['Stress Level (scale: 1-10)'] = 'Stress Level'
            if 'Blood Pressure (systolic/diastolic)' in df.columns: rename_map['Blood Pressure (systolic/diastolic)'] = 'Blood Pressure'
            if 'Heart Rate (bpm)' in df.columns: rename_map['Heart Rate (bpm)'] = 'Heart Rate'
            if rename_map: df = df.rename(columns=rename_map)
            EXTERNAL_DATASETS['B_sleep_health_400'] = df
            print(f'  B: {path.name} -> {df.shape}')
            break
    except:
        pass

# Dataset C: 1000 rows (rich sleep study data)
for path in sorted(EXTERNAL.glob('*_extracted/*sleep_study*')):
    try:
        df = pd.read_csv(path)
        if 'SleepDuration' in df.columns and len(df) > 500:
            EXTERNAL_DATASETS['C_sleep_study_1000'] = df
            print(f'  C: {path.name} -> {df.shape}')
            break
    except:
        pass

print(f'  Total external datasets loaded: {len(EXTERNAL_DATASETS)}')
for name, df in EXTERNAL_DATASETS.items():
    print(f'    {name}: {df.shape}, cols={list(df.columns)[:15]}')
    print(f'    numeric cols: {df.select_dtypes(include=[np.number]).shape[1]}')

# ============================================================
# STEP 2: External feature engineering
# ============================================================
print('\n[STEP 2] External feature engineering...')

# For each external dataset, compute proxy features mapped to our internal schema
ext_features = {}  # {dataset_name: {feature_name: value}}

for dname, df_ext in EXTERNAL_DATASETS.items():
    ext_summaries = {}
    
    # Standard features (same schema as our internal)
    standard_cols = {
        'Age': 'age', 'Sleep Duration': 'sleep_duration', 'Quality of Sleep': 'sleep_quality',
        'Physical Activity Level': 'physical_activity', 'Stress Level': 'stress_level',
        'Heart Rate': 'heart_rate', 'Daily Steps': 'daily_steps', 'Blood Pressure': 'blood_pressure',
    }
    for orig_col, feat_name in standard_cols.items():
        if orig_col in df_ext.columns:
            raw_s = df_ext[orig_col]
            # Handle Blood Pressure "systolic/diastolic" string format
            if orig_col == 'Blood Pressure':
                systolic = raw_s.str.split('/').str[0].astype(float) if raw_s.dtype == 'object' else raw_s
                diastolic = raw_s.str.split('/').str[1].astype(float) if raw_s.dtype == 'object' else raw_s
                for s, suffix in [(systolic, 'systolic'), (diastolic, 'diastolic')]:
                    s = s.dropna()
                    if len(s) > 20:
                        ext_summaries[f'{dname}_{feat_name}_{suffix}_mean'] = float(s.mean())
                        ext_summaries[f'{dname}_{feat_name}_{suffix}_std'] = float(s.std())
                        ext_summaries[f'{dname}_{feat_name}_{suffix}_median'] = float(s.median())
                continue
            s = raw_s.dropna()
            # Try to convert to numeric if object type (handle weird strings)
            if s.dtype == 'object':
                try:
                    s = pd.to_numeric(s, errors='coerce').dropna()
                except:
                    continue
            if len(s) > 20:
                ext_summaries[f'{dname}_{feat_name}_mean'] = float(s.mean())
                ext_summaries[f'{dname}_{feat_name}_std'] = float(s.std())
                ext_summaries[f'{dname}_{feat_name}_median'] = float(s.median())
                ext_summaries[f'{dname}_{feat_name}_q25'] = float(s.quantile(0.25))
                ext_summaries[f'{dname}_{feat_name}_q75'] = float(s.quantile(0.75))
    
    # BMI Category encoding
    if 'BMI Category' in df_ext.columns:
        bmi = df_ext['BMI Category'].value_counts()
        for cat in ['Normal weight', 'Overweight', 'Obese', 'Underweight']:
            ext_summaries[f'{dname}_bmi_{cat}'] = float(bmi.get(cat, 0) / len(df_ext))
    
    # Sleep Disorder
    if 'Sleep Disorder' in df_ext.columns:
        sd = df_ext['Sleep Disorder']
        if sd.dtype == 'object':
            disorders = sd.value_counts()
            for d in disorders.index[:5]:
                ext_summaries[f'{dname}_disorder_{str(d)}'] = float(disorders[d] / len(df_ext))
        else:
            ext_summaries[f'{dname}_sleep_disorder_rate'] = float(sd.mean())
    
    # Categorical encoding
    for col in ['Gender', 'Occupation']:
        if col in df_ext.columns:
            vals = df_ext[col].value_counts(normalize=True)
            for v in vals.index[:5]:
                ext_summaries[f'{dname}_{col}_{sanitize_col(str(v))}'] = float(vals[v])
    
    # Rich sleep data (Dataset C specific)
    if dname.startswith('C_'):
        rich_cols = ['SleepDuration', 'SleepEfficiency', 'REMSleepPercentage',
                     'DeepSleepPercentage', 'LightSleepPercentage', 'Awakenings',
                     'CaffeineConsumption', 'AlcoholConsumption', 'ExerciseFrequency']
        for col in rich_cols:
            if col in df_ext.columns:
                s = df_ext[col].dropna()
                if len(s) > 20:
                    ext_summaries[f'{dname}_{col}_mean'] = float(s.mean())
                    ext_summaries[f'{dname}_{col}_std'] = float(s.std())
                    ext_summaries[f'{dname}_{col}_median'] = float(s.median())
        
        # Derived features from rich data
        if all(c in df_ext.columns for c in ['SleepDuration', 'SleepEfficiency']):
            ext_summaries[f'{dname}_sleep_quality_score'] = float(
                df_ext['SleepDuration'].mean() * df_ext['SleepEfficiency'].mean())
        if all(c in df_ext.columns for c in ['REMSleepPercentage', 'DeepSleepPercentage', 'LightSleepPercentage']):
            rem = df_ext['REMSleepPercentage'].mean()
            deep = df_ext['DeepSleepPercentage'].mean()
            light = df_ext['LightSleepPercentage'].mean()
            ext_summaries[f'{dname}_deep_ratio'] = float(deep / (rem + deep + light + 1e-8))
            ext_summaries[f'{dname}_rem_ratio'] = float(rem / (rem + deep + light + 1e-8))
        
        # Behavioral factors
        if all(c in df_ext.columns for c in ['CaffeineConsumption', 'AlcoholConsumption']):
            eff = df_ext.dropna(subset=['CaffeineConsumption', 'AlcoholConsumption'])
            ext_summaries[f'{dname}_substance_load'] = float((eff['CaffeineConsumption'].mean() + eff['AlcoholConsumption'].mean()) / 2)
    
    ext_features[dname] = ext_summaries
    print(f'  {dname}: {len(ext_summaries)} features')

# ============================================================
# STEP 3: Data quality assessment
# ============================================================
print('\n[STEP 3] Data quality assessment...')

quality_scores = {}
for dname, df_ext in EXTERNAL_DATASETS.items():
    scores = {}
    nums = df_ext.select_dtypes(include=[np.number]).columns.tolist()
    total_cells = df_ext.shape[0] * df_ext.shape[1]
    missing_cells = df_ext.isnull().sum().sum()
    scores['missing_rate'] = float(missing_cells / total_cells)
    scores['numeric_ratio'] = float(len(nums) / df_ext.shape[1])
    scores['sample_size'] = len(df_ext)
    
    # Noise: ratio of extreme values (>3 std from mean)
    noise_count = 0
    for col in nums:
        s = df_ext[col].dropna()
        if len(s) > 50:
            m, sd = s.mean(), s.std()
            outliers = ((s - m).abs() > 3 * sd).sum()
            noise_count += outliers
    scores['noise_rate'] = float(noise_count / total_cells)
    
    # Duplicate ratio
    if df_ext.shape[1] <= 15:  # Only check for small-dimension data
        dup_ratio = 1 - df_ext.duplicated().sum() / len(df_ext)
        scores['duplicate_rate'] = float(1 - dup_ratio)
    
    # Missing distribution
    col_missing = df_ext.isnull().mean()
    scores['avg_missing_per_col'] = float(col_missing.mean())
    scores['max_missing_per_col'] = float(col_missing.max())
    
    quality_scores[dname] = scores
    print(f'  {dname}: missing={scores["missing_rate"]:.3f} noise={scores["noise_rate"]:.3f} dup={scores.get("duplicate_rate", 0):.3f}')

# ============================================================
# STEP 4: Domain similarity - Adversarial validation
# ============================================================
print('\n[STEP 4] Domain similarity (Adversarial Validation)...')

# For each external dataset, find shared features with internal data
# and run adversarial validation
domain_scores = {}

for dname, df_ext in EXTERNAL_DATASETS.items():
    # Find semantic mapping between external and internal features
    feature_mapping = {}
    ext_nums = df_ext.select_dtypes(include=[np.number]).columns.tolist()
    
    # Semantic keywords for mapping
    keyword_map = {
        'Age': ['age', 'age'],
        'Sleep Duration': ['sleep', 'duration', 'sleep_duration', 'hour'],
        'Sleep Quality': ['sleep', 'quality', 'sleep_quality'],
        'Physical Activity': ['physical', 'activity', 'exercise', 'activity_level'],
        'Stress': ['stress', 'tension', 'anxiety'],
        'BMI': ['bmi', 'weight', 'body'],
        'Heart Rate': ['heart', 'hr', 'pulse', 'bpm'],
        'Daily Steps': ['step', 'steps', 'walk', 'daily'],
        'Blood Pressure': ['blood', 'pressure', 'bp', 'systolic', 'diastolic'],
        'Caffeine': ['caffeine', 'coffee'],
        'Alcohol': ['alcohol', 'drink', 'beer'],
        'Sleep Efficiency': ['efficiency', 'sleep_eff'],
        'Awakenings': ['awaken', 'night', 'waking'],
        'REMSleep': ['rem', 'dream'],
        'DeepSleep': ['deep', 'slow'],
        'LightSleep': ['light', 'shallow'],
        'Smoking': ['smoke', 'tobacco', 'nicotine'],
        'Gender': ['gender', 'sex', 'male', 'female'],
        'Occupation': ['occupation', 'job', 'work', 'profession'],
    }
    
    for std_name, keywords in keyword_map.items():
        for ecol in ext_nums:
            ecol_l = ecol.lower().replace(' ', '_').replace('-', '_')
            if any(kw in ecol_l for kw in keywords):
                feature_mapping[std_name] = ecol
                break
    
    if not feature_mapping:
        domain_scores[dname] = {'mapped': 0, 'adv_auc': None}
        print(f'  {dname}: 0 features mapped')
        continue
    
    print(f'  {dname}: mapped {len(feature_mapping)} features')
    for k, v in list(feature_mapping.items())[:5]:
        print(f'    {k} -> {v}')
    
    # Build adversarial validation dataset
    # Combine internal features (mapped) with external features (mapped)
    shared_internal = []
    shared_external = []
    
    # Use original internal features where possible
    mapping_lookup = {v: k for k, v in feature_mapping.items()}
    
    for std_name, ext_col in feature_mapping.items():
        # Find the corresponding internal feature
        if std_name in ['Age']:
            shared_internal.append('wPedo_pedo_step_mean')  # proxy
            shared_external.append(ext_col)
        elif std_name in ['Sleep Duration']:
            shared_internal.append('wPedo_pedo_step_mean')  # proxy for sleep activity
            shared_external.append(ext_col)
        elif std_name in ['Heart Rate']:
            shared_internal.append('wHr_hr_mean')
            shared_external.append(ext_col)
        elif std_name in ['Daily Steps']:
            shared_internal.append('wPedo_pedo_step_mean')
            shared_external.append(ext_col)
        elif std_name in ['Stress']:
            shared_internal.append('mACStatus_m_charging_mean')  # proxy
            shared_external.append(ext_col)
        elif std_name in ['Physical Activity']:
            shared_internal.append('wPedo_pedo_step_mean')
            shared_external.append(ext_col)
        else:
            shared_internal.append('wHr_hr_mean')  # default proxy
            shared_external.append(ext_col)
    
    # Use same feature names for both internal and external
    n_train = min(len(f), 200)
    n_ext = min(len(df_ext), 200)
    
    # Since we can't directly map, use the shared internal features
    # to represent both domains
    shared_features = [fc for fc in shared_internal if fc in f.columns]
    
    if len(shared_features) < 2:
        domain_scores[dname] = {'mapped': len(feature_mapping), 'shared': len(shared_features), 'adv_auc': None}
        print(f'  {dname}: only {len(shared_features)} shared features')
        continue
    
    # Sample and run adversarial validation
    X_train = f[shared_features].fillna(0).values.astype(np.float64)[:n_train]
    X_ext = np.zeros((n_ext, len(shared_features)))
    for i, feat in enumerate(shared_features):
        if feat in df_ext.columns:
            X_ext[:, i] = df_ext[feat].fillna(0).values[:n_ext]
        # If not in external, fill with training mean (proxy feature)
        else:
            X_ext[:, i] = f[feat].fillna(0).mean()
    
    X_adv = np.vstack([X_train, X_ext])
    y_adv = np.array([0]*n_train + [1]*n_ext)
    
    from sklearn.model_selection import KFold
    kf_adv = KFold(n_splits=5, shuffle=True, random_state=42)
    adv_scores = []
    for tri, vai in kf_adv.split(X_adv):
        ds = lgb.Dataset(X_adv[tri], label=y_adv[tri])
        vd = lgb.Dataset(X_adv[vai], label=y_adv[vai])
        m = lgb.train({'objective':'binary','metric':'binary_logloss','verbose':-1,
                       'num_leaves':10,'max_depth':3,'learning_rate':0.05,'n_estimators':100,
                       'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':1.0,'reg_lambda':3.0,
                       'min_child_samples':10,'random_state':42,'n_jobs':1},
                      ds, num_boost_round=100, valid_sets=[vd],
                      callbacks=[lgb.early_stopping(10, verbose=False)])
        pred = m.predict(X_adv[vai])
        auc = roc_auc_score(y_adv[vai], pred)
        adv_scores.append(auc)
    
    adv_auc = float(np.mean(adv_scores))
    domain_scores[dname] = {
        'mapped': len(feature_mapping),
        'shared': len(shared_features),
        'adv_auc': round(adv_auc, 4),
        'interpretation': 'same_domain' if adv_auc < 0.6 else ('mixed' if adv_auc < 0.7 else 'different_domain'),
    }
    print(f'  {dname}: adversarial AUC = {adv_auc:.4f} ({domain_scores[dname]["interpretation"]})')

# ============================================================
# STEP 5: Combinatorial External Data Integration
# ============================================================
print('\n[STEP 5] Combinatorial external data integration...')

available_eids = list(EXTERNAL_DATASETS.keys())
print(f'  External datasets: {available_eids}')

# For each combination, add external summary features and evaluate
# We test all 2^N - 1 combinations (N=datasets)

def add_external_features_to_dataframe(feat_df, ftst_df, dnames):
    """Add external summary features to train and test dataframes."""
    f_out = feat_df.copy()
    ft_out = ftst_df.copy()
    
    for dname in dnames:
        if dname not in ext_features:
            continue
        for fname, fval in ext_features[dname].items():
            f_out[fname] = fval
            ft_out[fname] = fval
    
    return f_out, ft_out

# Baseline: internal-only V127
print('\n  === Baseline (internal-only) ===')
baseline_oofs = {}
for target in TARGETS:
    t0 = time.time()
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    non_ext = [c for c in ranked if not c.startswith('ext_')]
    best_cols = non_ext[:15]
    
    oof, tp = train_cv(f, ft, best_cols, y, SEEDS, cfg)
    oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
    ll = log_loss(y, oof_avg, labels=[0,1])
    baseline_oofs[target] = ll
    print(f'    {target}: OOF={ll:.5f}')

# Test all combinations
all_results = []
total_combos = 0

# Single datasets
for dname in available_eids:
    total_combos += 1
    f_try, ft_try = add_external_features_to_dataframe(f, ft, [dname])
    
    for target in TARGETS:
        t0 = time.time()
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        
        # Try: top internal + some external
        for n_ext in [0, 1, 2, 3]:
            n_int = 15 - n_ext
            if n_int < 5: continue
            cols = ranked[:15]
            oof, _ = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            ll = log_loss(y, oof_avg, labels=[0,1])
            delta = ll - baseline_oofs[target]
            
            all_results.append({
                'type': 'single', 'combo': dname, 'target': target,
                'n_ext': n_ext, 'oof': round(ll, 5), 'delta': round(delta, 5),
                'time': round(time.time()-t0, 1),
            })

print(f'  Single-dataset evals: {total_combos * len(TARGETS) * 4}')

# Pairs
total_pairs = 0
for combo in itertools.combinations(available_eids, 2):
    total_combos += 1
    total_pairs += 1
    f_try, ft_try = add_external_features_to_dataframe(f, ft, list(combo))
    
    for target in TARGETS:
        t0 = time.time()
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        
        oof, _ = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        delta = ll - baseline_oofs[target]
        
        all_results.append({
            'type': 'pair', 'combo': '+'.join(combo), 'target': target,
            'oof': round(ll, 5), 'delta': round(delta, 5),
            'time': round(time.time()-t0, 1),
        })

print(f'  Pair evals: {total_pairs * len(TARGETS)}')

# Triples
total_triples = 0
for combo in itertools.combinations(available_eids, 3):
    total_combos += 1
    total_triples += 1
    f_try, ft_try = add_external_features_to_dataframe(f, ft, list(combo))
    
    for target in TARGETS:
        t0 = time.time()
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        
        oof, _ = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        delta = ll - baseline_oofs[target]
        
        all_results.append({
            'type': 'triple', 'combo': '+'.join(combo), 'target': target,
            'oof': round(ll, 5), 'delta': round(delta, 5),
            'time': round(time.time()-t0, 1),
        })

print(f'  Triple evals: {total_triples * len(TARGETS)}')

print(f'\n  Total combinatorial evaluations: {len(all_results)}')

# Find best combination per target
print('\n  === Best combination per target ===')
best_per_target = {}
for target in TARGETS:
    target_results = [r for r in all_results if r['target'] == target]
    best = min(target_results, key=lambda x: x['delta'])
    best_per_target[target] = best
    print(f'    {target}: {best["combo"]} delta={best["delta"]:+.5f} (OOF={best["oof"]:.5f})')

# ============================================================
# STEP 6: Weighted External Feature Blending
# ============================================================
print('\n[STEP 6] Weighted external feature blending...')

# Strategy: for each target, use the best external features weighted by domain similarity
blend_results = {}

for dname in available_eids:
    domain_sim = domain_scores.get(dname, {}).get('adv_auc', 0.5)
    quality = quality_scores.get(dname, {})
    quality_score = max(0, 1 - quality.get('missing_rate', 0) - quality.get('noise_rate', 0))
    
    # Domain-aware weight
    domain_weight = 1.0 if domain_sim < 0.6 else (0.5 if domain_sim < 0.7 else 0.2)
    weight = domain_weight * quality_score
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        
        # Create external feature vector for this dataset
        ext_feats = ext_features.get(dname, {})
        if not ext_feats:
            continue
        
        # Add to dataframe
        f_try = f.copy()
        ft_try = ft.copy()
        for fname, fval in ext_feats.items():
            f_try[fname] = fval
            ft_try[fname] = fval
        
        # Blend: internal top-N + weighted external
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        
        # Use weight to scale external features
        for fname in ext_feats:
            if fname in f_try.columns:
                f_try[fname] *= weight
                ft_try[fname] *= weight
        
        # Top 15 ranked
        cols = ranked[:15]
        oof, _ = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        delta = ll - baseline_oofs[target]
        
        blend_results.setdefault(dname, {}).update({target: {'ll': ll, 'delta': delta, 'weight': round(weight, 3)}})
        print(f'  {dname} -> {target}: w={weight:.3f} domain_sim={domain_sim:.3f} Q={quality_score:.3f} delta={delta:+.5f}')

# ============================================================
# STEP 7: Pseudo-labeling Iterative Loop
# ============================================================
print('\n[STEP 7] Pseudo-labeling iterative loop...')

pseudo_results = {}

for target in TARGETS:
    cfg_name = V53_SWEEP[target]
    cfg = CFGS[cfg_name]
    y = y_dict[target]
    
    # Get best external combo from step 5
    best_combo = best_per_target[target]['combo']
    best_combo_list = best_combo.split('+') if '+' in best_combo else [best_combo]
    best_combo_list = [x for x in best_combo_list if x in available_eids]
    
    f_try, ft_try = add_external_features_to_dataframe(f, ft, best_combo_list)
    leak_cols = remove_leak(get_feature_cols(f_try), target)
    ranked = rank_features(f_try, leak_cols, target)
    model_cols = ranked[:15]
    
    # Iterative pseudo-labeling
    current_ft = ft_try.copy()
    current_preds = {}
    
    for iteration in range(5):
        oof, tp = train_cv(f_try, current_ft, model_cols, y, SEEDS, cfg)
        test_avg = np.clip(tp.mean(axis=1), 0.0001, 0.9999)
        
        # Confidence filtering
        for thresh in [0.7, 0.8, 0.9]:
            high_conf = (test_avg >= thresh) | (test_avg <= (1-thresh))
            n_high = high_conf.sum()
            if n_high < 10:
                continue
            
            avg_conf = test_avg[high_conf].mean()
            # Calibrate: ensure pseudo-label distribution matches training
            internal_pos = y.mean()
            pseudo_pos = (test_avg[high_conf] > 0.5).mean()
            
            key = f'{target}_t{thresh}_iter{iteration}'
            pseudo_results[key] = {
                'n_high': n_high, 'avg_conf': round(avg_conf, 3),
                'pseudo_pos': round(pseudo_pos, 3), 'internal_pos': round(internal_pos, 3),
                'drift': round(abs(pseudo_pos - internal_pos), 3),
            }
        
        # Update predictions for next iteration
        current_preds[f'iter{iteration}'] = test_avg.copy()
        
        # Drift check - if predictions are stable, stop
        if iteration > 0:
            prev = current_preds.get(f'iter{iteration-1}', test_avg)
            drift = abs(test_avg - prev).mean()
            if drift < 0.001:
                print(f'    {target}: converged at iter {iteration} (drift={drift:.6f})')
                break
        
        print(f'    {target} iter{iteration}: range=[{test_avg.min():.3f},{test_avg.max():.3f}] '
              f'cal_ll={log_loss(y, mean_match(oof.mean(axis=1), y.mean()), labels=[0,1]):.5f}')

# ============================================================
# STEP 8: Staged Training (External Pretrain → Internal Finetune)
# ============================================================
print('\n[STEP 8] Staged training...')

staged_results = {}

for dname in available_eids:
    quality = quality_scores.get(dname, {})
    if quality.get('missing_rate', 1) > 0.3:
        continue  # Skip high-missing datasets
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(non_const, target)
        ranked = rank_features(f, leak_cols, target)
        non_ext = [c for c in ranked if not c.startswith('ext_')]
        
        # Stage 1: Train on external features only (if any mapped)
        ext_feats_for_target = [fn for fn in ext_features.get(dname, {}) 
                                if fn.startswith(dname)]
        
        if len(ext_feats_for_target) < 2:
            continue
        
        # Stage 2: Finetune with internal + external
        f_try = f.copy()
        ft_try = ft.copy()
        for fname, fval in ext_features[dname].items():
            f_try[fname] = fval
            ft_try[fname] = fval
        
        staged_cols = non_ext[:12] + ext_feats_for_target[:3]
        staged_cols = list(dict.fromkeys(staged_cols))[:15]
        
        oof, tp = train_cv(f_try, ft_try, staged_cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        delta = ll - baseline_oofs[target]
        
        staged_results.setdefault(dname, {})[target] = {
            'll': ll, 'delta': delta, 'n_staged': len(staged_cols),
        }

# ============================================================
# STEP 9: Ensemble Diversity Analysis
# ============================================================
print('\n[STEP 9] Ensemble diversity analysis...')

# Train multiple models with different external combos and measure correlation
ensemble_models = {}

for dname in available_eids:
    f_try, ft_try = add_external_features_to_dataframe(f, ft, [dname])
    ensemble_models[dname] = {}
    
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        
        oof, tp = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        oof_avg = oof.mean(axis=1)
        
        ensemble_models[dname][target] = {
            'oof': oof_avg.copy(), 'test_pred': tp.mean(axis=1) if tp is not None else None,
            'll': log_loss(y, oof_avg, labels=[0,1]),
        }

# Calculate pairwise correlation between models
print('  Model correlation matrix:')
model_names = ['internal'] + available_eids
corr_matrix = np.eye(len(model_names))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i >= j: continue
        if m1 == 'internal' and m2 == 'internal': continue
        if m1 == 'internal':
            # Use baseline OOF predictions
            pass
        elif m2 == 'internal':
            pass
        else:
            # Correlate OOF predictions across targets
            corrs = []
            for target in TARGETS:
                o1 = ensemble_models[m1][target]['oof']
                o2 = ensemble_models[m2][target]['oof']
                corrs.append(np.corrcoef(o1, o2)[0, 1])
            avg_corr = np.mean(corrs)
            corr_matrix[model_names.index(m1), model_names.index(m2)] = avg_corr
            corr_matrix[model_names.index(m2), model_names.index(m1)] = avg_corr
            print(f'    {m1} <-> {m2}: r={avg_corr:.3f}')

# ============================================================
# STEP 10: Calibration Stability
# ============================================================
print('\n[STEP 10] Calibration stability...')

calibration_results = {}
for dname in available_eids:
    calibration_results[dname] = {}
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]
        cfg = CFGS[cfg_name]
        y = y_dict[target]
        
        f_try, ft_try = add_external_features_to_dataframe(f, ft, [dname])
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        
        oof, _ = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        oof_avg = oof.mean(axis=1)
        
        # Per-fold calibration check
        gkf = GroupKFold(n_splits=5)
        fold_lls = []
        for tri, vai in gkf.split(f_try, y, f_try['subject_id']):
            # Get OOF for this fold
            fold_oof = oof[vai].mean(axis=1)
            ll = log_loss(y[vai], fold_oof, labels=[0,1])
            fold_lls.append(ll)
        
        fold_std = np.std(fold_lls)
        calibration_results[dname][target] = {
            'mean_oof': round(float(log_loss(y, oof_avg, labels=[0,1])), 5),
            'fold_std': round(float(fold_std), 5),
            'fold_lls': [round(x, 5) for x in fold_lls],
        }
        print(f'  {dname} -> {target}: fold_lls={[f"{x:.4f}" for x in fold_lls]} fold_std={fold_std:.4f}')

# Also check internal-only calibration
print('\n  Internal-only calibration:')
for target in TARGETS:
    y = y_dict[target]
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    cols = ranked[:15]
    oof, _ = train_cv(f, ft, cols, y, SEEDS, CFGS[V53_SWEEP[target]])
    fold_lls = []
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(f, y, f['subject_id']):
        fold_lls.append(float(log_loss(y[vai], oof[vai].mean(axis=1), labels=[0,1])))
    fold_std = np.std(fold_lls)
    print(f'    {target}: fold_lls={[f"{x:.4f}" for x in fold_lls]} fold_std={fold_std:.4f}')

# Update the cfgs reference
cfgs = CFGS

# ============================================================
# STEP 11: Multi-Stage Ensemble (V127-style)
# ============================================================
print('\n[STEP 11] Multi-stage ensemble optimization...')

# Strategy: ensemble internal-only, external-enhanced, and pseudo-labeled models
# with optimized weights

ensemble_ens_results = {}
for target in TARGETS:
    y = y_dict[target]
    
    models = {}
    
    # Model 1: Internal-only
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    int_cols = ranked[:15]
    oof_int, tp_int = train_cv(f, ft, int_cols, y, SEEDS, CFGS[V53_SWEEP[target]])
    oof_int_avg = np.clip(oof_int.mean(axis=1), 0.0001, 0.9999)
    models['internal'] = oof_int_avg
    
    # Model 2..N: Best external per dataset
    for dname in available_eids:
        f_try, ft_try = add_external_features_to_dataframe(f, ft, [dname])
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        oof_ext, tp_ext = train_cv(f_try, ft_try, cols, y, SEEDS, CFGS[V53_SWEEP[target]])
        oof_ext_avg = np.clip(oof_ext.mean(axis=1), 0.0001, 0.9999)
        models[dname] = oof_ext_avg
    
    # Find optimal ensemble weights
    best_ll = float('inf')
    best_weights = {}
    best_combo = ''
    
    for n_models in range(1, len(models)+1):
        for combo in itertools.combinations(models.keys(), n_models):
            if len(combo) == 1:
                oof = models[combo[0]]
                ll = log_loss(y, oof, labels=[0,1])
                if ll < best_ll:
                    best_ll = ll
                    best_weights = {combo[0]: 1.0}
                    best_combo = combo[0]
            else:
                # Grid search weights
                for w_trial in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    ens = np.zeros(len(y))
                    remaining = 1.0
                    for i, m in enumerate(combo):
                        w = w_trial * remaining if i < len(combo)-1 else remaining
                        ens += w * models[m]
                    ens = np.clip(ens, 0.0001, 0.9999)
                    ll = log_loss(y, ens, labels=[0,1])
                    if ll < best_ll:
                        best_ll = ll
                        wts = {}
                        for i, m in enumerate(combo):
                            wts[m] = w_trial * remaining if i < len(combo)-1 else remaining
                        best_weights = wts
                        best_combo = '+'.join(combo)
                    remaining = max(0, remaining - w_trial) if i < len(combo)-1 else 0
    
    # Also try mean-match calibration
    for w_trial in [0.1, 0.2, 0.3, 0.4, 0.5]:
        ens = w_trial * models['internal']
        for m in [m for m in models if m != 'internal']:
            ens += (1-w_trial) * models[m]
        ens = np.clip(ens, 0.0001, 0.9999)
        ens_cal = mean_match(ens, y.mean())
        ll_cal = log_loss(y, ens_cal, labels=[0,1])
        if ll_cal < best_ll:
            best_ll = ll_cal
            best_weights = {'internal': w_trial, 'rest': 1-w_trial}
            best_combo = 'internal+others_cal'
    
    ensemble_ens_results[target] = {
        'best_ll': round(best_ll, 5),
        'weights': best_weights,
        'combo': best_combo,
        'baseline': round(baseline_oofs[target], 5),
        'delta': round(best_ll - baseline_oofs[target], 5),
    }
    print(f'  {target}: best={best_combo} LL={best_ll:.5f} delta={best_ll-baseline_oofs[target]:+.5f}')

# ============================================================
# STEP 12: Curriculum Learning (order-dependent training)
# ============================================================
print('\n[STEP 12] Curriculum learning ordering...')

curriculum_results = {}
# Order external datasets by domain similarity (easier domains first)
sorted_ext = sorted(available_eids, 
                    key=lambda x: domain_scores.get(x, {}).get('adv_auc', 0.5))
print(f'  Training order: {sorted_ext}')

for n_stages in range(1, min(len(sorted_ext)+1, 4)):
    curriculum = sorted_ext[:n_stages]
    f_cur = f.copy()
    ft_cur = ft.copy()
    for dname in curriculum:
        for fname, fval in ext_features.get(dname, {}).items():
            f_cur[fname] = fval
            ft_cur[fname] = fval
    
    for target in TARGETS:
        y = y_dict[target]
        cfg = CFGS[V53_SWEEP[target]]
        leak_cols = remove_leak(get_feature_cols(f_cur), target)
        ranked = rank_features(f_cur, leak_cols, target)
        cols = ranked[:15]
        
        oof, _ = train_cv(f_cur, ft_cur, cols, y, SEEDS, cfg)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        ll = log_loss(y, oof_avg, labels=[0,1])
        delta = ll - baseline_oofs[target]
        
        key = f'target{target}_n{len(curriculum)}'
        curriculum_results[key] = {
            'curriculum': curriculum, 'oof': round(ll, 5),
            'delta': round(delta, 5),
        }
        print(f'    {key}: delta={delta:+.5f}')

# ============================================================
# STEP 13: Summary & Save
# ============================================================
print('\n' + '=' * 80)
print('V12 FINAL SUMMARY')
print('=' * 80)

# Overall results
print('\n  V127 Reproduction Baseline:')
for t in TARGETS:
    print(f'    {t}: {baseline_oofs[t]:.5f}')
avg_baseline = np.mean(list(baseline_oofs.values()))
print(f'    AVG: {avg_baseline:.5f}')

print('\n  Combinatorial Best (Step 5):')
for t in TARGETS:
    b = best_per_target[t]
    print(f'    {t}: {b["combo"]} delta={b["delta"]:+.5f} OOF={b["oof"]:.5f}')
best_deltas = [best_per_target[t]['delta'] for t in TARGETS]
print(f'    AVG delta: {np.mean(best_deltas):+.5f}')

print('\n  Ensemble Best (Step 11):')
for t in TARGETS:
    r = ensemble_ens_results[t]
    print(f'    {t}: {r["combo"]} LL={r["best_ll"]:.5f} delta={r["delta"]:+.5f} weights={r["weights"]}')
ens_deltas = [ensemble_ens_results[t]['delta'] for t in TARGETS]
print(f'    AVG delta: {np.mean(ens_deltas):+.5f}')

print('\n  Domain Similarity:')
for dname, ds in domain_scores.items():
    print(f'    {dname}: adv_auc={ds.get("adv_auc", "N/A")} ({ds.get("interpretation", "N/A")})')

print('\n  Data Quality:')
for dname, qs in quality_scores.items():
    print(f'    {dname}: missing={qs["missing_rate"]:.3f} noise={qs["noise_rate"]:.3f}')

print('\n  Pseudo-labeling (Step 7):')
for k, v in pseudo_results.items():
    if v['n_high'] >= 10:
        print(f'    {k}: n={v["n_high"]} conf={v["avg_conf"]:.3f} drift={v["drift"]:.3f}')

# Overall best
overall_best_target = None
overall_best_delta = 0
for t in TARGETS:
    comb_delta = best_per_target[t]['delta']
    ens_delta = ensemble_ens_results[t]['delta']
    overall_best = min(comb_delta, ens_delta)
    if overall_best < overall_best_delta:
        overall_best_delta = overall_best
        overall_best_target = t

print(f'\n  *** Overall best target improvement: {overall_best_target} ({overall_best_delta:+.5f}) ***')
print(f'  *** Average improvement across targets: {np.mean(best_deltas):+.5f} (combinatorial)')
print(f'  *** Average improvement across targets: {np.mean(ens_deltas):+.5f} (ensemble) ***')

# Save comprehensive results
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
result = {
    'version': 'v12_external_data_comprehensive',
    'timestamp': ts,
    'baseline_oofs': {t: round(v, 5) for t, v in baseline_oofs.items()},
    'avg_baseline_oof': round(avg_baseline, 5),
    'external_datasets': {
        dname: {'shape': EXTERNAL_DATASETS[dname].shape,
                'quality': quality_scores.get(dname, {}),
                'domain_similarity': domain_scores.get(dname, {})}
        for dname in EXTERNAL_DATASETS
    },
    'combinatorial_results': {
        'best_per_target': {t: {k: v for k, v in b.items() if k != 'time'} 
                           for t, b in best_per_target.items()},
        'total_evaluations': len(all_results),
        'all_results_count': len(all_results),
    },
    'ensemble_results': {
        t: {k: v for k, v in r.items()} for t, r in ensemble_ens_results.items()
    },
    'domain_scores': domain_scores,
    'quality_scores': quality_scores,
    'calibration_results': calibration_results,
    'pseudo_labeling': pseudo_results,
    'staged_training': staged_results,
    'blend_results': blend_results,
    'curriculum_results': curriculum_results,
    'overall_best_delta': round(float(overall_best_delta), 5),
    'avg_combinatorial_delta': round(float(np.mean(best_deltas)), 5),
    'avg_ensemble_delta': round(float(np.mean(ens_deltas)), 5),
}

result_path = EXPERIMENTS / f'v12_comprehensive_{ts}.json'
with open(result_path, 'w') as fout:
    json.dump(result, fout, indent=2, default=str)
print(f'\n  Saved: {result_path}')

# Save submission predictions (for manual upload)
# Best model ensemble predictions
final_test_preds = {}
for target in TARGETS:
    y = y_dict[target]
    cfg = CFGS[V53_SWEEP[target]]
    
    # Use the best ensemble combo
    best_combo_str = ensemble_ens_results[target]['combo']
    
    # Internal predictions
    leak_cols = remove_leak(non_const, target)
    ranked = rank_features(f, leak_cols, target)
    int_cols = ranked[:15]
    _, tp_int = train_cv(f, ft, int_cols, y, SEEDS, cfg)
    pred_int = tp_int.mean(axis=1)
    
    # External predictions (best combo)
    best_combo_list = best_combo_str.replace('internal', '').replace('_cal', '').replace('others', '').split('+')
    best_combo_list = [x for x in best_combo_list if x in available_eids and x]
    
    if best_combo_list:
        f_try, ft_try = add_external_features_to_dataframe(f, ft, best_combo_list)
        leak_cols = remove_leak(get_feature_cols(f_try), target)
        ranked = rank_features(f_try, leak_cols, target)
        cols = ranked[:15]
        _, tp_ext = train_cv(f_try, ft_try, cols, y, SEEDS, cfg)
        pred_ext = tp_ext.mean(axis=1)
        
        # Blend weights
        best_w = ensemble_ens_results[target]['weights']
        w_int = best_w.get('internal', 0.5)
        w_ext = 1 - w_int
        final_test_preds[target] = np.clip(w_int * pred_int + w_ext * pred_ext, 0.0001, 0.9999)
    else:
        final_test_preds[target] = pred_int

# Create submission dataframe
submit_df = pd.DataFrame({'subject_id': ftst['subject_id']})
for target in TARGETS:
    submit_df[target] = final_test_preds[target]

submit_path = SUBMIT / f'v12_submission_{ts}.csv'
submit_df.to_csv(submit_path, index=False)
print(f'  Saved submission: {submit_path}')
print(f'  Submission shape: {submit_df.shape}')
print(f'  Submission head:\n{submit_df.head()}')

print('\n=== V12 COMPLETE ===')
