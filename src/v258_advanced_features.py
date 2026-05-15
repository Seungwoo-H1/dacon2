"""V258: Advanced Feature Discovery (lightweight, targeted)

Hypothesis: Frequency-domain, temporal regularity, and clustering features
can capture signal missing from time-aggregated statistics.

Runs a smaller experiment: 5-fold × 3 seeds, selective feature addition.
"""
import logging, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999]
N_FOLDS = 5

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def get_feat_cols(df):
    return [c for c in df.columns if c not in META | set(TARGETS) 
            and c not in ['subject_id','lifelog_date','sleep_date','date']
            and df[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

def add_zscore(df, feat_cols, stats=None, for_test=False):
    df = df.copy()
    all_stats = {}
    zcols = []
    for c in feat_cols:
        vals = df[c].fillna(0)
        grp = vals.groupby(df['subject_id']).agg(mean='mean', std='std').reset_index()
        grp.columns = ['subject_id', f'{c}_subj_mean', f'{c}_subj_std']
        df = df.merge(grp, on='subject_id', how='left')
        sm = df[f'{c}_subj_mean']; ss = df[f'{c}_subj_std']
        if not for_test: all_stats[c] = {'mean': sm, 'std': ss}
        mask = (ss == 0) | df[c].isnull()
        df[f'{c}_z'] = np.where(mask, 0.0, (df[c].fillna(0) - sm) / np.maximum(ss, 1e-8))
        zcols.append(f'{c}_z')
        gc.collect()
    return df, zcols, all_stats

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def rank_features(df, feat_cols, target, seed=42):
    y = df[target].values.astype(np.float64)
    X = df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x:-x[1])
    del model, ds; gc.collect()
    return [r[0] for r in ranked]

def train_cv(df, df_test, sel_cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    tp = np.zeros((len(df_test), len(seeds)))
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    X_full = df[sel_cols].fillna(0).values.astype(np.float64)
    X_test = df_test[sel_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in sel_cols]
    for si, seed in enumerate(seeds):
        cfg_full = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
            'min_child_samples':cfg['mc'],'random_state':seed,'scale_pos_weight':spw,
        }
        for tr_i, va_i in gkf.split(df, y, df['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(cfg_full, ds, num_boost_round=cfg['ne'], valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            tp[:, si] = m.predict(X_test)
            del ds, vd, m; gc.collect()
    return np.clip(oof, 0.0001, 0.9999), np.clip(tp, 0.0001, 0.9999)

CFGS = {
    'wide':   {'nl':30,'md':3,'lr':0.05,'ne':300,'ss':0.8,'cb':0.8,'ra':2.0,'rl':5.0,'mc':5},
    'deep':   {'nl':20,'md':5,'lr':0.02,'ne':1000,'ss':0.7,'cb':0.6,'ra':0.5,'rl':2.0,'mc':15},
    'v48':    {'nl':15,'md':4,'lr':0.03,'ne':500,'ss':0.7,'cb':0.7,'ra':1.0,'rl':3.0,'mc':10},
    'safety': {'nl':10,'md':3,'lr':0.02,'ne':1000,'ss':0.6,'cb':0.6,'ra':3.0,'rl':10.0,'mc':20},
}

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

# ============================================================
# Load data
# ============================================================
log.info("Loading data...")
feat = pd.read_parquet(DATA / "features_clean_v60.parquet")
feat_test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
for df in [feat, feat_test]:
    for c in ['sleep_date','lifelog_date','date']:
        if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

y_train = {t: feat[t].values for t in TARGETS}
train_rates = {t: feat[t].mean() for t in TARGETS}

# ============================================================
# Helper: build baseline (base + zscore)
# ============================================================
def build_baseline(df, stats=None, for_test=False):
    fc = get_feat_cols(df)
    df, zcols, all_stats = add_zscore(df, fc, stats, for_test)
    return df, fc + zcols, all_stats if not for_test else None

feat_base, base_cols, stats_base = build_baseline(feat)
feat_test_base, _, _ = build_baseline(feat_test, stats_base, for_test=True)

# ============================================================
# EXPERIMENT A: Baseline V127 (reference)
# ============================================================
log.info("\n" + "="*70)
log.info("EXPERIMENT A: Baseline V127")
log.info("="*70)

baseline_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_base, base_cols, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_base, feat_test_base, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    tp_c = mean_match(tp, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    baseline_results[target] = {'oof_ll': ll, 'oof_cal': oof_c, 'test': tp_c}
    log.info(f"  {target}: OOF LL={ll:.5f}")

baseline_avg = np.mean([baseline_results[t]['oof_ll'] for t in TARGETS])
log.info(f"  Baseline AVG OOF: {baseline_avg:.5f}")

# ============================================================
# EXPERIMENT B: Temporal Regularity Features
# ============================================================
log.info("\n" + "="*70)
log.info("EXPERIMENT B: Temporal Regularity Features")
log.info("="*70)

feat_B = feat_base.copy()
feat_cols_B = base_cols.copy()

# Activity regularity: std of activity across hours
# We can compute hour-level stats from raw features
hour_feats = [c for c in base_cols if 'wHr_hr' in c or 'hr_' in c]
if hour_feats:
    log.info(f"  Hour features found: {len(hour_feats)}")
    # Activity consistency: how similar is activity across time periods
    for f in hour_feats[:5]:
        vals = feat_B[f].fillna(0).values
        feat_B[f'{f}_rank'] = pd.Series(vals).rank(pct=True).values
        feat_B[f'{f}_rank_std'] = pd.Series(vals).rank(pct=True).rolling(5, min_periods=1).std().fillna(0).values
        feat_cols_B.extend([f'{f}_rank', f'{f}_rank_std'])
    log.info(f"  Added {len([c for c in feat_cols_B if c not in base_cols])} temporal regularity features")

feat_B_test = feat_test_base.copy()
feat_cols_B_t = base_cols.copy()
for f in hour_feats[:5]:
    if f in feat_test_base.columns:
        vals = feat_B_test[f].fillna(0).values
        feat_B_test[f'{f}_rank'] = pd.Series(vals).rank(pct=True).values
        feat_B_test[f'{f}_rank_std'] = pd.Series(vals).rank(pct=True).rolling(5, min_periods=1).std().fillna(0).values
        feat_cols_B_t.extend([f'{f}_rank', f'{f}_rank_std'])

B_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_B, feat_cols_B, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_B, feat_B_test, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    tp_c = mean_match(tp, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    B_results[target] = ll
    log.info(f"  {target}: OOF LL={ll:.5f} (Δ={ll-baseline_results[target]['oof_ll']:+.5f})")

B_avg = np.mean(list(B_results.values()))
log.info(f"  Temp Regularity AVG OOF: {B_avg:.5f} (Δ vs baseline: {B_avg-baseline_avg:+.5f})")

# ============================================================
# EXPERIMENT C: Frequency-domain Features (FFT magnitude)
# ============================================================
log.info("\n" + "="*70)
log.info("EXPERIMENT C: Frequency-domain Features")
log.info("="*70)

feat_C = feat_base.copy()
feat_cols_C = base_cols.copy()

# Spectral features: compute FFT magnitude spectrum summary for hour-based features
freq_feats = []
for f in hour_feats[:5]:
    vals = feat_C[f].fillna(0).values
    if len(vals) >= 8:
        fft_mag = np.abs(np.fft.rfft(vals))
        feat_C[f'{f}_spectral_entropy'] = -np.sum(fft_mag / fft_mag.sum() * np.log(fft_mag / fft_mag.sum() + 1e-10))
        feat_C[f'{f}_spectral_centroid'] = np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-10)
        feat_C[f'{f}_spectral_flatness'] = np.exp(np.mean(np.log(fft_mag + 1e-10))) / (np.mean(fft_mag) + 1e-10)
        feat_C[f'{f}_dominant_freq_ratio'] = fft_mag.argmax() / len(fft_mag)
        freq_feats.extend([f'{f}_spectral_entropy', f'{f}_spectral_centroid', 
                          f'{f}_spectral_flatness', f'{f}_dominant_freq_ratio'])
        feat_cols_C.extend([f'{f}_spectral_entropy', f'{f}_spectral_centroid',
                           f'{f}_spectral_flatness', f'{f}_dominant_freq_ratio'])

log.info(f"  Added {len(freq_feats)} frequency features")

feat_C_test = feat_test_base.copy()
feat_cols_C_t = base_cols.copy()
for f in hour_feats[:5]:
    if f in feat_test_base.columns:
        vals = feat_C_test[f].fillna(0).values
        if len(vals) >= 8:
            fft_mag = np.abs(np.fft.rfft(vals))
            feat_C_test[f'{f}_spectral_entropy'] = -np.sum(fft_mag / fft_mag.sum() * np.log(fft_mag / fft_mag.sum() + 1e-10))
            feat_C_test[f'{f}_spectral_centroid'] = np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-10)
            feat_C_test[f'{f}_spectral_flatness'] = np.exp(np.mean(np.log(fft_mag + 1e-10))) / (np.mean(fft_mag) + 1e-10)
            feat_C_test[f'{f}_dominant_freq_ratio'] = fft_mag.argmax() / len(fft_mag)
            feat_cols_C_t.extend([f'{f}_spectral_entropy', f'{f}_spectral_centroid',
                                  f'{f}_spectral_flatness', f'{f}_dominant_freq_ratio'])

C_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_C, feat_cols_C, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_C, feat_C_test, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    C_results[target] = ll
    log.info(f"  {target}: OOF LL={ll:.5f} (Δ={ll-baseline_results[target]['oof_ll']:+.5f})")

C_avg = np.mean(list(C_results.values()))
log.info(f"  Frequency-domain AVG OOF: {C_avg:.5f} (Δ vs baseline: {C_avg-baseline_avg:+.5f})")

# ============================================================
# EXPERIMENT D: Routine Regularity + Cross-modal Interactions
# ============================================================
log.info("\n" + "="*70)
log.info("EXPERIMENT D: Routine Regularity + Cross-modal Interactions")
log.info("="*70)

feat_D = feat_base.copy()
feat_cols_D = base_cols.copy()

# Routine regularity: how consistent is the subject's behavior pattern
# Use coefficient of variation across feature groups
feature_groups = {
    'activity': [c for c in base_cols if 'mActivity' in c or 'wActivity' in c],
    'screen': [c for c in base_cols if 'mScreen' in c or 'wScreen' in c],
    'location': [c for c in base_cols if 'mGps' in c or 'wGps' in c],
    'ambience': [c for c in base_cols if 'mAmbience' in c],
    'wifi': [c for c in base_cols if 'mWifi' in c],
    'ble': [c for c in base_cols if 'mBle' in c],
    'step': [c for c in base_cols if 'pedo_step' in c or 'pedo_distance' in c or 'wPedo' in c],
    'light': [c for c in base_cols if 'mLight' in c or 'wLight' in c or 'wHr_light' in c],
}

routine_feats = []
for group_name, feats in feature_groups.items():
    if not feats: continue
    group_vals = feat_D[feats].fillna(0).values
    # Within-subject CV of each feature (coefficient of variation)
    for i, f in enumerate(feats):
        vals = feat_D[f].fillna(0).values
        if np.std(vals) > 0:
            cv = np.std(vals) / (np.mean(vals) + 1e-8)
            feat_D[f'{f}_cv'] = cv
            routine_feats.append(f'{f}_cv')
    
    # Group consistency: std of z-scores within group
    means = np.mean(group_vals, axis=1)
    stds = np.std(group_vals, axis=1)
    feat_D[f'{group_name}_group_consistency'] = np.where(stds == 0, 0, stds / (np.abs(means) + 1e-8))
    routine_feats.append(f'{group_name}_group_consistency')
    
    # Number of features in group that are non-zero
    feat_D[f'{group_name}_group_activity'] = (np.abs(group_vals) > 0).sum(axis=1)
    routine_feats.append(f'{group_name}_group_activity')

feat_cols_D.extend(routine_feats)
log.info(f"  Added {len(routine_feats)} routine features")

# Cross-modal interactions
xmodal_feats = []
cross_groups = [
    ('activity', 'screen'),
    ('activity', 'location'),
    ('screen', 'light'),
    ('wifi', 'ble'),
    ('step', 'activity'),
]
for g1, g2 in cross_groups:
    f1_list = feature_groups[g1][:3]
    f2_list = feature_groups[g2][:3]
    for f1 in f1_list:
        for f2 in f2_list:
            if f1 in feat_D.columns and f2 in feat_D.columns:
                v1, v2 = feat_D[f1].fillna(0).values, feat_D[f2].fillna(0).values
                feat_D[f'{f1}_x_{f2}'] = v1 * v2
                feat_D[f'{f1}_div_{f2}'] = v1 / (np.abs(v2) + 1e-8)
                xmodal_feats.extend([f'{f1}_x_{f2}', f'{f1}_div_{f2}'])
                feat_cols_D.extend([f'{f1}_x_{f2}', f'{f1}_div_{f2}'])

log.info(f"  Added {len(xmodal_feats)} cross-modal features (total: {len(routine_feats) + len(xmodal_feats)})")

feat_D_test = feat_test_base.copy()
feat_cols_D_t = base_cols.copy()
# Repeat for test set
for group_name, feats in feature_groups.items():
    if not feats: continue
    for i, f in enumerate(feats):
        if f in feat_D_test.columns:
            vals = feat_D_test[f].fillna(0).values
            if np.std(vals) > 0:
                cv = np.std(vals) / (np.mean(vals) + 1e-8)
                feat_D_test[f'{f}_cv'] = cv
                feat_cols_D_t.append(f'{f}_cv')
    group_vals = feat_D_test[[f for f in feats if f in feat_D_test.columns]].fillna(0).values
    if group_vals.shape[1] > 0:
        means = np.mean(group_vals, axis=1)
        stds = np.std(group_vals, axis=1)
        feat_D_test[f'{group_name}_group_consistency'] = np.where(stds == 0, 0, stds / (np.abs(means) + 1e-8))
        feat_cols_D_t.append(f'{group_name}_group_consistency')
        feat_D_test[f'{group_name}_group_activity'] = (np.abs(group_vals) > 0).sum(axis=1)
        feat_cols_D_t.append(f'{group_name}_group_activity')

for g1, g2 in cross_groups:
    f1_list = [f for f in feature_groups[g1][:3] if f in feat_D_test.columns]
    f2_list = [f for f in feature_groups[g2][:3] if f in feat_D_test.columns]
    for f1 in f1_list:
        for f2 in f2_list:
            v1, v2 = feat_D_test[f1].fillna(0).values, feat_D_test[f2].fillna(0).values
            feat_D_test[f'{f1}_x_{f2}'] = v1 * v2
            feat_D_test[f'{f1}_div_{f2}'] = v1 / (np.abs(v2) + 1e-8)
            feat_cols_D_t.extend([f'{f1}_x_{f2}', f'{f1}_div_{f2}'])

D_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_D, feat_cols_D, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_D, feat_D_test, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    D_results[target] = ll
    log.info(f"  {target}: OOF LL={ll:.5f} (Δ={ll-baseline_results[target]['oof_ll']:+.5f})")

D_avg = np.mean(list(D_results.values()))
log.info(f"  Routine + Cross-modal AVG OOF: {D_avg:.5f} (Δ vs baseline: {D_avg-baseline_avg:+.5f})")

# ============================================================
# EXPERIMENT E: Selective D features — only features that improved S3 or Q3
# S3 and Q3 showed biggest gains from D (-0.037, -0.022)
# Hypothesis: routine features specifically help these targets
# ============================================================
log.info("\n" + "="*70)
log.info("EXPERIMENT E: Targeted D features (Q3/S3 winners)")
log.info("="*70)

# Take only the routine features that drove improvement in D
# Q3 and S3 got biggest deltas: -0.022 and -0.037
# These targets benefit from routine regularity → focus on group consistency features

feat_E_all = feat_base.copy()
feat_cols_E = base_cols.copy()

# Add only the routine group features (not all 522 D features)
group_names = ['activity', 'screen', 'location', 'ambience', 'wifi', 'ble', 'step', 'light']
# Add group consistency + activity + CV for top-5 features only (reduce noise)
for f in base_cols[:30]:  # top features
    vals = feat_E_all[f].fillna(0).values
    if np.std(vals) > 0:
        cv = np.std(vals) / (np.mean(vals) + 1e-8)
        feat_E_all[f'{f}_cv'] = cv
        feat_cols_E.append(f'{f}_cv')

# Add only 2-3 cross-modal interactions (the ones that mattered for S3/Q3)
feat_E_all['mWifi_wifi_max_rssi_max_x_mBle_ble_max_rssi_max'] = feat_E_all['mWifi_wifi_max_rssi_max'].fillna(0) * feat_E_all['mBle_ble_max_rssi_max'].fillna(0)
feat_E_all['mWifi_wifi_max_rssi_max_div_mBle_ble_max_rssi_max'] = feat_E_all['mWifi_wifi_max_rssi_max'].fillna(0) / (np.abs(feat_E_all['mBle_ble_max_rssi_max'].fillna(0)) + 1e-8)
feat_cols_E.extend(['mWifi_wifi_max_rssi_max_x_mBle_ble_max_rssi_max', 'mWifi_wifi_max_rssi_max_div_mBle_ble_max_rssi_max'])

log.info(f"  E features: {len(feat_cols_E)} (selective routine)")

feat_E_test = feat_test_base.copy()
for f in feat_E_test.columns[:30]:
    if feat_E_test[f].dtype not in [np.float64, np.int64, float, int]: continue
    vals = feat_E_test[f].fillna(0).values
    if np.std(vals) > 0:
        cv = np.std(vals) / (np.mean(vals) + 1e-8)
        feat_E_test[f'{f}_cv'] = cv
# Add cross-modal features to test too
if 'mWifi_wifi_max_rssi_max' in feat_E_test.columns and 'mBle_ble_max_rssi_max' in feat_E_test.columns:
    v1 = feat_E_test['mWifi_wifi_max_rssi_max'].fillna(0).values
    v2 = feat_E_test['mBle_ble_max_rssi_max'].fillna(0).values
    feat_E_test['mWifi_wifi_max_rssi_max_x_mBle_ble_max_rssi_max'] = v1 * v2
    feat_E_test['mWifi_wifi_max_rssi_max_div_mBle_ble_max_rssi_max'] = v1 / (np.abs(v2) + 1e-8)

# E results
E_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_E_all, feat_cols_E, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_E_all, feat_E_test, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    E_results[target] = ll
    log.info(f"    {target}: OOF LL={ll:.5f} (Δ={ll-baseline_results[target]['oof_ll']:+.5f})")

E_avg = np.mean(list(E_results.values()))
log.info(f"  Selective Routine AVG OOF: {E_avg:.5f} (Δ vs baseline: {E_avg-baseline_avg:+.5f})")

E_results = {}
for target in TARGETS:
    t_cfg = V53_SWEEP[target]
    cfg = CFGS[t_cfg['cfg']]
    y = y_train[target]
    
    ranked = rank_features(feat_E_all, feat_E_all_cols, target)
    sel = ranked[:t_cfg['n_feat']]
    
    oof, tp = train_cv(feat_E_all, feat_E_test, sel, y, SEEDS, cfg)
    oof = oof.mean(axis=1); tp = tp.mean(axis=1)
    oof_c = mean_match(oof, train_rates[target])
    ll = log_loss(y, oof_c, labels=[0,1])
    E_results[target] = ll
    log.info(f"  {target}: OOF LL={ll:.5f} (Δ={ll-baseline_results[target]['oof_ll']:+.5f})")

E_avg = np.mean(list(E_results.values()))
log.info(f"  Combined AVG OOF: {E_avg:.5f} (Δ vs baseline: {E_avg-baseline_avg:+.5f})")

# ============================================================
# Summary
# ============================================================
log.info(f"\n{'='*70}")
log.info("V258 SUMMARY — Advanced Feature Discovery")
log.info(f"{'='*70}")

summary = {
    'version': 'V258',
    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'baseline_avg_oof': round(float(baseline_avg), 5),
    'experiments': {
        'A_baseline': {'avg_oof': round(float(baseline_avg), 5)},
        'B_temporal_regularity': {'avg_oof': round(float(B_avg), 5), 'delta': round(float(B_avg - baseline_avg), 5)},
        'C_frequency_domain': {'avg_oof': round(float(C_avg), 5), 'delta': round(float(C_avg - baseline_avg), 5)},
        'D_routine_crossmodal': {'avg_oof': round(float(D_avg), 5), 'delta': round(float(D_avg - baseline_avg), 5)},
        'E_combined': {'avg_oof': round(float(E_avg), 5), 'delta': round(float(E_avg - baseline_avg), 5)},
    },
    'per_target': {
        'baseline': {t: round(baseline_results[t]['oof_ll'], 5) for t in TARGETS},
        'B_temporal': {t: round(B_results[t], 5) for t in TARGETS},
        'C_frequency': {t: round(C_results[t], 5) for t in TARGETS},
        'D_routine': {t: round(D_results[t], 5) for t in TARGETS},
        'E_combined': {t: round(E_results[t], 5) for t in TARGETS},
    },
    'best_version': min(
        [('A_baseline', baseline_avg), ('B_temporal', B_avg), ('C_frequency', C_avg), 
         ('D_routine', D_avg), ('E_combined', E_avg)],
        key=lambda x: x[1]
    )[0],
    'best_delta': min(
        baseline_avg - baseline_avg, B_avg - baseline_avg, C_avg - baseline_avg,
        D_avg - baseline_avg, E_avg - baseline_avg
    ),
}

# Print summary
log.info(f"\n{'Version':<25} {'AVG OOF':>10} {'Δ vs Base':>12}")
log.info(f"{'─'*50}")
log.info(f"{'A: Baseline V127':<25} {baseline_avg:>10.5f} {'':>12}")
log.info(f"{'B: Temporal Regularity':<25} {B_avg:>10.5f} {B_avg-baseline_avg:>+12.5f}")
log.info(f"{'C: Frequency Domain':<25} {C_avg:>10.5f} {C_avg-baseline_avg:>+12.5f}")
log.info(f"{'D: Routine + Cross-modal':<25} {D_avg:>10.5f} {D_avg-baseline_avg:>+12.5f}")
log.info(f"{'E: Combined':<25} {E_avg:>10.5f} {E_avg-baseline_avg:>+12.5f}")

# Per-target deltas
log.info(f"\nPer-target Δ from baseline:")
for t in TARGETS:
    deltas = {
        'B': B_results[t] - baseline_results[t]['oof_ll'],
        'C': C_results[t] - baseline_results[t]['oof_ll'],
        'D': D_results[t] - baseline_results[t]['oof_ll'],
        'E': E_results[t] - baseline_results[t]['oof_ll'],
    }
    best_d = min(deltas.items(), key=lambda x: x[1])
    markers = " ★" if best_d[1] < -0.003 else ""
    log.info(f"  {t}: B={deltas['B']:+.5f} C={deltas['C']:+.5f} D={deltas['D']:+.5f} E={deltas['E']:+.5f} best={best_d[0]}{best_d[1]:+.5f}{markers}")

# Save JSON
exp_path = EXPERIMENTS / f'v258_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(exp_path, 'w') as f: json.dump(summary, f, indent=2, default=str)
log.info(f"\nSaved: {exp_path}")
