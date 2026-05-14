"""V259: Frequency Domain Features — FFT, Cyclic Decomposition, Temporal Patterns

Per-subject daily time series analysis using the already-processed per-day aggregate
features. Each subject has ~33-57 daily rows, ordered chronologically.

Features created:
1. FFT: dominant frequency, amplitude spectrum, spectral energy, spectral entropy
2. Cyclic: trend+seasonal+residual decomposition, seasonal amplitude/phase
3. Temporal: day-of-week effects, autocorrelation (lag-1,7,14), PAC lags 1-7
4. Cross-modal: cross-spectrum between activity↔screen_time, heart_rate↔steps
5. Spectral entropy: Shannon entropy of PSD per subject per feature

GroupKFold(5) CV. Seed=42. 3 seeds for averaging.
"""
import logging, sys, gc, re, json, warnings, time, os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA_PROC = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999]
N_FOLDS = 5

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feat_cols(df):
    exclude = META_COLS | set(TARGETS) | {'subject_id','lifelog_date','sleep_date','date'}
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

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
    return df, zcols, all_stats if not for_test else None

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
    'Q1':  {'cfg': 'deep', 'n_feat': 19},
    'Q2':  {'cfg': 'deep', 'n_feat': 14},
    'Q3':  {'cfg': 'v48', 'n_feat': 11},
    'S1':  {'cfg': 'wide', 'n_feat': 21},
    'S2':  {'cfg': 'deep', 'n_feat': 19},
    'S3':  {'cfg': 'safety','n_feat': 23},
    'S4':  {'cfg': 'wide', 'n_feat': 20},
}

# ============================================================
# HELPER: autocorrelation and partial autocorrelation
# ============================================================
def autocorr(x, lag):
    """Autocorrelation coefficient at given lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n <= lag or np.std(x) < 1e-10:
        return 0.0
    return np.sum(x[:n-lag] * x[lag:]) / (np.sum(x**2) + 1e-10)

def partial_autocorr(x, lag):
    """Partial autocorrelation via Durbin-Levinson recursion."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if lag >= n or lag == 0:
        return 0.0
    var = np.sum(x**2) + 1e-10
    acf = np.zeros(lag + 1)
    for k in range(lag + 1):
        acf[k] = np.sum(x[:n-k] * x[k:]) / var
    phi = np.zeros(lag + 1)
    phi[1] = acf[1] / acf[0]
    for m in range(2, lag + 1):
        num = acf[m] - np.sum(phi[1:m] * acf[m-1:0:-1])
        den = acf[0] - np.sum(phi[1:m] * acf[1:m])
        phi[m] = num / (den + 1e-10)
    return phi[lag]

# ============================================================
# PER-SUBJECT TIME SERIES FEATURE ENGINEERING
# ============================================================
# Feature groups for cross-modal analysis
ACTIVITY_FEATURES = ['mActivity_m_activity_mean', 'mActivity_m_activity_std',
                     'mActivity_m_activity_count', 'mActivity_m_activity_sum']
SCREEN_FEATURES = ['mScreenStatus_m_screen_use_mean', 'mScreenStatus_m_screen_use_std',
                   'mScreenStatus_m_screen_use_count', 'mScreenStatus_m_screen_use_sum']
STEPS_FEATURES = ['wPedo_pedo_step_mean', 'wPedo_pedo_step_sum', 'wPedo_pedo_distance_mean',
                  'wPedo_pedo_distance_sum', 'wPedo_pedo_burned_calories_mean',
                  'wPedo_pedo_burned_calories_sum']
HEART_RATE_FEATURES = ['wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_count']
LIGHT_FEATURES = ['mLight_m_light_mean', 'mLight_m_light_std']
WIFI_FEATURES = ['mWifi_wifi_avg_rssi_mean', 'mWifi_wifi_max_rssi_mean',
                 'mWifi_wifi_strong_ratio_mean']
GPS_FEATURES = ['mGps_gps_avg_speed_mean', 'mGps_gps_avg_speed_std',
                'mGps_gps_count_mean', 'mGps_gps_max_speed_mean']

# Key features that are commonly available and informative
KEY_TIME_FEATURES = ACTIVITY_FEATURES + STEPS_FEATURES + HEART_RATE_FEATURES + SCREEN_FEATURES

def compute_time_series_features(daily_df):
    """Compute frequency-domain and temporal features from a single subject's daily time series.
    daily_df: sorted-by-date DataFrame with 'lifelog_date'/'date' column.
    Returns dict of {feature_name: value}.
    """
    feats = {}
    dates = daily_df.sort_values('lifelog_date') if 'lifelog_date' in daily_df.columns else daily_df.sort_values('date')
    n = len(dates)
    if n < 7:
        return feats

    # Key features for time series analysis (avoid z-score duplicates)
    key_features = [
        'mActivity_m_activity_mean', 'mActivity_m_activity_std',
        'wHr_hr_mean', 'wHr_hr_std',
        'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum', 'wPedo_pedo_distance_mean',
        'mScreenStatus_m_screen_use_mean', 'mScreenStatus_m_screen_use_std',
        'mLight_m_light_mean',
        'mWifi_wifi_avg_rssi_mean', 'mWifi_wifi_max_rssi_mean',
        'mBle_ble_avg_rssi_mean',
        'mGps_gps_avg_speed_mean', 'mGps_gps_count_mean',
        'mUsageStats_usage_total_time_mean', 'mUsageStats_usage_major_ratio_mean',
        'wLight_w_light_mean',
    ]
    numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
    analysis_cols = [c for c in key_features if c in numeric_cols]
    # Also include a broader set but deduplicated (no _zscore in name)
    broader = [c for c in numeric_cols 
               if c not in ('subject_id', 'lifelog_date', 'sleep_date', 'date')
               and '_zscore' not in c
               and c not in TARGETS]
    # Use broader set for FFT/cyclic/temporal, but limit to top features
    if len(broader) > 30:
        broader = broader[:30]
    all_cols = list(set(analysis_cols + broader))
    
    for col in all_cols:
        if col in ('subject_id', 'lifelog_date', 'sleep_date', 'date'):
            continue
        vals = daily_df[col].astype(np.float64).fillna(0).values
        
        # Need enough non-constant values
        if np.std(vals) < 1e-10 or n < 8:
            continue

        # ---- FFT Features ----
        # Detrend: remove linear trend
        t = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(t, vals, 1)
        detrended = vals - (slope * t + intercept)
        
        fft_vals = np.fft.rfft(detrended)
        fft_mag = np.abs(fft_vals)
        fft_phase = np.angle(fft_vals)
        
        # Dominant frequency (skip DC)
        if len(fft_mag) > 2:
            dom_idx = np.argmax(fft_mag[1:]) + 1
            feats[f'{col}_fft_dom_freq'] = dom_idx
            feats[f'{col}_fft_dom_freq_norm'] = dom_idx / (len(fft_mag) - 1)
            feats[f'{col}_fft_dom_amp'] = float(fft_mag[dom_idx])
            
            # Second dominant frequency
            fft_mag_copy = fft_mag.copy()
            fft_mag_copy[dom_idx] = 0
            if len(fft_mag_copy) > 2:
                dom2_idx = np.argmax(fft_mag_copy[1:]) + 1
                feats[f'{col}_fft_2nd_freq'] = dom2_idx
                feats[f'{col}_fft_2nd_amp'] = float(fft_mag_copy[dom2_idx])

        # Spectral energy
        feats[f'{col}_fft_total_energy'] = float(np.sum(fft_mag**2))
        
        # Low vs high frequency energy ratio
        half = len(fft_mag) // 2
        if half > 0:
            energy_low = np.sum(fft_mag[:half]**2)
            energy_high = np.sum(fft_mag[half:]**2)
            total_e = energy_low + energy_high + 1e-10
            feats[f'{col}_fft_low_energy_ratio'] = float(energy_low / total_e)
            feats[f'{col}_fft_high_energy_ratio'] = float(energy_high / total_e)
            feats[f'{col}_fft_energy_skew'] = float((energy_low - energy_high) / total_e)
        
        # DC component ratio
        feats[f'{col}_fft_dc_ratio'] = float(fft_mag[0] / (np.sum(fft_mag) + 1e-10))
        
        # Spectral entropy
        psd = fft_mag / (np.sum(fft_mag) + 1e-10)
        psd_pos = psd[psd > 0]
        feats[f'{col}_fft_spectral_entropy'] = float(-np.sum(psd_pos * np.log(psd_pos + 1e-10)))
        
        # Spectral centroid
        freqs = np.arange(len(fft_mag), dtype=np.float64)
        feats[f'{col}_fft_spectral_centroid'] = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10))
        
        # Spectral flatness
        feats[f'{col}_fft_spectral_flatness'] = float(
            np.exp(np.mean(np.log(fft_mag + 1e-10))) / (np.mean(fft_mag) + 1e-10))
        
        # Spectral roll-off (85% energy)
        cum_energy = np.cumsum(fft_mag**2)
        roll_idx = np.searchsorted(cum_energy, 0.85 * cum_energy[-1])
        feats[f'{col}_fft_spectral_roll_off'] = float(roll_idx / max(len(fft_mag) - 1, 1))
        
        # Crest factor (peak / RMS of detrended)
        rms = np.sqrt(np.mean(detrended**2))
        feats[f'{col}_fft_crest_factor'] = float(np.max(np.abs(detrended)) / (rms + 1e-10))
        
        # Max-to-mean spectrum ratio
        feats[f'{col}_fft_spectrum_kurtosis_proxy'] = float(
            np.max(fft_mag) / (np.mean(fft_mag) + 1e-10))
        
        # ---- Cyclic Decomposition (trend + seasonal + residual) ----
        trend_win = min(7, max(3, n // 3))
        trend = pd.Series(vals).rolling(trend_win, center=True, min_periods=1).mean().values
        detrended_seasonal = vals - trend
        
        # Seasonal: mean of detrended values at same position in window
        seasonal = np.zeros(n)
        for i in range(n):
            idx = i % trend_win
            mask = np.arange(n) % trend_win == idx
            seasonal[i] = np.mean(detrended_seasonal[mask]) if mask.any() else 0
        
        residual = detrended_seasonal - seasonal
        
        total_var = np.var(vals) + 1e-10
        feats[f'{col}_cyclic_trend_std'] = float(np.std(trend))
        feats[f'{col}_cyclic_seasonal_amplitude'] = float(np.max(seasonal) - np.min(seasonal))
        feats[f'{col}_cyclic_seasonal_std'] = float(np.std(seasonal))
        feats[f'{col}_cyclic_residual_std'] = float(np.std(residual))
        feats[f'{col}_cyclic_seasonal_var_ratio'] = float(np.var(seasonal) / total_var)
        feats[f'{col}_cyclic_trend_var_ratio'] = float(np.var(trend) / total_var)
        feats[f'{col}_cyclic_residual_var_ratio'] = float(np.var(residual) / total_var)
        
        # Seasonal phase
        if np.std(seasonal) > 0:
            phase_peak = np.argmax(np.abs(seasonal)) % trend_win
            feats[f'{col}_cyclic_seasonal_phase'] = float(phase_peak / trend_win)
        else:
            feats[f'{col}_cyclic_seasonal_phase'] = 0.5
        
        # Residual autocorrelation
        feats[f'{col}_cyclic_residual_ac1'] = float(autocorr(residual, 1))
        
        # ---- Temporal Pattern Features ----
        # Day of week effects
        dow = pd.to_datetime(daily_df['lifelog_date'] if 'lifelog_date' in daily_df.columns else daily_df['date']).dt.dayofweek.values
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Per-DOW mean
        dow_means = np.zeros(7)
        for d in range(7):
            mask = dow == d
            if mask.sum() >= 1:
                dow_means[d] = vals[mask].mean()
        feats[f'{col}_temporal_dow_var'] = float(np.var(dow_means))
        feats[f'{col}_temporal_dow_range'] = float(np.ptp(dow_means))
        
        # Weekend vs weekday
        wd_mask = dow < 5
        we_mask = dow >= 5
        wd_m = vals[wd_mask].mean() if wd_mask.sum() > 0 else 0
        we_m = vals[we_mask].mean() if we_mask.sum() > 0 else 0
        feats[f'{col}_temporal_we_wd_ratio'] = float(we_m / (wd_m + 1e-10))
        
        # Autocorrelation at lag 1, 7, 14
        centered = vals - np.mean(vals)
        for lag in [1, 7, 14]:
            feats[f'{col}_temporal_ac_lag{lag}'] = float(autocorr(centered, lag))
        
        # Partial autocorrelation lags 1-7
        for lag in range(1, 8):
            feats[f'{col}_temporal_pac_lag{lag}'] = float(partial_autocorr(centered, lag))
        
        # Volatility (mean absolute change)
        changes = np.diff(vals)
        feats[f'{col}_temporal_mean_abs_change'] = float(np.mean(np.abs(changes)))
        feats[f'{col}_temporal_change_cv'] = float(np.std(changes) / (np.mean(np.abs(changes)) + 1e-10))
        
        # Run length (consecutive days above/below median)
        med = np.median(vals)
        above = (vals > med).astype(int)
        if above.sum() > 0 and above.sum() < n:
            changes_sign = np.diff(above)
            n_runs = (np.abs(changes_sign) > 0).sum() + 1
            feats[f'{col}_temporal_n_runs'] = float(n_runs)
            feats[f'{col}_temporal_mean_run_length'] = float(n / max(n_runs, 1))
        else:
            feats[f'{col}_temporal_n_runs'] = 1.0
            feats[f'{col}_temporal_mean_run_length'] = float(n)
    
    # ---- Cross-modal frequency features ----
    # Cross-spectrum between key modality groups
    modality_maps = {
        'activity': [c for c in ACTIVITY_FEATURES if c in daily_df.columns],
        'screen': [c for c in SCREEN_FEATURES if c in daily_df.columns],
        'steps': [c for c in STEPS_FEATURES if c in daily_df.columns],
        'heart_rate': [c for c in HEART_RATE_FEATURES if c in daily_df.columns],
        'light': [c for c in LIGHT_FEATURES if c in daily_df.columns],
    }
    
    # Compute mean PSD per modality
    modality_psd = {}
    for mod, cols in modality_maps.items():
        if not cols or len(cols) < 1:
            continue
        all_mag = None
        for c in cols:
            vals = daily_df[c].astype(np.float64).fillna(0).values
            if np.std(vals) < 1e-10 or n < 8:
                continue
            t = np.arange(n, dtype=np.float64)
            slope, intercept = np.polyfit(t, vals, 1)
            detrended = vals - (slope * t + intercept)
            fft_mag = np.abs(np.fft.rfft(detrended))
            if all_mag is None:
                all_mag = fft_mag.copy()
            else:
                all_mag += fft_mag
        if all_mag is not None:
            all_mag /= len(cols)
            modality_psd[mod] = all_mag
            
            # Spectral entropy of modality-level PSD
            psd = all_mag / (np.sum(all_mag) + 1e-10)
            psd_pos = psd[psd > 0]
            feats[f'{mod}_modality_spectral_entropy'] = float(-np.sum(psd_pos * np.log(psd_pos + 1e-10)))
            feats[f'{mod}_modality_spectral_centroid'] = float(
                np.sum(np.arange(len(all_mag), dtype=float) * all_mag) / (np.sum(all_mag) + 1e-10))
            feats[f'{mod}_modality_low_energy_ratio'] = float(
                np.sum(all_mag[:len(all_mag)//2]**2) / (np.sum(all_mag**2) + 1e-10))
    
    # Cross-spectrum: activity vs screen (inverse relationship?)
    if 'activity' in modality_psd and 'screen' in modality_psd:
        a = modality_psd['activity']
        s = modality_psd['screen']
        # Cross-spectrum
        a_fft = np.fft.rfft(a - np.mean(a))
        s_fft = np.fft.rfft(s - np.mean(s))
        cross_spec = a_fft * np.conj(s_fft)
        feats['modality_activity_screen_coherence'] = float(np.abs(cross_spec).mean())
        
        # Phase difference at dominant frequency
        dom_idx = np.argmax(np.abs(a_fft[1:])) + 1
        if len(cross_spec) > dom_idx:
            phase_diff = np.angle(a_fft[dom_idx]) - np.angle(s_fft[dom_idx])
            feats['modality_activity_screen_phase_diff'] = float(np.abs(np.sin(phase_diff)))
        
        # Are they inversely related? (negative cross-spectrum at dominant freq)
        feats['modality_activity_screen_inverse'] = float(np.sign(np.real(cross_spec[dom_idx])))
    
    # Cross-spectrum: heart_rate vs steps
    if 'heart_rate' in modality_psd and 'steps' in modality_psd:
        h = modality_psd['heart_rate']
        s = modality_psd['steps']
        h_fft = np.fft.rfft(h - np.mean(h))
        s_fft = np.fft.rfft(s - np.mean(s))
        cross_spec = h_fft * np.conj(s_fft)
        feats['modality_hr_steps_coherence'] = float(np.abs(cross_spec).mean())
        dom_idx = np.argmax(np.abs(h_fft[1:])) + 1
        if len(cross_spec) > dom_idx:
            phase_diff = np.angle(h_fft[dom_idx]) - np.angle(s_fft[dom_idx])
            feats['modality_hr_steps_phase_diff'] = float(np.abs(np.sin(phase_diff)))
    
    return feats

# ============================================================
# MAIN
# ============================================================
def build_baseline_from_df(feat, feat_test):
    fc = get_feat_cols(feat)
    feat_b, zcols, _ = add_zscore(feat, fc)
    feat_test_b, _, _ = add_zscore(feat_test, fc)
    return feat_b, fc + zcols

def main():
    start_time = time.time()
    
    log.info("=" * 70)
    log.info("V259: Frequency Domain Features — FFT, Cyclic Decomposition")
    log.info("=" * 70)
    
    # ---- Load baseline data ----
    log.info("\n[1/5] Loading data...")
    feat = pd.read_parquet(DATA_PROC / "features_clean_v60.parquet")
    feat_test = pd.read_parquet(DATA_PROC / "test_features_clean_v60.parquet")
    for df in [feat, feat_test]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    y_train = {t: feat[t].values for t in TARGETS}
    train_rates = {t: float(feat[t].mean()) for t in TARGETS}
    
    feat_base, base_cols = build_baseline_from_df(feat, feat_test)
    feat_test_base = feat_base.copy()  # z-scores already added
    
    log.info(f"  Baseline features: {len(base_cols)} (base={len([c for c in base_cols if 'z' not in c])} + zscore={len([c for c in base_cols if 'z' in c])})")
    
    # ---- Compute per-subject time series features ----
    log.info("\n[2/5] Computing per-subject time series features...")
    all_freq_feats = {}  # subject_id -> feature dict
    new_feature_names = set()
    
    for sid in sorted(feat['subject_id'].unique()):
        sdf = feat_base[feat_base['subject_id'] == sid]
        feats = compute_time_series_features(sdf)
        if feats:
            all_freq_feats[sid] = feats
            new_feature_names.update(feats.keys())
            log.info(f"  {sid}: {len(feats)} features ({len(sdf)} days)")
    
    log.info(f"\n  Total new features: {len(new_feature_names)}")
    
    # Categorize
    fft_count = sum(1 for k in new_feature_names if '_fft_' in k)
    cyclic_count = sum(1 for k in new_feature_names if '_cyclic_' in k)
    temporal_count = sum(1 for k in new_feature_names if '_temporal_' in k)
    entropy_count = sum(1 for k in new_feature_names if 'spectral_entropy' in k)
    modality_count = sum(1 for k in new_feature_names if 'modality_' in k)
    
    log.info(f"  Breakdown: FFT={fft_count}, Cyclic={cyclic_count}, Temporal={temporal_count}, "
             f"Spectral entropy={entropy_count}, Modality={modality_count}")
    
    # ---- Create augmented dataframes ----
    log.info("\n[3/5] Creating augmented DataFrames...")
    
    # Build freq feature DataFrame (wide format: one row per subject)
    freq_rows = []
    for sid in sorted(feat_base['subject_id'].unique()):
        row = {'subject_id': sid}
        if sid in all_freq_feats:
            row.update(all_freq_feats[sid])
        freq_rows.append(row)
    
    freq_df = pd.DataFrame(freq_rows)
    freq_df['subject_id'] = freq_df['subject_id'].astype(str)
    feat_base['subject_id'] = feat_base['subject_id'].astype(str)
    feat_test_base['subject_id'] = feat_test_base['subject_id'].astype(str)
    
    # Merge with baseline — use set_index + concat for large column count
    freq_cols = [c for c in freq_df.columns if c != 'subject_id']
    freq_df_idx = freq_df.set_index('subject_id')
    feat_base_idx = feat_base.set_index('subject_id')
    feat_aug = feat_base_idx.join(freq_df_idx, how='left')
    feat_aug = feat_aug.reset_index()
    feat_aug = feat_aug.fillna(0)
    aug_cols = base_cols + freq_cols
    
    # Test set: same merge
    feat_test_base_idx = feat_test_base.set_index('subject_id')
    feat_test_aug = feat_test_base_idx.join(freq_df_idx, how='left')
    feat_test_aug = feat_test_aug.reset_index()
    feat_test_aug = feat_test_aug.fillna(0)
    
    log.info(f"  Train features: {len(aug_cols)} (baseline={len(base_cols)} + freq={len(aug_cols)-len(base_cols)})")
    
    # ---- Run CV experiments ----
    log.info("\n[4/5] Running 5-fold CV...")
    
    # A: Baseline
    log.info("  [A] Baseline (base + zscore)")
    baseline_results = {}
    for target in TARGETS:
        t_cfg = V53_SWEEP[target]
        cfg = CFGS[t_cfg['cfg']]
        y = y_train[target]
        ranked = rank_features(feat_base, base_cols, target)
        sel = ranked[:t_cfg['n_feat']]
        oof, tp = train_cv(feat_base, feat_test_base, sel, y, SEEDS, cfg)
        oof = oof.mean(axis=1)
        oof_c = mean_match(oof, train_rates[target])
        ll = float(log_loss(y, oof_c, labels=[0,1]))
        baseline_results[target] = ll
        log.info(f"    {target}: LL={ll:.5f}")
    
    baseline_avg = float(np.mean(list(baseline_results.values())))
    log.info(f"  Baseline AVG: {baseline_avg:.5f}")
    
    # B: Full frequency features (all new features available for selection)
    log.info("  [B] Frequency-augmented features")
    B_results = {}
    for target in TARGETS:
        t_cfg = V53_SWEEP[target]
        cfg = CFGS[t_cfg['cfg']]
        y = y_train[target]
        ranked = rank_features(feat_aug, aug_cols, target)
        sel = ranked[:t_cfg['n_feat']]
        oof, tp = train_cv(feat_aug, feat_test_aug, sel, y, SEEDS, cfg)
        oof = oof.mean(axis=1)
        oof_c = mean_match(oof, train_rates[target])
        ll = float(log_loss(y, oof_c, labels=[0,1]))
        B_results[target] = ll
        log.info(f"    {target}: LL={ll:.5f} (Δ={ll-baseline_results[target]:+.5f})")
    
    B_avg = float(np.mean(list(B_results.values())))
    log.info(f"  Frequency AVG: {B_avg:.5f} (Δ={B_avg-baseline_avg:+.5f})")
    
    # C: Baseline + top frequency-only features
    log.info("  [C] Baseline + top frequency-only features")
    C_results = {}
    for target in TARGETS:
        t_cfg = V53_SWEEP[target]
        cfg = CFGS[t_cfg['cfg']]
        y = y_train[target]
        
        # Rank all features, separate baseline vs freq
        ranked = rank_features(feat_aug, aug_cols, target)
        
        # Take n_feat baseline features
        base_selected = [c for c in ranked[:t_cfg['n_feat']] if '_z' in c or any(
            x in c for x in ['mActivity','wHr','wPedo','mLight','mScreen','mWifi','mBle','mGps','mAmbience','mACStatus','wLight','mUsageStats'])]
        
        # Add top freq features
        freq_in_ranking = [c for c in ranked if c not in base_cols]
        top_freq = freq_in_ranking[:min(5, len(freq_in_ranking))]
        sel = base_selected + top_freq
        sel = list(dict.fromkeys(sel))  # deduplicate, preserve order
        
        oof, tp = train_cv(feat_aug, feat_test_aug, sel, y, SEEDS, cfg)
        oof = oof.mean(axis=1)
        oof_c = mean_match(oof, train_rates[target])
        ll = float(log_loss(y, oof_c, labels=[0,1]))
        C_results[target] = ll
        log.info(f"    {target}: LL={ll:.5f} (Δ={ll-baseline_results[target]:+.5f})")
    
    C_avg = float(np.mean(list(C_results.values())))
    log.info(f"  Baseline+Freq AVG: {C_avg:.5f} (Δ={C_avg-baseline_avg:+.5f})")
    
    # D: Only temporal pattern features (subset)
    log.info("  [D] Baseline + temporal pattern features only")
    
    temporal_feat_names = [k for k in new_feature_names if '_temporal_' in k]
    log.info(f"    Using {len(temporal_feat_names)} temporal features")
    
    # Build D dataframe with only temporal features
    temporal_rows = []
    for sid in sorted(feat_base['subject_id'].unique()):
        row = {'subject_id': sid}
        if sid in all_freq_feats:
            for tf in temporal_feat_names:
                if tf in all_freq_feats[sid]:
                    row[tf] = all_freq_feats[sid][tf]
        temporal_rows.append(row)
    
    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df['subject_id'] = temporal_df['subject_id'].astype(str)
    temporal_cols = [c for c in temporal_df.columns if c != 'subject_id']
    temporal_idx = temporal_df.set_index('subject_id')
    feat_D = feat_base.set_index('subject_id').join(temporal_idx, how='left').reset_index()
    feat_D = feat_D.fillna(0)
    D_cols = base_cols + temporal_cols
    
    feat_D_test = feat_test_base.set_index('subject_id').join(temporal_idx, how='left').reset_index()
    feat_D_test = feat_D_test.fillna(0)
    
    D_results = {}
    for target in TARGETS:
        t_cfg = V53_SWEEP[target]
        cfg = CFGS[t_cfg['cfg']]
        y = y_train[target]
        ranked = rank_features(feat_D, D_cols, target)
        sel = ranked[:t_cfg['n_feat']]
        oof, tp = train_cv(feat_D, feat_D_test, sel, y, SEEDS, cfg)
        oof = oof.mean(axis=1)
        oof_c = mean_match(oof, train_rates[target])
        ll = float(log_loss(y, oof_c, labels=[0,1]))
        D_results[target] = ll
        log.info(f"    {target}: LL={ll:.5f} (Δ={ll-baseline_results[target]:+.5f})")
    
    D_avg = float(np.mean(list(D_results.values())))
    log.info(f"  Temporal AVG: {D_avg:.5f} (Δ={D_avg-baseline_avg:+.5f})")
    
    # ---- Summary ----
    elapsed = time.time() - start_time
    log.info("\n" + "=" * 70)
    log.info("V259 SUMMARY")
    log.info("=" * 70)
    
    result = {
        "version": "v259_frequency",
        "features_created": len(new_feature_names),
        "fft_features": fft_count,
        "cyclic_features": cyclic_count,
        "temporal_pattern_features": temporal_count,
        "spectral_entropy_features": entropy_count,
        "baseline_oof": round(baseline_avg, 5),
        "frequency_augmented_oof": round(B_avg, 5),
        "delta": round(B_avg - baseline_avg, 5),
        "per_target": {
            "baseline": {t: round(baseline_results[t], 5) for t in TARGETS},
            "frequency": {t: round(B_results[t], 5) for t in TARGETS},
            "baseline_plus_freq": {t: round(C_results[t], 5) for t in TARGETS},
            "temporal": {t: round(D_results[t], 5) for t in TARGETS},
        },
        "deltas_per_target": {
            "frequency": {t: round(B_results[t] - baseline_results[t], 5) for t in TARGETS},
            "baseline_plus_freq": {t: round(C_results[t] - baseline_results[t], 5) for t in TARGETS},
            "temporal": {t: round(D_results[t] - baseline_results[t], 5) for t in TARGETS},
        },
        "feature_breakdown": {
            "fft": fft_count,
            "cyclic": cyclic_count,
            "temporal": temporal_count,
            "spectral_entropy": entropy_count,
            "modality": modality_count,
        },
        "subject_coverage": len(all_freq_feats),
        "total_subjects": 10,
        "notes": (
            f"Per-subject daily time series FFT/cyclic/temporal analysis. "
            f"FFT: {fft_count} features (dominant freq, spectral energy, entropy, centroid, flatness, roll-off, crest). "
            f"Cyclic: {cyclic_count} (trend/seasonal/residual decomposition). "
            f"Temporal: {temporal_count} (DOW effects, AC/PAC lags, run length, volatility). "
            f"Modality cross-spectrum: {modality_count} (activity↔screen, HR↔steps coherence/phase). "
            f"Spectral entropy: {entropy_count}. "
            f"Δ freq={B_avg-baseline_avg:+.5f}, Δ baseline+freq={C_avg-baseline_avg:+.5f}, Δ temporal={D_avg-baseline_avg:+.5f}. "
            f"Best: {'frequency' if B_avg==min(B_avg,C_avg,D_avg) else 'temporal' if D_avg==min(B_avg,C_avg,D_avg) else 'baseline_plus_freq'}. "
            f"Elapsed: {elapsed:.1f}s"
        ),
    }
    
    exp_path = EXPERIMENTS / 'v259_frequency_features_result.json'
    with open(exp_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"\nSaved: {exp_path}")
    log.info(f"Elapsed: {elapsed:.1f}s")
    
    # Print summary table
    log.info(f"\n{'Version':<30} {'AVG OOF':>10} {'Δ vs Base':>12}")
    log.info(f"{'─'*55}")
    log.info(f"{'A: Baseline':<30} {baseline_avg:>10.5f} {'':>12}")
    log.info(f"{'B: Frequency (all)':<30} {B_avg:>10.5f} {B_avg-baseline_avg:>+12.5f}")
    log.info(f"{'C: Baseline+Freq':<30} {C_avg:>10.5f} {C_avg-baseline_avg:>+12.5f}")
    log.info(f"{'D: Temporal only':<30} {D_avg:>10.5f} {D_avg-baseline_avg:>+12.5f}")
    
    # Per-target deltas
    log.info(f"\nPer-target Δ from baseline:")
    for t in TARGETS:
        deltas = {'B': B_results[t]-baseline_results[t], 'C': C_results[t]-baseline_results[t], 'D': D_results[t]-baseline_results[t]}
        best = min(deltas.items(), key=lambda x: x[1])
        marker = " ★" if best[1] < -0.003 else ""
        log.info(f"  {t}: B={deltas['B']:+.5f} C={deltas['C']:+.5f} D={deltas['D']:+.5f} best={best[0]}{best[1]:+.5f}{marker}")

if __name__ == "__main__":
    main()
