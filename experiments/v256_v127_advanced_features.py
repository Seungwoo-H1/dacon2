"""
V256: Advanced Feature Discovery for V127

Experiment: Add frequency-domain, clustering, anomaly, temporal, interaction,
and routine features to the V127 baseline pipeline.

Hypothesis: Current features are time-based aggregation only. Additional signal
may exist in frequency domain, behavioral patterns, anomaly scores,
harmonic temporal features, cross-modal interactions, and routine regularity.

Pipeline: GroupKFold 5-fold × 5 seeds per target, OOF evaluation.
"""

import os, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT = ROOT / 'submissions'
for d in [EXPERIMENTS, SUBMIT]:
    d.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
SEEDS = [42, 7, 999, 777, 123]
N_FOLDS = 5

V53_SWEEP = {
    'Q1':  {'cfg': 'deep'},
    'Q2':  {'cfg': 'deep'},
    'Q3':  {'cfg': 'v48'},
    'S1':  {'cfg': 'wide'},
    'S2':  {'cfg': 'deep'},
    'S3':  {'cfg': 'safety'},
    'S4':  {'cfg': 'wide'},
}

CFGS = {
    'wide':   {'num_leaves':30,'max_depth':3,'learning_rate':0.05,'n_estimators':300,
               'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,
               'min_child_samples':5},
    'deep':   {'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,
               'min_child_samples':15},
    'v48':    {'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
               'min_child_samples':10},
    'safety': {'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':1000,
               'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':3.0,'reg_lambda':10.0,
               'min_child_samples':20},
}

LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    ex = META_COLS | set(TARGETS)
    cols = []
    for c in df.columns:
        if c in ex:
            continue
        dtype = df[c].dtype
        if dtype in [np.float64, np.int64, float, int, bool, np.bool_]:
            cols.append(c)
    return cols

def cfg_to_params(cfg_s, seed, spw):
    params = dict(cfg_s)
    params['scale_pos_weight'] = spw
    params['random_state'] = seed
    params['force_row_wise'] = True
    params['n_jobs'] = 1
    return params

def train_cv(feat, ftst, cols, y, seeds, cfg):
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(feat), dtype=np.float64)
    test_preds_list = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
        X_tr = feat.iloc[tr_idx][cols].values.astype(np.float64)
        y_tr = y[tr_idx].astype(np.float64)
        X_va = feat.iloc[va_idx][cols].values.astype(np.float64)
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)

        for seed in seeds:
            p = cfg_to_params(cfg, seed, spw)
            sn = [sanitize_col(c) for c in cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(p, ds)
            oof[va_idx] += m.predict(X_va) / len(seeds)

            Xt = ftst[cols].values.astype(np.float64)
            test_preds_list.append(m.predict(Xt))

    del X_tr, y_tr, X_va, ds
    gc.collect()
    return oof, np.mean(np.stack(test_preds_list), axis=0)


# ============================================================
# FEATURE EXTRACTORS
# ============================================================

def add_cyclic_frequency_features(df):
    """
    Feature Group 1: Cyclic/Frequency Features
    
    For each subject, use the numeric base features (non-zscore, non-hour, non-target, non-meta)
    to compute per-day frequency domain features from the time series.
    """
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    new_features = []
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    subjects = sorted(df['subject_id'].unique())
    all_new = []
    
    for subj in subjects:
        subj_df = df[df['subject_id'] == subj].sort_values('date').reset_index(drop=True)
        n_days = len(subj_df)
        
        if n_days < 7:
            continue
            
        for bcol in base_cols:
            if bcol not in subj_df.columns:
                continue
            vals = subj_df[bcol].values.astype(np.float64)
            # Replace NaN/inf for FFT
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip if constant
            if np.std(vals) < 1e-10:
                continue
                
            # FFT
            fft_vals = np.fft.rfft(vals)
            fft_magnitude = np.abs(fft_vals)
            fft_phase = np.angle(fft_vals)
            
            # Dominant frequency index (excluding DC)
            if len(fft_magnitude) > 1:
                dom_idx = np.argmax(fft_magnitude[1:]) + 1
            else:
                dom_idx = 0
            
            # Spectral power
            spectral_power = np.sum(fft_magnitude**2)
            
            # Spectral entropy
            power_norm = fft_magnitude / (np.sum(fft_magnitude) + 1e-10)
            spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            
            # Circadian periodicity: check strength of 1-cycle signal
            if len(fft_magnitude) > 1:
                circadian_power = fft_magnitude[1]  # first harmonic
                total_power = spectral_power + 1e-10
                circadian_ratio = circadian_power**2 / total_power
            else:
                circadian_ratio = 0.0
            
            new_feat_prefix = sanitize_col(bcol.replace('m_', '').replace('w_', ''))
            feat_name = f"freq_{new_feat_prefix}"
            
            new_features.append({
                'subject_id': subj,
                'date': subj_df['date'].iloc[0],
                f'{feat_name}_dom_freq_idx': dom_idx,
                f'{feat_name}_spectral_power': spectral_power,
                f'{feat_name}_spectral_entropy': spectral_entropy,
                f'{feat_name}_circadian_ratio': circadian_ratio,
            })
    
    if new_features:
        new_df = pd.DataFrame(new_features)
        # Expand single-row entries to all subject dates
        for subj in subjects:
            subj_dates = df[df['subject_id'] == subj]['date'].values
            subj_new = new_df[new_df['subject_id'] == subj]
            if len(subj_new) > 0:
                for col in subj_new.select_dtypes(include=[np.number]).columns:
                    df.loc[df['subject_id'] == subj, col] = subj_new[col].values[0]
                    new_features[-1].pop(col, None)
    
    return df


def add_day_of_year_harmonics(df):
    """
    Feature Group 4a: Day-of-year harmonic features (sin/cos with multiple harmonics)
    """
    df = df.copy()
    dates = pd.to_datetime(df['date'])
    doy = dates.dt.dayofyear.astype(float) / 365.0
    
    for harmonic in [1, 2, 3, 4]:
        df[f'doy_sin_h{harmonic}'] = np.sin(2 * np.pi * harmonic * doy)
        df[f'doy_cos_h{harmonic}'] = np.cos(2 * np.pi * harmonic * doy)
    
    return df


def add_ema_features(df, alphas=None):
    """
    Feature Group 4b: Exponential moving averages with multiple alpha values
    """
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    df = df.copy()
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    subjects = sorted(df['subject_id'].unique())
    
    for subj in subjects:
        mask = df['subject_id'] == subj
        subj_idx = df[mask].index
        if len(subj_idx) < 3:
            continue
        subj_df = df.loc[subj_idx].sort_values('date')
        
        for bcol in base_cols:
            if bcol not in subj_df.columns:
                continue
            vals = subj_df[bcol].values.astype(np.float64)
            vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
            
            for alpha in alphas:
                ema = np.empty_like(vals, dtype=np.float64)
                ema[0] = vals[0] if not np.isnan(vals[0]) else 0
                for i in range(1, len(vals)):
                    if np.isnan(vals[i]):
                        ema[i] = ema[i-1]
                    else:
                        ema[i] = alpha * vals[i] + (1 - alpha) * ema[i-1]
                
                feat_name = f"ema_{alpha}_{sanitize_col(bcol.replace('m_', '').replace('w_', ''))}"
                df.loc[subj_idx, feat_name] = ema
    
    return df


def add_hurst_exponent(df):
    """
    Feature Group 4c: Hurst exponent (long-term memory in time series)
    """
    df = df.copy()
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    subjects = sorted(df['subject_id'].unique())
    
    for subj in subjects:
        mask = df['subject_id'] == subj
        subj_idx = df[mask].index
        if len(subj_idx) < 10:
            df.loc[subj_idx, 'hurst_exponent'] = 0.5
            continue
        
        subj_df = df.loc[subj_idx].sort_values('date')
        
        # Compute Hurst exponent from a representative feature (total activity)
        hurst_vals = np.zeros(len(subj_idx))
        
        for i in range(len(subj_idx)):
            # Use activity + heart rate + steps for Hurst
            activity_val = 0
            if 'mActivity_m_activity_mean' in subj_df.columns:
                activity_val = subj_df['mActivity_m_activity_mean'].iloc[i]
            
            hr_val = 0
            if 'wHr_hr_mean' in subj_df.columns:
                hr_val = subj_df['wHr_hr_mean'].iloc[i]
            
            steps_val = 0
            if 'wPedo_pedo_step_mean' in subj_df.columns:
                steps_val = subj_df['wPedo_pedo_step_mean'].iloc[i]
            
            combined = np.array([activity_val, hr_val, steps_val])
            combined = np.nan_to_num(combined, nan=0.0)
            
            if np.std(combined) < 1e-10:
                hurst_vals[i] = 0.5
                continue
            
            # R/S analysis (simplified)
            n = len(combined)
            if n < 10:
                hurst_vals[i] = 0.5
                continue
            
            mean_val = np.mean(combined)
            deviations = combined - mean_val
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(combined) * np.sqrt(n)
            
            if S > 0:
                try:
                    hurst_vals[i] = np.log(R / S) / np.log(n)
                except:
                    hurst_vals[i] = 0.5
            else:
                hurst_vals[i] = 0.5
        
        df.loc[subj_idx, 'hurst_exponent'] = np.clip(hurst_vals, 0, 1)
    
    return df


def add_sample_entropy(df, m=2, r=0.2):
    """
    Feature Group 4d: Sample entropy (time series complexity)
    """
    df = df.copy()
    
    base_feature = 'mActivity_m_activity_mean'
    subjects = sorted(df['subject_id'].unique())
    
    for subj in subjects:
        mask = df['subject_id'] == subj
        subj_idx = df[mask].index
        subj_df = df.loc[subj_idx].sort_values('date')
        
        if base_feature not in subj_df.columns or len(subj_idx) < m + 2:
            df.loc[subj_idx, 'sample_entropy'] = 1.0
            continue
        
        vals = subj_df[base_feature].values.astype(np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        
        se = np.zeros(len(vals))
        for i in range(len(vals)):
            seg = vals[max(0,i-2):min(len(vals),i+3)] if len(vals) >= 3 else vals
            if len(seg) < m + 2:
                se[i] = 1.0
                continue
            
            try:
                # Simplified sample entropy calculation
                n = len(seg)
                m_bundles = []
                for j in range(n - m + 1):
                    m_bundles.append(seg[j:j+m])
                
                # Count matches
                counts = np.zeros(len(m_bundles))
                for j in range(len(m_bundles)):
                    for k in range(j + 1, len(m_bundles)):
                        dist = np.max(np.abs(m_bundles[j] - m_bundles[k]))
                        if dist < r * np.std(seg) + 1e-10:
                            counts[j] += 1
                
                phi_m = np.mean(counts) / (len(m_bundles) - 1)
                
                # m+1 bundles
                m1_bundles = []
                for j in range(n - m):
                    m1_bundles.append(seg[j:j+m+1])
                
                counts1 = np.zeros(len(m1_bundles))
                for j in range(len(m1_bundles)):
                    for k in range(j + 1, len(m1_bundles)):
                        dist = np.max(np.abs(m1_bundles[j] - m1_bundles[k]))
                        if dist < r * np.std(seg) + 1e-10:
                            counts1[j] += 1
                
                phi_m1 = np.mean(counts1) / (len(m1_bundles) - 1)
                
                if phi_m > 0 and phi_m1 > 0:
                    se[i] = -np.log(phi_m1 / phi_m)
                else:
                    se[i] = 1.0
            except:
                se[i] = 1.0
        
        se = np.clip(se, 0, 10)
        # Normalize per subject
        se_mean = np.nanmean(se) if np.any(~np.isnan(se)) else 1.0
        se_mean = min(se_mean, 5.0)  # cap
        
        df.loc[subj_idx, 'sample_entropy'] = se_mean
    
    return df


def add_clustering_features(df):
    """
    Feature Group 2: Clustering-Based Behavioral Embeddings
    
    Create per-user feature vectors from aggregated base features,
    cluster users, and add cluster labels + distances as features.
    """
    df = df.copy()
    
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    
    # Aggregate per-user: mean of each base feature across all days
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    # Also include some std features
    std_cols = [c for c in df.columns if c.startswith(('mActivity','wPedo','wHr','mLight')) 
                and '_std' in c and '_zscore' not in c and '_hour' not in c]
    base_cols.extend(std_cols)
    
    base_cols = list(set(base_cols))
    
    # Per-user aggregation
    user_feats = df.groupby('subject_id')[base_cols].agg(['mean', 'std', 'min', 'max'])
    user_feats.columns = ['_'.join(c) for c in user_feats.columns]
    user_feats = user_feats.reset_index()
    
    # Fill NaN
    user_feats = user_feats.fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_feats.drop('subject_id', axis=1))
    
    # KMeans with various k values
    for k in [3, 5, 7]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        distances = km.transform(X_scaled)
        
        user_feats[f'cluster_k{k}'] = labels
        
        # Add distance to each cluster centroid as features
        for c in range(k):
            user_feats[f'cluster_k{k}_dist_{c}'] = distances[:, c]
    
    # DBSCAN for density-based clustering
    db = DBSCAN(eps=1.5, min_samples=2)
    db_labels = db.fit_predict(X_scaled)
    user_feats['cluster_dbscan'] = db_labels
    
    # Merge back to main df
    df = df.merge(user_feats[['subject_id', 'cluster_k3', 'cluster_k5', 'cluster_k7', 'cluster_dbscan']], 
                  on='subject_id', how='left')
    
    # Add cluster distances
    for k in [3, 5, 7]:
        for c in range(k):
            col_name = f'cluster_k{k}_dist_{c}'
            if col_name in user_feats.columns:
                df = df.merge(user_feats[['subject_id', col_name]], on='subject_id', how='left')
    
    return df


def add_anomaly_reconstruction_features(df):
    """
    Feature Group 3: Anomaly/Reconstruction Features
    
    Simple autoencoder-like approach using PCA reconstruction error
    as anomaly score per day.
    """
    df = df.copy()
    
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    # Include some std features too
    std_cols = [c for c in df.columns if c.startswith(('mActivity','wPedo','wHr','mLight','mBle','mWifi')) 
                and '_std' in c and '_zscore' not in c and '_hour' not in c]
    base_cols.extend(std_cols)
    base_cols = list(set(base_cols))
    
    # Replace NaN with 0 for PCA
    X = df[base_cols].fillna(0).values.astype(np.float64)
    
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use PCA with 10 components
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Reconstruction error per sample
    recon_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    
    # Also compute per-subject reconstruction error
    recon_error_zscore = (recon_error - np.mean(recon_error)) / (np.std(recon_error) + 1e-10)
    
    df['reconstruction_error'] = recon_error
    df['reconstruction_error_zscore'] = recon_error_zscore
    
    # Per-subject mean reconstruction error (global anomaly indicator)
    subj_means = df.groupby('subject_id')['reconstruction_error'].transform('mean')
    df['reconstruction_error_subject_mean'] = subj_means
    
    # Per-subject std of reconstruction error (consistency of anomaly pattern)
    subj_stds = df.groupby('subject_id')['reconstruction_error'].transform('std')
    df['reconstruction_error_subject_std'] = subj_stds
    
    return df


def add_interaction_features(df):
    """
    Feature Group 5: Cross-Modal Interactions
    
    Non-linear interaction features between different modalities.
    """
    df = df.copy()
    
    # 1) activity × heart_rate interaction
    if 'mActivity_m_activity_mean' in df.columns and 'wHr_hr_mean' in df.columns:
        df['interaction_activity_hr'] = df['mActivity_m_activity_mean'] * df['wHr_hr_mean']
        df['interaction_activity_hr_ratio'] = df['mActivity_m_activity_mean'] / (df['wHr_hr_mean'] + 1e-10)
    
    # 2) GPS mobility × ambience correlation proxies
    # GPS mean speed × ambience noise ratio
    gps_cols = [c for c in df.columns if c.startswith('mGps_') and 'avg_speed_mean' in c and '_zscore' not in c]
    ambience_cols = [c for c in df.columns if c.startswith('mAmbience_') and ('speech_sum' in c or 'vehicle_sum' in c) and '_zscore' not in c]
    
    for gc in gps_cols:
        for ac in ambience_cols:
            short_g = gc.replace('mGps_', '').replace('_mean', '')
            short_a = ac.replace('mAmbience_', '').replace('_sum', '')
            df[f'interaction_gps_mob_{short_g}_amb_{short_a}'] = df[gc] * df[ac]
    
    # 3) Screen usage × WiFi connection state
    if 'mScreenStatus_m_screen_use_mean' in df.columns and 'mWifi_wifi_strong_ratio_mean' in df.columns:
        df['interaction_screen_wifi'] = df['mScreenStatus_m_screen_use_mean'] * df['mWifi_wifi_strong_ratio_mean']
    
    if 'mScreenStatus_m_screen_use_mean' in df.columns and 'mACStatus_m_charging_mean' in df.columns:
        df['interaction_screen_charging'] = df['mScreenStatus_m_screen_use_mean'] * df['mACStatus_m_charging_mean']
    
    # 4) BLE device diversity × GPS mobility
    ble_cols = [c for c in df.columns if c.startswith('mBle_') and 'device_count_mean' in c and '_zscore' not in c]
    for bc in ble_cols:
        short_b = bc.replace('mBle_', '').replace('_mean', '')
        if 'mGps_gps_avg_speed_mean' in df.columns:
            df[f'interaction_ble_{short_b}_gps_speed'] = df[bc] * df['mGps_gps_avg_speed_mean']
    
    # 5) Usage stats interactions
    if 'mUsageStats_usage_total_time_mean' in df.columns and 'mActivity_m_activity_mean' in df.columns:
        df['interaction_usage_activity'] = df['mUsageStats_usage_total_time_mean'] * df['mActivity_m_activity_mean']
    
    if 'mUsageStats_usage_game_ratio_mean' in df.columns and 'wHr_hr_mean' in df.columns:
        df['interaction_game_hr'] = df['mUsageStats_usage_game_ratio_mean'] * df['wHr_hr_mean']
    
    return df


def add_routine_regularity_features(df):
    """
    Feature Group 6: User Routine Regularity Score
    
    - std of daily activity hours
    - consistency of sleep timing (proxy: activity onset patterns)
    - routine predictability index
    """
    df = df.copy()
    
    feature_groups = ['mACStatus','mActivity','mLight','mScreenStatus','wLight','wPedo',
                       'mAmbience','mBle','mGps','mUsageStats','mWifi','wHr']
    
    base_cols = []
    for g in feature_groups:
        gcols = [c for c in df.columns if c.startswith(g + '_') and '_mean' in c and '_zscore' not in c and '_hour' not in c]
        base_cols.extend(gcols)
    
    std_cols = [c for c in df.columns if c.startswith(('mActivity','wPedo','wHr','mLight','mBle','mWifi')) 
                and '_std' in c and '_zscore' not in c and '_hour' not in c]
    base_cols.extend(std_cols)
    base_cols = list(set(base_cols))
    
    subjects = sorted(df['subject_id'].unique())
    
    for subj in subjects:
        mask = df['subject_id'] == subj
        subj_idx = df[mask].index
        
        # Activity std across days (lower = more routine)
        if 'mActivity_m_activity_std' in df.columns:
            act_stds = df.loc[subj_idx, 'mActivity_m_activity_std'].values
            df.loc[subj_idx, 'routine_activity_std'] = np.mean(act_stds)
            
            # Regularity score: 1 / (1 + std) — higher = more routine
            df.loc[subj_idx, 'routine_activity_regularity'] = 1.0 / (1.0 + np.mean(act_stds))
        
        # Step count consistency
        if 'wPedo_pedo_step_mean' in df.columns:
            steps = df.loc[subj_idx, 'wPedo_pedo_step_mean'].values
            step_cv = np.std(steps) / (np.mean(steps) + 1e-10)
            df.loc[subj_idx, 'routine_step_cv'] = step_cv
            df.loc[subj_idx, 'routine_step_consistency'] = 1.0 / (1.0 + step_cv)
        
        # Screen use regularity
        if 'mScreenStatus_m_screen_use_std' in df.columns:
            scr_stds = df.loc[subj_idx, 'mScreenStatus_m_screen_use_std'].values
            df.loc[subj_idx, 'routine_screen_regularity'] = 1.0 / (1.0 + np.mean(scr_stds))
        
        # HR consistency (sleep/wake regularity proxy)
        if 'wHr_hr_std' in df.columns:
            hr_stds = df.loc[subj_idx, 'wHr_hr_std'].values
            df.loc[subj_idx, 'routine_hr_regularity'] = 1.0 / (1.0 + np.mean(hr_stds))
        
        # Overall routine predictability: geometric mean of individual regularities
        reg_cols = ['routine_activity_regularity', 'routine_step_consistency', 
                     'routine_screen_regularity', 'routine_hr_regularity']
        valid_regs = [df.loc[subj_idx, c].values[0] for c in reg_cols if c in df.columns and df.loc[subj_idx, c].notna().any()]
        if valid_regs:
            predictability = np.prod(valid_regs) ** (1.0 / len(valid_regs))
            df.loc[subj_idx, 'routine_predictability_index'] = predictability
    
    # Fill any NaN
    reg_cols_fill = ['routine_activity_std', 'routine_activity_regularity', 'routine_step_cv',
                      'routine_step_consistency', 'routine_screen_regularity', 'routine_hr_regularity',
                      'routine_predictability_index']
    for c in reg_cols_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0.5)
    
    return df



# ============================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================

def run_feature_experiment(feature_addon_fn, feature_set_name, feat_df):
    """
    Train models with a specific feature set using GroupKFold 5-fold × 5 seeds.
    Returns per-target OOF log_loss and AVG OOF.
    """
    print(f"\n{'='*60}")
    print(f"Running: {feature_set_name}")
    print(f"{'='*60}")
    
    # Apply feature addon
    df_exp = feature_addon_fn(feat_df.copy())
    
    feature_cols = get_feature_cols(df_exp)
    
    # Remove leak columns
    leak_all = LEAK_S | LEAK_Q
    
    # Rank features per target and take top N
    all_ranks = {}
    for target in TARGETS:
        y = df_exp[target].values.astype(np.float64)
        X = df_exp[feature_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        cfg_name = V53_SWEEP[target]['cfg']
        base = CFGS[cfg_name]
        
        params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
                   'random_state': 42, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feature_cols]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        m = lgb.train(params, ds, num_boost_round=50)
        imp = m.feature_importance(importance_type='gain')
        ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
        all_ranks[target] = [r[0] for r in ranked]
        del m, ds, X
        gc.collect()
    
    # Use top 200 features (plenty for advanced features)
    n_feat = 200
    
    results = {}
    for target in TARGETS:
        print(f"\n  Target: {target}")
        y = df_exp[target].values.astype(np.float64)
        ranked_cols = all_ranks[target]
        
        # Add leak-safe filtering per target
        safe_cols = [c for c in ranked_cols[:n_feat] if c not in leak_all]
        if len(safe_cols) < 10:
            safe_cols = ranked_cols[:n_feat]
        
        print(f"    Features used: {len(safe_cols)}")
        
        oof = np.zeros(len(df_exp), dtype=np.float64)
        test_preds_list = []
        group = df_exp['subject_id'].values
        gkf = GroupKFold(n_splits=N_FOLDS)
        
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(df_exp, y, group)):
            X_tr = df_exp.iloc[tr_idx][safe_cols].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx].astype(np.float64)
            X_va = df_exp.iloc[va_idx][safe_cols].fillna(0).values.astype(np.float64)
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            
            for seed in SEEDS:
                p = cfg_to_params(CFGS[V53_SWEEP[target]['cfg']], seed, spw)
                sn = [sanitize_col(c) for c in safe_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(p, ds)
                oof[va_idx] += m.predict(X_va) / len(SEEDS)
                
                Xt = df_exp[safe_cols].fillna(0).values.astype(np.float64)
                test_preds_list.append(m.predict(Xt))
        
            del X_tr, y_tr, X_va, ds
            gc.collect()
        
        oof_clipped = np.clip(oof, 0.0001, 0.9999)
        test_mean = np.mean(np.stack(test_preds_list), axis=0)
        
        ll = log_loss(y, oof_clipped)
        results[target] = {
            'oof_logloss': ll,
            'n_features': len(safe_cols),
            'feature_names': safe_cols[:20],  # top 20
        }
        
        print(f"    OOF LogLoss: {ll:.6f}")
    
    avg_oof = np.mean([r['oof_logloss'] for r in results.values()])
    print(f"\n  AVG OOF LogLoss: {avg_oof:.6f}")
    
    return results, avg_oof, df_exp


def main():
    start_time = time.time()
    print(f"V256: Advanced Feature Discovery — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: features_clean_v60.parquet")
    
    # Load data
    df = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f"Loaded: {df.shape}")
    
    all_experiment_results = {}
    all_feature_sets = {}
    
    # ============================================================
    # EXPERIMENT A: Baseline V127 (no advanced features)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT A: Baseline V127 (reference)")
    print(f"{'='*70}")
    
    results_a, avg_a, df_a = run_feature_experiment(
        lambda x: x, 
        'A_Baseline_V127', 
        df
    )
    all_experiment_results['A_Baseline_V127'] = results_a
    all_experiment_results['A_Baseline_V127']['_avg_oof'] = avg_a
    all_feature_sets['A_Baseline_V127'] = df_a
    all_feature_sets['A_Baseline_V127']['_n_features'] = len(get_feature_cols(df))
    
    # ============================================================
    # EXPERIMENT B: Cyclic/Frequency Features only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT B: Cyclic/Frequency Features")
    print(f"{'='*70}")
    
    results_b, avg_b, df_b = run_feature_experiment(
        add_cyclic_frequency_features,
        'B_Frequency_Domain',
        df
    )
    all_experiment_results['B_Frequency_Domain'] = results_b
    all_experiment_results['B_Frequency_Domain']['_avg_oof'] = avg_b
    all_feature_sets['B_Frequency_Domain'] = df_b
    all_feature_sets['B_Frequency_Domain']['_n_features'] = len(get_feature_cols(df_b))
    
    # ============================================================
    # EXPERIMENT C: Day-of-Year Harmonics only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT C: Day-of-Year Harmonic Features")
    print(f"{'='*70}")
    
    results_c, avg_c, df_c = run_feature_experiment(
        add_day_of_year_harmonics,
        'C_DOY_Harmonics',
        df
    )
    all_experiment_results['C_DOY_Harmonics'] = results_c
    all_experiment_results['C_DOY_Harmonics']['_avg_oof'] = avg_c
    all_feature_sets['C_DOY_Harmonics'] = df_c
    all_feature_sets['C_DOY_Harmonics']['_n_features'] = len(get_feature_cols(df_c))
    
    # ============================================================
    # EXPERIMENT D: EMA Features only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT D: Exponential Moving Average Features")
    print(f"{'='*70}")
    
    results_d, avg_d, df_d = run_feature_experiment(
        add_ema_features,
        'D_EMA_Features',
        df
    )
    all_experiment_results['D_EMA_Features'] = results_d
    all_experiment_results['D_EMA_Features']['_avg_oof'] = avg_d
    all_feature_sets['D_EMA_Features'] = df_d
    all_feature_sets['D_EMA_Features']['_n_features'] = len(get_feature_cols(df_d))
    
    # ============================================================
    # EXPERIMENT E: Anomaly/Reconstruction Features only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT E: Anomaly/Reconstruction Features (PCA)")
    print(f"{'='*70}")
    
    results_e, avg_e, df_e = run_feature_experiment(
        add_anomaly_reconstruction_features,
        'E_Anomaly_Reconstruction',
        df
    )
    all_experiment_results['E_Anomaly_Reconstruction'] = results_e
    all_experiment_results['E_Anomaly_Reconstruction']['_avg_oof'] = avg_e
    all_feature_sets['E_Anomaly_Reconstruction'] = df_e
    all_feature_sets['E_Anomaly_Reconstruction']['_n_features'] = len(get_feature_cols(df_e))
    
    # ============================================================
    # EXPERIMENT F: Interaction Features only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT F: Cross-Modal Interaction Features")
    print(f"{'='*70}")
    
    results_f, avg_f, df_f = run_feature_experiment(
        add_interaction_features,
        'F_Interaction_Features',
        df
    )
    all_experiment_results['F_Interaction_Features'] = results_f
    all_experiment_results['F_Interaction_Features']['_avg_oof'] = avg_f
    all_feature_sets['F_Interaction_Features'] = df_f
    all_feature_sets['F_Interaction_Features']['_n_features'] = len(get_feature_cols(df_f))
    
    # ============================================================
    # EXPERIMENT G: Routine Regularity Features only
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT G: Routine Regularity Features")
    print(f"{'='*70}")
    
    results_g, avg_g, df_g = run_feature_experiment(
        add_routine_regularity_features,
        'G_Routine_Regularity',
        df
    )
    all_experiment_results['G_Routine_Regularity'] = results_g
    all_experiment_results['G_Routine_Regularity']['_avg_oof'] = avg_g
    all_feature_sets['G_Routine_Regularity'] = df_g
    all_feature_sets['G_Routine_Regularity']['_n_features'] = len(get_feature_cols(df_g))
    
    # ============================================================
    # EXPERIMENT H: Combined (B+C+D+E+F+G) - all advanced features
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT H: Combined Advanced Features (all)")
    print(f"{'='*70}")
    
    def combined_addon(x):
        df2 = add_day_of_year_harmonics(x)
        df2 = add_ema_features(df2)
        df2 = add_anomaly_reconstruction_features(df2)
        df2 = add_interaction_features(df2)
        df2 = add_routine_regularity_features(df2)
        return df2
    
    results_h, avg_h, df_h = run_feature_experiment(
        combined_addon,
        'H_Combined_Advanced',
        df
    )
    all_experiment_results['H_Combined_Advanced'] = results_h
    all_experiment_results['H_Combined_Advanced']['_avg_oof'] = avg_h
    all_feature_sets['H_Combined_Advanced'] = df_h
    all_feature_sets['H_Combined_Advanced']['_n_features'] = len(get_feature_cols(df_h))
    
    # ============================================================
    # EXPERIMENT I: Frequency + EMA (subset combo)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT I: Frequency + EMA + Harmonics")
    print(f"{'='*70}")
    
    def combo_addon(x):
        df2 = add_cyclic_frequency_features(x)
        df2 = add_day_of_year_harmonics(df2)
        df2 = add_ema_features(df2)
        return df2
    
    results_i, avg_i, df_i = run_feature_experiment(
        combo_addon,
        'I_Freq_EMA_DOY',
        df
    )
    all_experiment_results['I_Freq_EMA_DOY'] = results_i
    all_experiment_results['I_Freq_EMA_DOY']['_avg_oof'] = avg_i
    all_feature_sets['I_Freq_EMA_DOY'] = df_i
    all_feature_sets['I_Freq_EMA_DOY']['_n_features'] = len(get_feature_cols(df_i))
    
    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    summary = {}
    for exp_name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                      'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                      'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
        res = all_experiment_results[exp_name]
        avg = res['_avg_oof']
        nf = int(all_feature_sets[exp_name]['_n_features'])
        summary[exp_name] = {
            'avg_oof': round(avg, 6),
            'n_features': nf,
            'per_target': {t: round(res[t]['oof_logloss'], 6) for t in TARGETS},
        }
    
    # Print summary table
    print(f"\n{'Experiment':<30} {'AVG OOF':>10} {'Δ vs Baseline':>14} {'N Features':>12}")
    print("-" * 70)
    
    baseline_avg = summary['A_Baseline_V127']['avg_oof']
    for name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                  'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                  'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
        s = summary[name]
        delta = s['avg_oof'] - baseline_avg
        marker = "  ★★★" if delta < -0.005 else ("  ★★" if delta < -0.002 else ("  ★" if delta < 0 else ""))
        print(f"{name:<30} {s['avg_oof']:>10.6f} {delta:>+14.6f} {int(s['n_features']):>12} {marker}")
    
    # Per-target detailed table
    print(f"\n\nPer-Target OOF LogLoss:")
    print(f"{'Target':<8}", end="")
    for name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                  'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                  'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
        print(f" {name[:15]:>15}", end="")
    print()
    print("-" * (8 + 16 * 9))
    
    for t in TARGETS:
        print(f"{t:<8}", end="")
        for name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                      'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                      'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
            val = summary[name]['per_target'][t]
            print(f" {val:>15.6f}", end="")
        print()
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save results
    result_file = EXPERIMENTS / 'v127_advanced_feature_result.md'
    with open(result_file, 'w') as f:
        f.write(f"# DACon2 V127 개선 실험 #6: Advanced Feature Discovery\n\n")
        f.write(f"**실행일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST  \n")
        f.write(f"**스크립트**: experiments/v256_v127_advanced_features.py  \n")
        f.write(f"**데이터**: features_clean_v60.parquet (450 rows, 278 features, 10 subjects, 7 targets)\n\n")
        
        f.write("## 결론\n\n")
        
        # Find best
        best_name = min(summary.keys(), key=lambda k: summary[k]['avg_oof'])
        best_avg = summary[best_name]['avg_oof']
        best_delta = best_avg - baseline_avg
        f.write(f"**최적 실험**: {best_name}\n")
        f.write(f"- AVG OOF: **{best_avg:.6f}**\n")
        f.write(f"- Δ vs Baseline: **{best_delta:+.6f}**\n\n")
        
        if best_delta < -0.005:
            f.write("✅ **유의미한 개선** — advanced features가 signal 추가에 성공\n")
        elif best_delta < 0:
            f.write("⚠️ **미미한 개선** — advanced features가 일부 signal 추가 (의미 있을 수 있음)\n")
        else:
            f.write("❌ **개선 없음** — advanced features가 noise 추가 (baseline 유지 권장)\n\n")
        
        f.write("## 실험 구성\n\n")
        f.write("| # | 실험 | 특징 | AVG OOF | Δ vs Baseline |\n")
        f.write("|---|------|------|---------|---------------|\n")
        for name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                      'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                      'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
            s = summary[name]
            delta = s['avg_oof'] - baseline_avg
            marker = " ★★★" if delta < -0.005 else (" ★★" if delta < -0.002 else (" ★" if delta < 0 else ""))
            f.write(f"| {'1' if name=='A_Baseline_V127' else '2':<2} | **{name}** | {name.split('_')[1:] + name.split('_')[2:] if '_' in name else name} | {s['avg_oof']:.6f} | {delta:+.6f} | {marker} |\n")
        
        f.write("\n## Per-Target 상세 OOF\n\n")
        f.write("| Target | Baseline | Freq | DOY | EMA | Anomaly | Interact | Routine | Combined | Freq+EMA+DOY |\n")
        f.write("|--------|----------|------|-----|-----|---------|----------|---------|----------|---------------|\n")
        for t in TARGETS:
            vals = [summary[n]['per_target'][t] for n in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']]
            deltas = [f"{v - vals[0]:+.4f}" for v in vals]
            f.write(f"| {t} | {vals[0]:.6f} | {deltas[1]} | {deltas[2]} | {deltas[3]} | {deltas[4]} | {deltas[5]} | {deltas[6]} | {deltas[7]} | {deltas[8]} |\n")
        
        f.write(f"\n## 추가된 Feature 수\n\n")
        f.write("| 실험 | Total Features | Added |\n")
        f.write("|------|---------------|-------|\n")
        for name in ['A_Baseline_V127', 'B_Frequency_Domain', 'C_DOY_Harmonics',
                      'D_EMA_Features', 'E_Anomaly_Reconstruction', 'F_Interaction_Features',
                      'G_Routine_Regularity', 'H_Combined_Advanced', 'I_Freq_EMA_DOY']:
            nf = summary[name]['n_features']
            added = int(nf) - int(summary['A_Baseline_V127']['n_features'])
            f.write(f"| {name} | {nf} | +{added} |\n")
        
        f.write(f"\n## 시간\n\n총 실행 시간: {elapsed:.1f}초\n\n")
        f.write(f"## 발견사항\n\n")
        f.write("### Frequency Features (B)\n")
        f.write(f"- AVG OOF: {summary['B_Frequency_Domain']['avg_oof']:.6f} (Δ={summary['B_Frequency_Domain']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- Spectral entropy, circadian ratio, dominant frequency extracted per base feature\n\n")
        f.write("### DOY Harmonics (C)\n")
        f.write(f"- AVG OOF: {summary['C_DOY_Harmonics']['avg_oof']:.6f} (Δ={summary['C_DOY_Harmonics']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- sin/cos with harmonics 1-4 added (8 new features)\n\n")
        f.write("### EMA Features (D)\n")
        f.write(f"- AVG OOF: {summary['D_EMA_Features']['avg_oof']:.6f} (Δ={summary['D_EMA_Features']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- EMA with alpha=0.1,0.3,0.5,0.7,0.9 for each base feature\n\n")
        f.write("### Anomaly/Reconstruction (E)\n")
        f.write(f"- AVG OOF: {summary['E_Anomaly_Reconstruction']['avg_oof']:.6f} (Δ={summary['E_Anomaly_Reconstruction']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- PCA reconstruction error, subject-mean/std added\n\n")
        f.write("### Interaction Features (F)\n")
        f.write(f"- AVG OOF: {summary['F_Interaction_Features']['avg_oof']:.6f} (Δ={summary['F_Interaction_Features']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- Cross-modal interactions: activity×HR, screen×WiFi, GPS×ambience, etc.\n\n")
        f.write("### Routine Regularity (G)\n")
        f.write(f"- AVG OOF: {summary['G_Routine_Regularity']['avg_oof']:.6f} (Δ={summary['G_Routine_Regularity']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- Per-user regularity scores from activity, step, screen, HR consistency\n\n")
        f.write("### Combined Advanced (H)\n")
        f.write(f"- AVG OOF: {summary['H_Combined_Advanced']['avg_oof']:.6f} (Δ={summary['H_Combined_Advanced']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- All advanced features combined\n\n")
        f.write("### Freq+EMA+DOY (I)\n")
        f.write(f"- AVG OOF: {summary['I_Freq_EMA_DOY']['avg_oof']:.6f} (Δ={summary['I_Freq_EMA_DOY']['avg_oof']-baseline_avg:+.6f})\n")
        f.write(f"- Temporal frequency combo\n")
        
        f.write(f"\n---\n")
        f.write(f"**결과 파일**: {result_file}\n")
    
    print(f"\nResults saved to: {result_file}")
    
    # Save JSON for machine parsing
    json_file = EXPERIMENTS / f'v256_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved to: {json_file}")


if __name__ == '__main__':
    main()
