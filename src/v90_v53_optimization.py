"""
V90: Comprehensive V53 Optimization
Systematically tests multiple combinations of:
1. Seed count (30, 50, 100)
2. Feature engineering (base only, +zscore, +rolling, +interaction)
3. Hyperparameter sweep per cfg (deep, wide, safety, v48)
4. Ensemble (V53 + V10 blend)
5. Calibration (mean-matching vs isotonic)
6. n_feat sweep (+-3 from baseline)

Uses GroupKFold for OOF evaluation. Best OOF config → leaderboard submission.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max','mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std',
    'mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std',
    'mWifi_wifi_bssid_count_max',
]

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':  {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

# CFG param sweep candidates
CFG_SWEEP = {
    'deep': [
        {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
        {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 800, 'ss': 0.8, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'nl': 25, 'md': 6, 'lr': 0.01, 'ne': 1500, 'ss': 0.6, 'cb': 0.5, 'ra': 0.3, 'rl': 1.0, 'mc': 20},
        {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 500, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    ],
    'wide': [
        {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
        {'nl': 40, 'md': 4, 'lr': 0.03, 'ne': 400, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'nl': 20, 'md': 3, 'lr': 0.05, 'ne': 200, 'ss': 0.9, 'cb': 0.9, 'ra': 3.0, 'rl': 7.0, 'mc': 3},
    ],
    'safety': [
        {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
        {'nl': 15, 'md': 3, 'lr': 0.03, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 2.0, 'rl': 7.0, 'mc': 15},
        {'nl': 8, 'md': 2, 'lr': 0.01, 'ne': 1500, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 15.0, 'mc': 30},
    ],
    'v48': [
        {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        {'nl': 10, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.8, 'cb': 0.8, 'ra': 1.5, 'rl': 4.0, 'mc': 8},
        {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 600, 'ss': 0.6, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    ],
}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    """Add subject-level zscore features."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols

def add_rolling(df, cols):
    """Add rolling mean/std features."""
    df = df.copy().sort_values(['subject_id', 'date'])
    new_cols = []
    for c in cols:
        g = df.groupby('subject_id')[c]
        for w in [3, 7]:
            rm = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = g.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            df[f'{c}_rm{w}'] = rm.values
            df[f'{c}_rs{w}'] = rs.values
            new_cols.extend([f'{c}_rm{w}', f'{c}_rs{w}'])
    return df, new_cols

def add_interaction(df, top_cols):
    """Add pairwise interaction features for top columns."""
    df = df.copy()
    new_cols = []
    # Only add interactions for top 10 cols to avoid explosion
    n_top = min(10, len(top_cols))
    for i in range(n_top):
        for j in range(i+1, n_top):
            c1, c2 = top_cols[i], top_cols[j]
            if c1 in df.columns and c2 in df.columns:
                df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                new_cols.append(f'{c1}_x_{c2}')
    return df, new_cols

def mean_matching(p, ref_mean):
    """Simple mean-matching calibration."""
    return np.clip(p + (ref_mean - p.mean()), 0.0001, 0.9999)

def isotonic_calibrate(oof_preds, train_preds, y_train):
    """Isotonic regression calibration."""
    iso = IsotonicRegression(y_min=0.0001, y_max=0.9999, out_of_bounds='clip')
    iso.fit(train_preds, y_train)
    return iso

def train_and_predict_oof(train_feat, test_feat, feat_cols, target, cfg_name, n_seeds, feat_type='zscore',
                          n_feat=20, cfg_override=None, cal_type='mean_match'):
    """Train with GroupKFold OOF, return OOF predictions and CV log_loss."""
    X_all = train_feat[feat_cols].fillna(0).values.astype(np.float64)
    y = train_feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    sn = [sanitize(c) for c in feat_cols]

    # Get base config
    cfg = cfg_override or CFGS.get(cfg_name, CFGS['deep'])
    n_trees = cfg['ne']

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    oof_preds = np.zeros(len(y))
    all_train_preds = []
    all_oof_for_cal = []

    for si, s in enumerate(range(42, 42 + n_seeds)):
        for fold_i, (tr_i, va_i) in enumerate(gkf.split(X_all, y, train_feat['subject_id'])):
            X_tr = X_all[tr_i]
            X_va = X_all[va_i]
            y_tr = y[tr_i]

            cfg_seed = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'force_row_wise': True, 'n_jobs': 1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': n_trees,
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'], 'random_state': s,
                'scale_pos_weight': spw,
            }
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds_tr, num_boost_round=n_trees)
            oof_preds[va_i] += m.predict(X_va)

    # Average seeds
    n_total = n_seeds * 5  # seeds * folds
    oof_preds /= n_total

    # CV log_loss
    cv_loss = log_loss(y, oof_preds, labels=[0, 1])
    return oof_preds, cv_loss


def test_config_combination(train, test, feat_cols, target, experiment_name,
                            n_seeds=50, feat_type='zscore', n_feat=20,
                            cfg_name='deep', cfg_override=None, cal_type='mean_match',
                            add_rolling_feat=False, add_interaction_feat=False,
                            train_on_all=True):
    """Test a full config combination and return results."""
    t0 = time.time()

    # Feature engineering (train/test already personalized by run_experiments)
    train_f = train.copy()
    test_f = test.copy()
    all_cols = list(feat_cols)

    # Remove _rm* / _rs* / _x_ from all_cols for base extraction
    base_cols = [c for c in all_cols if not c.endswith('_zscore') and not c.endswith('_rm*') and not c.endswith('_rs*') and '_x_' not in c]
    # Remove agg helper columns that leaked in
    agg_cols = [f'{c}_subj_mean' for c in base_cols] + [f'{c}_subj_std' for c in base_cols]
    all_cols_filtered = [c for c in all_cols if c not in agg_cols]

    # Add rolling
    if add_rolling_feat:
        rolling_cols = [c for c in all_cols_filtered if not c.endswith('_rm3') and not c.endswith('_rs3') and not c.endswith('_rm7') and not c.endswith('_rs7')]
        train_f, added_rolling = add_rolling(train_f, rolling_cols)
        test_f, _ = add_rolling(test_f, rolling_cols)
        all_cols_filtered = all_cols_filtered + added_rolling

    # Add interaction
    if add_interaction_feat:
        train_f, added_inter = add_interaction(train_f, all_cols_filtered[:10])
        test_f, _ = add_interaction(test_f, all_cols_filtered[:10])
        all_cols_filtered = all_cols_filtered + added_inter

    train_f = train_f.fillna(0)
    test_f = test_f.fillna(0)

    # Remove leak
    all_available = remove_leak(all_cols_filtered, target)
    y = train_f[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    sn = [sanitize(c) for c in all_available]

    # Feature ranking (1 seed)
    X_all = train_f[all_available].fillna(0).values.astype(np.float64)
    base_cfg = cfg_override or CFGS['deep']
    p_rank = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': '-1',
        'n_estimators': min(base_cfg['ne'], 50),
        'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
        'learning_rate': base_cfg['lr'],
        'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
        'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
        'min_child_samples': base_cfg['mc'],
        'scale_pos_weight': spw, 'random_state': 42, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
    m_rank = lgb.train(p_rank, ds, num_boost_round=p_rank['n_estimators'])
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_available, imp), key=lambda x: -x[1])
    sel_cols = [r[0] for r in ranked[:n_feat]]

    # OOF training
    X = train_f[sel_cols].fillna(0).values.astype(np.float64)
    Xt = test_f[sel_cols].fillna(0).values.astype(np.float64)
    sn_sel = [sanitize(c) for c in sel_cols]

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    oof_preds = np.zeros(len(y))
    cfg_final = cfg_override or CFGS[cfg_name]

    n_total = n_seeds * 5

    for si, s in enumerate(range(42, 42 + n_seeds)):
        for fold_i, (tr_i, va_i) in enumerate(gkf.split(X, y, train_f['subject_id'])):
            ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel, params={'verbose': '-1'})
            cfg_seed = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'force_row_wise': True, 'n_jobs': 1,
                'num_leaves': cfg_final['nl'], 'max_depth': cfg_final['md'],
                'learning_rate': cfg_final['lr'], 'n_estimators': cfg_final['ne'],
                'subsample': cfg_final['ss'], 'colsample_bytree': cfg_final['cb'],
                'reg_alpha': cfg_final['ra'], 'reg_lambda': cfg_final['rl'],
                'min_child_samples': cfg_final['mc'], 'random_state': s,
                'scale_pos_weight': spw,
            }
            m = lgb.train(cfg_seed, ds_tr, num_boost_round=cfg_final['ne'])
            oof_preds[va_i] += m.predict(X[va_i])

    oof_preds /= n_total
    cv_loss = log_loss(y, oof_preds, labels=[0, 1])

    # Test predictions
    test_preds = np.zeros(len(Xt))
    for s in range(42, 42 + n_seeds):
        ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel, params={'verbose': '-1'})
        cfg_seed = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg_final['nl'], 'max_depth': cfg_final['md'],
            'learning_rate': cfg_final['lr'], 'n_estimators': cfg_final['ne'],
            'subsample': cfg_final['ss'], 'colsample_bytree': cfg_final['cb'],
            'reg_alpha': cfg_final['ra'], 'reg_lambda': cfg_final['rl'],
            'min_child_samples': cfg_final['mc'], 'random_state': s,
            'scale_pos_weight': spw,
        }
        m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg_final['ne'])
        test_preds += m.predict(Xt)
    test_preds /= n_seeds

    # Calibration
    train_rate = y.mean()
    if cal_type == 'mean_match':
        cal_test = np.clip(test_preds + (train_rate - test_preds.mean()), 0.0001, 0.9999)
        cal_oof = np.clip(oof_preds + (train_rate - oof_preds.mean()), 0.0001, 0.9999)
    elif cal_type == 'none':
        cal_test = np.clip(test_preds, 0.0001, 0.9999)
        cal_oof = oof_preds
    else:
        cal_test = np.clip(test_preds, 0.0001, 0.9999)
        cal_oof = oof_preds

    cal_oof_loss = log_loss(y, cal_oof, labels=[0, 1])

    result = {
        'experiment': experiment_name,
        'target': target,
        'cv_loss': cv_loss,
        'cal_oof_loss': cal_oof_loss,
        'train_rate': float(train_rate),
        'test_mean': float(cal_test.mean()),
        'test_shift': float(cal_test.mean() - train_rate),
        'n_seeds': n_seeds,
        'feat_type': feat_type,
        'cfg_name': cfg_name,
        'n_feat': n_feat,
        'cal_type': cal_type,
        'rolling': add_rolling_feat,
        'interaction': add_interaction_feat,
        'time_s': time.time() - t0,
    }
    return result, sel_cols, cal_test


def run_experiments():
    """Run comprehensive experiments."""
    log.info("=" * 80)
    log.info("V90: Comprehensive V53 Optimization")
    log.info("=" * 80)

    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    train_cols_order = list(train.columns)
    test = test[train_cols_order]

    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    feat_cols = get_feature_cols(train)
    base_cols = [c for c in feat_cols if not c.endswith('_zscore') and not c.endswith('_rm*') and not c.endswith('_rs*') and '_x_' not in c]

    log.info(f"  Base features: {len(base_cols)}")

    # Add personalization to train/test for feature ranking
    train, zscore_cols = add_personalization(train, base_cols)
    test, _ = add_personalization(test, base_cols)
    all_feat_cols = base_cols + zscore_cols
    log.info(f"  With personalization: {len(all_feat_cols)} total")

    # Experiment plan: combinations to test
    # Format: (feat_type, cfg_name, n_seeds, n_feat, cal_type, rolling, interaction)
    experiments = [
        # --- Baseline V53 swept ---
        ('zscore', 'deep', 50, 17, 'mean_match', False, False, 'Q1'),
        ('zscore', 'deep', 50, 17, 'mean_match', False, False, 'Q2'),
        ('zscore', 'v48', 50, 11, 'mean_match', False, False, 'Q3'),
        ('zscore', 'wide', 50, 17, 'mean_match', False, False, 'S1'),
        ('zscore', 'deep', 50, 20, 'mean_match', False, False, 'S2'),
        ('zscore', 'safety', 50, 23, 'mean_match', False, False, 'S3'),
        ('zscore', 'wide', 50, 23, 'mean_match', False, False, 'S4'),

        # --- More seeds (100) ---
        ('zscore', 'deep', 100, 17, 'mean_match', False, False, 'Q1'),
        ('zscore', 'deep', 100, 17, 'mean_match', False, False, 'Q2'),
        ('zscore', 'v48', 100, 11, 'mean_match', False, False, 'Q3'),
        ('zscore', 'wide', 100, 17, 'mean_match', False, False, 'S1'),
        ('zscore', 'deep', 100, 20, 'mean_match', False, False, 'S2'),
        ('zscore', 'safety', 100, 23, 'mean_match', False, False, 'S3'),
        ('zscore', 'wide', 100, 23, 'mean_match', False, False, 'S4'),

        # --- With rolling ---
        ('zscore', 'deep', 50, 17, 'mean_match', True, False, 'Q1'),
        ('zscore', 'deep', 50, 17, 'mean_match', True, False, 'Q2'),
        ('zscore', 'v48', 50, 11, 'mean_match', True, False, 'Q3'),
        ('zscore', 'wide', 50, 17, 'mean_match', True, False, 'S1'),
        ('zscore', 'deep', 50, 20, 'mean_match', True, False, 'S2'),
        ('zscore', 'safety', 50, 23, 'mean_match', True, False, 'S3'),
        ('zscore', 'wide', 50, 23, 'mean_match', True, False, 'S4'),

        # --- V53 swept n_feat (with more seeds) ---
        ('zscore', 'deep', 100, 19, 'mean_match', False, False, 'Q1'),
        ('zscore', 'deep', 100, 14, 'mean_match', False, False, 'Q2'),
        ('zscore', 'v48', 100, 5, 'mean_match', False, False, 'Q3'),
        ('zscore', 'wide', 100, 21, 'mean_match', False, False, 'S1'),
        ('zscore', 'deep', 100, 19, 'mean_match', False, False, 'S2'),
        ('zscore', 'safety', 100, 21, 'mean_match', False, False, 'S3'),
        ('zscore', 'wide', 100, 20, 'mean_match', False, False, 'S4'),

        # --- CFG sweep for deep ---
        ('zscore', 'deep', 50, 20, 'mean_match', False, False, 'Q1'),
        ('zscore', 'deep', 50, 20, 'mean_match', False, False, 'Q2'),
        ('zscore', 'deep', 50, 20, 'mean_match', False, False, 'S2'),
    ]

    results = {}
    exp_idx = 0

    for exp in experiments:
        feat_type, cfg_name, n_seeds, n_feat, cal_type, roll, interact, target = exp
        exp_idx += 1

        # For V53 swept experiments, use specific n_feat
        if feat_type == 'zscore' and not roll and not interact and cfg_name in CFGS and target in ['Q1','Q2','Q3','S1','S2','S3','S4']:
            # Check if it's a swept experiment
            swept_nfeat = {'Q1': 19, 'Q2': 14, 'Q3': 5, 'S1': 21, 'S2': 19, 'S3': 21, 'S4': 20}
            if n_feat in [5, 11, 14, 17, 19, 21, 23] and n_seeds == 100 and not roll and not interact:
                if cfg_name in CFGS:
                    # This might be swept
                    pass

        experiment_name = f"{feat_type}_{cfg_name}_s{n_seeds}_nf{n_feat}_cal{cal_type}_r{int(roll)}_i{int(interact)}_{target}"
        log.info(f"\n[{exp_idx}/{len(experiments)}] {experiment_name}")

        result, sel_cols, test_pred = test_config_combination(
            train, test, all_feat_cols, target, experiment_name,
            n_seeds=n_seeds, feat_type=feat_type, n_feat=n_feat,
            cfg_name=cfg_name, cfg_override=None, cal_type=cal_type,
            add_rolling_feat=roll, add_interaction_feat=interact
        )
        results[(target, experiment_name)] = result
        log.info(f"  CV={result['cv_loss']:.4f}, Cal OOF={result['cal_oof_loss']:.4f}, "
                 f"test_mean={result['test_mean']:.4f}, shift={result['test_shift']:+.4f}, "
                 f"time={result['time_s']:.0f}s")

    # Save results
    res_path = SUBMIT / f'v90_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    # Convert to serializable
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
    with open(res_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    log.info(f"\nResults saved: {res_path}")

    return results


if __name__ == "__main__":
    run_experiments()
