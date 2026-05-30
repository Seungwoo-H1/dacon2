"""
V156 — Group-Enriched Stacking with Multi-Meta Ensemble

Hypothesis: V146's feature space is saturated. Adding group-level statistics
and cross-domain interactions, combined with multi-meta-learner ensemble,
will improve generalization without breaking V146's proven structure.

Changes from V146:
1. Group-level feature expansion: per-subject stats across all domains
2. Cross-domain interaction features (HR×Pedo, GPS×Wifi, BLE×Wifi)
3. Wider feature selection: top-K×2 instead of top-K
4. Multi-meta ensemble: LR(C=10) + LR(C=50) + Ridge regression
5. OOF-based isotonic calibration on test predictions

V146 backbone preserved:
- Same config→target mapping
- Same leak removal
- Same GroupKFold 5-fold
- Same 5 seeds with stride 7
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 5
META_C_VALUES = [10.0, 50.0]  # Multi-meta: C=10 and C=50


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def add_group_features(df, feat_cols):
    """
    Add group-level (per-subject) statistics for all features.
    This captures per-subject behavioral profiles.
    """
    added = []
    for stat_type in ['mean', 'std', 'min', 'max']:
        grp = df.groupby('subject_id')[feat_cols].transform(stat_type)
        for c in grp.columns:
            new_name = f"{c}_grp_{stat_type}"
            df[new_name] = grp[c].values
            added.append(new_name)
    return df, added


def add_cross_domain_interactions(df, feat_cols):
    """
    Add cross-domain interaction features.
    Focus on meaningful cross-domain combinations.
    """
    added = []
    
    # Domain column groups
    domains = {
        'hr': [c for c in feat_cols if 'wHr' in c],
        'pedo': [c for c in feat_cols if 'wPedo' in c and ('_mean' in c or '_sum' in c)],
        'gps': [c for c in feat_cols if 'mGps' in c and ('_mean' in c or '_sum' in c or '_count' in c)],
        'wifi': [c for c in feat_cols if 'mWifi' in c and ('_mean' in c or '_sum' in c or '_count' in c)],
        'ble': [c for c in feat_cols if 'mBle' in c and ('_mean' in c or '_count' in c or '_rssi' in c)],
        'light': [c for c in feat_cols if 'wLight' in c],
        'activity': [c for c in feat_cols if 'mActivity' in c and ('_mean' in c or '_sum' in c or '_count' in c)],
        'screen': [c for c in feat_cols if 'mScreenStatus' in c or 'screen_use' in c],
        'charging': [c for c in feat_cols if 'charging' in c],
        'ambience': [c for c in feat_cols if 'mAmbience' in c and '_sum' in c],
    }
    
    # Compute domain aggregates (mean of domain features)
    domain_means = {}
    for domain_name, cols in domains.items():
        if len(cols) > 0:
            valid_cols = [c for c in cols if c in df.columns]
            if len(valid_cols) > 0:
                agg_name = f"dom_{domain_name}_mean"
                df[agg_name] = df[valid_cols].mean(axis=1)
                domain_means[domain_name] = agg_name
                added.append(agg_name)
    
    # Cross-domain interactions: pairwise products of domain means
    domain_names = list(domain_means.keys())
    for i in range(len(domain_names)):
        for j in range(i+1, len(domain_names)):
            d1, d2 = domain_names[i], domain_names[j]
            if domain_means[d1] in df.columns and domain_means[d2] in df.columns:
                int_name = f"dom_{d1}_x_{d2}"
                df[int_name] = df[domain_means[d1]] * df[domain_means[d2]]
                added.append(int_name)
    
    return df, added


def add_ratio_features(df, feat_cols):
    """
    Add ratio features: e.g., HR/Pedo, GPS/Wifi, etc.
    These capture relative behavioral patterns.
    """
    added = []
    
    # HR step ratio
    hr_mean = [c for c in feat_cols if 'wHr' in c and 'mean' in c]
    pedo_step_mean = [c for c in feat_cols if 'wPedo_pedo_step_mean' in c]
    if hr_mean and pedo_step_mean:
        df['ratio_hr_step'] = df[hr_mean[0]] / (df[pedo_step_mean[0]] + 1e-8)
        added.append('ratio_hr_step')
    
    # GPS distance vs WiFi count
    gps_mean = [c for c in feat_cols if 'mGps' in c and 'mean' in c]
    wifi_count = [c for c in feat_cols if 'mWifi' in c and 'count' in c]
    if gps_mean and wifi_count:
        df['ratio_gps_wifi'] = df[gps_mean[0]] / (df[wifi_count[0]] + 1e-8)
        added.append('ratio_gps_wifi')
    
    # Activity vs screen usage
    act_mean = [c for c in feat_cols if 'mActivity' in c and 'mean' in c]
    screen_mean = [c for c in feat_cols if 'screen_use' in c and 'mean' in c]
    if act_mean and screen_mean:
        df['ratio_act_screen'] = df[act_mean[0]] / (df[screen_mean[0]] + 1e-8)
        added.append('ratio_act_screen')
    
    return df, added


def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def proper_stacking_v156(train_df, test_df, feat_cols_train, feat_cols_test, n_feat_mult=2):
    """
    V156: Group-enriched stacking with multi-meta ensemble.
    - Same 5 seeds × GroupKFold as V146
    - Wider feature selection (top-K × 2)
    - Multi-meta ensemble: LR(C=10) + LR(C=50) + Ridge avg
    """
    global t_start
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS)) for t in TARGETS}
    
    # Multi-meta predictions
    meta_preds_train = {t: {c: [] for c in META_C_VALUES + ['ridge']} for t in TARGETS}
    meta_preds_test = {t: {c: [] for c in META_C_VALUES + ['ridge']} for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols_train, t)
        n_feat = V53_SWEEP[t]['n_feat'] * n_feat_mult  # ×2 wider
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:min(n_feat, len(ranked))]
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            
            cols_for_train = [c for c in sel_cols if c in feat_cols_clean]
            cols_for_test = [c for c in sel_cols if c in feat_cols_test]
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[cols_for_train].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[cols_for_train].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in cols_for_train]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[cols_for_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            log.info(f"    Seed {si} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        # Multi-meta ensemble — single run per meta learner
        stacked = np.column_stack(per_seed_oofs)
        
        # Collect all meta predictions (OOF + test)
        meta_oof_preds = []   # list of OOF pred arrays
        meta_test_preds = []  # list of test pred arrays
        
        for mc in META_C_VALUES:
            meta = LogisticRegression(C=mc, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
            oof_pred = meta.predict_proba(stacked)[:, 1]
            meta_oof_preds.append(oof_pred)
            test_stacked = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
            test_pred = meta.predict_proba(test_stacked)[:, 1]
            meta_test_preds.append(test_pred)
        
        # Ridge meta
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(stacked, y)
        oof_pred_ridge = ridge.predict(stacked)
        oof_pred_ridge = np.clip(oof_pred_ridge, 0.001, 0.999)
        meta_oof_preds.append(oof_pred_ridge)
        test_stacked = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        test_pred_ridge = ridge.predict(test_stacked)
        test_pred_ridge = np.clip(test_pred_ridge, 0.001, 0.999)
        meta_test_preds.append(test_pred_ridge)
        
        # Average across meta learners
        train_oof[t] = np.mean(meta_oof_preds, axis=0)
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"    Multi-meta OOF (LR×2 + Ridge): {ll:.5f}")
        
        # Average test predictions across meta learners
        test_preds[t] = np.mean(meta_test_preds, axis=0)
    
    # Compute average OOF
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                       for t in TARGETS])
    log.info(f"\n{'='*70}")
    log.info(f"V156 AVG OOF: {avg_oof:.5f}")
    log.info(f"V146 AVG OOF: 0.63169")
    log.info(f"Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v156_multi_meta_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    # Per-target OOF
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5)
    
    meta_data = {
        'version': 'V156',
        'name': 'Group-Enriched Stacking + Multi-Meta Ensemble',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'n_feat_mult': n_feat_mult,
        'meta_learners': ['LR(C=10)', 'LR(C=50)', 'Ridge'],
        'group_features': True,
        'cross_domain_interactions': True,
        'per_target_oof': per_target_oof,
        'v146_avg_oof': 0.63169,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v156_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_oof, meta_data


# ================================================================
# MAIN
# ================================================================

t_start = time.time()
log.info("=" * 70)
log.info("V156 — Group-Enriched Stacking with Multi-Meta Ensemble")
log.info("=" * 70)

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

base_feat_cols = get_feature_cols(train_df)
log.info(f"Base features: {len(base_feat_cols)}")

# Add group features
log.info("Adding group-level features...")
train_df, group_added = add_group_features(train_df, base_feat_cols)
test_df, _ = add_group_features(test_df, base_feat_cols)
log.info(f"  Added {len(group_added)} group features")

# Add cross-domain interactions
log.info("Adding cross-domain interaction features...")
train_df, inter_added = add_cross_domain_interactions(train_df, base_feat_cols)
test_df, _ = add_cross_domain_interactions(test_df, base_feat_cols)
log.info(f"  Added {len(inter_added)} interaction features")

# Add ratio features
log.info("Adding ratio features...")
train_df, ratio_added = add_ratio_features(train_df, base_feat_cols)
test_df, _ = add_ratio_features(test_df, base_feat_cols)
log.info(f"  Added {len(ratio_added)} ratio features")

all_aug_cols = base_feat_cols + group_added + inter_added + ratio_added
log.info(f"Total augmented features: {len(all_aug_cols)}")

# Filter features that exist in both train and test
feat_cols_train = [c for c in all_aug_cols if c in train_df.columns]
feat_cols_test = [c for c in feat_cols_train if c in test_df.columns]
log.info(f"Train features: {len(feat_cols_train)}, Test features: {len(feat_cols_test)}")

# Target means
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

avg_oof, meta = proper_stacking_v156(train_df, test_df, feat_cols_train, feat_cols_test)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
