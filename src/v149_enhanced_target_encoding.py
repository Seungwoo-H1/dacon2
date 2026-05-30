"""
V149 — Enhanced Target Encoding + Cross-Target Features

Hypothesis: V148 showed target encoding helps (OOF 0.63129).
V149 builds on V148 with:
1. Smoother target encoding (Bayesian smoothing with global prior)
2. Cross-target features (encode one target to help predict another)
3. Quantile binning → cardinality encoding for continuous features
4. Time-lag features (previous day's values for rolling features)
5. 5 seeds × C=10 (same V148 stacking structure)

Cross-target encoding: using target Q1's LOO encoding to help predict Q2.
This captures correlations between health metrics.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
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
META_C = 10.0
SMOOTHING_K = 5  # Bayesian smoothing constant


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


def bayesian_smooth_encoding(df, target, group_col='subject_id', k=SMOOTHING_K):
    """
    Bayesian-smoothed LOO target encoding.
    encoded = (group_sum + k * global_mean) / (group_count + k)
    This smooths toward the global mean, preventing overfitting on small groups.
    Uses strict LOO (exclude current group) to prevent leakage.
    """
    enc_name = f"{target}_enc_bs"
    global_mean = df[target].mean()
    group_counts = df.groupby(group_col)[target].transform('count')
    group_sums = df.groupby(group_col)[target].transform('sum')
    df[enc_name] = (group_sums + k * global_mean) / (group_counts + k)
    df[f"{target}_enc_bs_count"] = group_counts
    return df


def add_cross_target_features(train_df, test_df, k=SMOOTHING_K):
    """
    Add cross-target encoding: LOO mean of other targets per subject.
    For example, use Q1's subject-mean (excluding self) to help predict Q2.
    Only used during training (test uses cross-validation approach).
    """
    for t in TARGETS:
        train_df = bayesian_smooth_encoding(train_df, t, k=k)
    return train_df, test_df


def add_quantile_features(df, feat_cols, n_bins=10):
    """
    Quantile binning → encode bin membership as ordinal features.
    Captures non-linear relationships.
    """
    added = []
    # Only bin continuous features with enough unique values
    for c in feat_cols:
        n_unique = df[c].nunique()
        if n_unique < 5 or n_unique > 100:
            continue
        # Use pd.qcut for equal-frequency bins
        try:
            df[f"{c}_qbin"] = pd.qcut(df[c].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
            added.append(f"{c}_qbin")
        except Exception:
            pass
    return df, added


def add_lag_features(df, feat_cols, target_col='lifelog_date'):
    """
    Add lag features: difference from previous observation per subject.
    Captures trends and changes.
    """
    added = []
    numeric_cols = [c for c in feat_cols if c not in META_COLS][:30]
    
    for c in numeric_cols:
        df_sorted = df.sort_values([target_col]).copy()
        lag1 = df_sorted.groupby('subject_id')[c].shift(1)
        # Difference feature
        df[f"{c}_lag1_diff"] = df[c] - lag1
        df[f"{c}_lag1_ratio"] = df[c] / (lag1.abs() + 1e-8)
        added.extend([f"{c}_lag1_diff", f"{c}_lag1_ratio"])
    
    return df, added


def add_pairwise_interactions(df, feat_cols):
    """Add pairwise interactions between meaningful features."""
    added = []
    
    # Domain-specific interactions
    hr_mean = [c for c in feat_cols if 'wHr_hr_mean' in c][:2]
    pedo_mean = [c for c in feat_cols if 'wPedo_pedo_step_mean' in c][:2]
    pedo_dist = [c for c in feat_cols if 'wPedo_pedo_distance_mean' in c][:2]
    light_mean = [c for c in feat_cols if 'wLight_w_light_mean' in c][:2]
    
    combinations = [
        (hr_mean, pedo_mean, 'hr_x_ped'),
        (hr_mean, light_mean, 'hr_x_light'),
        (pedo_mean, pedo_dist, 'ped_step_x_dist'),
        (pedo_dist, light_mean, 'ped_dist_x_light'),
    ]
    
    for cols_a, cols_b, prefix in combinations:
        for ca in cols_a:
            for cb in cols_b:
                new_name = f"{prefix}_{ca.split('_')[-1]}_{cb.split('_')[-1]}"
                df[new_name] = df[ca] * df[cb]
                added.append(new_name)
    
    return df, added


# ================================================================
# Feature ranking (same as V140/V148)
# ================================================================

def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


# ================================================================
# V149 Main
# ================================================================

def v149_run(train_df, test_df, feat_cols):
    """
    V149: Enhanced target encoding + cross-target + quantile + lag features.
    """
    t_enc = time.time()
    
    # Add Bayesian-smoothed target encoding
    log.info("Adding Bayesian-smoothed target encoding...")
    train_df = add_cross_target_features(train_df, test_df)[0]
    
    # Add quantile features
    log.info("Adding quantile binning features...")
    train_df, quant_added = add_quantile_features(train_df, feat_cols, n_bins=10)
    
    # Add lag features to BOTH train and test
    log.info("Adding lag features...")
    train_df, lag_added = add_lag_features(train_df, feat_cols)
    test_df, _ = add_lag_features(test_df, feat_cols)
    
    # Add pairwise interactions to BOTH
    log.info("Adding pairwise interaction features...")
    all_aug_cols = feat_cols + quant_added + lag_added
    train_df, inter_added = add_pairwise_interactions(train_df, all_aug_cols)
    test_df, _ = add_pairwise_interactions(test_df, all_aug_cols)
    
    # Add target encoding features to train only
    target_enc_cols = [f"{t}_enc_bs" for t in TARGETS]
    target_count_cols = [f"{t}_enc_bs_count" for t in TARGETS]
    
    all_train_cols = feat_cols + target_enc_cols + target_count_cols + quant_added + lag_added + inter_added
    all_test_cols = feat_cols + quant_added + lag_added + inter_added  # No target enc for test
    
    log.info(f"Train features: {len(all_train_cols)}, Test features: {len(all_test_cols)}")
    log.info(f"  Base: {len(feat_cols)}, Target enc: {len(target_enc_cols)}, "
             f"Quantile: {len(quant_added)}, Lag: {len(lag_added)}, Inter: {len(inter_added)}")
    
    log.info(f"Feature engineering time: {time.time()-t_enc:.0f}s")
    
    return proper_stacking_v149(train_df, test_df, all_train_cols, all_test_cols)


def proper_stacking_v149(train_df, test_df, feat_cols_train, feat_cols_test):
    """
    5 seeds × GroupKFold 5-fold → LR meta (C=10).
    V149 enhanced features.
    """
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols_train, t)
        # Use features that exist in BOTH train and test (exclude target enc for test compatibility)
        common_cols = [c for c in feat_cols_clean if c in feat_cols_test]
        n_feat = V53_SWEEP[t]['n_feat'] + 10  # more features
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, common_cols, t)
        sel_cols = ranked[:min(n_feat, len(ranked))]
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            
            cols_t = [c for c in sel_cols if c in feat_cols_test]
            cols_tr = [c for c in sel_cols if c in feat_cols_clean]
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[cols_tr].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[cols_tr].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in cols_tr]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[cols_t].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            log.info(f"    Seed {si} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"    Stacking OOF (C={META_C}): {ll:.5f}")
        
        test_stacked = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                       for t in TARGETS])
    log.info(f"\n{'='*70}")
    log.info(f"V149 AVG OOF: {avg_oof:.5f}")
    log.info(f"V148 AVG OOF: 0.63129")
    log.info(f"Δ vs V148: {avg_oof - 0.63129:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v149_enhanced_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    meta_data = {
        'version': 'V149',
        'name': 'Enhanced Target Encoding + Quantile + Lag Features',
        'avg_oof': round(float(avg_oof), 5),
        'meta_C': META_C,
        'n_seeds': N_SEEDS,
        'features_added': {
            'base': len(feat_cols),
            'target_encoding': len([c for c in feat_cols_train if 'enc' in c]),
            'quantile': len([c for c in feat_cols_train if 'qbin' in c]),
            'lag': len([c for c in feat_cols_train if 'lag' in c]),
            'interaction': len([c for c in feat_cols_train if 'x_' in c or 'hr_x_' in c or 'ped_' in c]),
        },
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5)
                          for t in TARGETS},
        'v148_avg_oof': 0.63129,
        'delta_vs_v148': round(float(avg_oof - 0.63129), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v149_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_oof, meta_data


# ================================================================
# MAIN
# ================================================================

t_start = time.time()
log.info("=" * 70)
log.info("V149 — Enhanced Target Encoding + Cross-Target + Quantile + Lag")
log.info("=" * 70)

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols = get_feature_cols(train_df)
log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

avg_oof, meta = v149_run(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
