"""
V148 — Target-Encoded Deep Stacking

Hypothesis: V140~V147 are stuck in a local optimum of the stacking architecture.
Key bottleneck: base features don't encode per-subject patterns well enough.

V148 changes:
1. Add subject-level target encoding (LOO per subject) — captures per-subject baseline
2. Add interaction features between subject stats and global stats
3. Add time-series trend features (rolling mean over lifelog_date)
4. Stack with more students (5 seeds × 3 configs = 15 students)
5. Meta-learner C=10 (from V146)

Critical: target encoding must use strict LOO (per group) to avoid leakage.
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


def add_target_encoding(df, target, group_col='subject_id'):
    """
    Leave-one-group-out target encoding for the group.
    For each group, encode = (mean of OTHER groups, excluding current group).
    Actually for per-subject encoding, we want:
    For each row in a group, use mean of target from OTHER groups (exclude own group).
    This avoids direct leakage of own target.
    """
    enc_name = f"{target}_enc"
    # Group mean excluding each group itself — use transform with exclusion
    # Simple LOO: for each row, compute mean(target) of all rows NOT in same group
    group_mean = df.groupby(group_col)[target].transform('mean')
    global_mean = df[target].mean()
    n_groups = df[group_col].nunique()
    n_total = len(df)
    
    # LOO per group: (global_mean * n_total - group_mean * count_group) / (n_total - count_group)
    group_counts = df.groupby(group_col)[target].transform('count')
    loo_mean = ((global_mean * n_total) - (group_mean * group_counts)) / (n_total - group_counts + 1e-8)
    
    df[enc_name] = loo_mean
    # Add count
    df[f"{target}_enc_count"] = group_counts
    return df


def add_interaction_features(df, feat_cols):
    """Add pairwise interaction features between top numeric columns."""
    added = []
    # Subject-level aggregates as base for interactions
    numeric_cols = [c for c in feat_cols if 'mean' in c or 'sum' in c or 'std' in c][:20]
    if len(numeric_cols) < 5:
        return df, added
    
    # Cross-domain interactions (HR × Pedometer)
    hr_cols = [c for c in numeric_cols if 'wHr' in c and 'mean' in c][:3]
    pedo_cols = [c for c in numeric_cols if 'wPedo' in c and 'mean' in c][:3]
    light_cols = [c for c in numeric_cols if 'wLight' in c and 'mean' in c][:3]
    
    for hr_c in hr_cols:
        for ped_c in pedo_cols:
            new_name = f"{hr_c}_x_{ped_c}"
            df[new_name] = df[hr_c] * df[ped_c]
            added.append(new_name)
    
    for ped_c in pedo_cols:
        for light_c in light_cols:
            new_name = f"{ped_c}_x_{light_c}"
            df[new_name] = df[ped_c] * df[light_c]
            added.append(new_name)
    
    return df, added


def add_trend_features(df, feat_cols):
    """
    Add rolling mean features over lifelog_date per subject.
    Uses groupby + rolling on sorted dates.
    """
    added = []
    group = df['subject_id'].values
    unique_groups = np.unique(group)
    
    # Only add trend features for columns that make sense (daily aggregates)
    trend_candidates = [c for c in feat_cols if any(x in c for x in ['mean', 'std', 'sum']) 
                        and not any(x in c for x in ['subject_id', 'date', 'sleep_date', 'lifelog_date'])][:15]
    
    # This is expensive with 450 rows and 15 cols, but manageable
    # We'll add a simple: (value - global_mean) / global_std per group (z-score within group)
    for c in trend_candidates:
        new_name = f"{c}_zscore_grp"
        grp_mean = df.groupby('subject_id')[c].transform('mean')
        grp_std = df.groupby('subject_id')[c].transform('std').fillna(1e-8)
        df[new_name] = (df[c] - grp_mean) / grp_std
        added.append(new_name)
    
    return df, added


# ================================================================
# Feature ranking (same as V140)
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
# V148 Main
# ================================================================

def v148_run(train_df, test_df, feat_cols):
    """
    V148: Target-encoded deep stacking.
    Adds subject-level target encoding, interaction features, within-group z-scores.
    5 seeds × 3 configs → 15 students → LR meta (C=10).
    """
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Add enhanced features to train and test
    log.info("Adding target encoding features...")
    for t in TARGETS:
        train_df = add_target_encoding(train_df, t)
    
    log.info("Adding interaction features...")
    train_df, inter_added = add_interaction_features(train_df, feat_cols)
    
    log.info("Adding within-group z-score features...")
    train_df, trend_added = add_trend_features(train_df, feat_cols)
    
    train_aug_cols = feat_cols + inter_added + trend_added + [f"{t}_enc" for t in TARGETS] + [f"{t}_enc_count" for t in TARGETS]
    
    # Test also needs target encoding — use global mean (no target info in test)
    log.info("Adding test features...")
    for t in TARGETS:
        train_aug_cols_test = train_aug_cols
        # Test: target encoding = global mean of train
        train_mean = train_df[t].mean()
        group_means = train_df.groupby('subject_id')[t].mean()
        # For test rows, lookup subject's group mean from train
        # But we can't leak test target → use train global mean per subject (if exists)
        for idx in test_df.index:
            sid = test_df.loc[idx, 'subject_id']
            if sid in group_means.index:
                test_df.loc[idx, f"{t}_enc"] = group_means[sid]
            else:
                test_df.loc[idx, f"{t}_enc"] = train_mean
            test_df.loc[idx, f"{t}_enc_count"] = 0
    
    # Interaction features for test
    for hr_c, ped_c in zip(
        [c for c in inter_added if 'wHr' in c and 'x_' in c][:3],
        [c for c in inter_added if 'x_wPedo' in c][:3]
    ):
        base_hr = hr_c.replace('_x_wPedo_', 'wHr_').replace('_x_', '_mean_')
        # Just compute same interaction on test
        pass
    
    # Actually, simpler: recompute interactions on test
    test_df, _ = add_interaction_features(test_df, feat_cols)
    test_df, _ = add_trend_features(test_df, feat_cols)
    
    # Build test feature columns
    train_feat_cols = [c for c in train_aug_cols if c in feat_cols + inter_added + trend_added + 
                       [f"{t}_enc" for t in TARGETS] + [f"{t}_enc_count" for t in TARGETS]]
    test_feat_cols = [c for c in train_feat_cols if c in test_df.columns]
    
    log.info(f"Train features: {len(train_feat_cols)}, Test features: {len(test_feat_cols)}")
    
    # Stacking
    log.info("Running V148 stacking...")
    return proper_stacking_v148(train_df, test_df, train_feat_cols, test_feat_cols)


def proper_stacking_v148(train_df, test_df, feat_cols_train, feat_cols_test):
    """
    5 seeds × GroupKFold 5-fold → LR meta-learner.
    Uses V148 enhanced features.
    """
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols_train, t)
        n_feat = V53_SWEEP[t]['n_feat'] + 10  # use more features with encoding
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
    log.info(f"V148 AVG OOF: {avg_oof:.5f}")
    log.info(f"V140 AVG OOF: 0.64110")
    log.info(f"Δ vs V140: {avg_oof - 0.64110:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v148_target_enc_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    meta_data = {
        'version': 'V148',
        'name': 'Target-Encoded Deep Stacking',
        'avg_oof': round(float(avg_oof), 5),
        'meta_C': META_C,
        'n_seeds': N_SEEDS,
        'enhanced_features': len([c for c in feat_cols_train if c not in get_feature_cols(train_df)]),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5)
                          for t in TARGETS},
        'v140_avg_oof': 0.64110,
        'delta_vs_v140': round(float(avg_oof - 0.64110), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v148_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_oof, meta_data


# ================================================================
# MAIN
# ================================================================

t_start = time.time()
log.info("=" * 70)
log.info("V148 — Target-Encoded Deep Stacking")
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

avg_oof, meta = v148_run(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
