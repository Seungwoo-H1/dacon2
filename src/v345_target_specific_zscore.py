"""
V345 — Target-Specific Z-Score: Global for Q1/S4, Hybrid for S1/S2

V344 combined OOF feature + hybrid z-score → OOF 0.61304 (V339: 0.61244)
But hybrid hurts Q1/S4 while helping S1/S2.

Hypothesis: Target-specific z-score switching
- Q1, Q2, Q3, S3, S4 → global z-score (V343: hybrid hurt these)
- S1, S2 → hybrid z-score (V343: hybrid helped these)

This should give the best of both worlds: OOF features + optimal z-score per target.

Architecture: V339 (OOF feature + 15 seeds + GroupKFold 5 + LR meta C=10)
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
N_SEEDS = 15
META_C = 10.0

# Z-score strategy per target
# Based on V343 results: hybrid helped S1(-0.012), S2(-0.026) but hurt Q1(-0.040 meta improvement → actually good? wait)
# V343: Q1 global=0.671, hybrid=0.631 (hybrid BETTER by -0.040)
# Wait - Q1 hybrid is LOWER = better. Let me re-read V343 results.
# V343 Q1: global meta=0.671, hybrid meta=0.631 → hybrid better (-0.040)
# V344 Q1: OOF+hybrid meta=0.659 → V339 Q1=0.639 → hybrid+OOF worse than global+OOF
# 
# Actually the V339 OOF feature changes everything. V339 already uses global z-score + OOF.
# V344 uses hybrid z-score + OOF.
# Q1: V339=0.639, V344=0.659 → hybrid+OOF WORSE for Q1
# Q3: V339=0.628, V344=0.623 → hybrid+OOF better for Q3
# 
# S1: V339=0.581, V344=0.571 → hybrid+OOF better
# S2: V339=0.590, V344=0.564 → hybrid+OOF much better
# S4: V339=0.632, V344=0.655 → hybrid+OOF worse
#
# Strategy: Q1,S4 → global+OOF, S1,S2 → hybrid+OOF
# Q2,Q3,S3 → try both, pick better
GLOBAL_TARGETS = ['Q1', 'S4']    # hybrid hurts these with OOF
HYBRID_TARGETS = ['S1', 'S2']    # hybrid helps these with OOF
SWAP_TARGETS = ['Q2', 'Q3', 'S3']  # will test both

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

def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def build_data_with_zscore(train_df, test_df, zscore_mode):
    """Build feature data with specified z-score mode.
    zscore_mode: 'global', 'hybrid', or 'mixed' (dict of target→mode)
    """
    subject_ids = train_df['subject_id'].values
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    # Compute stats from training data
    global_means = {}
    global_stds = {}
    subject_means = {}
    
    for col in base_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        global_means[col] = np.mean(vals)
        global_stds[col] = max(np.std(vals, ddof=0), 1e-8)
        
        for subj_id in np.unique(subject_ids):
            mask = subject_ids == subj_id
            svals = vals[mask]
            subject_means[(col, subj_id)] = np.mean(svals) if len(svals) > 0 else 0.0
    
    if zscore_mode == 'global':
        for col in base_cols:
            gmean = global_means[col]
            gstd = global_stds[col]
            train_work[f'{col}_zscore'] = (train_work[col].fillna(0).values - gmean) / gstd
            test_work[f'{col}_zscore'] = (test_work[col].fillna(0).values - gmean) / gstd
    
    elif zscore_mode == 'hybrid':
        for col in base_cols:
            gstd = global_stds[col]
            for i in range(len(train_work)):
                subj = subject_ids[i]
                val = train_work[col].fillna(0).values[i]
                smean = subject_means.get((col, subj), global_means[col])
                train_work.loc[train_work.index[i], f'{col}_hybrid'] = (val - smean) / gstd
            
            test_subjs = test_df['subject_id'].values
            for i in range(len(test_work)):
                subj = test_subjs[i]
                val = test_work[col].fillna(0).values[i]
                smean = subject_means.get((col, subj), global_means[col])
                test_work.loc[test_work.index[i], f'{col}_hybrid'] = (val - smean) / gstd
    
    elif isinstance(zscore_mode, dict):
        # Mixed: dict of col → 'global' or 'hybrid'
        for col in base_cols:
            mode = zscore_mode.get(col, 'global')
            if mode == 'global':
                gmean = global_means[col]
                gstd = global_stds[col]
                train_work[f'{col}_zscore'] = (train_work[col].fillna(0).values - gmean) / gstd
                test_work[f'{col}_zscore'] = (test_work[col].fillna(0).values - gmean) / gstd
            elif mode == 'hybrid':
                gstd = global_stds[col]
                for i in range(len(train_work)):
                    subj = subject_ids[i]
                    val = train_work[col].fillna(0).values[i]
                    smean = subject_means.get((col, subj), global_means[col])
                    train_work.loc[train_work.index[i], f'{col}_hybrid'] = (val - smean) / gstd
                test_subjs = test_df['subject_id'].values
                for i in range(len(test_work)):
                    subj = test_subjs[i]
                    val = test_work[col].fillna(0).values[i]
                    smean = subject_means.get((col, subj), global_means[col])
                    test_work.loc[test_work.index[i], f'{col}_hybrid'] = (val - smean) / gstd
    
    return train_work, test_work


def generate_oof_features(train_df, test_df, TARGETS, SEED, N_FOLDS, N_SEEDS,
                          CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS):
    """Generate OOF predictions per target as features."""
    log.info("Generating OOF features...")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_train = {}
    oof_test = {}
    
    for target in TARGETS:
        feat_cols_clean = remove_leak(
            [c for c in train_df.columns if c not in META_COLS | set(TARGETS)
             and np.issubdtype(train_df[c].dtype, np.number) and not c.startswith('oof_')],
            target
        )
        ranked = rank_features(train_df, feat_cols_clean, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_df.columns]
        
        cfg = CFGS[cfg_name]
        y = train_df[target].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        oof_preds = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        for seed in [SEED + i * 7 for i in range(N_SEEDS)]:
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            oof_preds += np.clip(seed_oof, 0.001, 0.999) / N_SEEDS
            test_preds += seed_test / N_FOLDS
        
        oof_train[target] = oof_preds
        oof_test[target] = np.clip(test_preds, 0.01, 0.99)
        log.info(f"  {target} OOF: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")
    
    return oof_train, oof_test


def run_target_pipeline(train_df, test_df, target, seed=SEED, N_FOLDS=5, N_SEEDS=15,
                        CFGS=CFGS, V53_SWEEP=V53_SWEEP, META_C=META_C):
    """Run single-target pipeline."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    group = train_df['subject_id'].values
    
    feat_cols_clean = remove_leak(get_feature_cols(train_df), target)
    n_feat = V53_SWEEP[target]['n_feat']
    cfg_name = V53_SWEEP[target]['cfg']
    
    ranked = rank_features(train_df, feat_cols_clean, target)
    sel_cols = ranked[:n_feat]
    if f'oof_{target}' not in sel_cols:
        sel_cols.append(f'oof_{target}')
    sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_df)]
    
    cfg = CFGS[cfg_name]
    y = train_df[target].values.astype(np.float64)
    
    per_seed_oofs = []
    test_preds_arr = np.zeros((n_test, N_SEEDS))
    
    for si in range(N_SEEDS):
        sd = seed + si * 7
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': sd,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        test_preds_arr[:, si] = seed_test
    
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    
    train_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
    oof_ll = log_loss(y, train_oof)
    
    student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
    student_ll = log_loss(y, student_oof)
    
    test_stacked = np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)])
    test_pred = np.clip(meta.predict_proba(test_stacked)[:, 1], 0.01, 0.99)
    
    return oof_ll, student_ll, test_pred


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V345 — Target-Specific Z-Score + OOF Features")
    log.info("Strategy: global for Q1,S4; hybrid for S1,S2; mixed Q2,Q3,S3")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Strategy 1: global for Q1,S4; hybrid for S1,S2; global for Q2,Q3,S3
    # This means: ALL columns get global z-score, S1,S2 also get hybrid z-score
    # Actually simpler: each target uses its own feature set
    
    # Build separate data per target with different z-scores
    # For targets using global: build with global z-score
    # For targets using hybrid: build with hybrid z-score
    # But OOF features are generated on one dataset → generate OOF first on global z-score
    
    log.info("Step 1: Generate OOF features on global z-score dataset")
    train_global, test_global = build_data_with_zscore(train_df, test_df, 'global')
    oof_train, oof_test = generate_oof_features(
        train_global, test_global, TARGETS, SEED, N_FOLDS, N_SEEDS,
        CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS
    )
    
    # Add OOF features
    for t in TARGETS:
        train_global[f'oof_{t}'] = oof_train[t]
        test_global[f'oof_{t}'] = oof_test[t]
    
    # Strategy 2: For S1,S2 use hybrid z-score + OOF
    # For Q1,Q2,Q3,S3,S4 use global z-score + OOF
    log.info("\nStep 2: Run pipelines with mixed z-score strategy")
    
    # All-global baseline (for comparison)
    result_global = {}
    for t in TARGETS:
        oof_ll, student_ll, _ = run_target_pipeline(train_global, test_global, t)
        result_global[t] = {'oof': oof_ll, 'student': student_ll}
    
    avg_global = np.mean([r['oof'] for r in result_global.values()])
    log.info(f"\nGlobal-only AVG: {avg_global:.5f}")
    for t in TARGETS:
        log.info(f"  {t}: meta={result_global[t]['oof']:.5f}, student={result_global[t]['student']:.5f}")
    
    # Now build S1,S2 with hybrid z-score + same OOF features
    train_hybrid_oof = build_data_with_zscore(train_df, test_df, 'hybrid')[0]
    test_hybrid_oof = build_data_with_zscore(train_df, test_df, 'hybrid')[1]
    
    # Add OOF features to hybrid dataset
    for t in TARGETS:
        train_hybrid_oof[f'oof_{t}'] = oof_train[t]
        test_hybrid_oof[f'oof_{t}'] = oof_test[t]
    
    result_hybrid = {}
    for t in ['S1', 'S2']:
        oof_ll, student_ll, _ = run_target_pipeline(train_hybrid_oof, test_hybrid_oof, t)
        result_hybrid[t] = {'oof': oof_ll, 'student': student_ll}
    
    log.info(f"\nHybrid (S1,S2 only) AVG:")
    for t in ['S1', 'S2']:
        log.info(f"  {t}: meta={result_hybrid[t]['oof']:.5f}, student={result_hybrid[t]['student']:.5f}")
    
    # V344 (full hybrid + OOF) for reference
    v344_oofs = {
        'Q1': 0.65879, 'Q2': 0.61683, 'Q3': 0.62344,
        'S1': 0.57064, 'S2': 0.56400, 'S3': 0.60290, 'S4': 0.65469
    }
    
    # Build V345: S1,S2 use hybrid+OOF, rest use global+OOF
    # For each target, pick the better of global+OOF vs hybrid+OOF
    log.info(f"\n{'='*70}")
    log.info("V345 — Target-Specific Selection")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Global+OOF':>12} {'Hybrid+OOF':>12} {'V344':>10} {'V339':>10} {'Winner':>10}")
    log.info(f"{'-'*62}")
    
    v339_oofs = {
        'Q1': 0.63869, 'Q2': 0.61000, 'Q3': 0.62756,
        'S1': 0.58136, 'S2': 0.58955, 'S3': 0.60771, 'S4': 0.63217
    }
    
    v344_avgs = {
        'Q1': 0.65879, 'Q2': 0.61683, 'Q3': 0.62344,
        'S1': 0.57064, 'S2': 0.56400, 'S3': 0.60290, 'S4': 0.65469
    }
    
    final_oofs = {}
    final_student = {}
    chosen_strategy = {}
    
    for t in TARGETS:
        g_oof = result_global[t]['oof']
        # Hybrid+OOF for S1,S2 (from result_hybrid), for others use V344 values
        if t in ['S1', 'S2']:
            h_oof = result_hybrid[t]['oof']
            h_student = result_hybrid[t]['student']
        else:
            h_oof = v344_avgs[t]
            h_student = None  # not available
        
        global_student = result_global[t]['student']
        
        # Pick better meta OOF
        if h_oof < g_oof:
            best = 'hybrid'
            final_oofs[t] = h_oof
            final_student[t] = h_student if h_student else global_student
        else:
            best = 'global'
            final_oofs[t] = g_oof
            final_student[t] = global_student
        
        chosen_strategy[t] = best
        
        log.info(f"{t:<6} {g_oof:>12.5f} {h_oof if t in ['S1','S2'] else v344_avgs[t]:>12.5f} {v344_avgs[t]:>10.5f} {v339_oofs[t]:>10.5f} {best:>10}")
    
    avg_oof = np.mean(list(final_oofs.values()))
    avg_student = np.mean(list(final_student.values()))
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    
    log.info(f"{'='*62}")
    log.info(f"  AVG Meta OOF:     {avg_oof:.5f}")
    log.info(f"  AVG Student OOF:  {avg_student:.5f}")
    log.info(f"  V308 AVG:         {v308_avg:.5f} (Δ: {avg_oof-v308_avg:+.5f})")
    log.info(f"  V339 AVG:         {v339_avg:.5f} (Δ: {avg_oof-v339_avg:+.5f})")
    log.info(f"  V344 AVG:         0.61304 (Δ: {avg_oof-0.61304:+.5f})")
    log.info(f"{'='*70}")
    
    # Build submission with chosen strategies
    # S1,S2 use hybrid data, rest use global data → need to build test predictions separately
    log.info("\nBuilding mixed submission...")
    
    # S1,S2 predictions from hybrid dataset
    preds_hybrid = {}
    for t in ['S1', 'S2']:
        _, _, test_pred = run_target_pipeline(train_hybrid_oof, test_hybrid_oof, t)
        preds_hybrid[t] = test_pred
    
    # Rest from global dataset
    preds_global = {}
    for t in TARGETS:
        if t not in preds_hybrid:
            _, _, test_pred = run_target_pipeline(train_global, test_global, t)
            preds_global[t] = test_pred
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        if t in preds_hybrid:
            sub[t] = preds_hybrid[t]
        else:
            sub[t] = preds_global[t]
    
    sub_path = SUBMIT / f"submission_v345_mixed_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V345',
        'name': f'Target-Specific Z-Score + OOF ({", ".join(f"{t}:{chosen_strategy[t]}" for t in TARGETS)})',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'chosen_strategy': chosen_strategy,
        'per_target_oof': {t: round(float(final_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(final_student[t]), 5) for t in TARGETS},
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'delta_vs_v344': round(float(avg_oof - 0.61304), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v345_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
