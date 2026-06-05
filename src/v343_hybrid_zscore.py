"""
V343 — Hybrid Z-Score: Within-Person Deviation, Global Scale

V309 failed because per-subject z-score removed between-person signal.
But within-person variability is still noise that hurts student models.

New idea: (x - subject_mean) / global_std
- Keeps between-person distances (global_std preserves scale)
- Removes within-person noise (subject_mean centering)
- Subject with higher-than-usual HR is captured, AND
  subjects are still comparable to each other

Hypothesis: This hybrid captures "anomalous behavior for THIS person"
while keeping inter-person differences. Should help student models
because it's a cleaner signal.

Changes from V308:
1. Replace global z-score with hybrid z-score
2. Same V308 architecture (15 seeds, GroupKFold 5, LR meta C=10)
3. Compare global vs hybrid z-score ablation
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


def compute_hybrid_zscore(train_df, test_df):
    """Compute hybrid z-scores: (x - subject_mean) / global_std
    
    Within-person centering (subject_mean) removes intra-person variability noise.
    Global scale (global_std) preserves inter-person distances.
    """
    log.info("Computing hybrid z-scores: (x - subject_mean) / global_std")
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    train_result = train_df.copy()
    test_result = test_df.copy()
    
    subject_ids = train_df['subject_id'].values
    
    for col in base_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        
        # Compute subject means from training data
        subject_means = {}
        for subj_id in np.unique(subject_ids):
            mask = subject_ids == subj_id
            vals = train_vals[mask]
            subject_means[subj_id] = np.mean(vals) if len(vals) > 0 else 0.0
        
        # Compute global std from training data
        global_std = np.std(train_vals, ddof=0)
        if global_std < 1e-8:
            global_std = 1e-8
        
        # Apply hybrid z-score to train
        hybrid_train = np.zeros_like(train_vals)
        for i in range(len(train_vals)):
            hybrid_train[i] = (train_vals[i] - subject_means[subject_ids[i]]) / global_std
        
        zc_train = f'{col}_hybrid'
        train_result[zc_train] = hybrid_train
        
        # Apply to test: use test subject IDs and global std
        # For test, we need subject_mean from TRAINING data (same subject)
        test_subjects = test_df['subject_id'].values
        hybrid_test = np.zeros_like(test_vals)
        for i in range(len(test_vals)):
            subj = test_subjects[i]
            if subj in subject_means:
                hybrid_test[i] = (test_vals[i] - subject_means[subj]) / global_std
            else:
                # Unknown subject: use global mean instead
                global_mean = np.mean(train_vals)
                hybrid_test[i] = (test_vals[i] - global_mean) / global_std
        
        zc_test = f'{col}_hybrid'
        test_result[zc_test] = hybrid_test
    
    zscore_cols = [f'{c}_hybrid' for c in base_cols]
    log.info(f"  Generated {len(zscore_cols)} hybrid z-score features")
    return train_result, test_result, zscore_cols


def compute_global_zscore(train_df, test_df):
    """Compute global z-scores: (x - global_mean) / global_std (V308 method)."""
    log.info("Computing global z-scores: (x - global_mean) / global_std")
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    train_result = train_df.copy()
    test_result = test_df.copy()
    
    for col in base_cols:
        if col not in test_df.columns:
            continue
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        
        train_result[f'{col}_zscore'] = (train_vals - mean) / std
        test_result[f'{col}_zscore'] = (test_vals - mean) / std
    
    zscore_cols = [f'{c}_zscore' for c in base_cols]
    log.info(f"  Generated {len(zscore_cols)} global z-score features")
    return train_result, test_result, zscore_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V343 — Hybrid Z-Score: Within-Person Deviation, Global Scale")
    log.info("Hypothesis: hybrid z-score removes intra-person noise while")
    log.info("            preserving inter-person signal")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"Base features: {len(base_cols)}")
    log.info(f"Subjects: {train_df['subject_id'].nunique()}")
    log.info(f"Rows per subject: {len(train_df) / train_df['subject_id'].nunique():.0f}")
    
    # Run TWO experiments in parallel:
    # Experiment A: Global z-score (V308 baseline)
    # Experiment B: Hybrid z-score (V343)
    # Same pipeline, same seeds → direct comparison
    
    log.info("\nRunning Experiment A: Global z-score (V308 baseline)")
    log.info("-" * 50)
    
    train_global = train_df.copy()
    test_global = test_df.copy()
    train_global, test_global, _ = compute_global_zscore(train_global, test_global)
    
    # Run pipeline on global z-score
    result_global = run_pipeline(train_global, test_global, SEED, N_FOLDS, N_SEEDS,
                                  CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS, TARGETS)
    
    log.info("\nRunning Experiment B: Hybrid z-score (V343)")
    log.info("-" * 50)
    
    train_hybrid = train_df.copy()
    test_hybrid = test_df.copy()
    train_hybrid, test_hybrid, _ = compute_hybrid_zscore(train_hybrid, test_hybrid)
    
    # Run pipeline on hybrid z-score
    result_hybrid = run_pipeline(train_hybrid, test_hybrid, SEED, N_FOLDS, N_SEEDS,
                                  CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS, TARGETS)
    
    # Compare
    v308_avg = 0.62235
    
    avg_oof_global = np.mean(list(result_global['oofs'].values()))
    avg_oof_hybrid = np.mean(list(result_hybrid['oofs'].values()))
    avg_student_global = np.mean(list(result_global['student_oofs'].values()))
    avg_student_hybrid = np.mean(list(result_hybrid['student_oofs'].values()))
    
    log.info(f"\n{'='*70}")
    log.info("V343 COMPARISON: Global z-score vs Hybrid z-score")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Global':>10} {'Hybrid':>10} {'Δ':>8} {'ΔGlobal':>8}")
    log.info(f"{'-'*46}")
    for t in TARGETS:
        global_oof = result_global['oofs'][t]
        hybrid_oof = result_hybrid['oofs'][t]
        log.info(f"{t:<6} {global_oof:>10.5f} {hybrid_oof:>10.5f} {hybrid_oof-global_oof:>+8.5f} {hybrid_oof-v308_avg:>+8.5f}")
    
    log.info(f"{'-'*46}")
    log.info(f"  Global AVG: {avg_oof_global:.5f} (Δ vs V308: {avg_oof_global-v308_avg:+.5f})")
    log.info(f"  Hybrid AVG: {avg_oof_hybrid:.5f} (Δ vs V308: {avg_oof_hybrid-v308_avg:+.5f})")
    log.info(f"  Global Student AVG: {avg_student_global:.5f}")
    log.info(f"  Hybrid Student AVG: {avg_student_hybrid:.5f}")
    log.info(f"  Hybrid ΔStudent vs Global: {avg_student_hybrid - avg_student_global:+.5f}")
    log.info(f"{'='*70}")
    
    # Choose better method
    if avg_oof_hybrid < avg_oof_global:
        winner = 'hybrid'
        winner_result = result_hybrid
        winner_name = 'Hybrid z-score'
        log.info(f"  WINNER: Hybrid z-score (Δ vs global: {avg_oof_hybrid - avg_oof_global:+.5f})")
    else:
        winner = 'global'
        winner_result = result_global
        winner_name = 'Global z-score (V308)'
        log.info(f"  WINNER: Global z-score (Δ vs hybrid: {avg_oof_global - avg_oof_hybrid:+.5f})")
    
    # Build submission with winner
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = winner_result['test_df']['subject_id'].values
    sub['sleep_date'] = winner_result['test_df']['sleep_date'].values
    sub['lifelog_date'] = winner_result['test_df']['lifelog_date'].values
    for t in TARGETS:
        sub[t] = winner_result['preds'][t]
    
    sub_path = SUBMIT / f"submission_v343_{winner}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V343',
        'name': f'Hybrid Z-Score ({winner_name} wins)',
        'winner': winner,
        'avg_oof_global': round(float(avg_oof_global), 5),
        'avg_oof_hybrid': round(float(avg_oof_hybrid), 5),
        'avg_student_global': round(float(avg_student_global), 5),
        'avg_student_hybrid': round(float(avg_student_hybrid), 5),
        'delta_vs_v308': round(float(avg_oof_hybrid - v308_avg), 5),
        'per_target_oof': {t: round(float(winner_result['oofs'][t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(winner_result['student_oofs'][t]), 5) for t in TARGETS},
        'comparison': {
            'global_avg': round(float(avg_oof_global), 5),
            'hybrid_avg': round(float(avg_oof_hybrid), 5),
            'global_student': round(float(avg_student_global), 5),
            'hybrid_student': round(float(avg_student_hybrid), 5),
        },
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v343_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof_hybrid, meta_data


def run_pipeline(train_df, test_df, SEED, N_FOLDS, N_SEEDS, CFGS, V53_SWEEP,
                 LEAK_S, LEAK_Q, META_COLS, TARGETS):
    """Run full V308-style pipeline on pre-processed data."""
    log.info(f"  Features: {len(get_feature_cols(train_df))}")
    
    feat_cols = get_feature_cols(train_df)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    all_oofs = {}
    all_student_oofs = {}
    all_preds = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_df)]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        test_preds_arr = np.zeros((n_test, N_SEEDS))
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
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
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds_arr[:, si] = seed_test
        
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof = meta.predict_proba(stacked)[:, 1]
        oof_ll = log_loss(y, np.clip(train_oof, 0.001, 0.999))
        all_oofs[t] = oof_ll
        
        student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        test_stacked = np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)])
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        all_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"    {t}: student={student_ll:.5f}, meta={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
    
    return {
        'oofs': all_oofs,
        'student_oofs': all_student_oofs,
        'preds': all_preds,
        'train_df': train_df,
        'test_df': test_df,
    }


if __name__ == '__main__':
    main()
