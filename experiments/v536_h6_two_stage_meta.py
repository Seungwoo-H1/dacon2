#!/usr/bin/env python3
"""
V536 H6 — Two-Stage Meta (Ridge → LGBM)

Hypothesis: Ridge is linear and misses non-linear patterns.
Two-stage approach: Ridge → LGBM meta captures non-linear calibration.

Stage 1: Ridge (α=0.01) on [mean_pred, std_pred] → produces intermediate predictions
Stage 2: LGBM meta on [Ridge_pred, mean_pred, std_pred]

Grid search over LGBM meta hyperparameters:
  iterations=[100, 300, 500]
  learning_rate=[0.01, 0.05, 0.1]
  depth=[3, 4, 5]
  reg_alpha=[0, 1, 10]
  reg_lambda=[1, 10, 50]

Baseline: Single-stage Ridge (V534, avg_gap=-0.02824)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from itertools import product
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
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

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
SEED = 42
N_FOLDS = 5
N_SEEDS = 13

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count','wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count','wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

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
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05,
        'n_estimators': 50, 'scale_pos_weight': spw, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def train_base_models(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est):
    """Train XGB + LGBM base models, return (oof_xgb, oof_lgbm, test_xgb, test_lgbm)."""
    import xgboost as xgb
    
    X_tr = X_tr.astype(np.float64)
    X_va = X_va.astype(np.float64)
    X_test = X_test.astype(np.float64)
    
    # XGB
    xgb_params = {
        'n_estimators': n_est, 'max_depth': 4, 'learning_rate': 0.04,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3,
        'random_state': seed, 'n_jobs': 1, 'verbosity': 0
    }
    ds_tr_xgb = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    ds_va_xgb = xgb.DMatrix(X_va, feature_names=feat_names)
    ds_te_xgb = xgb.DMatrix(X_test, feature_names=feat_names)
    m_xgb = xgb.train(xgb_params, ds_tr_xgb, num_boost_round=n_est)
    
    # LGBM
    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    lgbm_params = {
        'n_estimators': n_est, 'num_leaves': 30, 'max_depth': 3,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5,
        'scale_pos_weight': spw, 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1, 'verbose': -1
    }
    sn = [sanitize_col(c) for c in feat_names]
    ds_tr_lgb = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
    m_lgb = lgb.train(lgbm_params, ds_tr_lgb, num_boost_round=n_est)
    
    oof_xgb = np.clip(m_xgb.predict(ds_va_xgb), 0.001, 0.999)
    oof_lgbm = np.clip(m_lgb.predict(X_va), 0.001, 0.999)
    test_xgb = m_xgb.predict(ds_te_xgb)
    test_lgbm = m_lgb.predict(X_test)
    
    return oof_xgb, oof_lgbm, test_xgb, test_lgbm


# V534 config (single best config, matching V535 submission)
V534_CONFIG = {
    'Q1':  {'n_feat': 3,  'n_est': 600},
    'Q2':  {'n_feat': 10, 'n_est': 800},
    'Q3':  {'n_feat': 7,  'n_est': 500},
    'S1':  {'n_feat': 3,  'n_est': 500},
    'S2':  {'n_feat': 7,  'n_est': 500},
    'S3':  {'n_feat': 23, 'n_est': 1000},
    'S4':  {'n_feat': 20, 'n_est': 300},
}

# LGBM meta hyperparameter grid
META_GRID = list(product(
    [100, 300, 500],   # iterations
    [0.01, 0.05, 0.1], # learning_rate
    [3, 4, 5],          # depth
    [0, 1, 10],         # reg_alpha
    [1, 10, 50],        # reg_lambda
))
META_PARAM_NAMES = ['iterations', 'learning_rate', 'depth', 'reg_alpha', 'reg_lambda']


def two_stage_meta_train(oofs_2d, y, lgbm_iter, lgbm_lr, lgbm_depth, lgbm_ra, lgbm_rl):
    """
    Two-stage meta training:
    Stage 1: Ridge on [mean, std]
    Stage 2: LGBM on [ridge_pred, mean, std]
    
    Uses group-wise CV to avoid leakage.
    Returns (avg_student_ll, meta_ll, gap).
    """
    n_train = len(y)
    
    # Stage 1: Ridge predictions via GroupKFold
    ridge_preds = np.zeros(n_train)
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # We need a grouping variable — use subject_id-based folds
    # For the meta stage, we'll use the same fold structure
    # Generate CV indices for stage 1
    fold_assignments = np.zeros(n_train, dtype=int) - 1
    
    # Rebuild folds using subject_id pattern from base training
    # Use simple GroupKFold on train indices
    subjects = np.repeat(0, n_train)  # dummy grouping
    # Actually we'll use KFold for meta (all training data, no subject grouping at meta level)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    fold_id = 0
    for tr_idx, va_idx in kf.split(np.zeros(n_train)):
        # Build mean+std features
        oofs_fold = oofs_2d[va_idx]
        mean_f = np.mean(oofs_fold, axis=1)
        std_f = np.std(oofs_fold, axis=1)
        X_fold_meta = np.column_stack([mean_f, std_f])
        
        # Train Ridge on training part of this fold
        tr_oofs = oofs_2d[tr_idx]
        tr_mean = np.mean(tr_oofs, axis=1)
        tr_std = np.std(tr_oofs, axis=1)
        X_tr_meta = np.column_stack([tr_mean, tr_std])
        
        ridge_model = Ridge(alpha=0.01)
        ridge_model.fit(X_tr_meta, y[tr_idx])
        ridge_preds[va_idx] = ridge_model.predict(X_fold_meta)
        fold_assignments[va_idx] = fold_id
        fold_id += 1
    
    # Stage 2: LGBM meta on [ridge_pred, mean, std]
    mean_all = np.mean(oofs_2d, axis=1)
    std_all = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([ridge_preds, mean_all, std_all])
    
    # Train LGBM via CV
    lgbm_meta_params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': 2**lgbm_depth, 'learning_rate': lgbm_lr,
        'reg_alpha': lgbm_ra, 'reg_lambda': lgbm_rl,
        'subsample': 0.8, 'colsample_bytree': 0.8,
    }
    
    cv_lls = []
    for tr_idx, va_idx in kf.split(np.zeros(n_train)):
        X_tr = X_meta[tr_idx]
        X_va = X_meta[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        full_params = {**lgbm_meta_params, 'scale_pos_weight': spw}
        
        sn = ['ridge_pred', 'mean_pred', 'std_pred']
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(full_params, ds, num_boost_round=lgbm_iter)
        
        va_pred = np.clip(m.predict(X_va), 0.001, 0.999)
        ll = log_loss(y_va, va_pred)
        cv_lls.append(ll)
    
    meta_cv_ll = np.mean(cv_lls)
    
    # Also compute avg_student (mean of individual model LLs)
    avg_student_ll = np.mean([log_loss(y, np.clip(o, 0.001, 0.999)) for o in oofs_2d.T])
    
    gap = avg_student_ll - meta_cv_ll
    return meta_cv_ll, avg_student_ll, gap


def ridge_baseline(oofs_2d, y):
    """Single-stage Ridge baseline (V534)."""
    mean_all = np.mean(oofs_2d, axis=1)
    std_all = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([mean_all, std_all])
    
    meta = Ridge(alpha=0.01)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y, train_proba)
    
    avg_student_ll = np.mean([log_loss(y, np.clip(o, 0.001, 0.999)) for o in oofs_2d.T])
    
    return meta_ll, avg_student_ll, avg_student_ll - meta_ll


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V536 H6 — Two-Stage Meta (Ridge → LGBM)")
    log.info("=" * 70)
    log.info("Stage 1: Ridge (α=0.01) on [mean, std]")
    log.info("Stage 2: LGBM on [ridge_pred, mean, std]")
    log.info(f"LGBM grid: {len(META_GRID)} combinations")
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score (same as V535)
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m, s = np.mean(tv), np.std(tv, ddof=0)
        if s < 1e-8: s = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m) / s
        test_df[zc] = (ev - m) / s
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(train_df)  # same column set
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Pre-rank features
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)
    
    # Generate base predictions (XGB + LGBM) for all targets
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}
    
    for target in TARGETS:
        bc = V534_CONFIG[target]
        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names
        
        n_est = bc['n_est']
        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)
        
        seed_oofs = []
        seed_test_preds = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 11
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                oof_xgb, oof_lgbm, test_xgb, test_lgbm = train_base_models(
                    seed, X_tr, y_tr, X_va, X_test_full, feat_names, n_est
                )
                
                # Optimal blend weight per seed
                ll_xgb = log_loss(y[va_idx], np.clip(oof_xgb, 0.001, 0.999))
                ll_lgbm = log_loss(y[va_idx], np.clip(oof_lgbm, 0.001, 0.999))
                inv_xgb = 1.0 / max(ll_xgb, 1e-10)
                inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
                wx = inv_xgb / (inv_xgb + inv_lgbm)
                
                blended_oof = wx * oof_xgb + (1 - wx) * oof_lgbm
                seed_oof[va_idx] = blended_oof
                
                # Same weight for test
                # Use average weight from all folds
                seed_test += (wx * test_xgb + (1 - wx) * test_lgbm)
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            
            seed_oofs.append(seed_oof)
            seed_test_preds.append(seed_test)
        
        all_seed_oofs[target] = seed_oofs
        all_test_preds[target] = seed_test_preds
        log.info(f"  {target}: {N_SEEDS} seeds done (n_feat={bc['n_feat']}, n_est={n_est})")
    
    # === Baseline: Single-stage Ridge (V534) ===
    log.info("\n" + "=" * 70)
    log.info("BASELINE: Single-stage Ridge (V534)")
    log.info("=" * 70)
    
    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    baseline_gaps = {}
    for t in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[t])
        _, _, gap = ridge_baseline(oofs_2d, train_df[t].values)
        baseline_gaps[t] = gap
    
    baseline_avg = np.mean(list(baseline_gaps.values()))
    baseline_vs308 = sum(1 for t in TARGETS if baseline_gaps[t] < v308_gaps[t])
    log.info(f"Baseline avg_gap: {baseline_avg:.5f}")
    log.info(f"Baseline vs308: {baseline_vs308}/7")
    for t in TARGETS:
        vs = "✅" if baseline_gaps[t] < v308_gaps[t] else "❌"
        log.info(f"  {t}: gap={baseline_gaps[t]:+.5f} (V308={v308_gaps[t]:.3f}) {vs}")
    
    # === Two-Stage Meta: Grid Search ===
    log.info("\n" + "=" * 70)
    log.info("TWO-STAGE META: Grid Search")
    log.info("=" * 70)
    log.info(f"Grid size: {len(META_GRID)} combinations")
    log.info(f"Estimated time per target: {len(META_GRID) * 0.3:.0f}s")
    
    best_target_meta = {}
    best_target_meta_params = {}
    total_two_stage_gap = 0
    all_target_results = {}
    
    for ti, target in enumerate(TARGETS):
        t1 = time.time()
        log.info(f"\n--- Target {target} ({ti+1}/{len(TARGETS)}) ---")
        
        oofs_2d = np.column_stack(all_seed_oofs[target])
        y = train_df[target].values
        
        best_gap = float('inf')
        best_params = None
        
        for gi, params in enumerate(META_GRID):
            lgbm_iter, lgbm_lr, lgbm_depth, lgbm_ra, lgbm_rl = params
            meta_ll, avg_ll, gap = two_stage_meta_train(
                oofs_2d, y, lgbm_iter, lgbm_lr, lgbm_depth, lgbm_ra, lgbm_rl
            )
            
            if gap < best_gap:
                best_gap = gap
                best_params = params
            
            # Progress indicator
            if (gi + 1) % 27 == 0:
                elapsed = time.time() - t1
                rate = (gi + 1) / elapsed if elapsed > 0 else 0
                eta = (len(META_GRID) - gi - 1) / rate if rate > 0 else 0
                log.info(f"  [{gi+1}/{len(META_GRID)}] best_gap={best_gap:.5f} | rate={rate:.1f}/s ETA={eta:.0f}s")
        
        best_iter, best_lr, best_depth, best_ra, best_rl = best_params
        log.info(f"  ✅ {target}: best gap={best_gap:.5f}")
        log.info(f"     params: iter={best_iter}, lr={best_lr}, depth={best_depth}, ra={best_ra}, rl={best_rl}")
        
        best_target_meta[target] = best_gap
        best_target_meta_params[target] = {
            'iterations': int(best_iter), 'learning_rate': float(best_lr),
            'depth': int(best_depth), 'reg_alpha': float(best_ra), 'reg_lambda': float(best_rl)
        }
        total_two_stage_gap += best_gap
        
        elapsed = time.time() - t1
        log.info(f"  Time: {elapsed:.1f}s")
        
        # Free memory
        gc.collect()
    
    avg_two_stage_gap = total_two_stage_gap / 7
    two_stage_vs308 = sum(1 for t in TARGETS if best_target_meta[t] < v308_gaps[t])
    
    # === Comparison ===
    log.info("\n" + "=" * 70)
    log.info("FINAL COMPARISON")
    log.info("=" * 70)
    
    log.info(f"\n{'Target':>5} | {'V534 Ridge':>12} | {'V536 H6 2-Stage':>14} | {'Delta':>8} | {'vs308'}")
    log.info("-" * 70)
    
    for t in TARGETS:
        delta = best_target_meta[t] - baseline_gaps[t]
        vs = "✅" if best_target_meta[t] < v308_gaps[t] else "❌"
        vs308_improved = "📈" if delta < 0 else ("📉" if delta > 0 else "=")
        log.info(f"  {t:>3} | {baseline_gaps[t]:>+11.5f} | {best_target_meta[t]:>+13.5f} | {delta:>+7.5f} | {vs} {vs308_improved}")
    
    log.info("-" * 70)
    improvement = baseline_avg - avg_two_stage_gap
    log.info(f"  {'AVG':>5} | {baseline_avg:>+11.5f} | {avg_two_stage_gap:>+13.5f} | {improvement:>+7.5f} | {baseline_vs308}/7 → {two_stage_vs308}/7")
    
    log.info(f"\nBaseline avg_gap: {baseline_avg:.5f}")
    log.info(f"Two-stage avg_gap: {avg_two_stage_gap:.5f}")
    log.info(f"Improvement: {improvement:+.5f}")
    log.info(f"Expected LB: {0.62235 - avg_two_stage_gap:.5f}")
    
    # Check if V536 H6 beats V534
    if avg_two_stage_gap < baseline_avg:
        log.info(f"\n🏆 V536 H6 beats V534 Ridge by {abs(improvement):.5f}!")
    else:
        log.info(f"\n⚠️  V536 H6 did NOT beat V534 Ridge (worse by {abs(improvement):.5f})")
    
    # === Save Submission ===
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        sub_df[t] = np.mean(all_test_preds[t], axis=0)
    
    sub_path = SUBMIT / f'submission_v536_h6_two_stage_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n📁 Submission saved: {sub_path}")
    
    # === Save Results ===
    result = {
        'version': 'V536_H6',
        'hypothesis': 'two_stage_meta',
        'name': 'Two-Stage Meta (Ridge → LGBM)',
        'description': 'Stage 1: Ridge on [mean, std], Stage 2: LGBM on [ridge_pred, mean, std]',
        'v534_config': {k: {'n_feat': v['n_feat'], 'n_est': v['n_est']} for k, v in V534_CONFIG.items()},
        'meta_stage1': {'type': 'Ridge', 'alpha': 0.01},
        'meta_stage2': {'type': 'LGBM', 'grid_size': len(META_GRID)},
        'baseline': {
            'type': 'single_stage_Ridge',
            'avg_gap': float(baseline_avg),
            'target_gaps': {k: float(v) for k, v in baseline_gaps.items()},
            'vs308': baseline_vs308,
        },
        'two_stage_best': {
            'avg_gap': float(avg_two_stage_gap),
            'target_gaps': {k: float(v) for k, v in best_target_meta.items()},
            'target_params': {k: str(v) for k, v in best_target_meta_params.items()},
            'vs308': two_stage_vs308,
        },
        'comparison': {
            'baseline_avg_gap': float(baseline_avg),
            'two_stage_avg_gap': float(avg_two_stage_gap),
            'improvement': float(improvement),
            'beats_baseline': avg_two_stage_gap < baseline_avg,
            'baseline_vs308': baseline_vs308,
            'two_stage_vs308': two_stage_vs308,
        },
        'expected_lb': float(0.62235 - avg_two_stage_gap),
        'timestamp': ts,
        'submission_path': str(sub_path.name),
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v536_h6_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    try:
        result = main()
    except Exception as e:
        log.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save failure info
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        result = {
            'version': 'V536_H6',
            'hypothesis': 'two_stage_meta',
            'status': 'FAILED',
            'error': str(e),
            'timestamp': ts,
        }
        result_path = EXPERIMENTS / f'v536_h6_{ts}.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Error result saved: {result_path}")
        sys.exit(1)
