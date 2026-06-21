#!/usr/bin/env python3
"""
V536 H1: Target-Specific Ridge Alpha

Hypothesis: Global Ridge α=0.01 is not optimal for all targets.
S1/S2 have large negative gaps (over-calibration) → different α per target.

Method:
- Use V534 BEST config (Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20)
- Same ensemble (XGB + LGBM, 13 seeds, 5-fold)
- Grid search Ridge alpha per target: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
- Pick best α per target based on OOF gap
- Report combined avg_gap with optimal per-target α combo

Expected: If S1/S2 benefit from higher α (more regularization = less over-calibration),
this should improve overall avg_gap.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

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
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
          'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
          'wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean',
          'wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
          'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean',
          'wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
          'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

SEED = 42
N_FOLDS = 5
N_SEEDS = 13
ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
V308_GAPS = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}

# V534 BEST config
V534_CONFIG = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}

XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}


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

def train_one(seed, X_tr, y_tr, X_va, X_test, feat_names, learner, n_est, **mp):
    if learner == 'xgb':
        params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
        ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
        ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
        m = xgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(ds_va), m.predict(ds_te)
    else:
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
                 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feat_names]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(X_va), m.predict(X_test)


def compute_gap_for_alpha(oofs_arr, y, alpha):
    """Compute gap for a single Ridge alpha with mean+std meta."""
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = Ridge(alpha=alpha)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y, train_proba)
    avg_student = np.mean([log_loss(y, np.clip(so, 0.001, 0.999)) for so in oofs_arr])
    return avg_student - meta_ll


def compute_train_proba_for_alpha(oofs_arr, y, alpha):
    """Same as above but also returns the train_proba for meta fitting."""
    oofs_2d = np.column_stack(oofs_arr)
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = Ridge(alpha=alpha)
    meta.fit(X_meta, y)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    return np.clip(train_proba, 0.001, 0.999)


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V536 H1 — Target-Specific Ridge Alpha")
    log.info("=" * 70)
    log.info("Config: V534 BEST (Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20)")
    log.info(f"Alpha grid: {ALPHA_GRID}")
    log.info("Baseline: V534 avg_gap=-0.02824 (global α=0.01)")

    try:
        train_df = pd.read_parquet(DATA / "features.parquet")
        test_df = pd.read_parquet(DATA / "test_features.parquet")
    except Exception as e:
        log.error(f"Failed to load data: {e}")
        return None

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # Z-score
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
    test_feat_cols = get_feature_cols(train_df)
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Pre-rank features
    log.info("\nPre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)
        log.info(f"  {target}: top-5 = {ranked_features[target][:5]}")

    # Generate predictions for all targets
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}

    for target in TARGETS:
        log.info(f"\n{'='*50}")
        log.info(f"Target: {target}")
        bc = V534_CONFIG[target]
        log.info(f"  Config: n_feat={bc['n_feat']}, xgb={bc['xgb_cfg']}, lgbm={bc['lgbm_cfg']}, n_est={bc['n_est']}")

        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names

        n_est = bc['n_est']
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[bc['xgb_cfg']]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[bc['lgbm_cfg']]}

        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        xgb_seed_oofs = []
        lgbm_seed_oofs = []
        xgb_test_preds = []
        lgbm_test_preds = []

        for si in range(N_SEEDS):
            seed = SEED + si * 11
            seed_oof_xgb = np.zeros(n_train)
            seed_oof_lgbm = np.zeros(n_train)
            seed_test_xgb = np.zeros(n_test)
            seed_test_lgbm = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]

                pvxgb, ttxgb = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'xgb', n_est, **xgb_params)
                pvlgbm, ttlgbm = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'lgbm', n_est, **lgbm_params)

                seed_oof_xgb[va_idx] = pvxgb
                seed_oof_lgbm[va_idx] = pvlgbm
                seed_test_xgb += ttxgb
                seed_test_lgbm += ttlgbm

            seed_oof_xgb = np.clip(seed_oof_xgb, 0.001, 0.999)
            seed_oof_lgbm = np.clip(seed_oof_lgbm, 0.001, 0.999)
            seed_test_xgb /= N_FOLDS
            seed_test_lgbm /= N_FOLDS

            xgb_seed_oofs.append(seed_oof_xgb)
            lgbm_seed_oofs.append(seed_oof_lgbm)
            xgb_test_preds.append(seed_test_xgb)
            lgbm_test_preds.append(seed_test_lgbm)

            if (si + 1) % 5 == 0:
                log.info(f"  seed {si+1}/{N_SEEDS} done")

        # Compute blend weight
        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_seed_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_seed_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10)
        inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        wx = inv_xgb / (inv_xgb + inv_lgbm)

        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_test_preds, lgbm_test_preds)]

        all_seed_oofs[target] = blended_oofs
        all_test_preds[target] = blended_tests

        log.info(f"  ✓ Done: wx={wx:.3f}, blend over {N_SEEDS} seeds")

    # === H1: Target-specific Ridge alpha search ===
    log.info("\n" + "=" * 70)
    log.info("H1: Target-Specific Ridge Alpha Search")
    log.info("=" * 70)

    best_alpha_per_target = {}
    best_gap_per_target = {}
    alpha_search_results = {t: {} for t in TARGETS}

    for target in TARGETS:
        oofs_arr = all_seed_oofs[target]
        y = train_df[target].values
        log.info(f"\n  {target}: Alpha search...")

        for alpha in ALPHA_GRID:
            gap = compute_gap_for_alpha(oofs_arr, y, alpha)
            alpha_search_results[target][alpha] = gap
            log.info(f"    α={alpha:.3f}: gap={gap:+.5f}")

        # Find best alpha
        alpha_gaps = alpha_search_results[target]
        best_alpha = min(alpha_gaps, key=alpha_gaps.get)
        best_gap = alpha_gaps[best_alpha]
        best_alpha_per_target[target] = best_alpha
        best_gap_per_target[target] = best_gap
        log.info(f"  → Best α for {target}: {best_alpha:.3f} (gap={best_gap:+.5f})")

    # Compute total avg_gap with per-target alphas
    total_gap = sum(best_gap_per_target.values())
    avg_gap_h1 = total_gap / 7

    # Also compute baseline (global α=0.01) for comparison
    baseline_gap = {}
    for target in TARGETS:
        baseline_gap[target] = compute_gap_for_alpha(all_seed_oofs[target], train_df[target].values, 0.01)
    avg_gap_baseline = sum(baseline_gap.values()) / 7

    vs308_h1 = sum(1 for t in TARGETS if best_gap_per_target[t] < V308_GAPS[t])
    vs308_baseline = sum(1 for t in TARGETS if baseline_gap[t] < V308_GAPS[t])

    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)

    log.info(f"\n  {'Target':>4}  {'Alpha':>7}  {'H1 Gap':>8}  {'Baseline':>9}  {'Δ vs Base':>10}  {'Beat V308?'}")
    log.info(f"  {'─────':>4}  {'──────':>7}  {'───────':>8}  {'────────':>9}  {'─────────':>10}  {'─────────'}")
    for t in TARGETS:
        h1_g = best_gap_per_target[t]
        bl_g = baseline_gap[t]
        diff = h1_g - bl_g
        vs = "✅" if h1_g < V308_GAPS[t] else "❌"
        sign = "+" if diff >= 0 else ""
        log.info(f"  {t:>4}  {best_alpha_per_target[t]:>7.3f}  {h1_g:>+8.5f}  {bl_g:>+9.5f}  {sign}{diff:>+9.5f}  {vs}")

    log.info(f"\n  Baseline (global α=0.01): avg_gap={avg_gap_baseline:+.5f}, vs308={vs308_baseline}/7")
    log.info(f"  H1 (per-target α):        avg_gap={avg_gap_h1:+.5f}, vs308={vs308_h1}/7")
    improvement = avg_gap_baseline - avg_gap_h1
    log.info(f"  Improvement: {improvement:+.5f}")

    vs308_h1 = sum(1 for t in TARGETS if best_gap_per_target[t] < V308_GAPS[t])
    if vs308_h1 > vs308_baseline:
        log.info(f"  🎯 V308 improved: {vs308_h1}/7 vs {vs308_baseline}/7 baseline")

    # Save submission with best per-target alpha predictions
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Compute test predictions using the best per-target alpha
    # (The ensemble predictions are already computed; just apply the best alpha for meta fitting)
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        test_pred_mean = np.mean(all_test_preds[t], axis=0)
        test_pred_std = np.std(all_test_preds[t], axis=0)
        X_meta_test = np.column_stack([test_pred_mean, test_pred_std])

        oofs_2d = np.column_stack(all_seed_oofs[t])
        avg_pred_train = np.mean(oofs_2d, axis=1)
        std_pred_train = np.std(oofs_2d, axis=1)
        X_meta_train = np.column_stack([avg_pred_train, std_pred_train])

        meta = Ridge(alpha=best_alpha_per_target[t])
        meta.fit(X_meta_train, train_df[t].values)
        train_pred_raw = meta.predict(X_meta_train)
        pmin, pmax = train_pred_raw.min(), train_pred_raw.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred_raw) * 0.5
        else:
            train_proba = (train_pred_raw - pmin) / (pmax - pmin)
        train_proba = np.clip(train_proba, 0.001, 0.999)

        test_pred_raw = meta.predict(X_meta_test)
        tmin, tmax = test_pred_raw.min(), test_pred_raw.max()
        if tmax - tmin < 1e-10:
            test_proba = np.ones_like(test_pred_raw) * 0.5
        else:
            test_proba = (test_pred_raw - tmin) / (tmax - tmin)
        test_proba = np.clip(test_proba, 0.001, 0.999)
        sub_df[t] = test_proba

    sub_path = SUBMIT / f'submission_v536_h1_target_ridge_alpha_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n📁 Submission saved: {sub_path}")

    # Save result JSON
    result = {
        'version': 'V536_H1',
        'name': 'Target-Specific Ridge Alpha',
        'hypothesis': 'target_specific_ridge_alpha',
        'config': {k: dict(v) for k, v in V534_CONFIG.items()},
        'best_alpha_per_target': {k: float(v) for k, v in best_alpha_per_target.items()},
        'alpha_grid': ALPHA_GRID,
        'avg_gap_h1': float(avg_gap_h1),
        'avg_gap_baseline': float(avg_gap_baseline),
        'improvement': float(improvement),
        'target_gaps_h1': {k: float(v) for k, v in best_gap_per_target.items()},
        'target_gaps_baseline': {k: float(v) for k, v in baseline_gap.items()},
        'alpha_search_details': {t: {str(k): float(v) for k, v in v.items()} for t, v in alpha_search_results.items()},
        'vs308_h1': vs308_h1,
        'vs308_baseline': vs308_baseline,
        'expected_lb': float(0.62235 - avg_gap_h1),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v536_h1_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Result saved: {result_path}")

    # Also save to memory directory
    memory_result_path = ROOT / 'memory' / 'v536_h1_results.json'
    with open(memory_result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Memory result saved: {memory_result_path}")

    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    main()
