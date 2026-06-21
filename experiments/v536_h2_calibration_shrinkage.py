#!/usr/bin/env python3
"""
V536-H2: Calibration Shrinkage (Post-hoc Gap Correction)

Hypothesis: S1/S2 gap이 매우음수(-0.096, -0.070) → 모델이 지나치게 자신감 있음
→ prediction을 training mean으로 shrink하면 gap이 0에 가까워짐.

Method: corrected_pred = mean + shrink * (pred - mean)
Target별 optimal shrink을 grid search [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OOF에서 평가 후 test에 적용.

Base: V534 config (same as V535 submission)
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
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count','wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count','wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

SEED = 42
N_FOLDS = 5
N_SEEDS = 13

# Calibration shrinkage grid per target
SHRINK_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

# V534 BEST config (same as V535)
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


def apply_calibration_shrink(preds, target_mean, shrink):
    """corrected_pred = mean + shrink * (pred - mean)"""
    return target_mean + shrink * (preds - target_mean)


def compute_gap_with_shrink(oofs_2d, y_true, shrink):
    """Compute gap for a single shrink value."""
    n_seeds = oofs_2d.shape[1]
    # Average OOF across seeds first
    avg_pred = np.mean(oofs_2d, axis=1)
    std_pred = np.std(oofs_2d, axis=1)
    X_meta = np.column_stack([avg_pred, std_pred])
    meta = Ridge(alpha=0.01)
    meta.fit(X_meta, y_true)
    train_pred = meta.predict(X_meta)
    pmin, pmax = train_pred.min(), train_pred.max()
    if pmax - pmin < 1e-10:
        train_proba = np.ones_like(train_pred) * 0.5
    else:
        train_proba = (train_pred - pmin) / (pmax - pmin)
    train_proba = np.clip(train_proba, 0.001, 0.999)
    meta_ll = log_loss(y_true, train_proba)

    # Shrink each OOF prediction
    oofs_shrunk = np.zeros_like(oofs_2d)
    for si in range(n_seeds):
        oofs_shrunk[:, si] = apply_calibration_shrink(oofs_2d[:, si], np.mean(y_true), shrink)
        oofs_shrunk[:, si] = np.clip(oofs_shrunk[:, si], 0.001, 0.999)
    avg_student = np.mean([log_loss(y_true, oofs_shrunk[:, si]) for si in range(n_seeds)])

    gap = avg_student - meta_ll
    return gap, meta_ll, avg_student


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V536-H2: Calibration Shrinkage (Post-hoc Gap Correction)")
    log.info("=" * 70)

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

    # Pre-rank
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # Generate predictions for all targets
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}
    train_means = {}

    for target in TARGETS:
        bc = V534_CONFIG[target]
        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names

        n_est = bc['n_est']
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[bc['xgb_cfg']]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[bc['lgbm_cfg']]}

        y = train_df[target].values.astype(np.float64)
        train_means[target] = np.mean(y)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        log.info(f"\n--- {target}: Training with n_est={n_est}, n_feat={bc['n_feat']} ---")

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

            log.info(f"    seed {si:2d} ({SEED+si*11:5d}): xgb_oof_mean={np.mean(seed_oof_xgb):.4f}, lgbm_oof_mean={np.mean(seed_oof_lgbm):.4f}, train_mean={np.mean(y):.4f}")

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

        log.info(f"  {target}: wx={wx:.3f}, blend_w_xgb_ll={ll_xgb:.5f}, blend_w_lgbm_ll={ll_lgbm:.5f}")
        log.info(f"  train_mean={train_means[target]:.5f}")

    # ================================================================
    # H2: Calibration Shrinkage — Grid Search per Target
    # ================================================================
    log.info("\n" + "=" * 70)
    log.info("V536-H2: Calibration Shrinkage Grid Search")
    log.info("=" * 70)

    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}

    best_shrinks = {}
    best_target_gaps = {}
    shrink_details = {}

    for target in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[target])
        y_true = train_df[target].values
        t_mean = train_means[target]

        log.info(f"\n--- {target}: shrink grid search (train_mean={t_mean:.5f}) ---")

        best_gap = float('inf')
        best_shrink = 1.0
        shrink_results = []

        for shrink in SHRINK_GRID:
            gap, meta_ll, avg_student = compute_gap_with_shrink(oofs_2d, y_true, shrink)
            shrink_results.append({
                'shrink': shrink,
                'gap': float(gap),
                'meta_ll': float(meta_ll),
                'avg_student_ll': float(avg_student),
                'vs308': '✅' if gap < v308_gaps[target] else '❌'
            })
            log.info(f"  shrink={shrink:.1f}: gap={gap:+.5f} (avg_student={avg_student:.5f}, meta_ll={meta_ll:.5f}) {shrink_results[-1]['vs308']}")
            if gap < best_gap:
                best_gap = gap
                best_shrink = shrink

        best_shrinks[target] = float(best_shrink)
        best_target_gaps[target] = float(best_gap)
        shrink_details[target] = shrink_results
        log.info(f"  🏆 {target}: best_shrink={best_shrink:.1f}, best_gap={best_gap:+.5f}")

    # ================================================================
    # Report
    # ================================================================
    total_gap = sum(best_target_gaps.values())
    avg_gap = total_gap / 7
    vs308 = sum(1 for t in TARGETS if best_target_gaps[t] < v308_gaps[t])

    log.info("\n" + "=" * 70)
    log.info("V536-H2: Final Results (Best Shrink per Target)")
    log.info("=" * 70)
    log.info(f"{'Target':<6} {'Best Shrink':>12} {'New Gap':>10} {'V308 Gap':>10} {'Status':>7}")
    log.info("-" * 48)
    for t in TARGETS:
        vs = "✅" if best_target_gaps[t] < v308_gaps[t] else "❌"
        log.info(f"{t:<6} {best_shrinks[t]:>12.1f} {best_target_gaps[t]:>10.5f} {v308_gaps[t]:>10.3f} {vs:>7}")
    log.info(f"\n  avg_gap: {avg_gap:.5f}")
    log.info(f"  vs308: {vs308}/7")
    log.info(f"  Expected LB: {0.62235 - avg_gap:.5f}")

    # ================================================================
    # Apply best shrinks to test predictions and save
    # ================================================================
    log.info("\nApplying calibration shrinkage to test predictions...")
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})

    for t in TARGETS:
        shrunk_tests = []
        for tp in all_test_preds[t]:
            shrunk = apply_calibration_shrink(tp, train_means[t], best_shrinks[t])
            shrunk_tests.append(np.clip(shrunk, 0.0, 1.0))
        sub_df[t] = np.mean(shrunk_tests, axis=0)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    shrink_str = '_'.join([f'{t}{int(best_shrinks[t]*10)}' for t in TARGETS])
    sub_path = SUBMIT / f'submission_v536_h2_shrink_{shrink_str}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"📁 Submission saved: {sub_path}")

    # Save result
    result = {
        'version': 'V536_H2',
        'hypothesis': 'calibration_shrinkage',
        'name': 'Post-hoc Gap Correction via Calibration Shrinkage',
        'config': {k: v for k, v in V534_CONFIG.items()},
        'shrink_grid': SHRINK_GRID,
        'best_shrinks': {k: float(v) for k, v in best_shrinks.items()},
        'avg_gap': float(avg_gap),
        'target_gaps': {k: float(v) for k, v in best_target_gaps.items()},
        'v308_gaps': v308_gaps,
        'vs308': int(vs308),
        'expected_lb': float(0.62235 - avg_gap),
        'train_means': {k: float(v) for k, v in train_means.items()},
        'shrink_details': shrink_details,
        'submission_file': str(sub_path.name),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    result_path = EXPERIMENTS / f'v536_h2_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"📁 Result saved: {result_path}")

    # Also save to memory format
    memory_path = EXPERIMENTS / f'v536_h2_results.json'
    memory_result = {
        'version': 'V536_H2',
        'hypothesis': 'calibration_shrinkage',
        'best_shrinks': {k: float(v) for k, v in best_shrinks.items()},
        'target_gaps': {k: float(v) for k, v in best_target_gaps.items()},
        'avg_gap': float(avg_gap),
        'vs308': int(vs308),
        'expected_lb': float(0.62235 - avg_gap),
        'submission_file': str(sub_path.name),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }
    with open(memory_path, 'w') as f:
        json.dump(memory_result, f, indent=2, ensure_ascii=False)
    log.info(f"📁 Memory result saved: {memory_path}")

    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    result = main()
    if result:
        print("\n" + "=" * 70)
        print("🏆 V536-H2 SUMMARY")
        print("=" * 70)
        print(f"  Best shrinks: {result['best_shrinks']}")
        print(f"  Target gaps:  {result['target_gaps']}")
        print(f"  avg_gap: {result['avg_gap']:.5f}")
        print(f"  vs308: {result['vs308']}/7 targets improved")
        print(f"  Expected LB: {result['expected_lb']:.5f}")
        print(f"  Submission: {result['submission_file']}")
        print("=" * 70)
