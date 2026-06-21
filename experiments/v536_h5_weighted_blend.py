#!/usr/bin/env python3
"""
V536-H5: OOF-Based Weighted Blend

Hypothesis: Equal blend (0.5/0.5) is suboptimal. Per-target OOF-based weighting
of XGB/LGBM should improve avg_gap.

Methods:
1. Equal blend (0.5/0.5) — baseline
2. OOF-based weight: weight_XGB = exp(-oof_XGB) / (exp(-oof_XGB) + exp(-oof_LGBM))
3. Fixed weight grid per target type (Q vs S): [0.3/0.7, 0.4/0.6, 0.5/0.5, 0.6/0.4, 0.7/0.3]

Uses V534 BEST config as base.
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


def run_target_pipeline(target, config, ranked_features, test_feat_cols,
                        train_df, test_df, gkf, xgb_cfgs, lgbm_cfgs):
    """Run one target with all seeds, return raw XGB/LGBM OOFs and test preds."""
    n_train = len(train_df)
    n_test = len(test_df)
    bc = config[target]
    sel_cols = ranked_features[target][:bc['n_feat']]
    feat_names = [c for c in sel_cols if c in test_feat_cols]
    if len(feat_names) != len(sel_cols):
        sel_cols = feat_names

    n_est = bc['n_est']
    xgb_params = {'n_estimators': n_est, **xgb_cfgs[bc['xgb_cfg']]}
    lgbm_params = {'n_estimators': n_est, **lgbm_cfgs[bc['lgbm_cfg']]}

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

    return {
        'target': target,
        'xgb_seed_oofs': xgb_seed_oofs,
        'lgbm_seed_oofs': lgbm_seed_oofs,
        'xgb_test_preds': xgb_test_preds,
        'lgbm_test_preds': lgbm_test_preds,
        'y': y,
        'n_est': n_est,
    }


def compute_oof_weight(xgb_oofs, lgbm_oofs, y):
    """Compute OOF-based blend weight using exp(-log_loss)."""
    ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs])
    ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs])
    # OOF-based: weight = exp(-loss) / sum(exp(-loss))
    weight_xgb = np.exp(-ll_xgb) / (np.exp(-ll_xgb) + np.exp(-ll_lgbm))
    return weight_xgb, ll_xgb, ll_lgbm


def compute_gap_with_blend(oofs_arr, y, alpha=0.01):
    """Compute gap with Ridge meta on blended OOFs."""
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
    return avg_student, meta_ll, avg_student - meta_ll


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V536-H5: OOF-Based Weighted Blend")
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
    test_feat_cols = get_feature_cols(test_df)
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Pre-rank
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # Run all targets (XGB and LGBM raw predictions)
    log.info("\n--- Training all targets (XGB + LGBM, 13 seeds, 5 folds) ---")
    all_data = {}
    for target in TARGETS:
        log.info(f"  Training {target}...")
        all_data[target] = run_target_pipeline(
            target, V534_CONFIG, ranked_features, test_feat_cols,
            train_df, test_df, gkf, XGB_CFGS, LGBM_CFGS
        )
        log.info(f"  {target}: done (n_est={all_data[target]['n_est']})")

    # ---- Method 1: Equal blend (0.5/0.5) baseline ----
    log.info("\n" + "=" * 70)
    log.info("Method 1: Equal blend (0.5/0.5) — baseline")
    log.info("=" * 70)
    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    equal_results = {}
    for target in TARGETS:
        d = all_data[target]
        w = 0.5
        blended_oofs = [w * xo + (1-w) * lo for xo, lo in zip(d['xgb_seed_oofs'], d['lgbm_seed_oofs'])]
        _, _, gap = compute_gap_with_blend(blended_oofs, d['y'])
        equal_results[target] = {'weight_xgb': w, 'blend': '0.5/0.5', 'gap': gap}
        v308_ok = "✅" if gap < v308_gaps[target] else "❌"
        log.info(f"  {target}: gap={gap:+.5f} (V308={v308_gaps[target]:.3f}) {v308_ok}")

    avg_equal = np.mean([v['gap'] for v in equal_results.values()])
    vs308_equal = sum(1 for t, v in equal_results.items() if v['gap'] < v308_gaps[t])
    log.info(f"  avg_gap: {avg_equal:+.5f}, vs308: {vs308_equal}/7")

    # ---- Method 2: OOF-based weight ----
    log.info("\n" + "=" * 70)
    log.info("Method 2: OOF-based weight (exp(-log_loss))")
    log.info("=" * 70)
    oof_results = {}
    for target in TARGETS:
        d = all_data[target]
        wx, ll_xgb, ll_lgbm = compute_oof_weight(d['xgb_seed_oofs'], d['lgbm_seed_oofs'], d['y'])
        w = float(wx)
        blended_oofs = [w * xo + (1-w) * lo for xo, lo in zip(d['xgb_seed_oofs'], d['lgbm_seed_oofs'])]
        _, _, gap = compute_gap_with_blend(blended_oofs, d['y'])
        oof_results[target] = {'weight_xgb': w, 'blend': f'{w:.3f}/{1-w:.3f}', 'gap': gap, 'll_xgb': ll_xgb, 'll_lgbm': ll_lgbm}
        v308_ok = "✅" if gap < v308_gaps[target] else "❌"
        log.info(f"  {target}: wx={w:.3f} (ll_xgb={ll_xgb:.4f}, ll_lgbm={ll_lgbm:.4f}), gap={gap:+.5f} (V308={v308_gaps[target]:.3f}) {v308_ok}")

    avg_oof = np.mean([v['gap'] for v in oof_results.values()])
    vs308_oof = sum(1 for t, v in oof_results.items() if v['gap'] < v308_gaps[t])
    log.info(f"  avg_gap: {avg_oof:+.5f}, vs308: {vs308_oof}/7")

    # ---- Method 3: Fixed weight grid per target type (Q vs S) ----
    log.info("\n" + "=" * 70)
    log.info("Method 3: Fixed weight grid per target type")
    log.info("=" * 70)

    Q_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]  # XGB weights for Q targets
    S_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]  # XGB weights for S targets

    grid_results = {}
    for qw in Q_WEIGHTS:
        for sw in S_WEIGHTS:
            total_gap = 0
            target_gaps_grid = {}
            for target in TARGETS:
                d = all_data[target]
                w = qw if target.startswith('Q') else sw
                blended_oofs = [w * xo + (1-w) * lo for xo, lo in zip(d['xgb_seed_oofs'], d['lgbm_seed_oofs'])]
                _, _, gap = compute_gap_with_blend(blended_oofs, d['y'])
                target_gaps_grid[target] = gap
                total_gap += gap

            avg_g = total_gap / 7
            vs308_g = sum(1 for t, g in target_gaps_grid.items() if g < v308_gaps[t])
            key = f'Q{qw}/S{sw}'
            grid_results[key] = {
                'Q_weight_xgb': qw, 'S_weight_xgb': sw,
                'avg_gap': avg_g, 'target_gaps': target_gaps_grid, 'vs308': vs308_g
            }
            v_marker = " 🎯" if avg_g < avg_equal else ""
            log.info(f"  {key}: avg_gap={avg_g:+.5f}, vs308={vs308_g}/7{v_marker}")

    # Find best grid
    best_grid_key = min(grid_results, key=lambda k: grid_results[k]['avg_gap'])
    best_grid = grid_results[best_grid_key]
    log.info(f"\n  Best grid: {best_grid_key} avg_gap={best_grid['avg_gap']:+.5f}")

    # ---- Compare all methods ----
    log.info("\n" + "=" * 70)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 70)

    all_methods = {
        'Equal (0.5/0.5)': avg_equal,
        'OOF-weighted': avg_oof,
        f'Best Grid ({best_grid_key})': best_grid['avg_gap'],
    }

    for name, avg_g in sorted(all_methods.items(), key=lambda x: x[1]):
        improved = "✅" if avg_g < avg_equal else "❌"
        delta = avg_g - avg_equal
        log.info(f"  {name}: avg_gap={avg_g:+.5f} (Δ from equal={delta:+.5f}) {improved}")

    # ---- Best overall ----
    best_method_name = min(all_methods, key=all_methods.get)
    best_avg = all_methods[best_method_name]

    log.info(f"\n🏆 Best method: {best_method_name} with avg_gap={best_avg:+.5f}")

    # ---- Save submission with best method ----
    log.info("\n--- Generating submission with best method ---")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})

    for target in TARGETS:
        d = all_data[target]
        if best_method_name == 'Equal (0.5/0.5)':
            w = 0.5
        elif best_method_name == 'OOF-weighted':
            w = oof_results[target]['weight_xgb']
        else:
            w = best_grid['target_gaps'].get('_qw')  # won't use this path directly
            # Actually extract from best_grid
            if target.startswith('Q'):
                w = best_grid['Q_weight_xgb']
            else:
                w = best_grid['S_weight_xgb']

        blended_test = [w * xt + (1-w) * lt for xt, lt in zip(d['xgb_test_preds'], d['lgbm_test_preds'])]
        sub_df[target] = np.mean(blended_test, axis=0)

    sub_path = SUBMIT / f'submission_v536_h5_{best_method_name.replace(" ", "_").replace("/", "_")}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"📁 Submission saved: {sub_path}")

    # ---- Save results ----
    # Detailed per-target results
    detailed_equal = {t: {'weight_xgb': v['weight_xgb'], 'blend': v['blend'], 'gap': v['gap']} for t, v in equal_results.items()}
    detailed_oof = {t: {'weight_xgb': v['weight_xgb'], 'blend': v['blend'], 'gap': v['gap'], 'll_xgb': v['ll_xgb'], 'll_lgbm': v['ll_lgbm']} for t, v in oof_results.items()}

    result = {
        'version': 'V536_H5',
        'hypothesis': 'oof_weighted_blend',
        'description': 'OOF-based weighted blend of XGB/LGBM — H5 of V536',
        'v308_gaps': v308_gaps,
        'methods': {
            'equal_blend': {
                'avg_gap': float(avg_equal),
                'vs308': vs308_equal,
                'target_gaps': {t: float(v['gap']) for t, v in equal_results.items()},
            },
            'oof_weighted': {
                'avg_gap': float(avg_oof),
                'vs308': vs308_oof,
                'target_gaps': {t: float(v['gap']) for t, v in oof_results.items()},
                'weights': {t: float(v['weight_xgb']) for t, v in oof_results.items()},
                'loglosses': {
                    t: {'xgb': float(v['ll_xgb']), 'lgbm': float(v['ll_lgbm'])}
                    for t, v in oof_results.items()
                },
            },
            'grid_search': {
                'results': {k: {
                    'avg_gap': float(v['avg_gap']),
                    'vs308': v['vs308'],
                    'target_gaps': {t: float(vv) for t, vv in v['target_gaps'].items()},
                } for k, v in grid_results.items()},
                'best': best_grid_key,
                'best_avg_gap': float(best_grid['avg_gap']),
            },
        },
        'comparison': {k: float(v) for k, v in all_methods.items()},
        'best_method': best_method_name,
        'best_avg_gap': float(best_avg),
        'config': {k: v for k, v in V534_CONFIG.items()},
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }

    result_path = EXPERIMENTS / f'v536_h5_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")

    return result

if __name__ == '__main__':
    result = main()
    if result:
        # Also save to the canonical results file
        import shutil
        src = EXPERIMENTS / f'v536_h5_{result["timestamp"]}.json'
        dst = ROOT / 'memory' / 'v536_h5_results.json'
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        log.info(f"Results also saved to: {dst}")
