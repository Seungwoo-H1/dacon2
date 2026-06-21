#!/usr/bin/env python3
"""
V536-H4 — CatBoost as Target-Specific Learner

Hypothesis: CatBoost generalizes better than XGB on small n=450,
especially for S1, S2 (small n_feat targets that overfit most).

Variants:
- Variant B: XGB for Q, CatBoost for S (swap out LGBM→CatBoost for S targets)
- Variant C: CatBoost for S1, S2 only (most overfit targets)
- Baseline: V534 config (XGB+LGBM blend) for reference

CatBoost params: iterations=300, depth=6, learning_rate=0.03, random_seed=42
Ridge α=0.01 meta (same as V534)

Reduced to 10 seeds for CatBoost targets to keep runtime reasonable.
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
import catboost as cb

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
N_SEEDS = 13  # for XGB/LGBM
N_SEEDS_CB = 10  # reduced for CatBoost (slower)

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


def train_one_catboost(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, depth=6, learning_rate=0.03):
    """Train CatBoost model."""
    sn = [sanitize_col(c) for c in feat_names]
    train_pool = cb.Pool(X_tr, label=y_tr, feature_names=sn)
    test_pool = cb.Pool(X_test, feature_names=sn)
    
    model = cb.CatBoostClassifier(
        iterations=n_est,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=seed,
        verbose=-1,
        allow_writing_files=False,
        loss_function='Logloss',
    )
    model.fit(train_pool, silent=True)
    
    probs_va = model.predict_proba(X_va)[:, 1]
    probs_te = model.predict_proba(X_test)[:, 1]
    return probs_va, probs_te


def train_one(seed, X_tr, y_tr, X_va, X_test, feat_names, learner, n_est, **mp):
    if learner == 'xgb':
        params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
        ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
        ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
        m = xgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(ds_va), m.predict(ds_te)
    elif learner == 'catboost':
        depth = mp.get('depth', 6)
        lr = mp.get('learning_rate', 0.03)
        return train_one_catboost(seed, X_tr, y_tr, X_va, X_test, feat_names, n_est, depth, lr)
    else:
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
                 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feat_names]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(X_va), m.predict(X_test)


# V534 BEST config (baseline)
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


def run_variant(target_configs, variant_name, train_df, test_df, gkf, ranked_features,
                test_feat_cols, N_FOLDS, N_SEEDS, meta_alpha=0.01):
    """
    Run one variant. Each target uses a single learner (XGB, LGBM, or CatBoost).
    
    For CatBoost targets, uses fewer seeds (N_SEEDS_CB=10).
    """
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}

    for ti, target in enumerate(TARGETS):
        bc = target_configs[target]
        learner = bc['learner']
        n_feat = bc['n_feat']
        n_est = bc['n_est']
        cfg_key = bc['cfg_key']
        
        # Use fewer seeds for CatBoost
        n_seeds = N_SEEDS_CB if learner == 'catboost' else N_SEEDS

        sel_cols = ranked_features[target][:n_feat]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols):
            sel_cols = feat_names

        y = train_df[target].values.astype(np.float64)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        seed_oofs = []
        test_preds = []

        for si in range(n_seeds):
            seed = SEED + si * 11
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]

                pv, pt = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, learner, n_est,
                                   **(XGB_CFGS.get(cfg_key, LGBM_CFGS.get(cfg_key, {}))))

                seed_oof[va_idx] = pv
                seed_test += pt

            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS

            seed_oofs.append(seed_oof)
            test_preds.append(seed_test)

            learner_tag = learner.upper()
            sys.stdout.write(f"\r  {variant_name} {target}: {learner_tag} seed {si+1}/{n_seeds}")
            sys.stdout.flush()

        all_seed_oofs[target] = seed_oofs
        all_test_preds[target] = test_preds

        learner_tag = learner.upper()
        log.info(f"\r  {variant_name} {target}: done ({learner_tag} n_feat={n_feat} n_est={n_est} n_seeds={n_seeds})")

    # Compute gaps with Ridge meta
    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    total_gap = 0
    target_gaps = {}
    
    for t in TARGETS:
        oofs_2d = np.column_stack(all_seed_oofs[t])
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        meta = Ridge(alpha=meta_alpha)
        meta.fit(X_meta, train_df[t].values)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_proba = np.clip(train_proba, 0.001, 0.999)
        meta_ll = log_loss(train_df[t].values, train_proba)
        avg_student = np.mean([log_loss(train_df[t].values, np.clip(so, 0.001, 0.999)) for so in all_seed_oofs[t]])
        gap = avg_student - meta_ll
        target_gaps[t] = gap
        total_gap += gap
        vs = "✅" if gap < v308_gaps[t] else "❌"
        log.info(f"  {t}: gap={gap:+.5f} (V308={v308_gaps[t]:.3f}) {vs}")

    avg_gap = total_gap / 7
    vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
    log.info(f"\n  {variant_name} avg_gap: {avg_gap:.5f}, vs308: {vs308}/7")
    log.info(f"  {variant_name} Expected LB: {0.62235 - avg_gap:.5f}")

    return {
        'variant': variant_name,
        'avg_gap': float(avg_gap),
        'target_gaps': {k: float(v) for k, v in target_gaps.items()},
        'vs308': vs308,
        'expected_lb': float(0.62235 - avg_gap),
        'all_seed_oofs': all_seed_oofs,
        'all_test_preds': all_test_preds,
    }


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V536-H4 — CatBoost as Target-Specific Learner")
    log.info("=" * 70)
    log.info("Hypothesis: CatBoost generalizes better than XGB on small n=450")
    log.info("CatBoost: iterations=300, depth=6, lr=0.03 | Seeds: XGB=13, CB=10")

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

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

    # Pre-rank features
    log.info("Pre-ranking features...")
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # ─── Variant B: XGB for Q, CatBoost for S ───
    log.info("\n" + "=" * 60)
    log.info("Variant B: XGB for Q, CatBoost for S")
    log.info("=" * 60)

    var_b_configs = {}
    for t in TARGETS:
        bc = V534_CONFIG[t]
        if t.startswith('Q'):
            var_b_configs[t] = {
                'learner': 'xgb',
                'n_feat': bc['n_feat'],
                'cfg_key': bc['xgb_cfg'],
                'n_est': bc['n_est'],
            }
        else:  # S targets → CatBoost, n_est=300 (faster)
            var_b_configs[t] = {
                'learner': 'catboost',
                'n_feat': bc['n_feat'],
                'cfg_key': bc['xgb_cfg'],
                'n_est': 300,  # Reduced from 500 for runtime
            }

    for t in TARGETS:
        c = var_b_configs[t]
        log.info(f"  {t}: n_feat={c['n_feat']}, learner={c['learner']}, n_est={c['n_est']}")

    result_b = run_variant(var_b_configs, "Variant_B_XGB-Q_CatBoost-S",
                           train_df, test_df, gkf, ranked_features, test_feat_cols,
                           N_FOLDS, N_SEEDS, meta_alpha=0.01)

    # ─── Variant C: CatBoost for S1, S2 only ───
    log.info("\n" + "=" * 60)
    log.info("Variant C: CatBoost for S1, S2 only")
    log.info("=" * 60)

    var_c_configs = {}
    for t in TARGETS:
        bc = V534_CONFIG[t]
        if t in ('S1', 'S2'):
            var_c_configs[t] = {
                'learner': 'catboost',
                'n_feat': bc['n_feat'],
                'cfg_key': bc['xgb_cfg'],
                'n_est': 300,
            }
        elif t.startswith('Q'):
            var_c_configs[t] = {
                'learner': 'xgb',
                'n_feat': bc['n_feat'],
                'cfg_key': bc['xgb_cfg'],
                'n_est': bc['n_est'],
            }
        else:  # S3, S4 → LGBM (same as V534)
            var_c_configs[t] = {
                'learner': 'lgbm',
                'n_feat': bc['n_feat'],
                'cfg_key': bc['lgbm_cfg'],
                'n_est': bc['n_est'],
            }

    for t in TARGETS:
        c = var_c_configs[t]
        log.info(f"  {t}: n_feat={c['n_feat']}, learner={c['learner']}, n_est={c['n_est']}")

    result_c = run_variant(var_c_configs, "Variant_C_CatBoost-S1S2",
                           train_df, test_df, gkf, ranked_features, test_feat_cols,
                           N_FOLDS, N_SEEDS, meta_alpha=0.01)

    # ─── Baseline: V534 config (XGB+LGBM blend) ───
    log.info("\n" + "=" * 60)
    log.info("Baseline: V534 (XGB+LGBM blend) — for comparison")
    log.info("=" * 60)
    
    n_train = len(train_df)
    n_test = len(test_df)
    baseline_oofs = {t: [] for t in TARGETS}
    baseline_test_preds = {t: [] for t in TARGETS}

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

        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_seed_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_seed_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10)
        inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        wx = inv_xgb / (inv_xgb + inv_lgbm)

        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_seed_oofs, lgbm_seed_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_test_preds, lgbm_test_preds)]

        baseline_oofs[target] = blended_oofs
        baseline_test_preds[target] = blended_tests

        log.info(f"  {target}: blend wx={wx:.3f}")

    # Baseline gaps
    log.info("\n  Baseline gaps (V534 XGB+LGBM):")
    v308_gaps = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}
    baseline_total_gap = 0
    baseline_target_gaps = {}
    
    for t in TARGETS:
        oofs_2d = np.column_stack(baseline_oofs[t])
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        meta = Ridge(alpha=0.01)
        meta.fit(X_meta, train_df[t].values)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_proba = np.clip(train_proba, 0.001, 0.999)
        meta_ll = log_loss(train_df[t].values, train_proba)
        avg_student = np.mean([log_loss(train_df[t].values, np.clip(so, 0.001, 0.999)) for so in baseline_oofs[t]])
        gap = avg_student - meta_ll
        baseline_target_gaps[t] = gap
        baseline_total_gap += gap
        vs = "✅" if gap < v308_gaps[t] else "❌"
        log.info(f"    {t}: gap={gap:+.5f} (V308={v308_gaps[t]:.3f}) {vs}")

    baseline_avg_gap = baseline_total_gap / 7
    baseline_vs308 = sum(1 for t in TARGETS if baseline_target_gaps[t] < v308_gaps[t])
    log.info(f"\n  Baseline avg_gap: {baseline_avg_gap:.5f}, vs308: {baseline_vs308}/7")
    log.info(f"  Baseline Expected LB: {0.62235 - baseline_avg_gap:.5f}")

    # ─── Save submission (best variant) ───
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = [result_b, result_c]
    best = min(results, key=lambda r: r['avg_gap'])
    log.info(f"\n🏆 Best variant: {best['variant']} (avg_gap={best['avg_gap']:.5f})")

    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    for t in TARGETS:
        sub_df[t] = np.mean(best['all_test_preds'][t], axis=0)

    sub_path = SUBMIT / f'submission_v536_h4_{best["variant"].replace(" ", "").replace("-", "_")}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n📁 Submission saved: {sub_path}")

    # ─── Save results ───
    result_summary = {
        'version': 'V536_H4',
        'hypothesis': 'catboost_learner',
        'description': 'CatBoost as target-specific learner, replacing XGB/LGBM',
        'variants': [],
        'baseline_v534': {
            'avg_gap': float(baseline_avg_gap),
            'target_gaps': {k: float(v) for k, v in baseline_target_gaps.items()},
            'vs308': baseline_vs308,
            'expected_lb': float(0.62235 - baseline_avg_gap),
        },
        'best_variant': best['variant'],
        'best_avg_gap': float(best['avg_gap']),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 1),
    }

    for r in results:
        variant_data = {
            'variant': r['variant'],
            'avg_gap': r['avg_gap'],
            'target_gaps': r['target_gaps'],
            'vs308': r['vs308'],
            'expected_lb': r['expected_lb'],
        }
        result_summary['variants'].append(variant_data)

    result_path = EXPERIMENTS / f'v536_h4_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2, default=str)
    log.info(f"Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result_summary

if __name__ == '__main__':
    main()
