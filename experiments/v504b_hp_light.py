#!/usr/bin/env python3
"""
V504b — V496 + Lighter HP Search (5 configs × 2 k values = 10 per target)
Fixes from V504: too many configs. Reduced to fast sweep.
Also: fixes CatBoost use_best_model issue (use correct param name).
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
except ImportError:
    print("ERROR: Required packages not installed")
    sys.exit(1)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout, force=True)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

LEAK_REMOVE = {
    'wHr_hr_median',
    'wLight_w_light_sum',
    'mActivity_m_activity_sum',
}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_', n)


def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def per_subject_zscore(df, feature_cols, subject_col='subject_id'):
    result = df[feature_cols].copy()
    for col in feature_cols:
        for subj, grp in df.groupby(subject_col)[col]:
            mean = grp.mean()
            std = grp.std()
            if std < 1e-8:
                result.loc[grp.index, col] = 0
            else:
                result.loc[grp.index, col] = (grp.values - mean) / std
    return result


def run_ensemble(X_tr, y_tr, X_va, y_va, X_te, lgb_cfg, xgb_cfg, cb_cfg, spw, top_sn):
    """Run 3-model ensemble, return oof_preds and test_preds."""
    oof_lgb = np.zeros(len(y_va))
    oof_xgb = np.zeros(len(y_va))
    oof_cb = np.zeros(len(X_te))

    # LGBM
    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=top_sn, params={'verbose': '-1'})
    m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=lgb_cfg['n_estimators'])
    oof_lgb[:] = m_lgb.predict(X_va)
    final_lgb = m_lgb.predict(X_te)

    # XGB
    m_xgb = xgb.XGBClassifier(**xgb_cfg)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[:] = m_xgb.predict_proba(X_va)[:, 1]
    final_xgb = m_xgb.predict_proba(X_te)[:, 1]

    # CB
    cb_cfg_run = {**cb_cfg, 'eval_set': [(X_va, y_va)]}
    m_cb = cb.CatBoostClassifier(**cb_cfg)
    m_cb.fit(X_tr, y_tr, verbose=False)
    oof_cb[:] = m_cb.predict_proba(X_va)[:, 1]
    final_cb = m_cb.predict_proba(X_te)[:, 1]

    oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
    return oof_avg, logloss(y_va, oof_avg), final_lgb, final_xgb, final_cb


def main():
    # Force flush
    import builtins
    old_print = builtins.print
    def print_flush(*args, **kwargs):
        old_print(*args, **kwargs)
        kwargs.get('file', sys.stdout).flush()
    builtins.print = print_flush

    t_start = time.time()
    print_flush("=" * 70)
    print_flush("V504b — V496 + Light HP Search (5 configs × 2 k values)")
    print_flush("=" * 70)

    # ── 1. Load data ──
    print_flush("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")

    target_cols_set = set(TARGETS)
    meta_cols = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    feature_cols_all = [c for c in train.columns
                        if c not in target_cols_set and c not in meta_cols
                        and np.issubdtype(train[c].dtype, np.number)]
    test_numeric = [c for c in test.columns if np.issubdtype(test[c].dtype, np.number)]
    common_cols = [c for c in feature_cols_all if c in test_numeric]
    leak_cols = [c for c in common_cols if c not in LEAK_REMOVE]

    train_sub = train[leak_cols + list(target_cols_set) + ['subject_id']]
    test_cols_for_model = [c for c in leak_cols if c in test.columns]
    test = test[['subject_id'] + test_cols_for_model] if 'subject_id' in test.columns else test[test_cols_for_model]

    groups = train_sub['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    # ── 2. Per-subject z-score + original ──
    print_flush("\n--- 2. Per-subject z-score normalization ---")
    train_z = per_subject_zscore(train_sub, leak_cols)
    test_z = per_subject_zscore(test, leak_cols)

    train_orig = train_sub[leak_cols].fillna(0).values.astype(np.float64)
    test_orig = test[leak_cols].fillna(0).values.astype(np.float64)
    train_z_vals = train_z.fillna(0).values.astype(np.float64)
    test_z_vals = test_z.fillna(0).values.astype(np.float64)

    X_train = np.hstack([train_orig, train_z_vals])
    X_test = np.hstack([test_orig, test_z_vals])

    feature_names = leak_cols + [f'{c}_zscore' for c in leak_cols]
    sn = [sanitize(c) for c in feature_names]
    print_flush(f"  Combined features: {X_train.shape[1]} ({len(leak_cols)} orig + {len(leak_cols)} zscore)")

    # ── 3. HP configs (5 configs, light) ──
    hp_configs = [
        # Aggressive regularization (conservative)
        dict(lgb=dict(num_leaves=10, max_depth=3, lr=0.01, n_est=1200, subsample=0.5, colsample=0.5, ra=10, rl=15, mcs=20),
             xgb=dict(max_depth=3, lr=0.01, n_est=1200, subsample=0.5, colsample=0.5, ra=10, lam=15, mcw=8),
             cb=dict(iter=1200, lr=0.01, depth=3, subsample=0.5, colsample=0.5, l2r=15, mdl=20)),
        # Medium
        dict(lgb=dict(num_leaves=15, max_depth=4, lr=0.02, n_est=800, subsample=0.7, colsample=0.7, ra=2, rl=5, mcs=10),
             xgb=dict(max_depth=4, lr=0.02, n_est=800, subsample=0.7, colsample=0.7, ra=2, lam=5, mcw=3),
             cb=dict(iter=800, lr=0.02, depth=4, subsample=0.7, colsample=0.7, l2r=5, mdl=10)),
        # Light regularization (more expressive)
        dict(lgb=dict(num_leaves=25, max_depth=5, lr=0.03, n_est=500, subsample=0.8, colsample=0.8, ra=0.5, rl=2, mcs=5),
             xgb=dict(max_depth=5, lr=0.03, n_est=500, subsample=0.8, colsample=0.8, ra=1, lam=2, mcw=1),
             cb=dict(iter=500, lr=0.03, depth=5, subsample=0.8, colsample=0.8, l2r=2, mdl=5)),
        # Very aggressive (minimal overfit)
        dict(lgb=dict(num_leaves=8, max_depth=3, lr=0.008, n_est=2000, subsample=0.4, colsample=0.4, ra=20, rl=30, mcs=30),
             xgb=dict(max_depth=3, lr=0.008, n_est=2000, subsample=0.4, colsample=0.4, ra=20, lam=25, mcw=12),
             cb=dict(iter=2000, lr=0.008, depth=3, subsample=0.4, colsample=0.4, l2r=25, mdl=30)),
        # Balanced
        dict(lgb=dict(num_leaves=18, max_depth=4, lr=0.015, n_est=1000, subsample=0.6, colsample=0.6, ra=5, rl=8, mcs=12),
             xgb=dict(max_depth=4, lr=0.015, n_est=1000, subsample=0.6, colsample=0.6, ra=5, lam=8, mcw=4),
             cb=dict(iter=1000, lr=0.015, depth=4, subsample=0.6, colsample=0.6, l2r=8, mdl=12)),
    ]

    predictions = {}
    target_results = {}

    for ti, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"--- [{ti+1}/7] {target} (rate={train_sub[target].mean():.3f}) ---")
        y = train_sub[target].values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        t1 = time.time()

        # Feature ranking: LGBM 50 rounds (fast)
        log.info("  Ranking (LGBM 50 rounds)...")
        ds_rank = lgb.Dataset(X_train, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                                'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.03,
                                'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
                                'random_state': 42, 'min_child_samples': 10},
                               ds_rank, num_boost_round=50)
        imp = np.array(model_rank.feature_importance(importance_type='gain'))
        imp_arr = np.zeros(X_train.shape[1])
        imp_arr[:len(imp)] = imp
        ranked_idx = np.argsort(-imp_arr)
        log.info(f"  Rank done ({time.time()-t1:.1f}s)")

        # HP search: k=[20, 40], 5 HP configs
        best_avg_oof = float('inf')
        best_config = None
        best_oof_lgb = None
        best_oof_xgb = None
        best_oof_cb = None

        for k in [20, 40]:
            k = min(k, len(ranked_idx))
            top_idx = ranked_idx[:k]
            top_sn = [sn[i] for i in top_idx]
            X_top = X_train[:, top_idx]

            for ci, hc in enumerate(hp_configs):
                # Normalize param names
                lgb_cfg = {
                    'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1,
                    'n_estimators': hc['lgb']['n_est'],
                    'learning_rate': hc['lgb']['lr'],
                    'num_leaves': hc['lgb']['num_leaves'],
                    'max_depth': hc['lgb']['max_depth'],
                    'subsample': hc['lgb']['subsample'],
                    'colsample_bytree': hc['lgb']['colsample'],
                    'reg_alpha': hc['lgb']['ra'],
                    'reg_lambda': hc['lgb']['rl'],
                    'min_child_samples': hc['lgb']['mcs'],
                }
                xgb_cfg = {
                    'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss',
                    'n_estimators': hc['xgb']['n_est'],
                    'learning_rate': hc['xgb']['lr'],
                    'max_depth': hc['xgb']['max_depth'],
                    'subsample': hc['xgb']['subsample'],
                    'colsample_bytree': hc['xgb']['colsample'],
                    'reg_alpha': hc['xgb']['ra'],
                    'reg_lambda': hc['xgb']['lam'],
                    'max_weight_per_category': hc['xgb']['mcw'],
                }
                cb_cfg = {
                    'random_state': 42, 'loss_function': 'Logloss', 'verbose': False,
                    'n_estimators': hc['cb']['iter'],
                    'learning_rate': hc['cb']['lr'],
                    'depth': hc['cb']['depth'],
                    'subsample': hc['cb']['subsample'],
                    'colsample_bylevel': hc['cb']['colsample'],
                    'l2_leaf_reg': hc['cb']['l2r'],
                    'model_size_reg': hc['cb']['mdl'],
                }

                oof_lgb = np.zeros(len(y))
                oof_xgb = np.zeros(len(y))
                oof_cb = np.zeros(len(y))

                for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                    X_tr, X_va = X_top[tr], X_top[va]
                    y_tr, y_va = y[tr], y[va]

                    # LGBM
                    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=top_sn, params={'verbose': '-1'})
                    m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=lgb_cfg['n_estimators'])
                    oof_lgb[va] = m_lgb.predict(X_va)

                    # XGB
                    m_xgb = xgb.XGBClassifier(**xgb_cfg)
                    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    oof_xgb[va] = m_xgb.predict_proba(X_va)[:, 1]

                    # CB
                    m_cb = cb.CatBoostClassifier(**cb_cfg)
                    m_cb.fit(X_tr, y_tr, verbose=False)
                    oof_cb[va] = m_cb.predict_proba(X_va)[:, 1]

                avg_oof = logloss(y, (oof_lgb + oof_xgb + oof_cb) / 3.0)

                if fold == 4:  # Only print last fold to reduce output
                    log.info(f"    k={k} hp[{ci}]: AVG={avg_oof:.4f}")

                if avg_oof < best_avg_oof:
                    best_avg_oof = avg_oof
                    best_config = (ci, k, top_idx.copy(), top_sn.copy(), lgb_cfg, xgb_cfg, cb_cfg)
                    best_oof_lgb = oof_lgb.copy()
                    best_oof_xgb = oof_xgb.copy()
                    best_oof_cb = oof_cb.copy()

        ci, k, top_idx, top_sn, best_lgb_cfg, best_xgb_cfg, best_cb_cfg = best_config
        log.info(f"  ✓ Best: k={k} hp[{ci}] AVG={best_avg_oof:.4f}")

        # Final predictions: multi-seed ensemble with best config
        log.info("  Building final ensemble (4 seeds)...")
        final_lgb = np.zeros(len(X_test))
        final_xgb = np.zeros(len(X_test))
        final_cb = np.zeros(len(X_test))

        seeds = [42, 123, 456, 789]
        n_total = len(seeds) * 5

        for seed in seeds:
            cfg_l = {**best_lgb_cfg, 'random_state': seed}
            cfg_x = {**best_xgb_cfg, 'random_state': seed}
            cfg_c = {**best_cb_cfg, 'random_state': seed}
            X_top_all = X_train[:, top_idx]
            X_te_all = X_test[:, top_idx]

            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                ds_tr = lgb.Dataset(X_top_all[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                m_lgb = lgb.train(cfg_l, ds_tr, num_boost_round=cfg_l['n_estimators'])
                final_lgb += m_lgb.predict(X_te_all) / n_total

                m_xgb = xgb.XGBClassifier(**cfg_x)
                m_xgb.fit(X_top_all[tr], y[tr], verbose=False)
                final_xgb += m_xgb.predict_proba(X_te_all)[:, 1] / n_total

                m_cb = cb.CatBoostClassifier(**cfg_c)
                m_cb.fit(X_top_all[tr], y[tr], verbose=False)
                final_cb += m_cb.predict_proba(X_te_all)[:, 1] / n_total

        test_preds = (final_lgb + final_xgb + final_cb) / 3.0
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'best_k': int(k),
            'best_hp_idx': ci,
            'best_avg_oof': float(best_avg_oof),
            'lgb_oof': float(logloss(y, best_oof_lgb)),
            'xgb_oof': float(logloss(y, best_oof_xgb)),
            'cb_oof': float(logloss(y, best_oof_cb)),
            'time': time.time() - t1,
        }
        log.info(f"  {target}: k={k} hp[{ci}] AVG={best_avg_oof:.4f} L={target_results[target]['lgb_oof']:.4f} X={target_results[target]['xgb_oof']:.4f} C={target_results[target]['cb_oof']:.4f} T={time.time()-t1:.0f}s")
        del X_top_all, X_te_all, final_lgb, final_xgb, final_cb
        gc.collect()

    # Summary
    avg_oof = np.mean([v['best_avg_oof'] for v in target_results.values()])
    print_flush(f"\n{'='*70}")
    print_flush("V504b RESULTS")
    print_flush(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        print_flush(f"  {t}: k={r['best_k']} hp[{r['best_hp_idx']}] AVG={r['best_avg_oof']:.4f}")
    print_flush(f"  AVG OOF: {avg_oof:.4f}")
    print_flush(f"  Total: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v504b_hp_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    print_flush(f"  Submission: {sub_path}")

    # Save meta
    meta = {
        'version': 'V504b_hp_light',
        'name': 'V496 + Light HP Search (5 configs × 2 k values)',
        'avg_oof': float(avg_oof),
        'n_features_base': len(leak_cols),
        'n_features_combined': X_train.shape[1],
        'target_results': target_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v504b_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print_flush(f"  Meta: {meta_path}")
    print_flush(f"  DONE.")


if __name__ == "__main__":
    main()
