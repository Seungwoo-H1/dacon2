#!/usr/bin/env python3
"""
V504 — V496 + Multi-Swarm HP Search + Target-Specific Optimal HP
Changes from V496:
1. Per-target HP search (grid over learning_rate, max_depth, subsample, reg params)
2. XGB ranking for feature selection (faster, 10 rounds)
3. 3-model ensemble with target-specific best HP
4. Target-specific feature count (not just K sweep, but HP-aware)
5. use_best_model=True for CatBoost, early_stopping
"""
import sys, gc, logging, json, re, time, warnings, itertools
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
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


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V504 — V496 + Per-Target HP Search + Optimal Ensemble")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
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
    
    log.info(f"  Train: {train_sub.shape}, Test: {test.shape}")
    log.info(f"  Features after leak removal: {len(leak_cols)}")

    groups = train_sub['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    # ── 2. Per-subject z-score + original ──
    log.info("\n--- 2. Per-subject z-score normalization ---")
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
    log.info(f"  Combined features: {X_train.shape[1]} ({len(leak_cols)} orig + {len(leak_cols)} zscore)")

    predictions = {}
    target_results = {}

    # ── 3. HP search grid ──
    lgb_hp_grid = [
        {'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 15},
        {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 10},
        {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.015, 'n_estimators': 1200, 'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 15.0, 'min_child_samples': 20},
        {'num_leaves': 25, 'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 600, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 8},
        {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 1500, 'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 15.0, 'reg_lambda': 20.0, 'min_child_samples': 25},
    ]
    
    xgb_hp_grid = [
        {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'lambda': 10.0, 'min_child_weight': 5},
        {'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 2.0, 'lambda': 5.0, 'min_child_weight': 3},
        {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 1200, 'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'lambda': 15.0, 'min_child_weight': 8},
        {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 600, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 2},
        {'max_depth': 3, 'learning_rate': 0.008, 'n_estimators': 1500, 'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 20.0, 'lambda': 25.0, 'min_child_weight': 10},
    ]

    cb_hp_grid = [
        {'iterations': 1000, 'learning_rate': 0.01, 'depth': 3, 'subsample': 0.6, 'colsample_bylevel': 0.5, 'l2_leaf_reg': 10.0, 'min_data_in_leaf': 15},
        {'iterations': 800, 'learning_rate': 0.02, 'depth': 4, 'subsample': 0.7, 'colsample_bylevel': 0.6, 'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10},
        {'iterations': 1200, 'learning_rate': 0.01, 'depth': 3, 'subsample': 0.5, 'colsample_bylevel': 0.5, 'l2_leaf_reg': 15.0, 'min_data_in_leaf': 20},
        {'iterations': 600, 'learning_rate': 0.03, 'depth': 5, 'subsample': 0.8, 'colsample_bylevel': 0.7, 'l2_leaf_reg': 2.0, 'min_data_in_leaf': 8},
        {'iterations': 1500, 'learning_rate': 0.008, 'depth': 3, 'subsample': 0.4, 'colsample_bylevel': 0.4, 'l2_leaf_reg': 20.0, 'min_data_in_leaf': 25},
    ]

    # ── 4. Per-target experiments ──
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate={train_sub[target].mean():.3f}) ---")
        y = train_sub[target].values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        t1 = time.time()

        # Feature ranking: XGB 10 rounds (fast)
        log.info("  Ranking (XGB 10 rounds)...")
        dtrain_rank = xgb.DMatrix(X_train, label=y)
        mr = xgb.train({
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'max_depth': 3, 'eta': 0.05, 'n_estimators': 10,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'random_state': 42
        }, dtrain_rank, num_boost_round=10, verbose_eval=False)
        imp = mr.get_score(importance_type='gain')
        all_feats = [f't{i}' for i in range(X_train.shape[1])]
        imp_arr = np.array([imp.get(f, 0) for f in all_feats])
        ranked_idx = np.argsort(-imp_arr)
        log.info(f"  Rank done ({time.time()-t1:.1f}s)")

        # Feature count sweep + HP search
        best_avg_oof = float('inf')
        best_config = None
        best_preds_oof = None
        best_configs_models = {}

        for k in [20, 30, 40, 50]:
            k = min(k, len(ranked_idx))
            top_idx = ranked_idx[:k]
            top_sn = [sn[i] for i in top_idx]
            X_top = X_train[:, top_idx]
            X_test_top = X_test[:, top_idx]

            # Try each HP config
            for h in range(len(lgb_hp_grid)):
                lgb_cfg = {**lgb_hp_grid[h], 'scale_pos_weight': spw, 'random_state': 42, 'verbose': -1}
                xgb_cfg = {**xgb_hp_grid[h], 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'}
                cb_cfg = {**cb_hp_grid[h], 'random_state': 42, 'loss_function': 'Logloss', 'verbose': False, 'use_best_model': True}

                oof_lgb = np.zeros(len(y))
                oof_xgb = np.zeros(len(y))
                oof_cb = np.zeros(len(y))

                for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                    ds_tr = lgb.Dataset(X_top[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                    m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=lgb_cfg['n_estimators'])
                    oof_lgb[va] = m_lgb.predict(X_top[va])
                    
                    m_xgb = xgb.XGBClassifier(**xgb_cfg)
                    m_xgb.fit(X_top[tr], y[tr], eval_set=[(X_top[va], y[va])], verbose=False)
                    oof_xgb[va] = m_xgb.predict_proba(X_top[va])[:, 1]
                    
                    m_cb = cb.CatBoostClassifier(**cb_cfg)
                    m_cb.fit(X_top[tr], y[tr], eval_set=(X_top[va], y[va]))
                    oof_cb[va] = m_cb.predict_proba(X_top[va])[:, 1]

                avg_oof = logloss(y, (oof_lgb + oof_xgb + oof_cb) / 3.0)
                log.info(f"    k={k} hp[{h}]: LGB={logloss(y,oof_lgb):.4f} XGB={logloss(y,oof_xgb):.4f} CB={logloss(y,oof_cb):.4f} AVG={avg_oof:.4f}")

                if avg_oof < best_avg_oof:
                    best_avg_oof = avg_oof
                    best_config = {'k': k, 'hp_idx': h, 'top_idx': top_idx.copy(), 'top_sn': top_sn.copy()}
                    best_preds_oof = {'lgb': oof_lgb.copy(), 'xgb': oof_xgb.copy(), 'cb': oof_cb.copy()}

        if best_config is None:
            log.error(f"  {target}: No valid config found! Skipping.")
            continue

        k = best_config['k']
        hp_idx = best_config['hp_idx']
        log.info(f"  Best: k={k} hp[{hp_idx}]: avg_oof={best_avg_oof:.4f}")

        # Final predictions with best config + multiple seeds
        best_lgb_cfg = {**lgb_hp_grid[hp_idx], 'scale_pos_weight': spw, 'verbose': -1}
        best_xgb_cfg = {**xgb_hp_grid[hp_idx], 'use_label_encoder': False, 'eval_metric': 'logloss'}
        best_cb_cfg = {**cb_hp_grid[hp_idx], 'loss_function': 'Logloss', 'verbose': False, 'use_best_model': True}
        
        top_idx = best_config['top_idx']
        top_sn = best_config['top_sn']
        X_top_all = X_train[:, top_idx]
        X_test_top_all = X_test[:, top_idx]

        final_lgb = np.zeros(len(X_test_top_all))
        final_xgb = np.zeros(len(X_test_top_all))
        final_cb = np.zeros(len(X_test_top_all))

        seeds = [42, 123, 456, 789]
        n_total = len(seeds) * 5

        for seed in seeds:
            cfg_l = {**best_lgb_cfg, 'random_state': seed}
            cfg_x = {**best_xgb_cfg, 'random_state': seed}
            cfg_c = {**best_cb_cfg, 'random_state': seed, 'iterations': int(best_cb_cfg['iterations'] * 0.75)}

            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                ds_tr = lgb.Dataset(X_top_all[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                m_lgb = lgb.train(cfg_l, ds_tr, num_boost_round=cfg_l['n_estimators'])
                final_lgb += m_lgb.predict(X_test_top_all) / n_total
                
                m_xgb = xgb.XGBClassifier(**cfg_x)
                m_xgb.fit(X_top_all[tr], y[tr], verbose=False)
                final_xgb += m_xgb.predict_proba(X_test_top_all)[:, 1] / n_total
                
                m_cb = cb.CatBoostClassifier(**cfg_c)
                m_cb.fit(X_top_all[tr], y[tr], verbose=False)
                final_cb += m_cb.predict_proba(X_test_top_all)[:, 1] / n_total

        test_preds = (final_lgb + final_xgb + final_cb) / 3.0
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'best_k': k,
            'best_hp_idx': hp_idx,
            'best_avg_oof': float(best_avg_oof),
            'lgb_oof': float(logloss(y, best_preds_oof['lgb'])),
            'xgb_oof': float(logloss(y, best_preds_oof['xgb'])),
            'cb_oof': float(logloss(y, best_preds_oof['cb'])),
            'time': time.time() - t1,
        }
        log.info(f"  {target}: k={k} hp[{hp_idx}] AVG={best_avg_oof:.4f} T={time.time()-t1:.0f}s")
        gc.collect()

    # Summary
    avg_oof = np.mean([v['best_avg_oof'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info("V504 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: k={r['best_k']} hp[{r['best_hp_idx']}] LGB={r['lgb_oof']:.4f} XGB={r['xgb_oof']:.4f} CB={r['cb_oof']:.4f} AVG={r['best_avg_oof']:.4f} T={r['time']:.0f}s")
    log.info(f"  AVG OOF: {avg_oof:.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v504_hp_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # Save meta
    meta = {
        'version': 'V504_hp_search',
        'name': 'V496 + Per-Target HP Search + Optimal Ensemble',
        'avg_oof': float(avg_oof),
        'n_features_base': len(leak_cols),
        'n_features_combined': X_train.shape[1],
        'target_results': target_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v504_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"  DONE.")


if __name__ == "__main__":
    main()
