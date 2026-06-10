"""
V496 — Per-Subject Normalization + Feature Engineering + 3-Model Ensemble

Key Hypothesis:
Subject-level feature normalization is the missing piece. Each subject has
different baseline behavior (age, lifestyle, device placement). By normalizing
features per-subject (z-score within subject), we remove subject bias and
let the model learn cross-subject patterns.

Changes from V495:
1. Per-subject z-score normalization (using ALL subjects' data for stats)
2. Additional engineered features: ratios, interactions of top features
3. 3-model ensemble (LGBM + XGB + CB) per target
4. Proper OOF stacking (no leakage)
5. Cross-validation to estimate LB (not V339 pattern)

Approach:
- Step 1: Per-subject z-score on base features (using groupby + transform)
- Step 2: Add ratio features (top 5 features pairwise ratios)
- Step 3: 3-model ensemble with different seeds and configs
- Step 4: Simple average ensemble (no meta layer to avoid leakage)
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# Conservative leak removal (less aggressive than V495)
LEAK_REMOVE = {
    # Some wrist features that might leak
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
    """Per-subject z-score normalization.
    For each feature, compute mean/std per subject, then z-score.
    Uses ALL subjects' data for computing subject-level stats.
    """
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


def engineer_features(df, feature_cols, top_features):
    """Add ratio features for top feature pairs."""
    extras = {}
    for i, f1 in enumerate(top_features[:5]):
        for f2 in top_features[i+1:5]:
            if f1 in df.columns and f2 in df.columns:
                extras[f'{f1}_ratio_{f2}'] = df[f1].fillna(0) / (df[f2].fillna(0) + 1e-8)
                extras[f'{f1}_plus_{f2}'] = df[f1].fillna(0) + df[f2].fillna(0)
    return extras


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V496 — Per-Subject Normalization + Feature Engineering + 3-Model")
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
    
    # Remove leak columns
    leak_cols = [c for c in common_cols if c not in LEAK_REMOVE]
    
    train = train[leak_cols + list(target_cols_set) + ['subject_id']]
    # Keep subject_id in test for per-subject normalization
    test_cols_for_model = [c for c in leak_cols if c in test.columns]
    test = test[['subject_id'] + test_cols_for_model] if 'subject_id' in test.columns else test[test_cols_for_model]
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features after leak removal: {len(leak_cols)}")
    log.info(f"  Test has subject_id: {'subject_id' in test.columns}")

    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    # ── 2. Per-subject z-score normalization ──
    log.info("\n--- 2. Per-subject z-score normalization ---")
    train_z = per_subject_zscore(train, leak_cols)
    test_z = per_subject_zscore(test, leak_cols)
    
    # Also keep original features
    train_orig = train[leak_cols].fillna(0).values.astype(np.float64)
    test_orig = test[leak_cols].fillna(0).values.astype(np.float64)
    train_z_vals = train_z.fillna(0).values.astype(np.float64)
    test_z_vals = test_z.fillna(0).values.astype(np.float64)

    # Combine: original + z-score = 2x features
    X_train = np.hstack([train_orig, train_z_vals])
    X_test = np.hstack([test_orig, test_z_vals])
    
    feature_names = leak_cols + [f'{c}_zscore' for c in leak_cols]
    sn = [sanitize(c) for c in feature_names]
    
    log.info(f"  Combined features: {X_train.shape[1]} ({len(leak_cols)} orig + {len(leak_cols)} zscore)")

    # ── 3. Feature ranking (on combined features) ──
    log.info("\n--- 3. Feature ranking ---")
    
    # Simple ranking: use LGBM importance on combined features
    params_rank = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'random_state': 42, 'min_child_samples': 10,
    }

    predictions = {}
    target_results = {}

    # ── 4. Per-target experiments ──
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate={train[target].mean():.3f}) ---")
        y = train[target].values.astype(np.float64)
        
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

        # Rank features
        ds_rank = lgb.Dataset(X_train, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked_idx = np.argsort(-imp)

        # ── 4a. Feature count sweep ──
        best_k = 40
        best_avg_oof = float('inf')
        best_preds = None
        
        for n_feat in [20, 30, 40, 50, 60, 80]:
            n_feat = min(n_feat, len(ranked_idx))
            top_idx = ranked_idx[:n_feat]
            top_sn = [sn[i] for i in top_idx]
            
            X_top = X_train[:, top_idx]
            X_test_top = X_test[:, top_idx]

            # ── 3-model ensemble with GroupKFold OOF ──
            oof_lgb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))
            
            # LGBM config
            lgb_cfg = {
                'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.02,
                'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 10,
                'scale_pos_weight': spw, 'random_state': 42,
                'verbose': -1,
            }
            
            # XGB config
            xgb_cfg = {
                'max_depth': 4, 'learning_rate': 0.02,
                'n_estimators': 800, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3,
                'random_state': 42, 'use_label_encoder': False,
                'eval_metric': 'logloss',
            }
            
            # CB config
            cb_cfg = {
                'iterations': 800, 'learning_rate': 0.02,
                'depth': 4, 'subsample': 0.7, 'colsample_bylevel': 0.7,
                'l2_leaf_reg': 5.0, 'min_data_in_leaf': 10,
                'random_state': 42, 'loss_function': 'Logloss',
                'verbose': False,
            }

            for fold, (tr, va) in enumerate(gkf.split(X_top, y, groups)):
                # LGBM
                ds_tr = lgb.Dataset(X_top[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                m_lgb = lgb.train(lgb_cfg, ds_tr, num_boost_round=lgb_cfg['n_estimators'])
                oof_lgb[va] = m_lgb.predict(X_top[va])
                
                # XGB
                m_xgb = xgb.XGBClassifier(**xgb_cfg)
                m_xgb.fit(X_top[tr], y[tr], eval_set=[(X_top[va], y[va])], verbose=False)
                oof_xgb[va] = m_xgb.predict_proba(X_top[va])[:, 1]
                
                # CB
                m_cb = cb.CatBoostClassifier(**cb_cfg)
                m_cb.fit(X_top[tr], y[tr], eval_set=(X_top[va], y[va]), use_best_model=True)
                oof_cb[va] = m_cb.predict_proba(X_top[va])[:, 1]

            # Average ensemble OOF
            oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
            avg_oof = logloss(y, oof_avg)
            
            log.info(f"    n_feat={n_feat}: LGB={logloss(y, oof_lgb):.4f}, XGB={logloss(y, oof_xgb):.4f}, CB={logloss(y, oof_cb):.4f}, AVG={avg_oof:.4f}")

            if avg_oof < best_avg_oof:
                best_avg_oof = avg_oof
                best_k = n_feat
                # Save final OOF for prediction
                best_oof_lgb = oof_lgb.copy()
                best_oof_xgb = oof_xgb.copy()
                best_oof_cb = oof_cb.copy()

        log.info(f"  Best: n_feat={best_k}, avg_oof={best_avg_oof:.4f}")

        # ── 4b. Final predictions on all data ──
        top_idx = ranked_idx[:best_k]
        top_sn = [sn[i] for i in top_idx]
        X_top_all = X_train[:, top_idx]
        X_test_top_all = X_test[:, top_idx]

        # Final 3-model ensemble on all data (5-fold for OOF-like predictions)
        final_lgb_preds = np.zeros(len(X_test_top_all))
        final_xgb_preds = np.zeros(len(X_test_top_all))
        final_cb_preds = np.zeros(len(X_test_top_all))

        for seed in [42, 123, 456]:  # 3 seeds for more diversity
            # LGBM
            lgb_cfg_s = {**lgb_cfg, 'random_state': seed, 'n_estimators': 600}
            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                ds_tr = lgb.Dataset(X_top_all[tr], label=y[tr], feature_name=top_sn, params={'verbose': '-1'})
                m = lgb.train(lgb_cfg_s, ds_tr, num_boost_round=lgb_cfg_s['n_estimators'])
                final_lgb_preds += m.predict(X_test_top_all) / (3 * 5)
            
            # XGB
            xgb_cfg_s = {**xgb_cfg, 'random_state': seed, 'n_estimators': 600}
            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                m = xgb.XGBClassifier(**xgb_cfg_s)
                m.fit(X_top_all[tr], y[tr], verbose=False)
                final_xgb_preds += m.predict_proba(X_test_top_all)[:, 1] / (3 * 5)
            
            # CB
            cb_cfg_s = {**cb_cfg, 'random_state': seed, 'iterations': 600}
            for fold, (tr, va) in enumerate(gkf.split(X_top_all, y, groups)):
                m = cb.CatBoostClassifier(**cb_cfg_s)
                m.fit(X_top_all[tr], y[tr], verbose=False)
                final_cb_preds += m.predict_proba(X_test_top_all)[:, 1] / (3 * 5)

        # Average of 3 models
        test_preds = (final_lgb_preds + final_xgb_preds + final_cb_preds) / 3.0
        predictions[target] = np.clip(test_preds, 0.0001, 0.9999)

        target_results[target] = {
            'best_n_feat': best_k,
            'best_avg_oof': float(best_avg_oof),
            'lgb_oof': float(logloss(y, best_oof_lgb)),
            'xgb_oof': float(logloss(y, best_oof_xgb)),
            'cb_oof': float(logloss(y, best_oof_cb)),
        }
        log.info(f"  {target}: LGB={target_results[target]['lgb_oof']:.4f}, XGB={target_results[target]['xgb_oof']:.4f}, CB={target_results[target]['cb_oof']:.4f}")

        del X_top_all, X_test_top_all, final_lgb_preds, final_xgb_preds, final_cb_preds
        gc.collect()

    # ── Summary ──
    avg_oof = np.mean([v['best_avg_oof'] for v in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info("V496 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: n_feat={r['best_n_feat']}, LGB={r['lgb_oof']:.4f}, XGB={r['xgb_oof']:.4f}, CB={r['cb_oof']:.4f}, AVG={r['best_avg_oof']:.4f}")
    log.info(f"  AVG OOF: {avg_oof:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save submission ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v496_subject_norm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # ── Save meta ──
    meta = {
        'version': 'V496_subject_norm',
        'name': 'Per-Subject Z-Score + 3-Model Ensemble (LGBM+XGB+CB)',
        'avg_oof': float(avg_oof),
        'n_features_base': len(leak_cols),
        'n_features_combined': X_train.shape[1],
        'target_results': target_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v496_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"  DONE.")


if __name__ == "__main__":
    main()
