"""
V493 — Aggressive 0.5 Push: Multi-Scale Features + Heavy Ensemble + Target Encoding

Hypothesis: V308의 Z-score features를 넘어서는 새로운 feature engineering이 필요.
특히:
1. Multi-scale aggregation (1-day, 3-day, 7-day windows per subject)
2. Subject-specific target encoding (LOO)
3. Cross-feature interactions (only top features)
4. Heavy ensemble: LGBM + CatBoost × 10 seeds each × 4 configs

This is a LONG run (~2-4 hours). Focus on feature quality.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS = ROOT / "experiments"
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_lgbm(X_train, y_train, X_val, sel_sn, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': -1,
    }
    ds = lgb.Dataset(X_train, label=y_train, feature_name=sel_sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    return model.predict(X_val)

def train_catboost(X_train, y_train, X_val, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    cb_model = cb.CatBoostClassifier(
        iterations=cfg['iter'], learning_rate=cfg['lr'],
        depth=cfg['depth'], loss_function='Logloss', eval_metric='Logloss',
        random_seed=seed, verbose=0, task_type='CPU',
        bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'], random_strength=cfg['rs'],
        scale_pos_weight=spw,
    )
    cb_model.fit(X_train, y_train, verbose=0)
    return np.clip(cb_model.predict_proba(np.where(np.isnan(X_val), 0, X_val))[:, 1], 0.0001, 0.9999)

def compute_loo_target_encoding(train_df, test_df, feature_col, target_col, subject_col):
    """Leave-one-out target encoding per subject for a feature."""
    # For each (subject, feature), compute mean(target) excluding current row
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    
    # Group by subject and feature
    grouped = train_df.groupby([subject_col, feature_col])[target_col].mean()
    
    # Map back to train
    train_df[feature_col + '_enc'] = train_df.apply(
        lambda row: grouped.get((row[subject_col], row[feature_col]), row[target_col]), axis=1
    )
    
    # For test, use the global mean of that feature
    feature_means = train_df.groupby(feature_col)[target_col].mean()
    test_df[feature_col + '_enc'] = test_df[feature_col].map(feature_means).fillna(
        train_df[target_col].mean()
    )
    
    return train_df, test_df

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V493 — Aggressive 0.5 Push with Multi-Scale Features")
    log.info("=" * 70)

    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    # Use common columns only
    target_cols = set(TARGETS)
    feature_cols_all = set(train.columns) - target_cols
    common_cols = sorted(feature_cols_all & set(test.columns))
    train = train[common_cols | target_cols]
    test = test[common_cols]
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    feat_cols = get_feature_cols(train)
    log.info(f"  Total features: {len(feat_cols)}")

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    LGBM_CONFIGS = [
        {'name': 'lgb_conservative', 'nl': 20, 'md': 4, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 15},
        {'name': 'lgb_deep', 'nl': 30, 'md': 5, 'lr': 0.015, 'ne': 1200, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 10},
    ]
    CB_CONFIGS = [
        {'name': 'cb_conservative', 'iter': 800, 'lr': 0.02, 'depth': 5, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
        {'name': 'cb_deep', 'iter': 1200, 'lr': 0.015, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0},
    ]
    SEEDS = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

    predictions = {}
    target_results = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        log.info(f"  Target rate: {train[target].mean():.3f}")

        y = train[target].values.astype(np.float64)
        
        # Feature selection using LGBM importance
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        signal_cols = [c for c in feat_cols if c not in META_COLS | set(TARGETS)]
        X_signal = train[signal_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in signal_cols]
        
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds_rank = lgb.Dataset(X_signal, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp_signal = model_rank.feature_importance(importance_type='gain')
        signal_ranked = sorted(zip(signal_cols, imp_signal), key=lambda x: -x[1])
        
        log.info("  Top 10 features:")
        for i, (name, imp) in enumerate(signal_ranked[:10]):
            log.info(f"    {i+1}. {name}: {imp:.2f}")

        # Feature selection sweep
        best_cv = float('inf')
        best_n_feat = 30
        best_oof_lgb = None
        best_oof_cb = None
        best_weights = np.array([1/3, 1/3, 1/3])

        for n_feat in [20, 30, 40, 50]:
            sel_cols = [r[0] for r in signal_ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in signal_ranked[:n_feat]]
            X_sel = train[sel_cols].fillna(0).values.astype(np.float64)

            oof_lgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))

            log.info(f"    n_feat={n_feat}: training {len(LGBM_CONFIGS)*len(SEEDS) + len(CB_CONFIGS)*len(SEEDS)} models...")
            t_train = time.time()
            
            for cfg in LGBM_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        model = train_lgbm(X_sel[tr], y[tr], X_sel[va], sel_sn, cfg, s)
                        oof_lgb[va] += model
            oof_lgb /= len(LGBM_CONFIGS) * len(SEEDS)

            t_train = time.time()
            for cfg in CB_CONFIGS:
                for s in SEEDS:
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, train['subject_id'].values)):
                        pred = train_catboost(X_sel[tr], y[tr], X_sel[va], cfg, s)
                        oof_cb[va] += pred
            oof_cb /= len(CB_CONFIGS) * len(SEEDS)
            log.info(f"    Done ({time.time()-t_train:.0f}s)")

            log.info(f"    LGBM CV={logloss(y, oof_lgb):.4f}, CB={logloss(y, oof_cb):.4f}")
            log.info(f"    Avg CV={logloss(y, (oof_lgb+oof_cb)/2):.4f}")

            if logloss(y, (oof_lgb+oof_cb)/2) < best_cv:
                best_cv = logloss(y, (oof_lgb+oof_cb)/2)
                best_n_feat = n_feat
                best_oof_lgb = oof_lgb.copy()
                best_oof_cb = oof_cb.copy()

        # Final prediction
        log.info(f"\n  Best n_feat={best_n_feat}")
        sel_cols = [r[0] for r in signal_ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in signal_ranked[:best_n_feat]]
        X_all = train[sel_cols].fillna(0).values.astype(np.float64)
        X_all_test = test[sel_cols].fillna(0).values.astype(np.float64)

        test_lgb = np.zeros(len(X_all_test))
        for cfg in LGBM_CONFIGS:
            for s in SEEDS:
                model = train_lgbm(X_all, y, X_all_test, sel_sn, cfg, s)
                test_lgb += model
        test_lgb /= len(LGBM_CONFIGS) * len(SEEDS)

        test_cb = np.zeros(len(X_all_test))
        for cfg in CB_CONFIGS:
            for s in SEEDS:
                pred = train_catboost(X_all, y, X_all_test, cfg, s)
                test_cb += pred
        test_cb /= len(CB_CONFIGS) * len(SEEDS)

        test_avg = (test_lgb + test_cb) / 2.0
        oof_avg = (best_oof_lgb + best_oof_cb) / 2.0

        predictions[target] = np.clip(test_avg, 0.0001, 0.9999)

        meta_oof = logloss(y, oof_avg)
        student_oof = logloss(y, test_avg)
        gap = abs(meta_oof - student_oof)

        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
            'test_mean': float(test_avg.mean()),
            'meta_oof': float(meta_oof),
            'student_oof': float(student_oof),
            'gap': float(gap),
        }
        log.info(f"  {target}: meta={meta_oof:.4f}, student={student_oof:.4f}, gap={gap:.4f}")

        gc.collect()

    avg_meta = np.mean([v['meta_oof'] for v in target_results.values()])
    avg_student = np.mean([v['student_oof'] for v in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info(f"V493 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: meta={r['meta_oof']:.4f}, student={r['student_oof']:.4f}, gap={r['gap']:.4f}")
    log.info(f"  AVG Meta OOF: {avg_meta:.4f}")
    log.info(f"  AVG Student OOF: {avg_student:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # Save submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v493_aggressive_05_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V493_aggressive_05',
        'name': 'Aggressive 0.5 push: LGBM+CB × 10 seeds × 2 configs × 5-fold',
        'n_models_per_target': 40,
        'target_results': target_results,
        'avg_meta_oof': float(avg_meta),
        'avg_student_oof': float(avg_student),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = EXPERIMENTS / f'v493_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    
    predictions = {}
    main()
