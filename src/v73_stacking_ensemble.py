"""
V73 — Stacking Ensemble + Multi-seed + Per-target Sweep

Strategy:
1. Leakage-clean features (V61 pipeline)
2. Multi-model: LGBM + CatBoost + XGBoost (OOF stacking)
3. Stacking: LR meta-learner on model OOF predictions
4. Per-target: n_feat sweep (10-30), 20 seeds per model
5. GroupKFold 5-fold (subject-level)
6. Post-processing: mean-match calibration

Architecture:
  Level 0: LGBM(20 seeds) + CatBoost(20 seeds) + XGBoost(20 seeds) → averaged OOF
  Level 1: LogisticRegression → calibrated stacking predictions
  Level 2: Weighted blend of stacked + simple average
  
Key: Only train on OOF for CV, full data for submission.
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
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# ── Leakage columns ──
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}
NIGHTTIME_LEAK = {
    'mScreenStatus_hour_night', 'mACStatus_hour_night',
    'mScreenStatus_hour_morning', 'wLight_w_light_sum',
    'mACStatus_charging_sum', 'mACStatus_charging_max',
}
SLEEP_DIRECT_LEAK = {
    'mGps_gps_avg_speed_max', 'mGps_gps_count_mean',
    'mActivity_m_activity_sum', 'mActivity_m_activity_max',
    'mActivity_m_activity_min',
}

N_FEAT_RANGE = [10, 15, 20, 25, 30]
N_SEEDS = 20

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    leak = set()
    if target.startswith('S'):
        leak = LEAK_S | NIGHTTIME_LEAK | SLEEP_DIRECT_LEAK
    elif target.startswith('Q'):
        leak = LEAK_Q | NIGHTTIME_LEAK
    return [c for c in cols if c not in leak]

def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ─── Single model trainers ───

def train_lgbm_model(X_train, y_train, sel_sn, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds = lgb.Dataset(X_train, label=y_train, feature_name=sel_sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=cfg['ne'])
    return model


def train_catboost_model(X_train, y_train, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    model = cb.CatBoostClassifier(
        iterations=cfg['iter'], learning_rate=cfg['lr'],
        depth=cfg['depth'], loss_function='Logloss', eval_metric='Logloss',
        random_seed=seed, verbose=0, task_type='CPU',
        bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'], random_strength=cfg['rs'],
        scale_pos_weight=spw,
    )
    model.fit(X_train, y_train, verbose=0)
    return model


def train_xgboost_model(X_train, y_train, cfg, seed):
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        max_depth=cfg['md'], learning_rate=cfg['lr'],
        n_estimators=cfg['ne'], subsample=cfg['ss'],
        colsample_bytree=cfg['cb'],
        reg_alpha=cfg['ra'], reg_lambda=cfg['rl'],
        min_child_weight=cfg['mc'], random_state=seed,
        scale_pos_weight=spw, tree_method='hist',
        verbosity=0, n_jobs=1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ── Model configs ──
LGBM_CFG = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CB_CFG = {'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0}
XGB_CFG = {'md': 5, 'lr': 0.02, 'ne': 800, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V73 — Stacking Ensemble (LGBM+CatBoost+XGB) × 20 seeds")
    log.info("=" * 70)

    # ── 1. Load data ──
    log.info("\n--- 1. Load data ---")
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    log.info(f"  Train: {train.shape}, Test: {test.shape}, Features: {len(feat_cols)}")

    target_results = {}
    predictions = {}

    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} ---")
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        log.info(f"  Leakage-clean features: {len(leak_cols)}")

        best_n_feat = 20
        best_cv = float('inf')
        best_test_lgb = None
        best_test_cb = None
        best_test_xgb = None

        for n_feat in N_FEAT_RANGE:
            # Feature ranking
            X_all = train[leak_cols].fillna(0).values.astype(np.float64)
            spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            sn = [sanitize(c) for c in leak_cols]
            
            params_rank = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
                'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
                'reg_alpha': 0.5, 'reg_lambda': 2.0,
                'scale_pos_weight': spw, 'random_state': 42,
                'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1,
            }
            ds_rank = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
            model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
            imp = model_rank.feature_importance(importance_type='gain')
            ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
            sel_cols = [r[0] for r in ranked[:n_feat]]
            sel_sn = [sanitize(r[0]) for r in ranked[:n_feat]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]
            del model_rank, ds_rank, X_all
            gc.collect()

            X_sel = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
            
            # ── OOF stacking: get per-model OOF ──
            oof_lgb = np.zeros(len(y))
            oof_cb = np.zeros(len(y))
            oof_xgb = np.zeros(len(y))

            t_train = time.time()

            # LGBM OOF
            for s in range(N_SEEDS):
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    m = train_lgbm_model(X_sel[tr], y[tr], sel_sn, LGBM_CFG, s)
                    oof_lgb[va] += m.predict(X_sel[va])
            oof_lgb /= N_SEEDS
            log.info(f"    LGBM OOF done ({time.time()-t_train:.0f}s)")
            gc.collect()

            # CatBoost OOF
            t_cb = time.time()
            for s in range(N_SEEDS):
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    m = train_catboost_model(X_sel[tr], y[tr], CB_CFG, s)
                    oof_cb[va] += np.clip(m.predict_proba(np.where(np.isnan(X_sel[va]), 0, X_sel[va]))[:, 1], 0.0001, 0.9999)
            oof_cb /= N_SEEDS
            log.info(f"    CatBoost OOF done ({time.time()-t_cb:.0f}s)")
            gc.collect()

            # XGBoost OOF
            t_xgb = time.time()
            for s in range(N_SEEDS):
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    m = train_xgboost_model(X_sel[tr], y[tr], XGB_CFG, s)
                    oof_xgb[va] += np.clip(m.predict_proba(np.where(np.isnan(X_sel[va]), 0, X_sel[va]))[:, 1], 0.0001, 0.9999)
            oof_xgb /= N_SEEDS
            log.info(f"    XGBoost OOF done ({time.time()-t_xgb:.0f}s)")
            gc.collect()

            # ── Simple average ──
            oof_avg = (oof_lgb + oof_cb + oof_xgb) / 3.0
            cv_avg = logloss(y, oof_avg)

            # ── Stacking: LR meta-learner ──
            oof_stack = np.column_stack([oof_lgb, oof_cb, oof_xgb])
            best_c_loss = float('inf')
            best_c = 1.0
            oof_stack_pred = np.zeros(len(y))
            for c_val in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
                meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=1000, random_state=42)
                meta.fit(oof_stack, y)
                p = np.clip(meta.predict_proba(oof_stack)[:, 1], 1e-15, 1-1e-15)
                loss = logloss(y, p)
                if loss < best_c_loss:
                    best_c_loss = loss
                    best_c = c_val
                    oof_stack_pred = p.copy()

            cv_stack = best_c_loss
            log.info(f"    n_feat={n_feat}: avg_cv={cv_avg:.4f}, stack_cv={cv_stack:.4f} (C={best_c})")

            # Use stacking result
            if cv_stack < best_cv:
                best_cv = cv_stack
                best_n_feat = n_feat
                # Save OOF for final test prediction
                best_oof_lgb = oof_lgb.copy()
                best_oof_cb = oof_cb.copy()
                best_oof_xgb = oof_xgb.copy()

            del oof_stack, oof_lgb, oof_cb, oof_xgb, oof_stack_pred, X_sel
            gc.collect()

        log.info(f"\n  ✅ Best: n_feat={best_n_feat}, stack_cv={best_cv:.4f}")
        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'per_target_rate': float(train[target].mean()),
        }

        # ── Final: train on ALL data, produce test predictions ──
        log.info(f"  Training final on all data (n_feat={best_n_feat})...")
        sel_cols = [r[0] for r in ranked[:best_n_feat]]
        sel_sn = [sanitize(r[0]) for r in ranked[:best_n_feat]]
        sel_idx = [leak_cols.index(c) for c in sel_cols]
        X_all = train[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]

        test_lgb = np.zeros(len(X_test))
        for s in range(N_SEEDS):
            m = train_lgbm_model(X_all, y, sel_sn, LGBM_CFG, s)
            test_lgb += m.predict(X_test)
        test_lgb /= N_SEEDS

        test_cb = np.zeros(len(X_test))
        for s in range(N_SEEDS):
            m = train_catboost_model(X_all, y, CB_CFG, s)
            test_cb += np.clip(m.predict_proba(np.where(np.isnan(X_test), 0, X_test))[:, 1], 0.0001, 0.9999)
        test_cb /= N_SEEDS

        test_xgb = np.zeros(len(X_test))
        for s in range(N_SEEDS):
            m = train_xgboost_model(X_all, y, XGB_CFG, s)
            test_xgb += np.clip(m.predict_proba(np.where(np.isnan(X_test), 0, X_test))[:, 1], 0.0001, 0.9999)
        test_xgb /= N_SEEDS

        # Stacking meta-learner on full OOF
        oof_stack_final = np.column_stack([best_oof_lgb, best_oof_cb, best_oof_xgb])
        meta = LogisticRegression(C=best_c, solver='lbfgs', max_iter=1000, random_state=42)
        meta.fit(oof_stack_final, y)
        test_stack = np.clip(meta.predict_proba(
            np.column_stack([test_lgb, test_cb, test_xgb])
        )[:, 1], 0.0001, 0.9999)

        # Blend: 50% stack + 50% simple avg
        test_blend = 0.5 * test_stack + 0.5 * (test_lgb + test_cb + test_xgb) / 3.0

        predictions[target] = np.clip(test_blend, 0.0001, 0.9999)
        target_results[target]['test_mean'] = float(test_blend.mean())
        target_results[target]['n_seeds'] = N_SEEDS

        log.info(f"  {target}: cv={best_cv:.4f}, test_mean={test_blend.mean():.4f}")
        del sel_cols, X_all, X_test, test_lgb, test_cb, test_xgb
        gc.collect()

    # ── Summary ──
    avg_cv = np.mean([v['best_cv'] for v in target_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V73 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: stack_cv={r['best_cv']:.4f} (n_feat={r['best_n_feat']})")
    log.info(f"  AVG STACK CV: {avg_cv:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v73_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V73_stacking_ensemble',
        'name': 'LGBM+CatBoost+XGB stacking ensemble × 20 seeds',
        'cv_method': 'GroupKFold_5fold',
        'n_models_per_target': 60,  # 3 models × 20 seeds
        'n_feat_sweep': N_FEAT_RANGE,
        'target_results': target_results,
        'avg_cv': float(avg_cv),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v73_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")


if __name__ == "__main__":
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb
    
    main()
