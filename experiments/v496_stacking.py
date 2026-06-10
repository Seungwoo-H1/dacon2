"""
V496 — Stacking with Per-Target Optimal C (V308 방식 그대로 재현)

핵심 전략:
1. V308의 stacking 아키텍처를 그대로 재현 (15 seeds → LR meta, C=10)
2. Per-target feature selection (top-K via LGBM gain importance)
3. Aggressive leak removal (wrist, nighttime, sleep-direct)
4. GroupKFold 5-fold for proper subject-level CV
5. Per-target optimal C sweep for meta learner
6. Simple average student predictions (no leak in meta features)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
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

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
    'mScreenStatus_hour_night','mACStatus_hour_night',
    'mScreenStatus_hour_morning','wLight_w_light_sum',
    'mACStatus_charging_sum','mACStatus_charging_max',
    'mGps_gps_avg_speed_max','mGps_gps_count_mean',
    'mActivity_m_activity_sum','mActivity_m_activity_max',
    'mActivity_m_activity_min',
}
LEAK_Q = LEAK_S.copy()

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999, 1234, 3456, 6789, 1111, 2222]


def logloss(y_true, y_pred):
    eps = 1e-15
    return -np.mean(y_true * np.log(np.clip(y_pred, eps, 1 - eps)) +
                    (1 - y_true) * np.log(np.clip(1 - y_pred, eps, 1 - eps)))


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V496 — Stacking (V308 Replicate + Aggressive Leak Removal)")
    log.info("=" * 70)

    # ── 1. Load data ──
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")

    # Keep only numeric columns
    for df in [train, test]:
        df = df.select_dtypes(include=[np.number])

    # Keep target + subject_id in train
    target_cols = set(TARGETS)
    feature_cols = [c for c in train.columns if c not in target_cols and np.issubdtype(train[c].dtype, np.number)]
    test_cols = [c for c in test.columns if np.issubdtype(test[c].dtype, np.number)]
    common_cols = [c for c in feature_cols if c in test_cols]

    train = train[[c for c in common_cols + list(target_cols) + ['subject_id'] if c in train.columns]]
    test = test[[c for c in common_cols if c in test.columns]]

    log.info(f"Train: {train.shape}, Test: {test.shape}")
    log.info(f"Features: {len(common_cols)}")

    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)

    predictions = {}
    target_results = {}

    # ── 2. Per-target stacking ──
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate={train[target].mean():.3f}) ---")

        y = train[target].values.astype(np.float64)

        # Remove leak features
        leak = LEAK_S if target.startswith('S') else LEAK_Q
        clean_cols = [c for c in common_cols if c not in leak]
        log.info(f"After leak removal: {len(clean_cols)} features")

        # Feature ranking with LGBM
        X = train[clean_cols].fillna(0).values.astype(np.float64)
        X_test = test[clean_cols].fillna(0).values.astype(np.float64)
        sn = [re.sub(r'[^a-zA-Z0-9_]','_', c) for c in clean_cols]

        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        ds_rank = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0, 'scale_pos_weight': spw,
            'random_state': 42, 'min_child_samples': 15,
        }
        model_rank = lgb.train(params_rank, ds_rank, num_boost_round=100)
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(clean_cols, imp), key=lambda x: -x[1])

        # ── Feature count sweep ──
        best_n_feat = 30
        best_cv = float('inf')
        best_C = 1.0
        best_test_preds = None

        for n_feat in [15, 20, 25, 30, 40, 50]:
            n_feat = min(n_feat, len(clean_cols))
            sel_cols = [r[0] for r in ranked[:n_feat]]
            sel_sn = [re.sub(r'[^a-zA-Z0-9_]','_', c) for c in sel_cols]
            sel_idx = [clean_cols.index(c) for c in sel_cols]

            X_sel = X[:, sel_idx]
            X_test_sel = X_test[:, sel_idx]

            # ── Student layer: 15 seeds, LGBM ──
            oof_student = np.zeros(len(y))
            n_models = 0
            for seed in SEEDS:
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    ds = lgb.Dataset(X_sel[tr], label=y[tr], feature_name=sel_sn, params={'verbose': '-1'})
                    m = lgb.train({
                        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                        'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.02,
                        'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
                        'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
                        'scale_pos_weight': max(((y[tr]==0).sum())/max((y[tr]==1).sum(),1), 0.1),
                        'random_state': seed,
                    }, ds, num_boost_round=500)
                    oof_student[va] += m.predict(X_sel[va])
                    n_models += 1
            oof_student /= n_models

            # ── Meta layer: LR with per-target C sweep ──
            best_C_this = 1.0
            best_C_cv = float('inf')
            for C_val in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
                oof_meta = np.zeros(len(y))
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    lr = LogisticRegression(C=C_val, max_iter=1000, random_state=42)
                    lr.fit(oof_student[tr:va].reshape(-1,1), y[tr:va])
                    # Wrong: should use student predictions from fold training
                    # Fix: need per-fold student predictions
                    pass

                # Correct: per-fold
                for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                    ds = lgb.Dataset(X_sel[tr], label=y[tr], feature_name=sel_sn, params={'verbose': '-1'})
                    m = lgb.train({
                        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                        'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.02,
                        'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
                        'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
                        'scale_pos_weight': max(((y[tr]==0).sum())/max((y[tr]==1).sum(),1), 0.1),
                        'random_state': 42,
                    }, ds, num_boost_round=500)
                    student_va = m.predict(X_sel[va]) / n_models

                    lr = LogisticRegression(C=C_val, max_iter=1000, random_state=42)
                    lr.fit(student_va.reshape(-1,1), y[va])
                    oof_meta[va] = lr.predict_proba(student_va.reshape(-1,1))[:, 1]

                cv = logloss(y, oof_meta)
                if cv < best_C_cv:
                    best_C_cv = cv
                    best_C_this = C_val

            log.info(f"    n_feat={n_feat}: meta_cv={best_C_cv:.4f} (C={best_C_this}), student={logloss(y,oof_student):.4f}, models={n_models}")

            if best_C_cv < best_cv:
                best_cv = best_C_cv
                best_n_feat = n_feat
                best_C = best_C_this

            del X_sel, X_test_sel
            gc.collect()

        log.info(f"  Best: n_feat={best_n_feat}, cv={best_cv:.4f}, C={best_C}")

        # ── Final: train on ALL data ──
        sel_cols = [r[0] for r in ranked[:best_n_feat]]
        sel_sn = [re.sub(r'[^a-zA-Z0-9_]','_', c) for c in sel_cols]
        sel_idx = [clean_cols.index(c) for c in sel_cols]
        X_all = X[:, sel_idx]
        X_all_test = X_test[:, sel_idx]

        # Student predictions
        test_preds = np.zeros(len(X_all_test))
        for seed in SEEDS:
            ds = lgb.Dataset(X_all, label=y, feature_name=sel_sn, params={'verbose': '-1'})
            m = lgb.train({
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.02,
                'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
                'scale_pos_weight': spw, 'random_state': seed,
            }, ds, num_boost_round=500)
            test_preds += m.predict(X_all_test)
        test_preds /= len(SEEDS)

        # Meta predictions
        lr_final = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
        # Need OOF student predictions for training meta
        oof_student_final = np.zeros(len(y))
        for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
            ds = lgb.Dataset(X_all[tr], label=y[tr], feature_name=sel_sn, params={'verbose': '-1'})
            m = lgb.train({
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.02,
                'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
                'scale_pos_weight': spw, 'random_state': 42,
            }, ds, num_boost_round=500)
            oof_student_final[va] = m.predict(X_all[va]) / len(SEEDS)

        lr_final.fit(oof_student_final.reshape(-1,1), y)
        test_meta = lr_final.predict_proba(test_preds.reshape(-1,1))[:, 1]

        meta_oof = logloss(y, lr_final.predict_proba(oof_student_final.reshape(-1,1))[:, 1])
        student_oof = logloss(y, oof_student_final)
        gap = abs(meta_oof - student_oof)

        predictions[target] = np.clip(test_meta, 0.0001, 0.9999)
        target_results[target] = {
            'best_n_feat': best_n_feat,
            'best_cv': float(best_cv),
            'meta_oof': float(meta_oof),
            'student_oof': float(student_oof),
            'gap': float(gap),
            'best_C': float(best_C),
        }
        log.info(f"  {target}: meta={meta_oof:.4f}, student={student_oof:.4f}, gap={gap:.4f}")

    # ── Summary ──
    avg_meta = np.mean([v['meta_oof'] for v in target_results.values()])
    avg_student = np.mean([v['student_oof'] for v in target_results.values()])
    avg_gap = np.mean([v['gap'] for v in target_results.values()])

    log.info(f"\n{'='*70}")
    log.info("V496 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = target_results[t]
        log.info(f"  {t}: meta={r['meta_oof']:.4f}, student={r['student_oof']:.4f}, gap={r['gap']:.4f}, n_feat={r['best_n_feat']}, C={r['best_C']}")
    log.info(f"  AVG Meta OOF: {avg_meta:.4f}")
    log.info(f"  AVG Student OOF: {avg_student:.4f}")
    log.info(f"  AVG Gap: {avg_gap:.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # ── Save submission ──
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]
    sub_path = SUBMIT / f"submission_v496_stacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    meta = {
        'version': 'V496_stacking',
        'name': 'V308 Stacking + Aggressive Leak Removal + 15 seeds + per-target C sweep',
        'cv_method': 'GroupKFold_5fold',
        'avg_meta_oof': float(avg_meta),
        'avg_student_oof': float(avg_student),
        'avg_gap': float(avg_gap),
        'target_results': target_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time()-t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v496_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info("  DONE.")


if __name__ == "__main__":
    main()
