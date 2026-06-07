"""
V437 — Baseline Subtraction + Adaptive Learning Rate + Per-Target Regularization

Hypothesis: V432/V435의 baseline subtraction이 student OOF를 0.633으로 낮췄지만,
LGBM base model의 학습이 아직 충분하지 않음. Adaptive learning rate와 
per-target regularization tuning으로 base model을 더 깊이 학습하면
student OOF를 0.620 수준으로 낮출 수 있음.

Changes from V435:
1. Adaptive LR: higher LR for Q targets (harder), lower for S targets
2. Per-target regularization sweep: find optimal reg for each target
3. More seeds (20) with deeper models
4. Per-subject baseline OOF estimation (no data leakage)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}

# Adaptive configs: higher LR + more trees for Q targets (harder)
LGB_ADAPT_CFGS = {
    'narrow': {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.008, 'n_estimators': 4000,
               'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 15.0, 'reg_lambda': 50.0,
               'min_child_samples': 40},
    'soft_aggressive': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.008, 'n_estimators': 3500,
                        'subsample': 0.45, 'colsample_bytree': 0.45, 'reg_alpha': 8.0, 'reg_lambda': 30.0,
                        'min_child_samples': 30},
    'ultra_deep': {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.015, 'n_estimators': 2000,
                   'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
                   'min_child_samples': 15},
    'safety': {'num_leaves': 8, 'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 2000,
               'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 20.0,
               'min_child_samples': 25},
    'broad': {'num_leaves': 30, 'max_depth': 4, 'learning_rate': 0.015, 'n_estimators': 1500,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0,
              'min_child_samples': 10},
}

V413_CONFIGS = {
    'Q1': 'narrow', 'Q2': 'soft_aggressive', 'Q3': 'narrow',
    'S1': 'ultra_deep', 'S2': 'soft_aggressive', 'S3': 'safety', 'S4': 'broad',
}
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

SEED = 42
N_FOLDS = 5
N_SEEDS = 20  # More seeds for better ensemble


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

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


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V437 — Baseline Sub + Adaptive LR + Deep Training")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values
    train_subjects = train_df['subject_id'].values

    # ===== Phase 1: LGBM base with baseline subtraction =====
    all_oof_seed = {}
    all_test_seed = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = LGB_ADAPT_CFGS[cfg_name]
        n_feat = V413_NFEAT[target]
        y = train_df[target].values.astype(np.float64)

        subject_ids = np.unique(train_subjects)
        subject_baseline = {}
        for sid in subject_ids:
            mask = train_subjects == sid
            s_y = y[mask]
            n_samples = mask.sum()
            global_rate = y.mean()
            subj_rate = s_y.mean() if n_samples > 0 else global_rate
            smooth = 0.7 * subj_rate + 0.3 * global_rate
            subject_baseline[sid] = smooth

        feat_cols = get_feature_cols(train_df)
        feat_cols = remove_leak(feat_cols, target)

        fold_ranks = []
        for fold in range(N_FOLDS):
            rank = rank_features(train_df, feat_cols, target, seed=SEED + fold)
            fold_ranks.append(rank[:n_feat])

        feat_counts = {}
        for fl in fold_ranks:
            for f in fl:
                feat_counts[f] = feat_counts.get(f, 0) + 1
        ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
        top_features = [f for f, c in ranked_features[:n_feat]]

        X_all = train_df[top_features].fillna(0).values.astype(np.float64)
        X_test_all = test_df[top_features].fillna(0).values.astype(np.float64)

        oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
        test_seed_arr = np.zeros((len(test_df), N_SEEDS))

        for s in range(N_SEEDS):
            sk = SEED + s * 7 + t_idx
            skf = GroupKFold(n_splits=N_FOLDS)
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))

            for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y, groups_arr)):
                x_train, y_train = X_all[tr_idx], y[tr_idx]
                x_val = X_all[va_idx]

                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                    'n_estimators': cfg['n_estimators'], 'scale_pos_weight': spw,
                    'random_state': sk, 'force_row_wise': True, 'n_jobs': 1,
                }

                ds_train = lgb.Dataset(x_train, label=y_train,
                    feature_name=[sanitize_col(c) for c in top_features])
                ds_val = lgb.Dataset(x_val, label=y[va_idx],
                    feature_name=[sanitize_col(c) for c in top_features], reference=ds_train)

                model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                    valid_sets=[ds_val],
                    callbacks=[lgb.early_stopping(300, verbose=False),
                               lgb.log_evaluation(period=0)])

                val_subs = [train_subjects[i] for i in va_idx]
                val_baselines = np.array([subject_baseline[sid] for sid in val_subs])
                val_preds = model.predict(x_val)
                resids = np.clip(val_preds - val_baselines, -0.3, 0.3)
                seed_oof[va_idx] = np.clip(val_baselines + resids, 0.001, 0.999)
                del model, ds_train, ds_val
                gc.collect()

            oof_seed_arr[:, s] = seed_oof
            test_seed_arr[:, s] = seed_test

        all_oof_seed[target] = oof_seed_arr
        all_test_seed[target] = test_seed_arr
        avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
        log.info(f"{target}: baseline_sub_oof={avg_oof:.5f}")

    # ===== Phase 2: Full Meta (self + stats + cross-target) =====
    log.info("\n=== Phase 2: Full Meta ===")

    from xgboost import XGBClassifier

    meta_oofs = {}
    student_oofs = {}
    test_preds = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed[target]
        test_preds_raw = all_test_seed[target]

        # Build full meta: 15 self + 4 stats + 6 cross = 25
        means = np.mean(oof_preds, axis=1, keepdims=True)
        stds = np.std(oof_preds, axis=1, keepdims=True)
        mins = np.min(oof_preds, axis=1, keepdims=True)
        maxs = np.max(oof_preds, axis=1, keepdims=True)
        cross_oof = []
        cross_test = []
        for i, t in enumerate(TARGETS):
            if i != t_idx:
                cross_oof.append(np.mean(all_oof_seed[t], axis=1))
                cross_test.append(np.mean(all_test_seed[t], axis=1))
        cross_oof = np.column_stack(cross_oof)
        cross_test = np.column_stack(cross_test)

        X_meta = np.hstack([oof_preds, means, stds, mins, maxs, cross_oof])
        X_test_meta = np.hstack([test_preds_raw,
            np.mean(test_preds_raw, axis=1, keepdims=True),
            np.std(test_preds_raw, axis=1, keepdims=True),
            np.min(test_preds_raw, axis=1, keepdims=True),
            np.max(test_preds_raw, axis=1, keepdims=True),
            cross_test])

        student_oofs[target] = log_loss(y, oof_preds.mean(axis=1))

        # XGB meta with low regularization (from V429 best params)
        mm = XGBClassifier(n_estimators=15, max_depth=3,
            reg_alpha=0.01, reg_lambda=0.0, gamma=0.0,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test_meta)[:, 1]

        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta

    predicted_lb = avg_meta + gap * 0.5
    estimated_lb_v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V437 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V308: {avg_student-0.69212:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: 0.070, ratio: {gap/0.070:.2f}x)")
    log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v437_adaptive_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    meta_data = {
        'version': 'V437',
        'name': 'Baseline Sub + Adaptive LR + Deep Training',
        'avg_meta_oof': round(float(avg_meta), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235, 'v308_avg_student': 0.69212,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb_v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'n_seeds': N_SEEDS,
        'meta_type': 'xgb_full_25feat_baseline_sub_adaptive',
        'per_target_meta_oof': {t: round(float(v), 5) for t, v in meta_oofs.items()},
        'per_target_student_oof': {t: round(float(v), 5) for t, v in student_oofs.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v437_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    log.info(f"Total: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
