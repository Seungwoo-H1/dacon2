"""
V436 — Ensemble V432 (self+stats) + V435 (full cross-target)

Hypothesis: V432 meta gap=1.12x, V435 meta gap=1.32x. By finding the 
ensemble weight that minimizes the V339 pattern LB (= meta + gap*0.85),
we can find the best gap balance between the two meta approaches.

Key insight: V432 has better gap calibration but higher meta OOF.
V435 has lower meta OOF but worse gap. Ensemble should find sweet spot.
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

LGB_CFGS = {
    'narrow': {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
               'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 8.0, 'reg_lambda': 30.0,
               'min_child_samples': 35},
    'soft_aggressive': {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.012, 'n_estimators': 2000,
                        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 4.0, 'reg_lambda': 15.0,
                        'min_child_samples': 25},
    'ultra_deep': {'num_leaves': 25, 'max_depth': 5, 'learning_rate': 0.025, 'n_estimators': 1000,
                   'subsample': 0.75, 'colsample_bytree': 0.65, 'reg_alpha': 0.3, 'reg_lambda': 1.5,
                   'min_child_samples': 12},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0,
               'min_child_samples': 20},
    'broad': {'num_leaves': 40, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 800,
              'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
              'min_child_samples': 8},
}

V413_CONFIGS = {
    'Q1': 'narrow', 'Q2': 'soft_aggressive', 'Q3': 'narrow',
    'S1': 'ultra_deep', 'S2': 'soft_aggressive', 'S3': 'safety', 'S4': 'broad',
}
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15


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
    log.info("V436 — Ensemble V432 (self+stats) + V435 (full cross-target)")
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
        cfg = LGB_CFGS[cfg_name]
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
                    callbacks=[lgb.early_stopping(200, verbose=False),
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

    # ===== Phase 2: Meta A (self+stats) and Meta B (full cross-target) =====
    log.info("\n=== Phase 2: Meta A (V432 self+stats) vs Meta B (V435 full) ===")

    from xgboost import XGBClassifier

    def build_meta_A(oof_p, test_p):
        """15 self + 4 stats = 19 features."""
        X_A = np.hstack([oof_p,
            np.mean(oof_p, axis=1, keepdims=True),
            np.std(oof_p, axis=1, keepdims=True),
            np.min(oof_p, axis=1, keepdims=True),
            np.max(oof_p, axis=1, keepdims=True)])
        X_test_A = np.hstack([test_p,
            np.mean(test_p, axis=1, keepdims=True),
            np.std(test_p, axis=1, keepdims=True),
            np.min(test_p, axis=1, keepdims=True),
            np.max(test_p, axis=1, keepdims=True)])
        return X_A, X_test_A

    def build_meta_B(oof_p, test_p, oof_all, test_all, targets_list, target_idx):
        """15 self + 4 stats + 6 cross-target = 25 features."""
        cross_oof = []
        cross_test = []
        for i, t in enumerate(targets_list):
            if i != target_idx:
                cross_oof.append(np.mean(oof_all[t], axis=1))
                cross_test.append(np.mean(test_all[t], axis=1))
        cross_oof = np.column_stack(cross_oof)
        cross_test = np.column_stack(cross_test)
        return np.hstack([oof_p,
            np.mean(oof_p, axis=1, keepdims=True),
            np.std(oof_p, axis=1, keepdims=True),
            np.min(oof_p, axis=1, keepdims=True),
            np.max(oof_p, axis=1, keepdims=True),
            cross_oof]), np.hstack([test_p,
            np.mean(test_p, axis=1, keepdims=True),
            np.std(test_p, axis=1, keepdims=True),
            np.min(test_p, axis=1, keepdims=True),
            np.max(test_p, axis=1, keepdims=True),
            cross_test])

    meta_oofs_A = {}
    meta_oofs_B = {}
    student_oofs = {}
    test_preds_A = {}
    test_preds_B = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed[target]
        test_preds_raw = all_test_seed[target]

        X_A, X_test_A = build_meta_A(oof_preds, test_preds_raw)
        X_B, X_test_B = build_meta_B(oof_preds, test_preds_raw, all_oof_seed, all_test_seed, TARGETS, t_idx)

        mm_A = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm_A.fit(X_A, y)

        mm_B = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm_B.fit(X_B, y)

        meta_oofs_A[target] = log_loss(y, mm_A.predict_proba(X_A)[:, 1])
        meta_oofs_B[target] = log_loss(y, mm_B.predict_proba(X_B)[:, 1])
        test_preds_A[target] = mm_A.predict_proba(X_test_A)[:, 1]
        test_preds_B[target] = mm_B.predict_proba(X_test_B)[:, 1]
        student_oofs[target] = log_loss(y, oof_preds.mean(axis=1))

        log.info(f"  {target}: A={meta_oofs_A[target]:.5f}, B={meta_oofs_B[target]:.5f}, "
                 f"Δ={meta_oofs_B[target]-meta_oofs_A[target]:+.5f}")

    avg_A = np.mean(list(meta_oofs_A.values()))
    avg_B = np.mean(list(meta_oofs_B.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap_A = avg_student - avg_A
    gap_B = avg_student - avg_B

    v339_A = avg_A + gap_A * 0.85
    v339_B = avg_B + gap_B * 0.85

    log.info(f"\n{'='*70}")
    log.info(f"  Meta A (self+stats):   avg={avg_A:.5f}, gap={gap_A:.5f} ({gap_A/0.070:.2f}x), V339={v339_A:.5f}")
    log.info(f"  Meta B (full):         avg={avg_B:.5f}, gap={gap_B:.5f} ({gap_B/0.070:.2f}x), V339={v339_B:.5f}")
    log.info(f"  Student:               avg={avg_student:.5f}")
    log.info(f"{'='*70}")

    # Pick best by V339 LB
    if v339_A <= v339_B:
        best_meta = 'A'
        final_oofs = meta_oofs_A
        final_test = test_preds_A
        final_avg = avg_A
        final_gap = gap_A
    else:
        best_meta = 'B'
        final_oofs = meta_oofs_B
        final_test = test_preds_B
        final_avg = avg_B
        final_gap = gap_B

    estimated_lb = final_avg + final_gap * 0.85

    log.info(f"\n  Best: Meta {best_meta} (V339 LB={estimated_lb:.5f})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = final_test[t]

    sub_path = SUBMIT / f"submission_v436_meta{best_meta}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")

    meta_data = {
        'version': 'V436',
        'name': 'Ensemble V432+V435 Meta Comparison',
        'meta_A': {'avg': round(avg_A, 5), 'gap': round(gap_A, 5), 'v339': round(v339_A, 5)},
        'meta_B': {'avg': round(avg_B, 5), 'gap': round(gap_B, 5), 'v339': round(v339_B, 5)},
        'best': best_meta,
        'final_avg_meta': round(float(final_avg), 5),
        'final_gap': round(float(final_gap), 5),
        'v339_lb': round(float(estimated_lb), 5),
        'avg_student': round(float(avg_student), 5),
        'v308_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v436_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    log.info(f"Total: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
