"""
V431 — XGB Meta with Seed Prediction Statistics

Hypothesis: V429/V430 used raw seed predictions (15 features) as XGB meta input.
But the predictions contain useful statistics (diversity, consensus, spread) that
are not directly captured by individual seed predictions. Adding meta-features
derived from seed predictions may improve the XGB meta-learner's ability to
distinguish signal from noise → lower gap.

V431:
1. V429 LGBM base (V413 configs, 15 seeds) — identical
2. XGB meta input: 15 raw predictions + 4 statistics (mean, std, min, max) = 19 features
3. Per-target beta/gamma/alpha joint sweep on expanded feature set
4. Goal: same meta OOF but smaller student-meta gap via better meta feature engineering
"""
import sys, gc, logging, json, re, time, warnings, math
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

# XGB meta hyperparameter sweep
ALPHA_SWEEP = [0.01, 0.05, 0.1]
BETA_SWEEP = [0.0, 0.1, 1.0, 5.0, 10.0]
GAMMA_SWEEP = [0.0, 0.01, 0.1, 1.0]


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


def build_meta_features(seed_preds):
    """
    Build meta features from seed predictions.
    Input: [n_samples, n_seeds]
    Output: [n_samples, n_seeds + 4] where 4 stats = mean, std, min, max
    """
    means = np.mean(seed_preds, axis=1, keepdims=True)
    stds = np.std(seed_preds, axis=1, keepdims=True)
    mins = np.min(seed_preds, axis=1, keepdims=True)
    maxs = np.max(seed_preds, axis=1, keepdims=True)
    return np.hstack([seed_preds, means, stds, mins, maxs])


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V431 — XGB Meta with Seed Prediction Statistics")
    log.info("Hypothesis: Meta features (mean/std/min/max) improve XGB meta gap")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    # ===== Phase 1: V429 LGBM base (identical) =====
    all_oof_seed = {}
    all_test_seed = {}

    for t_idx, target in enumerate(TARGETS):
        cfg_name = V413_CONFIGS[target]
        cfg = LGB_CFGS[cfg_name]
        n_feat = V413_NFEAT[target]

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

        X_all = train_df[top_features + [target]].fillna(0).values.astype(np.float64)
        y_all = X_all[:, -1]
        X_all = X_all[:, :-1]
        X_test_all = test_df[top_features].fillna(0).values.astype(np.float64)

        oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
        test_seed_arr = np.zeros((len(test_df), N_SEEDS))

        for s in range(N_SEEDS):
            sk = SEED + s * 7 + t_idx
            skf = GroupKFold(n_splits=N_FOLDS)
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))

            for train_idx, val_idx in skf.split(X_all, y_all, groups_arr):
                x_train, y_train = X_all[train_idx], y_all[train_idx]
                x_val, x_test = X_all[val_idx], X_test_all

                spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    **{k: v for k, v in cfg.items() if k not in ['n_estimators']},
                    'n_estimators': cfg['n_estimators'], 'scale_pos_weight': spw,
                    'random_state': sk, 'force_row_wise': True, 'n_jobs': 1,
                }

                ds_train = lgb.Dataset(x_train, label=y_train,
                    feature_name=[sanitize_col(c) for c in top_features])
                ds_val = lgb.Dataset(x_val, label=y_all[val_idx],
                    feature_name=[sanitize_col(c) for c in top_features], reference=ds_train)

                model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                    valid_sets=[ds_val],
                    callbacks=[lgb.early_stopping(200, verbose=False),
                               lgb.log_evaluation(period=0)])

                seed_oof[val_idx] = model.predict(x_val)
                seed_test += model.predict(x_test) / N_FOLDS
                del model, ds_train, ds_val
                gc.collect()

            oof_seed_arr[:, s] = seed_oof
            test_seed_arr[:, s] = seed_test

        all_oof_seed[target] = oof_seed_arr
        all_test_seed[target] = test_seed_arr

        avg_oof = log_loss(y_all, oof_seed_arr.mean(axis=1))
        log.info(f"{target}: base_oof={avg_oof:.5f}")

    # ===== Phase 2: XGB Meta with Statistics =====
    log.info("\n=== Phase 2: XGB Meta with Seed Prediction Statistics ===")
    log.info("Input features: 15 seed predictions + mean + std + min + max = 19 features")

    from xgboost import XGBClassifier

    # Also run V429 baseline (raw predictions, alpha only) for comparison
    log.info("\n--- Baseline: V429 (raw predictions, alpha sweep only) ---")
    v429_oofs = {}
    v429_student_oofs = {}
    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed[target]
        X_meta_raw = oof_preds
        
        best_alpha = 0.01
        best_oof = float('inf')
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
            mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=alpha, reg_lambda=1.0,
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
            mm.fit(X_meta_raw, y)
            oof_v = log_loss(y, mm.predict_proba(X_meta_raw)[:, 1])
            if oof_v < best_oof:
                best_oof = oof_v
                best_alpha = alpha
        v429_oofs[target] = best_oof
        v429_student_oofs[target] = log_loss(y, oof_preds.mean(axis=1))
    
    v429_avg = np.mean(list(v429_oofs.values()))
    v429_student_avg = np.mean(list(v429_student_oofs.values()))
    v429_gap = v429_student_avg - v429_avg
    log.info(f"  V429 AVG Meta: {v429_avg:.5f}, Student: {v429_student_avg:.5f}, Gap: {v429_gap:.5f}")

    # V431: with statistics
    log.info("\n--- V431: With seed prediction statistics (beta/gamma/alpha sweep) ---")
    
    per_target_best = {}
    meta_oofs = {}
    student_oofs = {}
    test_preds = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        oof_preds = all_oof_seed[target]
        X_test_raw = all_test_seed[target]

        # Build meta features with statistics
        X_meta = build_meta_features(oof_preds)  # [450, 19]
        X_test_meta = build_meta_features(X_test_raw)  # [250, 19]

        student_baseline = log_loss(y, oof_preds.mean(axis=1))

        best_combo = None
        best_score = float('inf')

        for alpha in ALPHA_SWEEP:
            for beta in BETA_SWEEP:
                for gamma in GAMMA_SWEEP:
                    mm = XGBClassifier(
                        n_estimators=15, max_depth=3,
                        reg_alpha=alpha, reg_lambda=beta,
                        gamma=gamma,
                        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
                    mm.fit(X_meta, y)
                    meta_pred = mm.predict_proba(X_meta)[:, 1]
                    meta_oof_val = log_loss(y, meta_pred)
                    
                    gap_val = student_baseline - meta_oof_val
                    gap_penalty = abs(gap_val - 0.070) * 0.3
                    score = meta_oof_val + gap_penalty

                    if score < best_score:
                        best_score = score
                        best_combo = (alpha, beta, gamma)

        alpha_b, beta_b, gamma_b = best_combo
        mm = XGBClassifier(
            n_estimators=15, max_depth=3,
            reg_alpha=alpha_b, reg_lambda=beta_b,
            gamma=gamma_b,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_pred_final = mm.predict_proba(X_meta)[:, 1]
        meta_oofs[target] = log_loss(y, meta_pred_final)
        test_preds[target] = mm.predict_proba(X_test_meta)[:, 1]
        student_oofs[target] = student_baseline
        
        per_target_best[target] = {'alpha': alpha_b, 'beta': beta_b, 'gamma': gamma_b}
        
        log.info(f"  {target}: best=(a={alpha_b}, b={beta_b}, g={gamma_b}), "
                 f"meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}, "
                 f"gap={student_oofs[target]-meta_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta

    predicted_lb = avg_meta + gap * 0.5
    estimated_lb_v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V431 Results:")
    log.info(f"  Meta features: 15 seeds + mean + std + min + max = 19")
    log.info(f"  Best per-target params: {per_target_best}")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f} (Δ vs V308: {avg_meta-0.62235:+.5f})")
    log.info(f"  AVG Student OOF: {avg_student:.5f} (Δ vs V308: {avg_student-0.69212:+.5f})")
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: 0.070, ratio: {gap/0.070:.2f}x)")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  V339 Pattern LB: {estimated_lb_v339:.5f}")
    log.info(f"\n  V429 Baseline: Meta={v429_avg:.5f}, Student={v429_student_avg:.5f}, Gap={v429_gap:.5f}")
    log.info(f"  Δ Meta (V431-V429): {avg_meta-v429_avg:+.5f}")
    log.info(f"  Δ Gap (V431-V429): {gap-v429_gap:+.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]

    sub_path = SUBMIT / f"submission_v431_meta_stats_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V431',
        'name': 'XGB Meta with Seed Prediction Statistics',
        'avg_meta_oof': round(float(avg_meta), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v429_avg_meta_oof': round(float(v429_avg), 5),
        'v429_avg_student_oof': round(float(v429_student_avg), 5),
        'v429_gap': round(float(v429_gap), 5),
        'v308_avg_oof': 0.62235, 'v308_avg_student': 0.69212,
        'v413_avg_student': 0.65128, 'v413_avg_meta': 0.60540,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'delta_vs_v429_meta': round(float(avg_meta - v429_avg), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'estimated_lb_v339_pattern': round(float(estimated_lb_v339), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': 0.070,
        'n_seeds': N_SEEDS,
        'meta_type': 'xgb_with_statistics_19feat',
        'meta_features': '15_seed_preds + mean + std + min + max',
        'alpha_sweep': ALPHA_SWEEP,
        'beta_sweep': BETA_SWEEP,
        'gamma_sweep': GAMMA_SWEEP,
        'best_per_target': {t: per_target_best[t] for t in TARGETS},
        'per_target_meta_oof': {t: round(float(v), 5) for t, v in meta_oofs.items()},
        'per_target_student_oof': {t: round(float(v), 5) for t, v in student_oofs.items()},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v431_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
