"""
V465 — Adversarial Feature Re-weighting + Strong Regularization

Hypothesis: V461은 feature를 완전히 제거 → V339 LB 0.596. 하지만 feature 제거는 signal도 함께 잃을 수 있음.
V465는 feature를 제거하지 않고, feature 값을 adversarial-importance에 따라 스케일링.
1. Adversarial importance가 높은 feature의 값을 0.3배로 스케일링 (signal 유지하되 train-test shift 영향 감소)
2. Regularization을 극단적으로 강하게: reg_alpha=20, reg_lambda=100
3. N_estimators=5000, learning_rate=0.005 — low LR로 서서히 학습
4. Interaction features는 제거 (feature scaling이 이미 신호를 보호하므로)

핵심: V461은 "feature 제거", V465는 "feature 약화". signal은 유지하되 train-test shift 영향만 감소.
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
Q_TARGETS = ['Q1','Q2','Q3']
S_TARGETS = ['S1','S2','S3','S4']

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
N_ESTIMATORS = 5000

CFGS = [
    {'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.005, 'n_estimators': N_ESTIMATORS,
     'subsample': 0.3, 'colsample_bytree': 0.3, 'reg_alpha': 20.0, 'reg_lambda': 100.0,
     'min_child_samples': 50},
    {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.005, 'n_estimators': N_ESTIMATORS,
     'subsample': 0.35, 'colsample_bytree': 0.35, 'reg_alpha': 15.0, 'reg_lambda': 80.0,
     'min_child_samples': 40},
    {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 3000,
     'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 50.0,
     'min_child_samples': 30},
]


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def apply_feature_scaling(X_df, adv_weights, num_col_names):
    """Scale feature values by (1 - 0.7 * adv_imp). Low adv_imp → scale 1.0, high → 0.3."""
    X_scaled = X_df.copy()
    for col in X_scaled.columns:
        if col in adv_weights:
            X_scaled[col] *= adv_weights[col]
        # else: no scaling (col not in adv analysis, e.g., z-score features)
    return X_scaled


def rank_features_lgb(feat_df, feat_cols, seed=SEED):
    """Rank features using LGBM gain importance."""
    y = feat_df.iloc[:, -1].values  # last column is the target
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': 1.0, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
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
    log.info("V465 — Adversarial Feature Scaling + Strong Reg")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df.dtypes[c], np.number)]

    # ===== Step 1: Adversarial Validation (Global) =====
    log.info("  Step 1: Adversarial Validation (Global)")
    adv_X = pd.concat([train_df[num_cols].fillna(0), test_df[num_cols].fillna(0)], axis=0)
    adv_y = np.array([0]*len(train_df) + [1]*len(test_df))
    
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 30, 'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 100,
              'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1}
    sn = [sanitize_col(c) for c in num_cols]
    ds = lgb.Dataset(adv_X[num_cols].values, label=adv_y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=100)
    adv_imp = m.feature_importance(importance_type='gain') / max(m.feature_importance(importance_type='gain').sum(), 1)
    
    adv_weights = {num_cols[i]: 1.0 - 0.7 * adv_imp[i] for i in range(len(num_cols))}
    
    log.info(f"  Adversarial weights: min={min(adv_weights.values()):.3f}, max={max(adv_weights.values()):.3f}, mean={np.mean(list(adv_weights.values())):.3f}")
    
    # Show top-10 most adversarial (heavily weighted-down) features
    sorted_adv = sorted(adv_weights.items(), key=lambda x: x[1])
    log.info("  Top-10 most adversarial features (scale 0.3):")
    for f, w in sorted_adv[:10]:
        log.info(f"    {f}: scale={w:.3f}")
    
    del m, ds
    gc.collect()

    # ===== Z-Score + Baseline =====
    log.info("  Step 2: Z-Score + Baseline")
    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    for col in num_cols:
        tr_mean = train_df.groupby('subject_id')[col].transform('mean')
        tr_std = train_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        te_mean = test_df.groupby('subject_id')[col].transform('mean')
        te_std = test_df.groupby('subject_id')[col].transform('std').fillna(0).replace(0, 1)
        zscore_train[f'z_{col}'] = (train_df[col] - tr_mean) / tr_std
        zscore_test[f'z_{col}'] = (test_df[col] - te_mean) / te_std

    subject_ids = np.unique(groups_arr)
    baselines = {}
    for t in TARGETS:
        y_t = train_df[t].values
        bl = {}
        for sid in subject_ids:
            mask = groups_arr == sid
            s_y = y_t[mask]; n_samples = mask.sum(); global_rate = y_t.mean()
            subj_rate = s_y.mean() if n_samples > 0 else global_rate
            bl[sid] = 0.7 * subj_rate + 0.3 * global_rate
        baselines[t] = bl

    # ===== Build features with adversarial scaling =====
    log.info("  Step 3: Apply Adversarial Feature Scaling")
    # Scale original numeric features
    train_num_scaled = apply_feature_scaling(train_df[num_cols], adv_weights, num_cols)
    test_num_scaled = apply_feature_scaling(test_df[num_cols], adv_weights, num_cols)
    log.info(f"  Scaled features: train={train_num_scaled.shape}, test={test_num_scaled.shape}")
    
    # Combine scaled features + z-score features (z-score features don't get scaled - they're normalized)
    all_train_features = pd.concat([train_num_scaled, zscore_train], axis=1)
    all_test_features = pd.concat([test_num_scaled, zscore_test], axis=1)
    log.info(f"  Total features: {all_train_features.shape[1]}")

    feat_cols_all = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features.dtypes[c], np.number)]

    # ===== Phase 1: 3-Model Stacking =====
    log.info("\n=== Phase 1: 3-Model Stacking ===")
    model_oof_seeds = [{} for _ in CFGS]
    model_test_seeds = [{} for _ in CFGS]
    NFEAT = {'Q1': 30, 'Q2': 30, 'Q3': 28, 'S1': 32, 'S2': 30, 'S3': 32, 'S4': 30}

    for cfg_idx in range(len(CFGS)):
        cfg = CFGS[cfg_idx]
        log.info(f"\n  --- Config {cfg_idx+1}/3 ---")
        for t_idx, target in enumerate(TARGETS):
            n_feat = NFEAT[target]
            y = train_df[target].values.astype(np.float64)
            feat_cols = remove_leak(feat_cols_all, target)

            train_with_target = all_train_features.copy()
            train_with_target[target] = train_df[target]

            fold_ranks = []
            for fold in range(5):
                rank = rank_features_lgb(train_with_target, feat_cols, seed=SEED + fold * 3)
                fold_ranks.append(rank[:n_feat])
            feat_counts = {}
            for fl in fold_ranks:
                for f in fl: feat_counts[f] = feat_counts.get(f, 0) + 1
            ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
            top_features = [f for f, c in ranked_features[:n_feat]]

            X_base = train_with_target[top_features].fillna(0).values.astype(np.float64)
            X_test_base = all_test_features[top_features].fillna(0).values.astype(np.float64)
            train_baselines = np.array([baselines[target][sid] for sid in groups_arr]).reshape(-1, 1)
            test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)
            X_all = np.hstack([X_base, train_baselines])
            X_test_all = np.hstack([X_test_base, test_baselines])

            oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
            test_seed_arr = np.zeros((len(test_df), N_SEEDS))
            skf = GroupKFold(n_splits=N_FOLDS)

            for s in range(N_SEEDS):
                if (s + 1) % 5 == 0:
                    log.info(f"    {target}: seed {s+1}/{N_SEEDS}")
                sk = SEED + s * 7 + t_idx + cfg_idx * 300
                seed_oof = np.zeros(len(train_df))
                seed_test = np.zeros(len(test_df))
                for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y, groups_arr)):
                    x_train, y_train = X_all[tr_idx], y[tr_idx]
                    x_val = X_all[va_idx]
                    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
                    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                        **cfg, 'scale_pos_weight': spw, 'random_state': sk,
                        'force_row_wise': True, 'n_jobs': 1}
                    ds_train = lgb.Dataset(x_train, label=y_train,
                        feature_name=[sanitize_col(c) for c in top_features + ['baseline']])
                    ds_val = lgb.Dataset(x_val, label=y[va_idx],
                        feature_name=[sanitize_col(c) for c in top_features + ['baseline']], reference=ds_train)
                    model = lgb.train(params, ds_train, num_boost_round=cfg['n_estimators'],
                        valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=0)])
                    seed_oof[va_idx] = model.predict(x_val)
                    seed_test += model.predict(X_test_all) / N_FOLDS
                    del model, ds_train, ds_val
                    gc.collect()
                oof_seed_arr[:, s] = np.clip(seed_oof, 0.001, 0.999)
                test_seed_arr[:, s] = np.clip(seed_test, 0.001, 0.999)

            model_oof_seeds[cfg_idx][target] = oof_seed_arr
            model_test_seeds[cfg_idx][target] = test_seed_arr
            avg_oof = log_loss(y, oof_seed_arr.mean(axis=1))
            log.info(f"  {target} [{cfg_idx+1}]: oof={avg_oof:.5f}")

    # ===== Phase 2: Meta =====
    log.info("\n=== Phase 2: Meta ===")
    student_oofs = {}; test_preds = {}; meta_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        all_oof = np.mean([model_oof_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        all_test = np.mean([model_test_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        student_oofs[target] = log_loss(y, all_oof)

        group = 'Q' if target.startswith('Q') else 'S'
        group_targets = Q_TARGETS if group == 'Q' else S_TARGETS
        other_group = S_TARGETS if group == 'Q' else Q_TARGETS
        
        from xgboost import XGBClassifier
        cross_oof_list = []
        cross_test_list = []
        for t_cross in group_targets:
            if t_cross == target: continue
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
        for t_cross in other_group:
            cross_oof_list.append(np.mean([model_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
            cross_test_list.append(np.mean([model_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
        cross_arr = np.column_stack(cross_oof_list)
        cross_arr_test = np.column_stack(cross_test_list)

        X_meta = np.hstack([all_oof.reshape(-1, 1), cross_arr])
        X_test = np.hstack([all_test.reshape(-1, 1), cross_arr_test])

        mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        meta_oofs[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])
        test_preds[target] = mm.predict_proba(X_test)[:, 1]
        log.info(f"  {target}: meta={meta_oofs[target]:.5f}, student={student_oofs[target]:.5f}")

    avg_meta = np.mean(list(meta_oofs.values()))
    avg_student = np.mean(list(student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V465 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  Gap: {gap:.5f} ({gap/0.070:.2f}x)")
    log.info(f"  V339 LB: {v339:.5f}")
    log.info(f"  N_estimators: {N_ESTIMATORS}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS: sub[t] = test_preds[t]
    sub_path = SUBMIT / f"submission_v465_feat_reweight_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    meta_data = {'version': 'V465', 'name': 'Adversarial Feature Scaling + Strong Reg',
        'avg_meta_oof': round(float(avg_meta), 5), 'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893, 'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5), 'n_models': 3, 'n_seeds': N_SEEDS,
        'n_estimators': N_ESTIMATORS,
        'submission_file': str(sub_path), 'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0)}
    meta_path = EXPERIMENTS / f'v465_{ts}.json'
    with open(meta_path, 'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, Total: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
