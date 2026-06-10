"""
V468 — Aggressive Per-Target Feature Reduction + V466 Consensus Baseline

Hypothesis: V466의 consensus selection(top 30)이 이미 good but, 더 작은 feature set(15-20)으로
과적합을 더 줄이면 student OOF가 추가로 낮아질 수 있음.
V466의 CV-internal adversarial + consensus ranking을 재사용하되, 
target별 top 15/20/25를 sweep하여 optimal K 찾기.

핵심: V466의 consensus는 유지하되, K를 aggressively 줄임 → overfitting 감소 → student ↓
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

CFGS = [
    {'num_leaves': 10, 'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 3000,
     'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 10.0, 'reg_lambda': 50.0,
     'min_child_samples': 40},
    {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
     'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 20.0,
     'min_child_samples': 25},
    {'num_leaves': 50, 'max_depth': 7, 'learning_rate': 0.03, 'n_estimators': 1000,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'min_child_samples': 8},
]

K_SWEPS = [15, 20, 25, 30]  # target별 feature 개수 sweep


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V468 — Aggressive Per-Target Feature Reduction + K Sweep")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df.dtypes[c], np.number)]

    # ===== Z-Score + Baseline =====
    log.info("  Step 1: Z-Score + Baseline")
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

    # ===== Interaction features =====
    for t in TARGETS:
        train_bl = np.array([baselines[t][sid] for sid in groups_arr])
        test_bl = np.array([baselines[t][sid] for sid in test_df['subject_id'].values])
        for col in num_cols:
            z_col = f'z_{col}'
            if z_col in zscore_train.columns:
                zscore_train[f'zb_{t}_{col}'] = zscore_train[z_col] * train_bl
                zscore_test[f'zb_{t}_{col}'] = zscore_test[z_col] * test_bl
                zscore_train[f'z3_{t}_{col}'] = zscore_train[z_col].values ** 3
                zscore_test[f'z3_{t}_{col}'] = zscore_test[z_col].values ** 3

    all_train_features = pd.concat([train_df[num_cols], zscore_train], axis=1)
    all_test_features = pd.concat([test_df[num_cols], zscore_test], axis=1)
    log.info(f"  Features: {all_train_features.shape[1]}")

    feat_cols_all = [c for c in all_train_features.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(all_train_features.dtypes[c], np.number)]

    # ===== Phase 1: 3-Model Stacking with K Sweep =====
    log.info("\n=== Phase 1: 3-Model Stacking (K Sweep) ===")
    
    # Store results for each K sweep
    best_config_k = None
    best_avg_meta = float('inf')
    best_avg_student = float('inf')
    best_gap = float('inf')
    best_meta_oof_seeds = None
    best_test_seeds = None
    best_meta_oofs = {}
    best_student_oofs = {}
    
    for K in K_SWEPS:
        log.info(f"\n  === K = {K} ===")
        model_oof_seeds = [{} for _ in CFGS]
        model_test_seeds = [{} for _ in CFGS]
        
        for cfg_idx in range(len(CFGS)):
            cfg = CFGS[cfg_idx]
            for t_idx, target in enumerate(TARGETS):
                y = train_df[target].values.astype(np.float64)
                feat_cols = remove_leak(feat_cols_all, target)

                train_with_target = all_train_features.copy()
                train_with_target[target] = train_df[target]

                # Feature ranking via importance
                spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                params_rank = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100,
                              'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1}
                
                X_rank = train_with_target[feat_cols].fillna(0).values.astype(np.float64)
                sn = [sanitize_col(c) for c in feat_cols]
                ds = lgb.Dataset(X_rank, label=y, feature_name=sn)
                m_rank = lgb.train(params_rank, ds, num_boost_round=100)
                imp = m_rank.feature_importance(importance_type='gain')
                ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
                top_features = [r[0] for r in ranked[:K]]
                del m_rank, ds, X_rank
                gc.collect()

                X_base = train_with_target[top_features].fillna(0).values.astype(np.float64)
                X_test_base = all_test_features[top_features].fillna(0).values.astype(np.float64)
                train_baselines = np.array([baselines[target][sid] for sid in groups_arr]).reshape(-1, 1)
                test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)
                X_all = np.hstack([X_base, train_baselines])
                X_test_all = np.hstack([X_test_base, test_baselines])

                oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
                test_seed_arr = np.zeros((len(test_df), N_SEEDS))
                skf_inner = GroupKFold(n_splits=N_FOLDS)

                for s in range(N_SEEDS):
                    if (s + 1) % 5 == 0:
                        log.info(f"    {target} K={K}: seed {s+1}/{N_SEEDS}")
                    sk = SEED + s * 7 + t_idx + cfg_idx * 300
                    seed_oof = np.zeros(len(train_df))
                    seed_test = np.zeros(len(test_df))
                    for fold, (tr_idx, va_idx) in enumerate(skf_inner.split(X_all, y, groups_arr)):
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
                log.info(f"  {target} K={K} [{cfg_idx+1}]: oof={avg_oof:.5f}")

        # Quick meta evaluation to find best K
        student_oofs_k = {}; meta_oofs_k = {}
        for t_idx, target in enumerate(TARGETS):
            y = train_df[target].values
            all_oof = np.mean([model_oof_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
            all_test = np.mean([model_test_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
            student_oofs_k[target] = log_loss(y, all_oof)
            
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
            meta_oofs_k[target] = log_loss(y, mm.predict_proba(X_meta)[:, 1])

        avg_meta_k = np.mean(list(meta_oofs_k.values()))
        avg_student_k = np.mean(list(student_oofs_k.values()))
        gap_k = avg_student_k - avg_meta_k
        log.info(f"  K={K}: meta={avg_meta_k:.5f}, student={avg_student_k:.5f}, gap={gap_k:.5f} ({gap_k/0.070:.2f}x)")

        # Track best by gap (prioritize safety)
        if gap_k < best_gap or (gap_k <= best_gap + 0.005 and avg_student_k < best_avg_student):
            best_gap = gap_k
            best_avg_meta = avg_meta_k
            best_avg_student = avg_student_k
            best_config_k = K
            best_meta_oof_seeds = model_oof_seeds
            best_test_seeds = model_test_seeds
            best_meta_oofs = meta_oofs_k
            best_student_oofs = student_oofs_k
            log.info(f"  -> New best: K={K}, gap={gap_k:.5f}")

    log.info(f"\n  Optimal K: {best_config_k}")

    # ===== Phase 2: Meta with optimal K =====
    log.info("\n=== Phase 2: Meta (optimal K) ===")
    test_preds = {}

    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values
        all_oof = np.mean([best_meta_oof_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)
        all_test = np.mean([best_test_seeds[i][target].mean(axis=1) for i in range(3)], axis=0)

        group = 'Q' if target.startswith('Q') else 'S'
        group_targets = Q_TARGETS if group == 'Q' else S_TARGETS
        other_group = S_TARGETS if group == 'Q' else Q_TARGETS
        
        from xgboost import XGBClassifier
        cross_oof_list = []
        cross_test_list = []
        for t_cross in group_targets:
            if t_cross == target: continue
            cross_oof_list.append(np.mean([best_meta_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
            cross_test_list.append(np.mean([best_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0))
        for t_cross in other_group:
            cross_oof_list.append(np.mean([best_meta_oof_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
            cross_test_list.append(np.mean([best_test_seeds[i][t_cross].mean(axis=1) for i in range(3)], axis=0) * 0.5)
        cross_arr = np.column_stack(cross_oof_list)
        cross_arr_test = np.column_stack(cross_test_list)
        X_meta = np.hstack([all_oof.reshape(-1, 1), cross_arr])
        X_test = np.hstack([all_test.reshape(-1, 1), cross_arr_test])
        mm = XGBClassifier(n_estimators=15, max_depth=3, reg_alpha=0.01, reg_lambda=0.0,
            gamma=0.0, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, min_child_weight=5, verbosity=0)
        mm.fit(X_meta, y)
        test_preds[target] = mm.predict_proba(X_test)[:, 1]
        log.info(f"  {target}: meta={best_meta_oofs[target]:.5f}, student={best_student_oofs[target]:.5f}")

    avg_meta = np.mean(list(best_meta_oofs.values()))
    avg_student = np.mean(list(best_student_oofs.values()))
    gap = avg_student - avg_meta
    v339 = avg_meta + gap * 0.85

    log.info(f"\n{'='*70}")
    log.info("V468 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  Gap: {gap:.5f} ({gap/0.070:.2f}x)")
    log.info(f"  V339 LB: {v339:.5f}")
    log.info(f"  Optimal K: {best_config_k}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS: sub[t] = test_preds[t]
    sub_path = SUBMIT / f"submission_v468_k_sweep_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    meta_data = {'version': 'V468', 'name': f'Aggressive Feature Reduction (K={best_config_k})',
        'avg_meta_oof': round(float(avg_meta), 5), 'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893, 'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5), 'n_models': 3, 'n_seeds': N_SEEDS,
        'submission_file': str(sub_path), 'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'optimal_k': best_config_k, 'k_sweep': K_SWEPS}
    meta_path = EXPERIMENTS / f'v468_{ts}.json'
    with open(meta_path, 'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, Total: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
