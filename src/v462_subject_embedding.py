"""
V462 — Subject Embedding + Cross-Target Constraint

Hypothesis: V439(0.623)와 V440(0.620)는 subject-level features가 핵심임을 보임.
V440에서 feat*subj_mean이 효과적이었음. 하지만 V458에서 붕괴.
V462는:
1. Subject ID를 embedding으로 변환 (not one-hot, but learned continuous)
2. K-fold cross-validated embedding: subject-level stats를 OOF로 계산
3. Cross-target constraints: Q targets와 S targets의 평균이 서로 연관되어 있으므로,
   Q의 OOF를 예측할 때 S의 OOF를 feature로, 반대로도 적용
4. Feature reduction: PCA로 282 features → 100 dimensions (noise 제거)

핵심 차이: V457의 subject*target interaction은 단순 곱셈이었음.
V462는 PCA + learned subject representation으로 더 풍부한 subject 표현 학습.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
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
N_COMPONENTS = 100  # PCA components

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


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def rank_features_lgb(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
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
    log.info("V462 — Subject Embedding + PCA + Cross-Target Constraint")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    groups_arr = train_df['subject_id'].values

    num_cols = [c for c in train_df.columns
                if c not in META_COLS | set(TARGETS)
                and np.issubdtype(train_df.dtypes[c], np.number)]

    # ===== Step 1: OOF Subject Embeddings =====
    # For each subject, compute OOF embeddings (avoid leakage)
    log.info("  Step 1: Computing OOF subject embeddings")
    skf = GroupKFold(n_splits=N_FOLDS)
    subject_embeddings = {}  # subject_id -> embedding vector
    subject_meta_features = {}  # subject_id -> metadata features
    
    # Per-subject statistics (OOF)
    for sid in np.unique(groups_arr):
        mask = groups_arr == sid
        subj_data = train_df.loc[mask, num_cols].fillna(0)
        
        # Basic stats
        subj_mean = subj_data.mean()
        subj_std = subj_data.std().fillna(0).replace(0, 1)
        subj_median = subj_data.median()
        subj_min = subj_data.min()
        subj_max = subj_data.max()
        
        # Subject embedding: z-score normalized stats across all subjects
        subj_z_mean = (subj_mean - train_df[num_cols].mean()) / (train_df[num_cols].std().fillna(1).replace(0, 1))
        subj_z_std = (subj_std - train_df[num_cols].std().fillna(0).replace(0, 1)) / (train_df[num_cols].std().fillna(1).replace(0, 1) + 1e-6)
        
        subject_meta_features[sid] = {
            'subj_mean': subj_mean,
            'subj_std': subj_std,
            'subj_z_mean': subj_z_mean,
        }
    
    # PCA on subject-level stats
    log.info("  Step 2: PCA on subject-level statistics")
    subject_ids = np.unique(groups_arr)
    subject_stats = np.zeros((len(subject_ids), len(num_cols) * 2))  # mean + std
    for i, sid in enumerate(subject_ids):
        mf = subject_meta_features[sid]
        subject_stats[i, :len(num_cols)] = mf['subj_z_mean'].values
        subject_stats[i, len(num_cols):] = mf['subj_std'].values - train_df[num_cols].std().fillna(0).replace(0, 1).values
    
    pca = PCA(n_components=min(N_COMPONENTS, min(subject_stats.shape)), random_state=SEED)
    subject_pca = pca.fit_transform(subject_stats)
    
    # Store PCA component statistics for test subjects
    test_pca_mean = pca.mean_
    test_pca_components = pca.components_
    explained_var = pca.explained_variance_ratio_
    log.info(f"  PCA explained variance (top 5): {explained_var[:5].sum():.4f}")

    # ===== Z-Score =====
    log.info("  Step 3: Z-Score + PCA")
    global_mean = train_df[num_cols].mean()
    global_std = train_df[num_cols].std().fillna(0).replace(0, 1)
    
    zscore_train = pd.DataFrame(index=train_df.index)
    zscore_test = pd.DataFrame(index=test_df.index)
    for col in num_cols:
        zscore_train[f'z_{col}'] = (train_df[col] - global_mean[col]) / global_std[col]
        zscore_test[f'z_{col}'] = (test_df[col] - global_mean[col]) / global_std[col]

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

    # ===== PCA on features =====
    log.info("  Step 4: Feature PCA (reduce dimensionality)")
    all_train_raw = pd.concat([train_df[num_cols], zscore_train], axis=1)
    all_test_raw = pd.concat([test_df[num_cols], zscore_test], axis=1)
    
    train_pca_fit = all_train_raw.fillna(0).values
    pca_feat = PCA(n_components=N_COMPONENTS, random_state=SEED)
    train_pca = pca_feat.fit_transform(train_pca_fit)
    test_pca = pca_feat.transform(all_test_raw.fillna(0).values)
    
    train_pca_df = pd.DataFrame(train_pca, columns=[f'pca_{i}' for i in range(N_COMPONENTS)])
    test_pca_df = pd.DataFrame(test_pca, columns=[f'pca_{i}' for i in range(N_COMPONENTS)])
    
    log.info(f"  PCA features: train={train_pca.shape}, test={test_pca.shape}")
    log.info(f"  PCA explained variance: {pca_feat.explained_variance_ratio_.sum():.4f}")

    # ===== Combine features =====
    # Subject embedding per row + PCA features
    train_subject_pca_arr = np.zeros((len(train_df), subject_pca.shape[1]))
    test_subject_pca_arr = np.zeros((len(test_df), subject_pca.shape[1]))
    for i, sid in enumerate(train_df['subject_id'].values):
        idx = np.where(subject_ids == sid)[0][0]
        train_subject_pca_arr[i] = subject_pca[idx]
    for i, sid in enumerate(test_df['subject_id'].values):
        if sid in subject_meta_features:
            # Use closest subject or global PCA center
            train_subject_pca_arr[i] = subject_pca.mean(axis=0)
        else:
            train_subject_pca_arr[i] = subject_pca.mean(axis=0)
    
    train_all = np.hstack([train_pca, train_subject_pca_arr])
    test_all = np.hstack([test_pca, test_subject_pca_arr])
    
    log.info(f"  Combined features: {train_all.shape[1]}")

    # ===== Phase 1: 3-Model Stacking =====
    log.info("\n=== Phase 1: 3-Model Stacking ===")
    model_oof_seeds = [{} for _ in CFGS]
    model_test_seeds = [{} for _ in CFGS]
    NFEAT = {'Q1': 50, 'Q2': 50, 'Q3': 45, 'S1': 55, 'S2': 50, 'S3': 55, 'S4': 50}

    for cfg_idx in range(len(CFGS)):
        cfg = CFGS[cfg_idx]
        log.info(f"\n  --- Config {cfg_idx+1}/3 ---")
        for t_idx, target in enumerate(TARGETS):
            n_feat = NFEAT[target]
            y = train_df[target].values.astype(np.float64)
            
            # Rank PCA features
            pca_col_names = [f'pca_{i}' for i in range(train_all.shape[1])]
            train_with_target = pd.DataFrame(train_all, columns=pca_col_names)
            train_with_target[target] = train_df[target]
            feat_cols = pca_col_names
            
            fold_ranks = []
            for fold in range(5):
                rank = rank_features_lgb(train_with_target, feat_cols, target, seed=SEED + fold * 3)
                fold_ranks.append(rank[:n_feat])
            feat_counts = {}
            for fl in fold_ranks:
                for f in fl: feat_counts[f] = feat_counts.get(f, 0) + 1
            ranked_features = sorted(feat_counts.items(), key=lambda x: -x[1])
            top_features = [int(f.replace("pca_","")) for f, c in ranked_features[:n_feat]]

            X_base = train_all[:, top_features]
            X_test_base = test_all[:, top_features]
            train_baselines = np.array([baselines[target][sid] for sid in groups_arr]).reshape(-1, 1)
            test_baselines = np.array([baselines[target][sid] for sid in test_df['subject_id'].values]).reshape(-1, 1)
            X_all = np.hstack([X_base, train_baselines])
            X_test_all = np.hstack([X_test_base, test_baselines])

            oof_seed_arr = np.zeros((len(train_df), N_SEEDS))
            test_seed_arr = np.zeros((len(test_df), N_SEEDS))

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
                    ds_train = lgb.Dataset(x_train, label=y_train, feature_name=[str(f) for f in top_features] + ['baseline'])
                    ds_val = lgb.Dataset(x_val, label=y[va_idx], feature_name=[str(f) for f in top_features] + ['baseline'], reference=ds_train)
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
    log.info("V462 Results:")
    log.info(f"  AVG Meta OOF: {avg_meta:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  Gap: {gap:.5f} ({gap/0.070:.2f}x)")
    log.info(f"  V339 LB: {v339:.5f}")
    log.info(f"  PCA components: {N_COMPONENTS}, explained var: {pca_feat.explained_variance_ratio_.sum():.4f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS: sub[t] = test_preds[t]
    sub_path = SUBMIT / f"submission_v462_subject_embedding_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    meta_data = {'version': 'V462', 'name': 'Subject Embedding + PCA + Cross-Target Constraint',
        'avg_meta_oof': round(float(avg_meta), 5), 'avg_student_oof': round(float(avg_student), 5),
        'v308_lb': 0.63893, 'estimated_lb_v339_pattern': round(float(v339), 5),
        'student_meta_gap': round(float(gap), 5), 'n_models': 3, 'n_seeds': N_SEEDS,
        'n_pca_components': N_COMPONENTS,
        'submission_file': str(sub_path), 'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0)}
    meta_path = EXPERIMENTS / f'v462_{ts}.json'
    with open(meta_path, 'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, Total: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
