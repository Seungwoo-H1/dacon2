"""
V327: Two-Level Stacking (L1 LR → L2 LR)

Hypothesis: V326 achieves OOF 0.59159 but has large student-Meta gap (0.10-0.15).
Two-level stacking might help by:
1. Grouping 15 seeds into 3 groups of 5
2. Training LR on each group → 3 intermediate predictions
3. Training final LR on 3 intermediate predictions + V321 seed predictions

This adds an extra layer of regularization, potentially reducing student-Meta gap.

Architecture:
- Level 0: V326 students (15 seeds, feature bagging)
- Level 1: 3 LR models (each on 5 seeds)
- Level 2: 1 LR model (on 3 L1 preds + 15 L0 preds = 18 features)

Expected OOF: 0.585-0.595 (small improvement via better regularization)
Risk: MEDIUM (extra complexity, possible underfitting)
Cost: ~120s
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0
FEATURE_BAG_FRACTION = 0.75
N_GROUPS = 3  # 15 seeds / 3 groups = 5 per group


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


def generate_zscore_features(train_df, test_df):
    """Generate global z-score features from train stats."""
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_base) & set(test_base)
    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        test_df = test_df.copy()
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
        train_df = train_df.copy()
        train_df[zc] = (vals - mean) / std
    return train_df, test_df


def generate_heavy_features(df):
    """Generate heavy features (interactions + per-subject z-scores)."""
    df = df.copy()
    new_cols = []

    # Interactions
    hr_cols = [c for c in df.columns if c.startswith('wHr_') and np.issubdtype(df[c].dtype, np.number)]
    pedo_cols = [c for c in df.columns if c.startswith('wPedo_') and np.issubdtype(df[c].dtype, np.number)]
    light_cols = [c for c in df.columns if c.startswith('mLight_') and np.issubdtype(df[c].dtype, np.number)]
    screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_') and np.issubdtype(df[c].dtype, np.number)]
    gps_cols = [c for c in df.columns if c.startswith('mGps_') and np.issubdtype(df[c].dtype, np.number)]
    ble_cols = [c for c in df.columns if c.startswith('mBle_') and np.issubdtype(df[c].dtype, np.number)]
    wifi_cols = [c for c in df.columns if c.startswith('mWifi_') and np.issubdtype(df[c].dtype, np.number)]
    usage_cols = [c for c in df.columns if c.startswith('mUsageStats_') and np.issubdtype(df[c].dtype, np.number)]

    if hr_cols and pedo_cols:
        df['hr_pedo_interaction'] = df[hr_cols].fillna(0).mean(axis=1) * df[pedo_cols].fillna(0).mean(axis=1)
        new_cols.append('hr_pedo_interaction')
    if light_cols and screen_cols:
        df['light_screen_interaction'] = df[light_cols].fillna(0).mean(axis=1) * df[screen_cols].fillna(0).mean(axis=1)
        new_cols.append('light_screen_interaction')
    if gps_cols and ble_cols:
        df['gps_ble_interaction'] = df[gps_cols].fillna(0).mean(axis=1) * df[ble_cols].fillna(0).mean(axis=1)
        new_cols.append('gps_ble_interaction')
    if wifi_cols and gps_cols:
        df['wifi_gps_interaction'] = df[wifi_cols].fillna(0).mean(axis=1) * df[gps_cols].fillna(0).mean(axis=1)
        new_cols.append('wifi_gps_interaction')

    # Ratio features
    pedo_steps = [c for c in pedo_cols if 'step' in c and 'sum' not in c]
    pedo_dist = [c for c in pedo_cols if 'distance' in c]
    if pedo_steps and pedo_dist:
        step_mean = df[pedo_steps].fillna(0).mean(axis=1)
        dist_mean = df[pedo_dist].fillna(0).mean(axis=1)
        df['step_length_ratio'] = (dist_mean + 1e-8) / (step_mean + 1e-8)
        new_cols.append('step_length_ratio')

    # Total activity proxy
    all_base = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                and not c.endswith('_zscore') and np.issubdtype(df[c].dtype, np.number)]
    df['total_activity_proxy'] = df[all_base].fillna(0).abs().sum(axis=1)
    new_cols.append('total_activity_proxy')

    # Per-subject z-scores
    base_cols = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and np.issubdtype(df[c].dtype, np.number)]
    for col in base_cols:
        zscored = df.groupby('subject_id')[col].transform(lambda g: (g - g.mean()) / max(g.std(ddof=0), 1e-8))
        zc = f'ps_zscore_{col}'
        df[zc] = zscored.values
        new_cols.append(zc)

    return df, new_cols


# V326 config
CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V327 — Two-Level Stacking (L1 Group LR → L2 Meta LR)")
    log.info("15 seeds → 3 groups of 5 → 3 LR → final LR on 18 features")
    log.info("=" * 70)

    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # Generate heavy features
    train_df, heavy_feat_names = generate_heavy_features(train_df)
    test_df, _ = generate_heavy_features(test_df)

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    base_cols = [c for c in train_feat_cols if '_zscore' not in c and c not in heavy_feat_names]

    log.info(f"Feature counts:")
    log.info(f"  Base: {len(base_cols)}")
    log.info(f"  Global zscore: {len([c for c in train_feat_cols if '_zscore' in c and c not in heavy_feat_names])}")
    log.info(f"  Heavy (interactions + per-subj zscore): {len(heavy_feat_names)}")
    log.info(f"  Total: {len(train_feat_cols)}")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)

    # ====== STEP 1: Run 15-seed V326 stacking ======
    log.info("\n" + "=" * 70)
    log.info("STEP 1: Training 15 V326-style students...")
    log.info("=" * 70)

    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    all_seed_test_preds = {t: [] for t in TARGETS}

    for t in TARGETS:
        log.info(f"\nTarget: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        ranked = rank_features(train_df, feat_cols_clean, t)
        candidate_feats = ranked

        for si in range(N_SEEDS):
            seed = SEED + si * 7

            rng = np.random.RandomState(seed)
            n_bag = max(int(len(candidate_feats) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]

            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)

            sel_cols = [c for c in bag_feats if c in test_feat_cols]

            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]

                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])

                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))

            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            all_seed_oofs[t].append(seed_oof.copy())
            all_seed_test_preds[t].append(seed_test.copy())

            if si < 3 or si == N_SEEDS - 1:
                s_oof = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")

        # Convert to array
        all_seed_oofs[t] = np.column_stack(all_seed_oofs[t])
        all_seed_test_preds[t] = np.column_stack(all_seed_test_preds[t])

    # V326-style LR meta (reference)
    log.info("\n" + "=" * 70)
    log.info("REFERENCE: V321 V326 single-level LR meta")
    log.info("=" * 70)

    v326_target_oofs = {}
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = all_seed_oofs[t]
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        v326_target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))

    v326_avg = np.mean(list(v326_target_oofs.values()))
    log.info(f"  V326 AVG OOF: {v326_avg:.5f}")

    # ====== STEP 2: Two-Level Stacking ======
    # Group seeds into N_GROUPS groups
    seeds_per_group = N_SEEDS // N_GROUPS
    groups = [[si for si in range(s * seeds_per_group, (s+1) * seeds_per_group)]
              for s in range(N_GROUPS)]
    # Handle remainder
    remaining_seeds = list(range(N_GROUPS * seeds_per_group, N_SEEDS))
    if remaining_seeds:
        for i, si in enumerate(remaining_seeds):
            groups[i % N_GROUPS].append(si)

    log.info(f"\n  Groups: {[len(g) for g in groups]}")

    log.info("\n" + "=" * 70)
    log.info("STEP 2: Two-Level Stacking")
    log.info("=" * 70)

    two_level_target_oofs = {}
    two_level_student_avg = {}

    for t in TARGETS:
        log.info(f"\nTarget: {t}")
        y = train_df[t].values.astype(np.float64)
        oof_matrix = all_seed_oofs[t]  # shape: (n_train, 15)

        # Level 1: Train LR on each group of seeds
        l1_preds_train = []  # shape: (n_train, N_GROUPS)
        l1_meta_models = []

        for gi, group_seeds in enumerate(groups):
            group_oof = oof_matrix[:, group_seeds]  # (n_train, seeds_per_group)
            l1_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED + gi)
            l1_meta.fit(group_oof, y)
            l1_pred = l1_meta.predict_proba(group_oof)[:, 1]
            l1_preds_train.append(l1_pred)
            l1_meta_models.append(l1_meta)
            l1_oof = log_loss(y, np.clip(l1_pred, 0.001, 0.999))
            log.info(f"    L1 Group {gi} (seeds {group_seeds}): OOF={l1_oof:.5f}")

        l1_matrix = np.column_stack(l1_preds_train)  # (n_train, 3)

        # Level 2: Train LR on L1 preds + all 15 seed preds
        l2_features = np.column_stack([l1_matrix, oof_matrix])  # (n_train, 18)
        l2_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED + 100)
        l2_meta.fit(l2_features, y)
        l2_pred = l2_meta.predict_proba(l2_features)[:, 1]
        two_level_target_oofs[t] = log_loss(y, np.clip(l2_pred, 0.001, 0.999))
        two_level_student_avg[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t].T])

        log.info(f"    L2 Meta: OOF={two_level_target_oofs[t]:.5f}")
        log.info(f"    L1 features: {l1_matrix.shape}, L2 features: {l2_features.shape}")
        log.info(f"    L2 weights (first 5): {l2_meta.coef_[0][:5]}")

    # L2 test prediction
    l2_test_preds = {t: [] for t in TARGETS}
    for t in TARGETS:
        # L1 test preds
        l1_test_list = []
        for gi, group_seeds in enumerate(groups):
            group_test = all_seed_test_preds[t][:, group_seeds]  # (n_test, seeds_per_group)
            l1_pred_test = l1_meta_models[gi].predict_proba(group_test)[:, 1]
            l1_test_list.append(l1_pred_test)

        l1_test_matrix = np.column_stack(l1_test_list)  # (n_test, 3)

        # L2 features for test
        test_oof_matrix = all_seed_test_preds[t]  # (n_test, 15)
        l2_test_features = np.column_stack([l1_test_matrix, test_oof_matrix])  # (n_test, 18)

        meta_model = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_model.fit(all_seed_oofs[t], train_df[t].values.astype(np.float64))
        l2_test_preds[t] = meta_model.predict_proba(test_oof_matrix)[:, 1]

    avg_two_level = np.mean(list(two_level_target_oofs.values()))
    avg_two_level_student = np.mean(list(two_level_student_avg.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V327 RESULTS (Two-Level Stacking)")
    log.info(f"{'='*70}")

    for t in TARGETS:
        gap = two_level_student_avg[t] - two_level_target_oofs[t]
        log.info(f"  {t}: OOF={two_level_target_oofs[t]:.5f} (student={two_level_student_avg[t]:.5f}, gap={gap:+.4f})")
    log.info(f"  AVG OOF: {avg_two_level:.5f}")
    log.info(f"  V326 AVG OOF: {v326_avg:.5f}")
    log.info(f"  Δ vs V326: {avg_two_level - v326_avg:+.5f}")
    log.info(f"  Student Avg: {avg_two_level_student:.5f}")
    log.info(f"  V321: 0.60569 | V326: 0.59159")

    pred_lb = avg_two_level + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")

    # Compare V326 vs Two-Level
    log.info(f"\n{'='*70}")
    log.info(f"V326 vs V327 Two-Level Comparison")
    log.info(f"{'='*70}")
    log.info(f"  V326 AVG: {v326_avg:.5f}")
    log.info(f"  V327 AVG: {avg_two_level:.5f}")
    log.info(f"  Improvement: {v326_avg - avg_two_level:+.5f}")
    log.info(f"{'='*70}")

    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = all_seed_oofs[t]
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(all_seed_test_preds[t])[:, 1]

    sub_path = SUBMIT / f"submission_v327_twostack_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V327',
        'name': 'Two-Level Stacking (L1 Group LR → L2 Meta LR)',
        'avg_oof': round(float(avg_two_level), 5),
        'v326_avg_oof': round(float(v326_avg), 5),
        'improvement_vs_v326': round(float(avg_two_level - v326_avg), 5),
        'avg_student_oof': round(float(avg_two_level_student), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'n_groups': N_GROUPS,
        'seeds_per_group': seeds_per_group,
        'meta_c': META_C,
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'v321_avg_oof': 0.60569,
        'v326_avg_oof_ref': round(float(v326_avg), 5),
        'delta_vs_v321': round(float(avg_two_level - 0.60569), 5),
        'per_target_oof': {t: round(float(two_level_target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(two_level_student_avg[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'Two-level stacking: 15 seeds → 3 groups of 5 → L1 LR → L2 LR on 18 features',
    }

    meta_path = EXPERIMENTS / f'v327_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_two_level, meta_data


if __name__ == '__main__':
    main()
