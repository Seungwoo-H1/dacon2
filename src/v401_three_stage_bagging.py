"""
V401: V401 — Three-Stage Stacking + 50% Bagging

Three-level hierarchy:
1. Stage 1: 15 seeds of V329-style students → OOF predictions
2. Stage 2: 15 seeds on augmented features (Stage 1 preds as features) → OOF predictions  
3. Stage 3: 15 seeds on doubly-augmented features (Stage 1 + Stage 2 preds as features) → OOF predictions
4. LR meta on Stage 3 students

Hypothesis: Each additional stacking layer extracts more signal from the ensemble diversity.
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
FEATURE_BAG_FRACTION = 0.50

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

def generate_base_features(train_df, test_df):
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
        if std < 1e-8: std = 1e-8
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[f'{col}_zscore'] = (vals - mean) / std
        test_df[f'{col}_zscore'] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std

    def add_interactions(df):
        df = df.copy()
        hr_cols = [c for c in df.columns if c.startswith('wHr_') and np.issubdtype(df[c].dtype, np.number)]
        pedo_cols = [c for c in df.columns if c.startswith('wPedo_') and np.issubdtype(df[c].dtype, np.number)]
        light_cols = [c for c in df.columns if c.startswith('mLight_') and np.issubdtype(df[c].dtype, np.number)]
        screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_') and np.issubdtype(df[c].dtype, np.number)]
        gps_cols = [c for c in df.columns if c.startswith('mGps_') and np.issubdtype(df[c].dtype, np.number)]
        ble_cols = [c for c in df.columns if c.startswith('mBle_') and np.issubdtype(df[c].dtype, np.number)]
        wifi_cols = [c for c in df.columns if c.startswith('mWifi_') and np.issubdtype(df[c].dtype, np.number)]
        if hr_cols and pedo_cols:
            df['hr_pedo_interact'] = df[hr_cols].fillna(0).mean(axis=1) * df[pedo_cols].fillna(0).mean(axis=1)
        if light_cols and screen_cols:
            df['light_screen_interact'] = df[light_cols].fillna(0).mean(axis=1) * df[screen_cols].fillna(0).mean(axis=1)
        if gps_cols and ble_cols:
            df['gps_ble_interact'] = df[gps_cols].fillna(0).mean(axis=1) * df[ble_cols].fillna(0).mean(axis=1)
        if wifi_cols and gps_cols:
            df['wifi_gps_interact'] = df[wifi_cols].fillna(0).mean(axis=1) * df[gps_cols].fillna(0).mean(axis=1)
        pedo_steps = [c for c in pedo_cols if 'step' in c and 'sum' not in c]
        pedo_dist = [c for c in pedo_cols if 'distance' in c]
        if pedo_steps and pedo_dist:
            df['step_length_ratio'] = df[pedo_dist].fillna(0).mean(axis=1) / (df[pedo_steps].fillna(0).mean(axis=1) + 1e-8)
        return df

    train_df = add_interactions(train_df)
    test_df = add_interactions(test_df)

    base_cols = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and not c.endswith('_interact')
                 and not c.endswith('_ratio') and np.issubdtype(train_df[c].dtype, np.number)]
    base_cols = base_cols[:60]

    for col in base_cols:
        for df_src, prefix in [(train_df, 'train'), (test_df, 'test')]:
            grp = df_src.groupby('subject_id')[col]
            for w in [3, 5]:
                df_src[f'ps_roll{w}_mean_{col}'] = grp.transform(lambda g: g.rolling(w, min_periods=1).mean()).values
                df_src[f'ps_roll{w}_std_{col}'] = grp.transform(lambda g: g.rolling(w, min_periods=1).std().fillna(0)).values
            ps_mean = grp.transform('mean')
            ps_std = grp.transform('std').fillna(0)
            ps_min = grp.transform('min')
            ps_max = grp.transform('max')
            ps_median = grp.transform('median')
            ps_iqr = grp.transform(lambda g: g.quantile(0.75) - g.quantile(0.25))
            for fn, fv in [('min',ps_min),('max',ps_max),('median',ps_median),('iqr',ps_iqr)]:
                df_src[f'ps_{fn}_{col}'] = fv.values
            df_src[f'ps_range_{col}'] = (ps_max - ps_min).values
            df_src[f'ps_cv_{col}'] = (ps_std / (ps_mean.abs() + 1e-8)).values
            df_src[f'ps_maxmin_ratio_{col}'] = (ps_max / (ps_min.abs() + 1e-8)).values
            abs_dev = grp.transform(lambda g: (g - g.mean()).abs())
            sq_dev = grp.transform(lambda g: (g - g.mean()) ** 2)
            outliers = grp.transform(lambda g: (g - g.mean()).abs() > 2 * max(g.std(ddof=0), 1e-8)).astype(float)
            for fn, fv in [('absdev',abs_dev),('sqdev',sq_dev),('outliers',outliers)]:
                df_src[f'ps_{fn}_{col}'] = fv.values
    return train_df, test_df

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
    log.info("V401 — V401 — Three-Stage Stacking + 50% Bagging")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    log.info("Generating base features...")
    train_df, test_df = generate_base_features(train_df, test_df)

    all_feat_cols = get_feature_cols(train_df)
    log.info(f"\nFeature counts: Total={len(all_feat_cols)}")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)

    # ========== STAGE 1: V329-style students ==========
    log.info("\nSTAGE 1: Training V329-style students...")
    stage1_oofs = {}

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(all_feat_cols, t)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        ranked = rank_features(train_df, feat_cols_clean, t)
        oofs = []
        for si in range(N_SEEDS):
            s = SEED + si * 7
            rng = np.random.RandomState(s)
            n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), V53_SWEEP[t]['n_feat'])
            bag = rng.choice(ranked, size=min(n_bag, len(ranked)), replace=False)
            seed_oof = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[list(bag)].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[list(bag)].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': s,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            oofs.append(np.clip(seed_oof, 0.001, 0.999))
        stage1_oofs[t] = oofs
        avg_soof = np.mean([log_loss(y, p) for p in oofs])
        log.info(f"  {t}: avg student OOF = {avg_soof:.5f}")

    # ========== STAGE 2: Students on Stage 1 predictions ==========
    log.info("\nSTAGE 2: Training students on Stage 1 predictions...")
    stage2_oofs = {}

    for t in TARGETS:
        log.info(f"\n  Target: {t}")
        y = train_df[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        # Build augmented features from Stage 1
        oof_matrix = np.column_stack(stage1_oofs[t])
        pred_mean = np.mean(oof_matrix, axis=1)
        pred_std = np.std(oof_matrix, axis=1)

        other_targets = [ot for ot in TARGETS if ot != t]
        other_pred = np.column_stack([np.mean(np.column_stack(stage1_oofs[ot]), axis=1) for ot in other_targets])

        feat_cols_clean = remove_leak(all_feat_cols, t)
        aug_df = train_df[feat_cols_clean].copy()
        aug_df['s1_' + t + '_pred_mean'] = pred_mean
        aug_df['s1_' + t + '_pred_std'] = pred_std
        for i, ot in enumerate(other_targets):
            aug_df[f's1_{ot}_pred_mean'] = other_pred[:, i]

        aug_names = list(aug_df.columns)

        # Rank and select
        X_aug = aug_df.values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn = [sanitize_col(c) for c in aug_names]
        ds = lgb.Dataset(X_aug, label=y, feature_name=sn)
        m_rank = lgb.train(params_rank, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked_aug = sorted(zip(aug_names, imp), key=lambda x: -x[1])
        ranked_names = [r[0] for r in ranked_aug]

        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked_names[:n_feat]
        log.info(f"  Selected: {n_feat} from {len(aug_names)}")
        log.info(f"  Top 5: {ranked_names[:5]}")

        # Train Stage 2 students
        oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(sel_cols) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(sel_cols, size=min(n_bag, len(sel_cols)), replace=False)
            seed_oof = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(aug_df, y, group)):
                X_tr = aug_df[bag].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = aug_df[bag].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            oofs.append(np.clip(seed_oof, 0.001, 0.999))

        stage2_oofs[t] = oofs
        avg_soof = np.mean([log_loss(y, p) for p in oofs])
        log.info(f"  {t}: avg student OOF = {avg_soof:.5f}")

    # ========== STAGE 3: Students on Stage 1 + Stage 2 predictions ==========
    log.info("\nSTAGE 3: Training students on Stage 1 + Stage 2 predictions...")
    stage3_oofs = {}

    for t in TARGETS:
        log.info(f"\n  Target: {t}")
        y = train_df[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        # Build doubly-augmented features
        oof_matrix_s1 = np.column_stack(stage1_oofs[t])
        pred_mean_s1 = np.mean(oof_matrix_s1, axis=1)
        pred_std_s1 = np.std(oof_matrix_s1, axis=1)

        oof_matrix_s2 = np.column_stack(stage2_oofs[t])
        pred_mean_s2 = np.mean(oof_matrix_s2, axis=1)
        pred_std_s2 = np.std(oof_matrix_s2, axis=1)

        other_targets = [ot for ot in TARGETS if ot != t]
        other_pred = np.column_stack([np.mean(np.column_stack(stage1_oofs[ot]), axis=1) for ot in other_targets])

        feat_cols_clean = remove_leak(all_feat_cols, t)
        aug_df = train_df[feat_cols_clean].copy()
        aug_df['s1_' + t + '_pred_mean'] = pred_mean_s1
        aug_df['s1_' + t + '_pred_std'] = pred_std_s1
        aug_df['s2_' + t + '_pred_mean'] = pred_mean_s2
        aug_df['s2_' + t + '_pred_std'] = pred_std_s2
        for i, ot in enumerate(other_targets):
            aug_df[f's1_{ot}_pred_mean'] = other_pred[:, i]

        aug_names = list(aug_df.columns)

        # Rank and select
        X_aug = aug_df.values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn = [sanitize_col(c) for c in aug_names]
        ds = lgb.Dataset(X_aug, label=y, feature_name=sn)
        m_rank = lgb.train(params_rank, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked_aug = sorted(zip(aug_names, imp), key=lambda x: -x[1])
        ranked_names = [r[0] for r in ranked_aug]

        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked_names[:n_feat]
        log.info(f"  Selected: {n_feat} from {len(aug_names)}")
        log.info(f"  Top 5: {ranked_names[:5]}")

        # Train Stage 3 students
        oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(sel_cols) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(sel_cols, size=min(n_bag, len(sel_cols)), replace=False)
            seed_oof = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(aug_df, y, group)):
                X_tr = aug_df[bag].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = aug_df[bag].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            oofs.append(np.clip(seed_oof, 0.001, 0.999))

        stage3_oofs[t] = oofs
        avg_soof = np.mean([log_loss(y, p) for p in oofs])
        log.info(f"  {t}: avg student OOF = {avg_soof:.5f}")

    # ========== Meta-learner on Stage 3 ==========
    log.info("\nMeta-learner on Stage 3 students...")
    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(np.column_stack(stage3_oofs[t]), y)
        train_pred = meta.predict_proba(np.column_stack(stage3_oofs[t]))[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in stage3_oofs[t]])
        meta_weights_info[t] = meta.coef_[0]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f})")

    avg_oof = np.mean(list(target_oofs.values()))
    avg_student = np.mean(list(student_avg_oofs.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V401 RESULTS — V401 — Three-Stage Stacking + 50% Bagging")
    log.info(f"{'='*70}")
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student: {avg_student:.5f}")
    log.info(f"  V329_cross_ps: 0.54365 | V329_cross_ps student: 0.64698")
    log.info(f"  Dv329: {avg_oof - 0.54365:+.5f} | Dstudent: {avg_student - 0.64698:+.5f}")

    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build submission
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        feat_cols_clean = remove_leak(all_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)

        # Stage 1 test predictions
        test_oofs_t = np.zeros((len(test_df), N_SEEDS))
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), V53_SWEEP[t]['n_feat'])
            bag = rng.choice(ranked, size=min(n_bag, len(ranked)), replace=False)
            seed_test = np.zeros(len(test_df))
            for fold, (tr_idx, va_idx) in enumerate(
                GroupKFold(n_splits=N_FOLDS).split(train_df, y, group)):
                X_tr = train_df[list(bag)].iloc[tr_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_test += m.predict(test_df[list(bag)].fillna(0).values.astype(np.float64))
            seed_test /= N_FOLDS
            test_oofs_t[:, si] = seed_test

        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack(stage3_oofs[t]), y)
        sub[t] = meta_t.predict_proba(test_oofs_t)[:, 1]

    sub_path = SUBMIT / f"submission_v400_three_stage_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V401',
        'name': 'V401 — Three-Stage Stacking + 50% Bagging',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(all_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'n_stages': 3,
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'Three-stage stacking: S1(base)→S2(S1 preds)→S3(S1+S2 preds)→LR meta',
    }

    meta_path = EXPERIMENTS / f'v400_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
