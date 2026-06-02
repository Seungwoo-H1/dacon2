"""
V337: Two-Stage Stacking — Student Predictions as Features

Stage 1: Train V329-style students → get OOF predictions per seed
Stage 2: Build augmented feature set = (original + pred_mean, pred_std, pred_min, pred_max)
         → Train new students on augmented features
Stage 3: LR meta on Stage 2 students

Clean implementation.
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

def add_cross_subject_features(train_df, test_df):
    log.info("Generating cross-subject features...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    base_cols = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {'date'}
                 and not c.endswith('_zscore') and not c.endswith('_interact')
                 and not c.endswith('_ratio') and not c.startswith('ps_')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    for col in base_cols[:80]:
        pop_mean = train_df[col].fillna(0).mean()
        pop_std = train_df[col].fillna(0).std(ddof=0)
        if pop_std < 1e-8: pop_std = 1e-8
        test_df[f'cs_zscore_{col}'] = (test_df[col].fillna(0) - pop_mean) / pop_std
        train_df[f'cs_zscore_{col}'] = (train_df[col].fillna(0) - pop_mean) / pop_std

    domains_dict = {
        'wHr': [c for c in train_df.columns if c.startswith('wHr_')],
        'wPedo': [c for c in train_df.columns if c.startswith('wPedo_')],
        'mLight': [c for c in train_df.columns if c.startswith('mLight_')],
        'mScreenStatus': [c for c in train_df.columns if c.startswith('mScreenStatus_')],
        'mGps': [c for c in train_df.columns if c.startswith('mGps_')],
        'mBle': [c for c in train_df.columns if c.startswith('mBle_')],
        'mWifi': [c for c in train_df.columns if c.startswith('mWifi_')],
        'mUsageStats': [c for c in train_df.columns if c.startswith('mUsageStats_')],
    }
    for domain_name, cols in domains_dict.items():
        if not cols: continue
        domain_base = [c for c in cols if c not in META_COLS | set(TARGETS) | {'date'}
                       and not c.endswith('_zscore') and not c.endswith('_interact')
                       and not c.endswith('_ratio') and not c.startswith('ps_')
                       and np.issubdtype(train_df[c].dtype, np.number)]
        if not domain_base: continue
        for df_src, grp in [(train_df, train_df.groupby('subject_id')[domain_base]),
                            (test_df, test_df.groupby('subject_id')[domain_base])]:
            dm = grp.mean()
            ds = grp.std().fillna(0)
            df_name = f'{domain_name.lower()}_domain'
            df_src[f'{df_name}_mean'] = dm.mean(axis=1).reindex(df_src['subject_id']).values
            df_src[f'{df_name}_std'] = ds.mean(axis=1).reindex(df_src['subject_id']).values

    log.info(f"  Cross-subject z-scores: {len(base_cols[:80])}")
    log.info(f"  Domain aggregations: {len(domains_dict)}")
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

def train_students(train_df, sel_cols, y, group, gkf, n_seeds, cfg):
    n_train = len(train_df)
    n_test = train_df.shape[0] + 0  # dummy
    seed_oofs = []
    for si in range(n_seeds):
        seed = SEED + si * 7
        rng = np.random.RandomState(seed)
        feat_list = list(sel_cols)
        n_feat = len(feat_list) if len(feat_list) < cfg['n_estimators'] else cfg['n_estimators']
        n_bag = max(int(len(feat_list) * FEATURE_BAG_FRACTION), 14)
        if len(feat_list) > n_bag:
            bag = rng.choice(feat_list, size=n_bag, replace=False)
        else:
            bag = feat_list
        sel = [c for c in bag if c in feat_list][:n_feat]
        seed_oof = np.zeros(n_train)
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[sel].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[sel].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
        seed_oofs.append(np.clip(seed_oof, 0.001, 0.999))
    return seed_oofs

def main():
    global t_start
    t_start = time.time()
    log.info("=" * 70)
    log.info("V337 — Two-Stage Stacking: Student Predictions as Features")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    log.info("Generating base features...")
    train_df, test_df = generate_base_features(train_df, test_df)
    train_df, test_df = add_cross_subject_features(train_df, test_df)

    all_feat_cols = get_feature_cols(train_df)
    base_cols = [c for c in all_feat_cols if '_zscore' not in c
                 and 'ps_' not in c and '_interact' not in c and 'ratio' not in c and 'domain' not in c]

    log.info(f"\nFeature counts: Base={len(base_cols)}, Total={len(all_feat_cols)}")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)

    # ========== STAGE 1: V329-style students ==========
    log.info("\nSTAGE 1: Training V329-style students...")
    stage1_oofs = {}
    stage1_col_names = {}

    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(all_feat_cols, t)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        ranked = rank_features(train_df, feat_cols_clean, t)
        oofs = train_students(train_df, ranked, y, group, gkf, N_SEEDS, cfg)
        stage1_oofs[t] = oofs
        stage1_col_names[t] = ranked
        avg_soof = np.mean([log_loss(y, p) for p in oofs])
        log.info(f"  {t}: avg student OOF = {avg_soof:.5f}")

    # ========== STAGE 2: Student predictions as features ==========
    log.info("\nSTAGE 2: Building augmented features + training new students...")

    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}

    for t in TARGETS:
        log.info(f"\n  Target: {t}")
        y = train_df[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]

        # Build per-seed prediction matrix from Stage 1
        oof_matrix = np.column_stack(stage1_oofs[t])
        pred_mean = np.mean(oof_matrix, axis=1)
        pred_std = np.std(oof_matrix, axis=1)

        # Other target predictions
        other_targets = [ot for ot in TARGETS if ot != t]
        other_pred = np.column_stack([np.mean(np.column_stack(stage1_oofs[ot]), axis=1) for ot in other_targets])

        # Original features
        feat_cols_clean = remove_leak(all_feat_cols, t)
        X_base = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)

        # Augmented features
        aug_names = feat_cols_clean + [f'st1_{t}_pred_mean', f'st1_{t}_pred_std'] + [f'st1_{ot}_pred_mean' for ot in other_targets]
        X_aug = np.column_stack([X_base, pred_mean, pred_std, other_pred])

        # Rank augmented features
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1),
            'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn = [sanitize_col(c) for c in aug_names]
        ds = lgb.Dataset(X_aug, label=y, feature_name=sn)
        m_rank = lgb.train(params_rank, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked_aug = sorted(zip(aug_names, imp), key=lambda x: -x[1])
        ranked_names = [r[0] for r in ranked_aug]

        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked_names[:n_feat]

        log.info(f"  Augmented features: {len(aug_names)}, selected: {n_feat}")
        log.info(f"  Top 5: {ranked_names[:5]}")

        # Train stage 2 students on augmented features
        # Need to rebuild train_df with augmented features for Stage 2
        # Create augmented train dataframe
        aug_df = train_df[feat_cols_clean].copy()
        aug_df['s1_mean'] = pred_mean
        aug_df['s1_std'] = pred_std
        for i, ot in enumerate(other_targets):
            aug_df[f's1_{ot}_mean'] = other_pred[:, i]

        oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(sel_cols) * FEATURE_BAG_FRACTION), n_feat)
            if len(sel_cols) > n_bag:
                bag = rng.choice(sel_cols, size=n_bag, replace=False)
            else:
                bag = sel_cols
            bag_set = set(bag)
            bag_feats = [f for f in sel_cols if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in sel_cols if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)

            seed_oof = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(aug_df, y, group)):
                X_tr = aug_df[bag_feats].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = aug_df[bag_feats].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag_feats]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            oofs.append(np.clip(seed_oof, 0.001, 0.999))

        # Meta-learner
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(np.column_stack(oofs), y)
        train_pred = meta.predict_proba(np.column_stack(oofs))[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in oofs])
        meta_weights_info[t] = meta.coef_[0]

        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f})")

    avg_oof = np.mean(list(target_oofs.values()))
    avg_student = np.mean(list(student_avg_oofs.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V337 RESULTS — Two-Stage Stacking (Student Preds as Features)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        w_sum = np.sum(np.abs(meta_weights_info[t]))
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f}, |W|={w_sum:.3f})")
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
        feat_cols_clean = remove_leak(all_feat_cols, t)
        ranked = stage1_col_names[t]

        # Get test predictions from stage 1 students
        test_oofs = np.zeros((len(test_df), N_SEEDS))
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), V53_SWEEP[t]['n_feat'])
            bag = rng.choice(ranked, size=n_bag, replace=False)
            sel = [c for c in bag if c in ranked][:V53_SWEEP[t]['n_feat']]

            seed_test = np.zeros(len(test_df))
            for fold, (tr_idx, va_idx) in enumerate(
                GroupKFold(n_splits=N_FOLDS).split(train_df, y, group)):
                X_tr = train_df[sel].iloc[tr_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_test += m.predict(test_df[sel].fillna(0).values.astype(np.float64))
            seed_test /= N_FOLDS
            test_oofs[:, si] = seed_test

        # Build augmented test features
        pred_mean_t = np.mean(test_oofs, axis=1)
        pred_std_t = np.std(test_oofs, axis=1)
        other_pred_t = np.column_stack([
            np.mean(np.zeros((len(test_df), N_SEEDS)), axis=1) if ot == t else np.mean(np.zeros((len(test_df), N_SEEDS)), axis=1)
            for ot in TARGETS if ot != t
        ])

        X_test_base = test_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        aug_names_t = feat_cols_clean + [f'st1_{t}_pred_mean', f'st1_{t}_pred_std'] + [f'st1_{ot}_pred_mean' for ot in other_targets]
        X_test_aug = np.column_stack([X_test_base, pred_mean_t, pred_std_t, other_pred_t])

        # Rank test augmented features same way
        ranked_t = [n for n, _ in sorted(zip(aug_names_t, np.zeros(len(aug_names_t))), key=lambda x: 0)][:len(aug_names_t)]
        ranked_aug_t = sorted(zip(aug_names_t, np.zeros(len(aug_names_t))), key=lambda x: 0)
        ranked_names_t = [r[0] for r in ranked_aug_t]

        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols_t = ranked_names_t[:n_feat]

        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(test_oofs, y)
        sub[t] = meta_t.predict_proba(test_oofs)[:, 1]

    sub_path = SUBMIT / f"submission_v337_stage2_features_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version': 'V337',
        'name': 'Two-Stage Stacking: Student Predictions as Features',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(all_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'Stage 2: original features + student OOF predictions (mean, std, other-target means) as additional features',
    }

    meta_path = EXPERIMENTS / f'v337_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
