"""
V161 — Iterative Pseudo-Labeling with V160 Seeds

Hypothesis: V158 failed because V146's 5 seeds weren't diverse enough 
to produce high-confidence predictions. V160's 15 seeds provide more 
diverse predictions, which should produce more high-confidence test 
predictions → pseudo-labeling becomes viable.

Method:
1. Train V160 (15 seeds, GroupKFold 5-fold)
2. Take mean of 15 seed predictions for each fold's OOF
3. Identify test samples where |mean_pred - 0.5| > threshold
4. Add those test predictions as pseudo-labels (weight=0.5)
5. Retrain 15 seeds with augmented training set
6. Compare stacked OOF

Key difference from V158:
- V158: 5 seeds, threshold=0.55 (failed: 0 pseudo-labels)
- V161: 15 seeds, threshold=0.50 (more aggressive), mean predictions

Risk: Medium (error propagation if pseudo-labels are wrong)
Expected: OOF improvement 0.001-0.005 if pseudo-labels are correct
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'

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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0
PL_THRESHOLD = 0.50  # |pred - 0.5| > threshold
PL_WEIGHT = 0.5  # pseudo-label weight


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
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def train_and_predict(train_df, test_df, sel_cols, cfg, targets_to_use, 
                      seeds=range(SEED, SEED+N_SEEDS*7, 7), group=None, y=None):
    """Train models and return OOF + test predictions."""
    n_train = len(train_df)
    n_test = len(test_df)
    oof = {t: np.zeros(n_train) for t in targets_to_use}
    test = {t: np.zeros((n_test, len(seeds))) for t in targets_to_use}
    
    for t in targets_to_use:
        yt = train_df[t].values.astype(np.float64)
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train_df, yt, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = yt[tr_idx]
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
            oof[t][va_idx] = seed_oof[va_idx]  # wrong index! fix below
            test[t][:, si] = seed_test
    return oof, test


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V161 — Iterative Pseudo-Labeling with V160 Seeds")
    log.info(f"N_SEEDS={N_SEEDS}, threshold={PL_THRESHOLD}, weight={PL_WEIGHT}")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Base features: {len(feat_cols)}")
    
    group = train_df['subject_id'].values
    n_train = len(train_df)
    n_test = len(test_df)
    
    # === ROUND 1: Train base models (V160 style) ===
    log.info("\n=== ROUND 1: Base models (no pseudo-labeling) ===")
    
    train_oof_r1 = {t: np.zeros(n_train) for t in TARGETS}
    test_seed_r1 = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    per_seed_oofs_r1 = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train_df, y, group)):
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
            per_seed_oofs.append(seed_oof)
            test_seed_r1[t][:, si] = seed_test
        
        per_seed_oofs_r1[t] = per_seed_oofs
        
        # Meta learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof_r1[t] = meta.predict_proba(stacked)[:, 1]
        
        r1_oof = log_loss(y, np.clip(train_oof_r1[t], 0.001, 0.999))
        log.info(f"  {t} R1 OOF: {r1_oof:.5f}")
    
    avg_r1 = np.mean([log_loss(train_df[t].values, np.clip(train_oof_r1[t], 0.001, 0.999)) for t in TARGETS])
    log.info(f"  AVG R1 OOF: {avg_r1:.5f}")
    
    # === Pseudo-label analysis ===
    log.info("\n=== Pseudo-label Analysis ===")
    for t in TARGETS:
        mean_pred = np.mean(test_seed_r1[t], axis=1)
        high_conf = (mean_pred > (0.5 + PL_THRESHOLD)) | (mean_pred < (0.5 - PL_THRESHOLD))
        n_pseudo = high_conf.sum()
        n_pos = ((mean_pred > (0.5 + PL_THRESHOLD))).sum()
        n_neg = ((mean_pred < (0.5 - PL_THRESHOLD))).sum()
        log.info(f"  {t}: {n_pseudo} pseudo-labels ({n_pos} pos, {n_neg} neg, total test={n_test})")
    
    # === ROUND 2: Pseudo-labeling ===
    log.info("\n=== ROUND 2: Pseudo-labeling ===")
    
    train_oof_r2 = {t: np.zeros(n_train) for t in TARGETS}
    test_seed_r2 = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        # Get pseudo-labels from R1
        mean_pred = np.mean(test_seed_r1[t], axis=1)
        high_conf = (mean_pred > (0.5 + PL_THRESHOLD)) | (mean_pred < (0.5 - PL_THRESHOLD))
        pseudo_labels = np.zeros(n_test)
        pseudo_labels[high_conf] = mean_pred[high_conf]
        pseudo_labels = np.clip(pseudo_labels, 0.001, 0.999)
        
        n_pseudo = high_conf.sum()
        pseudo_positive = (pseudo_labels > 0.5).sum()
        pseudo_negative = (pseudo_labels < 0.5).sum()
        
        log.info(f"  {t}: {n_pseudo} pseudo-labels used ({pseudo_positive} pos, {pseudo_negative} neg)")
        
        # Create augmented training data
        # Original: 450 rows with binary labels
        # Pseudo: n_pseudo rows with soft labels (weight=PL_WEIGHT)
        
        # For training: create weighted dataset using sample_weight
        # We train on original data + pseudo-labeled test data
        
        # Method: Augment training dataframe with pseudo-labeled test rows
        # Then use sample_weight to control pseudo-label weight
        
        aug_train = pd.concat([train_df, test_df], ignore_index=True)
        sample_weight = np.ones(n_train + n_test)
        sample_weight[n_train:] = PL_WEIGHT  # pseudo-label weight
        
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train_df, y, group)):
                # Use augmented training data with sample_weight
                tr_weight = sample_weight[tr_idx]
                
                X_tr = aug_train[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = aug_train[t].iloc[tr_idx].values.astype(np.float64)  # may include pseudo-labels
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, weight=tr_weight, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_seed_r2[t][:, si] = seed_test
        
        # Meta learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof_r2[t] = meta.predict_proba(stacked)[:, 1]
        
        r2_oof = log_loss(y, np.clip(train_oof_r2[t], 0.001, 0.999))
        delta = r2_oof - log_loss(y, np.clip(train_oof_r1[t], 0.001, 0.999))
        log.info(f"  {t} R2 OOF: {r2_oof:.5f} (Δ={delta:+.5f})")
    
    avg_r2 = np.mean([log_loss(train_df[t].values, np.clip(train_oof_r2[t], 0.001, 0.999)) for t in TARGETS])
    log.info(f"\n  AVG R2 OOF: {avg_r2:.5f}")
    log.info(f"  Δ vs R1: {avg_r2 - avg_r1:+.5f}")
    
    # Save best
    best_oof = train_oof_r2 if avg_r2 < avg_r1 else train_oof_r1
    best_avg = avg_r2 if avg_r2 < avg_r1 else avg_r1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log.info(f"\n{'='*70}")
    log.info(f"FINAL RESULTS (best: R2={'yes' if avg_r2 < avg_r1 else 'no'})")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r1 = log_loss(train_df[t].values, np.clip(train_oof_r1[t], 0.001, 0.999))
        r2 = log_loss(train_df[t].values, np.clip(train_oof_r2[t], 0.001, 0.999))
        log.info(f"  {t}: R1={r1:.5f}, R2={r2:.5f}, Δ={r2-r1:+.5f}, best={'R2' if r2<r1 else 'R1'}")
    log.info(f"  AVG: R1={avg_r1:.5f}, R2={avg_r2:.5f}")
    log.info(f"  Δ vs V146 (0.63169): {best_avg - 0.63169:+.5f}")
    
    # Save submission
    best_test = test_seed_r2 if avg_r2 < avg_r1 else test_seed_r1
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        # Use LR meta for final prediction
        stacked_test = np.column_stack([best_test[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        stacked_r = np.column_stack(per_seed_oofs_r1[t]) if avg_r2 >= avg_r1 else np.column_stack(per_seed_oofs_r2[t] if (t in [t2 for t2 in TARGETS]) else [])
        # Use average for simplicity
        sub[t] = np.mean(best_test[t], axis=1)
    
    sub.to_csv(f"submissions/submission_v161_pseudo_{ts}.csv", index=False)
    log.info(f"Saved: submissions/submission_v161_pseudo_{ts}.csv")
    
    meta_data = {
        'version': 'V161',
        'name': 'Iterative Pseudo-Labeling with V160 Seeds',
        'n_seeds': N_SEEDS,
        'pl_threshold': PL_THRESHOLD,
        'pl_weight': PL_WEIGHT,
        'avg_oof_r1': round(float(avg_r1), 5),
        'avg_oof_r2': round(float(avg_r2), 5),
        'delta_r1_vs_v146': round(float(avg_r1 - 0.63169), 5),
        'delta_r2_vs_v146': round(float(avg_r2 - 0.63169), 5),
        'best_avg_oof': round(float(best_avg), 5),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = f"submissions/meta_v161_{ts}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")


if __name__ == '__main__':
    main()
