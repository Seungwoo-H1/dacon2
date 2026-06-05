"""
V344 — OOF Feature + Hybrid Z-Score

Combination of V339 (OOF feature augmentation) and V343 (hybrid z-score).

V339: OOF feature → AVG OOF 0.61244 (-0.010 vs V308)
V343: Hybrid z-score → AVG OOF 0.61333 (-0.009 vs V308)
       Key: Q1 -0.040, S1 -0.012, S2 -0.026

Hypothesis: OOF features capture model calibration signal,
hybrid z-score captures cleaner per-subject deviation signal.
Together they should beat either alone.

Architecture: Same V308 (15 seeds, GroupKFold 5, LR meta C=10)
Features: base + hybrid_zscore + oof_feature per target
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
    """Rank features by LGBM gain importance."""
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


def generate_oof_features_per_target(train_df, test_df, TARGETS, SEED, N_FOLDS, N_SEEDS,
                                      CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS):
    """Generate OOF predictions per target as additional features.
    Uses hybrid z-score features (V343) as base.
    """
    log.info("Generating OOF features with hybrid z-score...")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # First, compute hybrid z-scores
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and not c.endswith('_hybrid')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    subject_ids = train_df['subject_id'].values
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    for col in base_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        
        # Subject means from training
        subject_means = {}
        for subj_id in np.unique(subject_ids):
            mask = subject_ids == subj_id
            vals = train_vals[mask]
            subject_means[subj_id] = np.mean(vals) if len(vals) > 0 else 0.0
        
        global_std = np.std(train_vals, ddof=0)
        if global_std < 1e-8:
            global_std = 1e-8
        
        for i in range(len(train_vals)):
            train_work[f'{col}_hybrid'] = (train_work.loc[train_work.index[i], col] - subject_means[subject_ids[i]]) / global_std
        
        test_subjects = test_df['subject_id'].values
        for i in range(len(test_vals)):
            subj = test_subjects[i]
            if subj in subject_means:
                test_work.loc[test_work.index[i], f'{col}_hybrid'] = (test_vals[i] - subject_means[subj]) / global_std
            else:
                global_mean = np.mean(train_vals)
                test_work.loc[test_work.index[i], f'{col}_hybrid'] = (test_vals[i] - global_mean) / global_std
    
    # Now generate OOF predictions using hybrid z-score features
    oof_features_train = {}
    oof_features_test = {}
    
    for target in TARGETS:
        # Features for this target (hybrid + base)
        feat_cols_clean = remove_leak(
            [c for c in train_work.columns if c not in META_COLS | set(TARGETS) 
             and np.issubdtype(train_work[c].dtype, np.number) and not c.startswith('oof_')],
            target
        )
        
        # Use top features from rank
        ranked = rank_features(train_work, feat_cols_clean, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in test_work.columns]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {target}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        y = train_work[target].values.astype(np.float64)
        group = train_work['subject_id'].values
        n_train = len(train_work)
        n_test = len(test_work)
        
        oof_preds = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        for seed in seeds:
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_work, y, group)):
                X_tr = train_work[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_work[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_work[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            oof_preds += seed_oof
            test_preds += seed_test
        
        oof_preds /= N_SEEDS
        test_preds /= N_SEEDS
        
        oof_features_train[target] = oof_preds
        oof_features_test[target] = np.clip(test_preds, 0.01, 0.99)
        
        log.info(f"  {target} OOF: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")
    
    return oof_features_train, oof_features_test, train_work, test_work


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V344 — OOF Feature + Hybrid Z-Score")
    log.info("Hypothesis: OOF features + hybrid z-score beats either alone")
    log.info("V339: OOF + global zscore → OOF 0.61244")
    log.info("V343: hybrid zscore → OOF 0.61333")
    log.info("V344: OOF + hybrid zscore → ?")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate OOF features with hybrid z-scores
    oof_train, oof_test, train_work, test_work = generate_oof_features_per_target(
        train_df, test_df, TARGETS, SEED, N_FOLDS, N_SEEDS,
        CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS
    )
    
    # Add OOF features to data
    for t in TARGETS:
        train_work[f'oof_{t}'] = oof_train[t]
        test_work[f'oof_{t}'] = oof_test[t]
    
    # Get final feature columns
    train_feat_cols = get_feature_cols(train_work)
    test_feat_cols = get_feature_cols(test_work)
    
    log.info(f"\nTrain: {len(train_feat_cols)} features (base+hybrid+oof)")
    log.info(f"Test: {len(test_feat_cols)} features")
    
    # Feature ranking with OOF features
    target_configs = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        # Include OOF features in ranking
        feat_cols_no_oof = [c for c in feat_cols_clean if not c.startswith('oof_')]
        ranked = rank_features(train_work, feat_cols_no_oof, t)
        
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        
        # Add OOF prediction of THIS target as extra feature
        if f'oof_{t}' not in sel_cols:
            sel_cols.append(f'oof_{t}')
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        target_configs[t] = {
            'cfg': CFGS[cfg_name],
            'features': sel_cols,
            'features_test': sel_cols_test,
        }
        log.info(f"  {t}: cfg={cfg_name}, {len(sel_cols_test)} features (+oof_{t})")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_work)
    n_test = len(test_work)
    group = train_work['subject_id'].values
    
    all_oofs = {}
    all_test_preds = {}
    all_student_oofs = {}
    
    for t in TARGETS:
        tc = target_configs[t]
        cfg = tc['cfg']
        feats = tc['features']
        feats_test = tc['features_test']
        
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} | features={len(feats)} | seeds={N_SEEDS}")
        
        y = train_work[t].values.astype(np.float64)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        train_oofs = np.zeros((n_train, N_SEEDS))
        test_preds = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_work, y, group)):
                X_tr = train_work[feats].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_work[feats].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in feats]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_work[feats_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            train_oofs[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
        
        stacked = np.column_stack(list(train_oofs.T))
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        stacked_test = np.column_stack([test_preds[:, si] for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"  {t}: student={student_ll:.5f}, meta={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    v343_avg = 0.61333
    
    log.info(f"\n{'='*70}")
    log.info(f"V344 RESULTS (OOF + Hybrid Z-Score)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Student OOF':>12} {'Meta OOF':>12} {'Gap':>8} {'ΔV308':>8} {'ΔV339':>8}")
    log.info(f"{'-'*58}")
    for t in TARGETS:
        log.info(f"{t:<6} {all_student_oofs[t]:>12.5f} {all_oofs[t]:>12.5f} {all_oofs[t]-all_student_oofs[t]:>+8.5f} {all_oofs[t]-v308_avg:>+8.5f} {all_oofs[t]-v339_avg:>+8.5f}")
    log.info(f"{'='*58}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF:    {v308_avg:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  V339 AVG OOF:    {v339_avg:.5f}")
    log.info(f"  Δ vs V339:       {avg_oof - v339_avg:+.5f}")
    log.info(f"  V343 AVG OOF:    {v343_avg:.5f}")
    log.info(f"  Δ vs V343:       {avg_oof - v343_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_work['subject_id'].values
    sub['sleep_date'] = test_work['sleep_date'].values
    sub['lifelog_date'] = test_work['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v344_oof_hybrid_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V344',
        'name': 'OOF Feature + Hybrid Z-Score',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'delta_vs_v343': round(float(avg_oof - v343_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v344_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
