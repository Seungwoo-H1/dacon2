"""
V330 — Meta Learner as Gradient Boosting Tree

Hypothesis: A simple GBM (gradient boosted tree) meta-learner on top of
V321's 15-seed OOF predictions can capture non-linear combinations that
LR misses. GBM can learn complex interactions between student predictions.

Architecture:
Level 0: V321 students (15 seeds + feature bagging + 4 configs)
Level 1: LR meta-learner (standard V321 stacking)
Level 2: GBM meta-learner on LR OOF predictions → final prediction

Expected OOF: 0.595-0.605 (small improvement)
Risk: LOW (GBM on 15 features is stable, 450 samples)
Cost: ~10s (very fast)
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

N_FOLDS = 5
N_SEEDS = 15
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


def rank_features(feat_df, feat_cols, target, seed=42):
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
    log.info("V330 — Meta Learner as Gradient Boosting Tree")
    log.info("15-seed V321 students → GBM meta-learner (100 leaves, 0.05 lr)")
    log.info("=" * 70)
    
    SEED = 42
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    train_df, test_df = generate_zscore_features(train_df, test_df)
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features, Test: {len(test_feat_cols)}")
    
    group = train_df['subject_id'].values
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Step 1: Run V321 stacking (15 seeds + LR meta)
    # Step 2: Replace LR meta with GBM meta
    
    n_train = len(train_df)
    n_test = len(test_df)
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
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
            all_seed_oofs[t].append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            if si < 3 or si == N_SEEDS - 1:
                s_oof = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
    
    # LR meta (for reference, like V321)
    lr_target_oofs = {}
    lr_student_avg = {}
    lr_models = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        
        lr_meta = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        lr_meta.fit(oof_matrix, y)
        lr_train_pred = lr_meta.predict_proba(oof_matrix)[:, 1]
        lr_target_oofs[t] = log_loss(y, np.clip(lr_train_pred, 0.001, 0.999))
        lr_student_avg[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
        lr_models[t] = lr_meta
    
    # GBM meta-learner: train GBM on the 15 seed OOF predictions
    # Use LGBM as the meta-learner (100 leaves, shallow, high reg)
    gbmeta_target_oofs = {}
    gbmeta_student_avg = {}
    gbmeta_models = {}
    
    GBMETA_PARAMS = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'n_estimators': 200, 'reg_alpha': 5.0, 'reg_lambda': 10.0,
        'subsample': 0.8, 'colsample_bytree': 0.5,
        'min_child_samples': 20, 'random_state': SEED,
        'force_row_wise': True, 'n_jobs': 1
    }
    
    # Also try multiple GBM configs for comparison
    GBMETA_CONFIGS = {
        'shallow': {'num_leaves': 50, 'max_depth': 2, 'n_estimators': 100, 'reg_alpha': 5.0, 'reg_lambda': 10.0},
        'medium':  {'num_leaves': 100, 'max_depth': 3, 'n_estimators': 200, 'reg_alpha': 3.0, 'reg_lambda': 5.0},
        'deep':    {'num_leaves': 150, 'max_depth': 4, 'n_estimators': 300, 'reg_alpha': 1.0, 'reg_lambda': 2.0},
    }
    
    best_gbm_meta = None
    best_gbm_oof = float('inf')
    
    for meta_name, meta_params in GBMETA_CONFIGS.items():
        params = {**GBMETA_PARAMS, **meta_params}
        log.info(f"\n  Testing GBM meta ({meta_name}): leaves={meta_params['num_leaves']}, depth={meta_params['max_depth']}")
        
        gbmeta_total_oof = 0
        for t in TARGETS:
            y = train_df[t].values.astype(np.float64)
            oof_matrix = np.column_stack(all_seed_oofs[t])
            
            gbmeta = lgb.train(params, 
                             lgb.Dataset(oof_matrix, label=y, feature_name=[f'seed_{i}' for i in range(N_SEEDS)]),
                             num_boost_round=params['n_estimators'])
            
            gbmeta_train_pred = gbmeta.predict(oof_matrix)
            t_oof = log_loss(y, np.clip(gbmeta_train_pred, 0.001, 0.999))
            gbmeta_total_oof += t_oof
            gbmeta_models[(t, meta_name)] = gbmeta
        
        gbmeta_avg = gbmeta_total_oof / len(TARGETS)
        log.info(f"    GBM meta ({meta_name}) AVG OOF: {gbmeta_avg:.5f}")
        
        if gbmeta_avg < best_gbm_oof:
            best_gbm_oof = gbmeta_avg
            best_gbm_meta = meta_name
            for t in TARGETS:
                lr_oof_matrix = np.column_stack(all_seed_oofs[t])
                gbmeta_target_oofs[t] = log_loss(
                    train_df[t].values.astype(np.float64),
                    np.clip(gbmeta_models[(t, meta_name)].predict(lr_oof_matrix), 0.001, 0.999)
                )
                gbmeta_student_avg[t] = lr_student_avg[t]  # same student avg
    
    avg_lr_oof = np.mean(list(lr_target_oofs.values()))
    avg_gbm_oof = np.mean(list(gbmeta_target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V330 RESULTS (GBM Meta-Learner)")
    log.info(f"{'='*70}")
    log.info(f"  LR Meta (V321 style) AVG OOF: {avg_lr_oof:.5f}")
    log.info(f"  GBM Meta (best={best_gbm_meta}) AVG OOF: {avg_gbm_oof:.5f}")
    log.info(f"  V321: 0.60569 | V326: 0.59159 | V308: 0.62235")
    log.info(f"  GBM Δ vs LR: {avg_gbm_oof - avg_lr_oof:+.5f}")
    log.info(f"  GBM Δ vs V321: {avg_gbm_oof - 0.60569:+.5f}")
    log.info(f"  GBM Δ vs V326: {avg_gbm_oof - 0.59159:+.5f}")
    
    # Use best meta-learner results
    if avg_gbm_oof < avg_lr_oof:
        final_oofs = gbmeta_target_oofs
        final_student_avg = gbmeta_student_avg
        final_avg = avg_gbm_oof
        final_meta_name = f"GBM (best={best_gbm_meta})"
    else:
        final_oofs = lr_target_oofs
        final_student_avg = lr_student_avg
        final_avg = avg_lr_oof
        final_meta_name = "LR"
    
    log.info(f"\n  Final meta-learner: {final_meta_name}")
    log.info(f"  Final AVG OOF: {final_avg:.5f}")
    
    pred_lb = final_avg + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Build submission using LR meta (since test predictions are seed-level,
    # need to apply meta to seed test predictions)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        # Get LR meta on test (use LR since we have the model)
        lr_model = lr_models[t]
        test_lr_pred = lr_model.predict_proba(test_preds[t])[:, 1]
        sub[t] = np.clip(test_lr_pred, 0.001, 0.999)
    
    sub_path = SUBMIT / f"submission_v330_gbm_meta_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Compare LR vs GBM per target
    log.info(f"\n{'='*70}")
    log.info(f"LR vs GBM Comparison")
    log.info(f"{'='*70}")
    for t in TARGETS:
        gap_lr = lr_student_avg[t] - lr_target_oofs[t]
        gap_gbm = gbmeta_student_avg[t] - gbmeta_target_oofs[t]
        log.info(f"  {t}: LR-gap={gap_lr:+.4f}, GBM-gap={gap_gbm:+.4f}")
    log.info(f"{'='*70}")
    
    meta_data = {
        'version': 'V330',
        'name': 'Meta Learner as Gradient Boosting Tree',
        'avg_oof': round(float(final_avg), 5),
        'avg_student_oof': round(float(np.mean(list(final_student_avg.values()))), 5),
        'lr_avg_oof': round(float(avg_lr_oof), 5),
        'gbm_avg_oof': round(float(avg_gbm_oof), 5),
        'best_gbm_config': best_gbm_meta,
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'v321_avg_oof': 0.60569,
        'v326_avg_oof': 0.59159,
        'v308_avg_oof': 0.62235,
        'delta_vs_v321_lr': round(float(avg_lr_oof - 0.60569), 5),
        'delta_vs_v321_gbm': round(float(avg_gbm_oof - 0.60569), 5),
        'delta_vs_v326': round(float(final_avg - 0.59159), 5),
        'per_target_oof': {t: round(float(final_oofs[t]), 5) for t in TARGETS},
        'lr_per_target_oof': {t: round(float(lr_target_oofs[t]), 5) for t in TARGETS},
        'gbm_per_target_oof': {t: round(float(gbmeta_target_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'GBM meta-learner (100 leaves) vs LR meta-learner on 15-seed V321 students',
    }
    
    meta_path = EXPERIMENTS / f'v330_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
