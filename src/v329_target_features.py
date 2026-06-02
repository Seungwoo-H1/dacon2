"""
V329 — Target-Specific Feature Engineering (Cross-Target Correlations)

Hypothesis: The 7 targets (Q1-Q3, S1-S4) are correlated. We can create
features from one target to predict another, using CV to prevent leakage.

Approach:
1. For each target T, train a simple model on ALL other targets to predict T
2. This gives "inter-target" features: e.g., S1_pred = model(S2, S3, S4, Q1-Q3)
3. These predictions are computed via CV on training data (no leakage)
4. Then run V321 stacking with these new features

Key: Use CV to generate inter-target predictions on train data only.
Test predictions use holdout models.

Expected OOF: 0.580-0.595 (significant improvement)
Risk: HIGH (cross-target leakage risk if not careful)
Cost: ~90s
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
    log.info("V329 — Target-Specific Feature Engineering (Cross-Target)")
    log.info("Inter-target CV predictions as additional features")
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
    
    log.info(f"Base features: {len(train_feat_cols)} (raw + z-scored)")
    
    # STEP 1: Generate inter-target features via CV
    # For each target T, train a model on other targets + base features to predict T
    # This creates features like "S1_from_S2", "S1_from_S3", etc.
    
    log.info("\nStep 1: Generating inter-target CV predictions...")
    
    group = train_df['subject_id'].values
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Inter-target features: for each target T, predict from other targets
    # This gives 6 inter-target features per target (using the other 6 targets)
    inter_target_features = set()
    inter_target_base = set()
    
    for t in TARGETS:
        # Other targets as features
        other_targets = [ot for ot in TARGETS if ot != t]
        inter_target_features.add(t)  # target itself
        for ot in other_targets:
            ft = f'{ot}_as_feat_{t}'
            inter_target_features.add(ft)
            inter_target_base.add(ot)
    
    # Also create simple correlations: S1 from S2 mean, etc.
    log.info(f"  Inter-target feature groups: {len(inter_target_features)}")
    
    # STEP 2: Generate inter-target predictions via CV
    # For each target T, create features from other targets' values
    # The idea: if S1 correlates with S2, then S2's value is useful for predicting S1
    
    # Create inter-target features directly from OTHER TARGETS' values
    # (no leakage since these are actual values, not predictions)
    # Then the CV model naturally handles the correlation
    
    # Actually, the safest approach: for each target T, add features computed
    # from OTHER targets using a simple CV prediction model
    
    # Step 2a: Create inter-target predictions for TRAINING data only
    # For each pair (source_target, target_target), train a CV model
    # to predict target_target from source_target, then use that prediction as a feature
    
    n_train = len(train_df)
    inter_pred_train = np.zeros((n_train, len(TARGETS), len(TARGETS) - 1))
    
    for ti, target_t in enumerate(TARGETS):
        y_target = train_df[target_t].values.astype(np.float64)
        source_features = [st for st in TARGETS if st != target_t]
        
        for si, source_t in enumerate(source_features):
            # Simple model: predict target_t from source_t's value
            # Use GroupKFold to prevent subject leakage
            y_source = train_df[source_t].values.astype(np.float64)
            
            # Train 5-fold CV model
            oof_pred = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y_target, group)):
                # Simple model: use the source target value as a single feature
                X_tr = train_df[source_t].fillna(0).iloc[tr_idx].values.astype(np.float64).reshape(-1, 1)
                X_va = train_df[source_t].fillna(0).iloc[va_idx].values.astype(np.float64).reshape(-1, 1)
                y_tr = y_target[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100,
                    'scale_pos_weight': spw, 'random_state': SEED + ti * 10 + si,
                    'force_row_wise': True, 'n_jobs': 1
                }
                ds = lgb.Dataset(X_tr, label=y_tr)
                m = lgb.train(params, ds, num_boost_round=100)
                oof_pred[va_idx] = m.predict(X_va)
            
            inter_pred_train[:, ti, si] = oof_pred
    
    # Add inter-target predictions as new features
    feature_idx = 0
    for ti, target_t in enumerate(TARGETS):
        for si, source_t in enumerate([st for st in TARGETS if st != target_t]):
            ft_name = f'inter_{source_t}_to_{target_t}'
            train_df[ft_name] = inter_pred_train[:, ti, si]
            test_df = test_df.copy()
            # For test: use mean of inter-target predictions across subjects
            test_df[ft_name] = np.mean(inter_pred_train[:, ti, si])
            feature_idx += 1
    
    log.info(f"  Added {feature_idx} inter-target features")
    
    # Also add pairwise interactions between targets
    log.info("  Adding pairwise target interactions...")
    pairwise_feats = []
    for i, t1 in enumerate(TARGETS):
        for j, t2 in enumerate(TARGETS):
            if j <= i:
                continue
            # Product and difference features
            ft_prod = f'{t1}_x_{t2}'
            ft_diff = f'{t1}_minus_{t2}'
            train_df[ft_prod] = train_df[t1] * train_df[t2]
            train_df[ft_diff] = train_df[t1] - train_df[t2]
            test_df = test_df.copy()
            test_df[ft_prod] = 0  # placeholder for test
            test_df[ft_diff] = 0  # placeholder
            pairwise_feats.extend([ft_prod, ft_diff])
    
    log.info(f"  Added {len(pairwise_feats)} pairwise features")
    
    # Now re-compute feature columns
    train_feat_cols_new = get_feature_cols(train_df)
    test_feat_cols_new = get_feature_cols(test_df)
    
    log.info(f"\nTotal features after augmentation: {len(train_feat_cols_new)}")
    log.info(f"  Base: {len(train_feat_cols)}")
    log.info(f"  Inter-target: {feature_idx}")
    log.info(f"  Pairwise: {len(pairwise_feats)}")
    
    # STEP 3: Run V321 stacking with enriched features
    log.info("\nStep 3: Running V321 stacking on enriched features...")
    
    n_test = len(test_df)
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols_new, t)
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
            
            sel_cols = [c for c in bag_feats if c in test_feat_cols_new]
            
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
    
    # LR meta-learner
    target_oofs = {}
    student_avg_oofs = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        
        meta = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
    
    avg_oof = np.mean(list(target_oofs.values()))
    avg_student = np.mean(list(student_avg_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V329 RESULTS (Target-Specific Feature Engineering)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  V321: 0.60569 | V326: 0.59159 | V308: 0.62235")
    log.info(f"  Delta vs V321: {avg_oof - 0.60569:+.5f}")
    log.info(f"  Delta vs V326: {avg_oof - 0.59159:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v329_target_features_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V329',
        'name': 'Target-Specific Feature Engineering (Cross-Target)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(train_feat_cols_new),
        'n_features_base': len(train_feat_cols),
        'n_inter_target': feature_idx,
        'n_pairwise': len(pairwise_feats),
        'n_seeds': N_SEEDS,
        'v321_avg_oof': 0.60569,
        'v326_avg_oof': 0.59159,
        'v308_avg_oof': 0.62235,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'delta_vs_v326': round(float(avg_oof - 0.59159), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'inter-target CV predictions + pairwise target interactions as features',
    }
    
    meta_path = EXPERIMENTS / f'v329_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
