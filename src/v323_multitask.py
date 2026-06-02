"""
V323 — Multi-Task Learning (Joint 7-Target Prediction)

Hypothesis: 7 targets share underlying health patterns. Predicting them
jointly allows the model to learn shared representations, improving
generalization on each target.

V323 approach:
1. Single LGBM model with 7 outputs (multi-output regression → threshold → binary)
2. Loss = mean of 7 target log-losses
3. This forces shared feature representations across targets
4. Alternative: train 7 LGBM models jointly with shared tree structure

Key insight: Current approach trains 7 independent models. Multi-task
shares inductive bias across targets. Similar subjects share similar
health patterns across Q and S domains.

Expected OOF: 0.58-0.60 (shared representation helps)
Expected LB: < 0.62 (better generalization)
Risk: Medium (might hurt individual target if targets are very different)
Cost: ~120s (7 outputs per model × 15 seeds)

Architecture: Multi-output LGBM (RegTree) → binary threshold → stacking
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


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V323 — Multi-Task Learning (Joint 7-Target Prediction)")
    log.info("Hypothesis: shared representation across targets → better generalization")
    log.info("V321: OOF=0.60569 (independent per-target models)")
    log.info("V323: 7-target joint model → shared features")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features
    log.info("Generating z-score features...")
    train_feat_cols_all = [c for c in train_df.columns
                           if c not in META_COLS | set(TARGETS)
                           and not c.endswith('_zscore')
                           and np.issubdtype(train_df[c].dtype, np.number)]
    test_feat_cols_all = [c for c in test_df.columns
                          if c not in META_COLS | set(TARGETS)
                          and not c.endswith('_zscore')
                          and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_feat_cols_all) & set(test_feat_cols_all)
    
    for col in common_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc_name = f'{col}_zscore'
        train_df[zc_name] = (train_vals - mean) / std
        test_df[zc_name] = (test_vals - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # ===== Multi-task training =====
    # For each seed, train ONE LGBM model with objective='binary' but
    # multiple target predictions. Use multi_target_tree method or
    # train a regression model and threshold.
    
    # Actually, LGBM supports objective='binary' only. 
    # Alternative: use objective='regression' with label weights per target.
    # Or: train separate models but share feature selection across targets.
    
    # V323 approach: Single shared feature set across all targets,
    # then train 7 independent models with SAME feature ranking.
    # The "multi-task" aspect is in FEATURE SELECTION, not model training.
    # Use consensus feature selection: features important for ANY target.
    
    log.info("\nFeature selection: consensus across all 7 targets")
    
    # Rank features for each target, then find consensus
    consensus_features = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        y = train_df[t].values.astype(np.float64)
        X = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn = [sanitize_col(c) for c in feat_cols_clean]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        m = lgb.train(params_rank, ds, num_boost_round=50)
        imp = m.feature_importance(importance_type='gain')
        ranked = sorted(zip(feat_cols_clean, imp), key=lambda x: -x[1])
        consensus_features[t] = [r[0] for r in ranked]
        
        # Score each feature: how many targets it appears in top-K
        for rank_pos, (feat, score) in enumerate(ranked):
            if feat not in consensus_features:
                consensus_features[feat] = {}
    
    # Find features that appear in top-30 for ANY target (union)
    union_top_features = set()
    for t in TARGETS:
        for i in range(min(30, len(consensus_features[t]))):
            union_top_features.add(consensus_features[t][i])
    
    log.info(f"Union top-30 features across targets: {len(union_top_features)}")
    
    # Also compute per-target OOF-LB patterns to inform multi-task
    # Use feature importance correlation across targets
    log.info("\nComputing feature importance correlation across targets...")
    feat_imp_matrix = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        y = train_df[t].values.astype(np.float64)
        X = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn = [sanitize_col(c) for c in feat_cols_clean]
        ds = lgb.Dataset(X, label=y, feature_name=sn)
        m = lgb.train(params_rank, ds, num_boost_round=50)
        feat_imp = dict(zip(feat_cols_clean, m.feature_importance(importance_type='gain')))
        feat_imp_matrix[t] = feat_imp
    
    # Feature selection: for each target, use top-K features from UNION set
    # This ensures shared representation
    multi_task_feat = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        union_set = union_top_features & set(feat_cols_clean)
        ranked_union = sorted(list(union_set), 
                             key=lambda f: feat_imp_matrix[t].get(f, 0), 
                             reverse=True)
        multi_task_feat[t] = ranked_union[:n_feat]
    
    log.info("Multi-task feature selection done")
    
    # Now train 15 seeds × 7 targets with feature bagging
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        cfg_name = V53_SWEEP[t]['cfg']
        feat_cols_clean = remove_leak(train_feat_cols, t)
        
        # Get ranked features (from multi-task consensus)
        ranked = sorted(list(multi_task_feat[t]),
                       key=lambda f: feat_imp_matrix[t].get(f, 0),
                       reverse=True)
        
        # Also rank from current target specifically
        specific_ranked = [f for f in feat_cols_clean]  # all features
        # Quick ranking
        X_all = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        spw_all = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw_all, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn_all = [sanitize_col(c) for c in feat_cols_clean]
        ds_all = lgb.Dataset(X_all, label=y, feature_name=sn_all)
        m_rank = lgb.train(params_rank, ds_all, num_boost_round=50)
        imp_rank = dict(zip(feat_cols_clean, m_rank.feature_importance(importance_type='gain')))
        specific_ranked = sorted(feat_cols_clean, key=lambda f: imp_rank.get(f, 0), reverse=True)
        
        # Combine: multi-task union features first, then specific
        candidate_feats = list(multi_task_feat[t]) + [f for f in specific_ranked if f not in multi_task_feat[t]]
        n_candidates = len(candidate_feats)
        
        log.info(f"    Config: {cfg_name}, n_feat: {V53_SWEEP[t]['n_feat']}")
        log.info(f"    Candidate features: {n_candidates}")
        
        cfg = CFGS[cfg_name]
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_feat = V53_SWEEP[t]['n_feat']
            n_bag = max(int(n_candidates * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
            
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in candidate_feats if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            
            sel_cols = bag_feats
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            if len(sel_cols_test) != len(sel_cols):
                sel_cols = sel_cols_test
            
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
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
    
    # Meta learner
    target_oofs = {}
    student_avg_oofs = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V323 RESULTS (Multi-task feature selection + bagging + LR(C={META_C}))")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146: 0.63169 | V308: 0.62235 | V312: 0.61448 | V321: 0.60569")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    
    v308_gap = 0.01658
    pred_lb = avg_oof + v308_gap + 0.003
    
    log.info(f"\n  Predicted LB: {pred_lb:.5f}")
    log.info(f"  V308 LB: 0.63893 | Δ: {pred_lb - 0.63893:+.5f}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v323_multitask_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V323',
        'name': 'Multi-Task Feature Selection + Stacking',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'v321_avg_oof': 0.60569,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'predicted_improvement_vs_v308': round(float(pred_lb - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'multi-task feature selection (union of top-30 across targets)',
    }
    
    meta_path = EXPERIMENTS / f'v323_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
