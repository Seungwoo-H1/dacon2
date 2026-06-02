"""
V321 — Stacking with Feature Bagging (Multi-Bag Diversity)

Hypothesis: Same 282 features for all 15 seeds limits ensemble diversity.
Even with different random_state, if feature space is identical,
students learn similar decision boundaries.

V321 approach:
1. Each seed gets a RANDOM SUBSET of features (70% of available)
2. Different seed → different feature bag (with some overlap)
3. Same cfg per target, same 15 seeds, same stacking structure
4. Ensures student diversity through feature space diversity

Key insight: V315 showed feature ranking is very stable. So feature
selection itself is not the bottleneck. But RANDOM BAGGING adds noise
that forces students to learn different patterns.

Expected OOF: 0.610-0.618 (better diversity → better meta learning)
Expected LB: < 0.638 (if diversity helps generalization)
Risk: Medium (feature bagging might hurt individual student performance)
Cost: ~60s (same as V308)
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
FEATURE_BAG_FRACTION = 0.75  # Each seed uses 75% of features


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


def generate_test_zscore(train_df, test_df):
    log.info("Generating test z-score features...")
    train_feat_cols = [c for c in train_df.columns
                       if c not in META_COLS | set(TARGETS)
                       and not c.endswith('_zscore')
                       and np.issubdtype(train_df[c].dtype, np.number)]
    test_feat_cols = [c for c in test_df.columns
                      if c not in META_COLS | set(TARGETS)
                      and not c.endswith('_zscore')
                      and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_feat_cols) & set(test_feat_cols)
    log.info(f"Common base columns for z-score: {len(common_cols)}")
    zscore_cols = []
    for col in common_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc_name = f'{col}_zscore'
        test_df[zc_name] = (test_vals - mean) / std
        zscore_cols.append(zc_name)
    log.info(f"Generated {len(zscore_cols)} z-score features for test")
    return test_df, zscore_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V321 — Stacking with Feature Bagging")
    log.info("Hypothesis: Feature diversity → better ensemble diversity")
    log.info(f"Bag fraction: {FEATURE_BAG_FRACTION} (each seed uses different subset)")
    log.info("V308: same 282 features for all 15 seeds → LB 0.63893")
    log.info("V321: different feature bag per seed → better diversity")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
    # Add z-scores to train
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df[zc] = (vals - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
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
        
        # Feature ranking (same for all seeds in this target)
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        # Use ALL ranked features as candidate pool (not just top-K)
        # Bagging will select subset from this pool
        candidate_feats = ranked  # all ranked features
        n_candidates = len(candidate_feats)
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        log.info(f"    Candidate features: {n_candidates}")
        
        cfg = CFGS[cfg_name]
        
        # Train 15 seeds with different feature bags
        all_seed_oofs_t = []
        all_seed_tests_t = []
        feature_bags_used = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            
            # Feature bag: random subset of candidates
            rng = np.random.RandomState(seed)
            n_bag = max(int(n_candidates * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
            feature_bags_used.append(len(bag))
            
            # Select top-K from the bag (based on global ranking)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            
            if len(bag_feats) < n_feat:
                # Pad with features not in bag if needed
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            
            sel_cols = bag_feats
            
            # Verify same columns in test
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
            all_seed_oofs_t.append(seed_oof)
            all_seed_tests_t.append(seed_test)
            
            s_oof = log_loss(y, seed_oof)
            if si < 5 or si % 3 == 0 or si == N_SEEDS - 1:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}, bag_size={len(bag_feats)}")
        
        test_preds[t] = np.column_stack(all_seed_tests_t)
        all_seed_oofs[t] = all_seed_oofs_t
        
        log.info(f"    Feature bag sizes: unique={len(set(feature_bags_used))}")
    
    # Level 1: LR meta-learner (same as V308)
    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
        meta_weights_info[t] = meta.coef_[0]
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V321 RESULTS (15 seeds, feature bagging, LR(C={META_C}))")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        w_sum = np.sum(np.abs(meta_weights_info[t]))
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f}, |W|={w_sum:.3f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146: 0.63169 | V308: 0.62235 | V312: 0.61448")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ vs V312: {avg_oof - 0.61448:+.5f}")
    
    # OOF-LB gap estimation
    v308_gap = 0.01658
    pred_lb = avg_oof + v308_gap + 0.003  # feature bagging might reduce gap slightly
    
    log.info(f"\n  OOF-LB Gap Estimation:")
    log.info(f"    Predicted LB: {pred_lb:.5f}")
    log.info(f"    V308 LB: 0.63893")
    log.info(f"    Δ vs V308 LB: {pred_lb - 0.63893:+.5f}")
    
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
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v321_feature_bagging_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V321',
        'name': f'Feature Bagging + Stacking ({FEATURE_BAG_FRACTION} bag fraction)',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'v312_avg_oof': 0.61448,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v312': round(float(avg_oof - 0.61448), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'predicted_improvement_vs_v308': round(float(pred_lb - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'feature bagging per seed for diversity',
    }
    
    meta_path = EXPERIMENTS / f'v321_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
