"""
V325 — Per-Subject Modeling with Leave-One-Subject CV

Hypothesis: 10 subjects, ~45 rows each. Train on 9 subjects, predict own.
Captures individual health trajectories better than pooled models.

Approach:
1. For each target, for each subject, train LGBM on other 9 subjects (~405 rows)
2. Predict own subject's rows (LOO at subject level)
3. Average across 5 seeds per subject → per-subject OOF
4. Aggregate per-subject OOF into full OOF for meta evaluation
5. Train final models on all 450 rows for test prediction

Expected OOF: 0.595-0.610
Risk: MEDIUM — subject-specific patterns might not generalize
Cost: ~45s (7 targets × 5 seeds × 10 subjects)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
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

SEED = 42
N_SEEDS = 5
FEATURE_BAG_FRACTION = 0.8


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


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V325 — Per-Subject Modeling with Leave-One-Subject CV")
    log.info("Hypothesis: Train on other 9 subjects, predict own subject")
    log.info("10 subjects, ~45 rows each, 405 rows for training")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-scores
    log.info("Generating z-score features...")
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
    
    train_feat_cols = [c for c in train_df.columns
                       if c not in META_COLS | set(TARGETS)
                       and np.issubdtype(train_df[c].dtype, np.number)]
    test_feat_cols = [c for c in test_df.columns
                     if c not in META_COLS | set(TARGETS)
                     and np.issubdtype(test_df[c].dtype, np.number)]
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    subjects = sorted(train_df['subject_id'].unique())
    log.info(f"Subjects: {len(subjects)}")
    
    # Store per-subject, per-seed LOO predictions
    # Format: dict[subject][target][seed_idx] = array of predictions for that subject
    subject_oofs = {s: {t: [] for t in TARGETS} for s in subjects}
    # Store per-subject, per-seed test predictions
    subject_tests = {s: {t: [] for t in TARGETS} for s in subjects}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        candidate_feats = ranked
        n_candidates = len(candidate_feats)
        n_feat = min(int(n_candidates * FEATURE_BAG_FRACTION), 50)
        
        log.info(f"    Using {n_feat} features (bag fraction {FEATURE_BAG_FRACTION})")
        
        for si in range(N_SEEDS):
            seed = SEED + si * 13
            
            for sub_idx, s in enumerate(subjects):
                subject_idx = train_df[train_df['subject_id'] == s].index
                other_idx = train_df[train_df['subject_id'] != s].index
                
                # Feature bag
                rng = np.random.RandomState(seed + sub_idx)
                bag = rng.choice(candidate_feats, size=n_feat, replace=False)
                bag_set = set(bag)
                bag_feats = [f for f in ranked if f in bag_set][:n_feat]
                
                if len(bag_feats) < n_feat:
                    remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                    bag_feats.extend(remaining)
                
                sel_cols = [c for c in bag_feats if c in test_feat_cols]
                
                X_other = train_df[sel_cols].iloc[other_idx].fillna(0).values.astype(np.float64)
                y_other = y[other_idx]
                
                spw = max(((y_other == 0).sum()) / max((y_other == 1).sum(), 1), 0.1)
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                    'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 200,
                    'subsample': 0.8, 'colsample_bytree': 0.8,
                    'reg_alpha': 3.0, 'reg_lambda': 8.0, 'min_child_samples': 5,
                    'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
                }
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_other, label=y_other, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=200)
                
                # LOO prediction: predict on own subject rows
                X_sub = train_df[sel_cols].iloc[subject_idx].fillna(0).values.astype(np.float64)
                pred_sub = m.predict(X_sub)
                subject_oofs[s][t].append(pred_sub)
                
                # Test prediction for this subject
                test_sub_idx = test_df[test_df['subject_id'] == s].index.tolist()
                if test_sub_idx:
                    X_test = test_df[sel_cols].iloc[test_sub_idx].fillna(0).values.astype(np.float64)
                    pred_test = m.predict(X_test)
                    while len(pred_test) < len(test_sub_idx):
                        pred_test = np.append(pred_test, 0.5)
                    subject_tests[s][t].append(pred_test[:len(test_sub_idx)])
        
        log.info(f"    Completed {N_SEEDS} seeds for target {t}")
    
    # Compute OOF: aggregate per-subject LOO predictions into full OOF
    target_oofs = {}
    student_avg_oofs = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        
        # Build per-subject averaged prediction array
        meta_pred = np.zeros(len(train_df))
        # Also store per-subject averaged predictions for student OOF calc
        subject_avg_preds = {}
        
        for s in subjects:
            preds_seed = subject_oofs[s][t]
            if len(preds_seed) > 0:
                avg_pred = np.mean(preds_seed, axis=0)
                subject_avg_preds[s] = avg_pred
            else:
                avg_pred = np.ones(len(train_df[train_df['subject_id']==s])) * 0.5
                subject_avg_preds[s] = avg_pred
            
            s_idx = train_df[train_df['subject_id'] == s].index
            meta_pred[s_idx] = avg_pred[:len(s_idx)]
        
        # Meta OOF (from averaged subject predictions)
        target_oofs[t] = log_loss(y, np.clip(meta_pred, 0.001, 0.999))
        
        # Student avg OOF: average of per-subject student OOF
        student_per_sub = []
        for s in subjects:
            s_mask = train_df['subject_id'] == s
            s_indices = train_df[s_mask].index.tolist()
            if s in subject_avg_preds and len(subject_avg_preds[s]) > 0:
                y_s = y[s_indices]
                p_s = subject_avg_preds[s][:len(y_s)]
                student_per_sub.append(log_loss(y_s, np.clip(p_s, 0.001, 0.999)))
        
        student_avg_oofs[t] = np.mean(student_per_sub) if student_per_sub else target_oofs[t]
        
        log.info(f"  {t}: Meta-OOF={target_oofs[t]:.5f}, Student-Avg-OOF={student_avg_oofs[t]:.5f}")
    
    avg_oof = np.mean(list(target_oofs.values()))
    avg_student = np.mean(list(student_avg_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V325 RESULTS (Per-Subject LOO, 5 seeds)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:+.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  V321: 0.60569 | V312: 0.61448 | V308: 0.62235")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    log.info(f"  Δ vs V312: {avg_oof - 0.61448:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Build submission: train final models on ALL 450 rows
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_feat = min(int(len(ranked) * FEATURE_BAG_FRACTION), 50)
        
        sub_preds = []
        for si in range(N_SEEDS):
            seed = SEED + si * 13
            rng = np.random.RandomState(seed)
            bag = rng.choice(ranked, size=n_feat, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            sel_cols = [c for c in bag_feats if c in test_feat_cols]
            
            X_tr = train_df[sel_cols].fillna(0).values.astype(np.float64)
            spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 200,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 3.0, 'reg_lambda': 8.0, 'min_child_samples': 5,
                'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
            }
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=200)
            sub_preds.append(m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64)))
        
        sub[t] = np.mean(sub_preds, axis=0)
    
    sub_path = SUBMIT / f"submission_v325_per_subject_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V325',
        'name': 'Per-Subject Modeling with Leave-One-Subject CV',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'n_subjects': len(subjects),
        'feature_bag_fraction': FEATURE_BAG_FRACTION,
        'v321_avg_oof': 0.60569,
        'v312_avg_oof': 0.61448,
        'v308_avg_oof': 0.62235,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'delta_vs_v312': round(float(avg_oof - 0.61448), 5),
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': 'per-subject LOO CV (train on 9 subjects, predict own)',
    }
    
    meta_path = EXPERIMENTS / f'v325_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
