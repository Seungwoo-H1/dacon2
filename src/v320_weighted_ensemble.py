"""
V320 — Weighted Student Ensemble (No Meta Learner)

Hypothesis: Meta learner (LR) is the bottleneck of stacking.
LR forces linear combination of student predictions, but optimal
combination might be non-linear. Also, LR overfits with 450 samples.

V320 approach:
1. Same 15 seeds, same V308 architecture
2. Instead of LR meta-learner: optimize per-student weights
3. Weight optimization via GroupKFold cross-validation:
   - For each fold: compute student OOF predictions on validation
   - Optimize weights on validation OOF predictions
   - Apply weights to train predictions
4. Test predictions: weighted average of 15 student predictions
5. Per-target weight optimization (same cfg per target for generalization)

Key insight from failures:
- V308 (LR C=10): LB 0.63893
- V312 (LR C=500): OOF 0.61448, LB unknown
- V316 (per-target cfg): LB 0.6505 (test overfitting!)
- Meta learner seems to be the problem, not the students

Expected OOF: 0.615-0.620 (similar to V312/V308)
Expected LB: < 0.638 (if weighted avg generalizes better)
Risk: Medium (weight optimization could overfit)
Cost: ~60s (same as V308)

Architecture: Same student training, DIFFERENT aggregation (weighted avg)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize

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


def optimize_weights(oof_students, y_true):
    """
    Optimize weights for ensemble averaging.
    Uses L-BFGS-B with sum-to-1 constraint and non-negative weights.
    """
    n_students = oof_students.shape[1]
    
    def objective(w):
        # Softmax to ensure positive weights
        w_pos = np.exp(w) / np.sum(np.exp(w))
        ensemble_pred = oof_students @ w_pos
        return log_loss(y_true, np.clip(ensemble_pred, 0.001, 0.999))
    
    x0 = np.zeros(n_students)  # start with equal weights
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12})
    
    w = np.exp(result.x) / np.sum(np.exp(result.x))
    best_oof = log_loss(y_true, oof_students @ w)
    
    return w, best_oof


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V320 — Weighted Student Ensemble (No Meta Learner)")
    log.info("Hypothesis: Weighted avg > LR stacking for generalization")
    log.info("V308: LR(C=10) meta → LB 0.63893")
    log.info("V312: LR(C=500) meta → OOF 0.61448")
    log.info("V320: optimize weights per target → better generalization")
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
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        
        cfg = CFGS[cfg_name]
        
        # Level 0: Train 15 seeds
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
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
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        all_seed_oofs[t] = per_seed_oofs
        
        # Level 1: Optimize weights
        # For OOF: use cross-validated student predictions per sample
        oof_matrix = np.column_stack(per_seed_oofs)  # (n_train, N_SEEDS)
        
        weights, oof_weighted = optimize_weights(oof_matrix, y)
        
        # Apply weights to train predictions
        train_pred = oof_matrix @ weights
        train_ll = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        
        log.info(f"    Optimized weights: {np.round(weights, 4)}")
        log.info(f"    Weighted ensemble OOF: {oof_weighted:.5f}")
        log.info(f"    Avg per-seed OOF: {np.mean([log_loss(y, p) for p in per_seed_oofs]):.5f}")
    
    # Compute overall results
    target_oofs = {}
    student_avg_oofs = {}
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        per_seed_list = all_seed_oofs[t]
        oof_matrix = np.column_stack(per_seed_list)
        weights, _ = optimize_weights(oof_matrix, y)
        train_pred = oof_matrix @ weights
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in per_seed_list])
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V320 RESULTS (15 seeds, weighted ensemble, no meta learner)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146: 0.63169 | V308: 0.62235 | V312: 0.61448")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ vs V312: {avg_oof - 0.61448:+.5f}")
    
    # OOF-LB gap estimation
    v308_gap = 0.01658
    pred_lb_v308_gap = avg_oof + v308_gap
    pred_lb_cons = avg_oof + v308_gap + 0.005
    
    log.info(f"\n  OOF-LB Gap Estimation:")
    log.info(f"    V308 actual gap: +{v308_gap:.5f}")
    log.info(f"    Predicted LB (V308 gap): {pred_lb_v308_gap:.5f}")
    log.info(f"    Predicted LB (conservative): {pred_lb_cons:.5f}")
    log.info(f"    V308 LB: 0.63893")
    log.info(f"    Δ vs V308 LB (conservative): {pred_lb_cons - 0.63893:+.5f}")
    
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test predictions: weighted average
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        weights, _ = optimize_weights(oof_matrix, y)
        sub[t] = test_preds[t] @ weights
    
    sub_path = SUBMIT / f"submission_v320_weighted_ensemble_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V320',
        'name': f'Weighted Student Ensemble (no meta learner)',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'v312_avg_oof': 0.61448,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v312': round(float(avg_oof - 0.61448), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'per_target_gap': {t: round(float(student_avg_oofs[t] - target_oofs[t]), 5) for t in TARGETS},
        'predicted_lb_conservative': round(float(pred_lb_cons), 5),
        'v308_actual_lb': 0.63893,
        'predicted_improvement_vs_v308': round(float(pred_lb_cons - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'aggregation_method': 'weighted_average',
        'meta_learner': 'none',
    }
    
    meta_path = EXPERIMENTS / f'v320_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
