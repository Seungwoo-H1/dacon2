"""
V314 — Per-Target Meta C Sweeping + 20 Seeds

Hypothesis: V312 found C=500 optimal for all targets with 15 seeds.
V313 showed 30 seeds + C=500 overfits (OOF 0.595 → LB 0.647).
Key insight: C and seeds have a trade-off relationship.
  - More seeds → more student diversity → more noise in stacking
  - High C (500) amplifies noise fitting

V314 approach:
1. Reduce seeds from 30 to 20 (moderate increase)
2. Per-target C sweeping: test C = [10, 50, 100, 200, 500] for each target
3. Select best C per target using OOF
4. Expected: lower OOF than V308 while maintaining OOF-LB gap ≤ 0.025

Architecture: Same V308 + 20 seeds + per-target C sweeping
Expected OOF: 0.610-0.615 (Δ -0.007~-0.012 vs V308)
Expected LB: < 0.635 (V308 0.63893 대비 개선)
Risk: Medium (per-target C overfitting to CV OOF)
Cost: ~120s (20 seeds × 7 targets × 5 C values sweep)
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
N_SEEDS = 20  # V312(15) → 20, V313(30) 과잉 → 절제
C_VALUES = [10, 50, 100, 200, 500]  # Per-target sweep


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


def generate_test_zscore(train_df, test_df):
    """Generate global z-score features for test set."""
    log.info("Generating test z-score features (global)...")
    
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


def train_single_seed(train_df, sel_cols, y, y_tr, train_idx, val_idx, test_df, 
                      test_idx, seed, cfg, N_FOLDS):
    """Train one seed, return OOF predictions and test predictions."""
    n_train = len(y)
    n_test = len(test_df)
    seed_oof = np.zeros(n_train)
    seed_test = np.zeros(n_test)
    student_oofs = []
    
    for fold, (tr_idx, va_idx) in enumerate([(train_idx, val_idx)]):  # single fold
        X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
        X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
        
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in sel_cols]
        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
        
        seed_oof[va_idx] = m.predict(X_va)
        seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
        student_oofs.append(log_loss(y[va_idx], np.clip(m.predict(X_va), 0.001, 0.999)))
    
    seed_oof = np.clip(seed_oof, 0.001, 0.999)
    seed_test = np.clip(seed_test / 1, 0.001, 0.999)  # already from single "fold"
    
    return seed_oof, seed_test, np.mean(student_oofs)


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V314 — Per-Target Meta C Sweeping + 20 Seeds")
    log.info("Hypothesis: Moderate seeds (20) + per-target C optimization")
    log.info("V308: 15 seeds, C=10, OOF=0.62235, LB=0.63893")
    log.info("V312: 15 seeds, C=500, OOF=0.61448")
    log.info("V313: 30 seeds, C=500, OOF=0.59512, LB=0.6467 (overfit!)")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features
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
    
    # Storage
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    student_oof_avg = {t: [] for t in TARGETS}
    per_seed_train_oofs = {t: [] for t in TARGETS}  # per-seed OOF for analysis
    best_c_per_target = {}
    best_oof_per_target = {}
    
    # Per-target C sweep
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        n_used = len(sel_cols)
        n_zs = sum(1 for c in sel_cols if c.endswith('_zscore'))
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        log.info(f"    Selected {n_used} features ({n_zs} zscore)")
        
        cfg = CFGS[cfg_name]
        
        # Train all seeds
        all_seed_oofs = []
        all_seed_tests = []
        seed_student_oofs = []
        
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
            all_seed_oofs.append(seed_oof)
            all_seed_tests.append(seed_test)
            
            s_oof = log_loss(y, seed_oof)
            seed_student_oofs.append(s_oof)
            
            if si < 5 or si % 3 == 0 or si == N_SEEDS - 1:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
        
        # C sweep: find best C per target
        best_c = None
        best_c_oof = float('inf')
        c_results = []
        
        for c_val in C_VALUES:
            stacked = np.column_stack(all_seed_oofs)
            meta = LogisticRegression(C=c_val, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
            train_pred = meta.predict_proba(stacked)[:, 1]
            ll = log_loss(y, np.clip(train_pred, 0.001, 0.999))
            c_results.append((c_val, ll))
            log.info(f"    C={c_val:5.1f}: OOF={ll:.5f}")
            
            if ll < best_c_oof:
                best_c_oof = ll
                best_c = c_val
        
        best_c_per_target[t] = best_c
        best_oof_per_target[t] = best_c_oof
        log.info(f"    ⭐ Best C={best_c} → OOF={best_c_oof:.5f}")
        
        # Use best C for final model
        stacked = np.column_stack(all_seed_oofs)
        meta = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        
        for i in range(N_SEEDS):
            test_preds[t][:, i] = all_seed_tests[i]
        student_oof_avg[t] = seed_student_oofs
        per_seed_train_oofs[t] = all_seed_oofs
        
        log.info(f"    Student avg OOF: {np.mean(seed_student_oofs):.5f} ± {np.std(seed_student_oofs):.5f}")
    
    # Compute overall results
    target_oofs = {}
    for t in TARGETS:
        target_oofs[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V314 RESULTS ({N_SEEDS} seeds, per-target C)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (best C={best_c_per_target[t]})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146 AVG OOF: 0.63169")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  V312 AVG OOF: 0.61448")
    log.info(f"  V313 AVG OOF: 0.59512")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ vs V312: {avg_oof - 0.61448:+.5f}")
    log.info(f"  Δ vs V313: {avg_oof - 0.59512:+.5f}")
    
    # Best C summary
    log.info(f"\n  Per-target best C:")
    for t in TARGETS:
        log.info(f"    {t}: C={best_c_per_target[t]}")
    
    # Overfitting analysis
    log.info(f"\n  Student-Meta gap analysis:")
    for t in TARGETS:
        s_avg = np.mean(student_oof_avg[t])
        gap = s_avg - target_oofs[t]
        log.info(f"    {t}: Student avg={s_avg:.5f}, Meta OOF={target_oofs[t]:.5f}, gap={gap:.5f}")
    
    global_student_avg = np.mean([np.mean(student_oof_avg[t]) for t in TARGETS])
    log.info(f"  Global Student avg: {global_student_avg:.5f}")
    
    # Expected LB estimate
    v308_gap = 0.01658
    pred_lb_v308_gap = avg_oof + v308_gap
    
    # V313 gap = 0.6467 - 0.59512 = 0.0516
    # V313 was too extreme (30 seeds + C=500)
    # V314 should be between V312 and V313 in gap
    estimated_gap = min(pred_lb_v308_gap - avg_oof, 0.035)  # cap at 0.035
    pred_lb_est = avg_oof + estimated_gap
    
    log.info(f"\n  OOF-LB Gap Estimation:")
    log.info(f"    V308 gap: +{v308_gap:.5f} (actual)")
    log.info(f"    V313 gap: +{0.6467 - 0.59512:.5f} (actual — overfit)")
    log.info(f"    Estimated V314 gap: +{estimated_gap:.5f}")
    log.info(f"    Predicted LB: {pred_lb_est:.5f}")
    log.info(f"    V308 LB: 0.63893")
    log.info(f"    Δ vs V308 LB: {pred_lb_est - 0.63893:+.5f}")
    
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=best_c_per_target[t], max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack([per_seed_train_oofs[t][i] for i in range(N_SEEDS)]), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v314_per_target_c_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V314',
        'name': f'Per-Target Meta C Sweeping + {N_SEEDS} Seeds',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'n_c_values_tested': len(C_VALUES),
        'c_values_tested': C_VALUES,
        'best_c_per_target': {t: int(best_c_per_target[t]) for t in TARGETS},
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'v312_avg_oof': 0.61448,
        'v313_avg_oof': 0.59512,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v312': round(float(avg_oof - 0.61448), 5),
        'delta_vs_v313': round(float(avg_oof - 0.59512), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'per_target_best_c': {t: int(best_c_per_target[t]) for t in TARGETS},
        'student_oof_avg': {t: round(float(np.mean(student_oof_avg[t])), 5) for t in TARGETS},
        'student_oof_std': {t: round(float(np.std(student_oof_avg[t])), 5) for t in TARGETS},
        'global_student_avg_oof': round(float(global_student_avg), 5),
        'predicted_lb_v308_gap': round(float(pred_lb_v308_gap), 5),
        'predicted_lb_estimated': round(float(pred_lb_est), 5),
        'v308_actual_lb': 0.63893,
        'v313_actual_lb': 0.6467,
        'predicted_improvement_vs_v308': round(float(pred_lb_est - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v314_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
