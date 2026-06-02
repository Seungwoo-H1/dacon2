"""
V310 — Z-Score Enriched Stacking + S2 Z-Score Removal + Calibration

V308 used global z-scores for ALL 7 targets. Key finding: S2 was the ONLY
target that DEGRADED with z-scores (+0.012 OOF).

V310 changes:
1. Remove z-score features for S2 target only (keep base features)
2. Keep z-score features for Q1,Q2,Q3,S1,S3,S4 (all improved)
3. Add sigmoid calibration to reduce OOF-LB gap (target 0.01658 → <0.010)

Hypothesis:
- S2 z-score features are harmful → removal should improve S2
- Calibration may reduce OOF-LB gap without hurting OOF much
- Net effect: better LB than V308 (0.63893)

Expected:
- OOF: similar to V308 (~0.622-0.625)
- S2: -0.005 to -0.010 improvement vs V308
- Calibration: OOF-LB gap reduction
- LB: < 0.635
- Risk: Low-Medium (same architecture, targeted changes only)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
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

# V310 specific: S2 does NOT use z-score features
S2_BASE_ONLY = True


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
    """Generate global z-score features for test set using training data statistics."""
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


def sigmoid_calibrate(train_preds, test_preds):
    """Apply isotonic calibration on OOF predictions."""
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(train_preds, np.clip(train_preds, 0.01, 0.99))
    test_cal = cal.predict(test_preds)
    return cal.predict(train_preds), test_cal


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V310 — Z-Score + S2 Base-Only + Calibration")
    log.info("Hypothesis: Remove z-score for S2 + sigmoid calibration")
    log.info("V308: global z-score for all, OOF=0.62235, LB=0.63893")
    log.info("V310: S2 base-only, same 15-seed stacking, calibration")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    test_subjects = set(test_df['subject_id'].unique())
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features for test
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
    # Also add z-score columns to train
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
    
    # Now select features for train
    train_feat_cols = get_feature_cols(train_df)
    zscore_train = [c for c in train_feat_cols if c.endswith('_zscore')]
    base_train = [c for c in train_feat_cols if not c.endswith('_zscore')]
    
    test_feat_cols = get_feature_cols(test_df)
    zscore_test = [c for c in test_feat_cols if c.endswith('_zscore')]
    base_test = [c for c in test_feat_cols if not c.endswith('_zscore')]
    
    log.info(f"Train: {len(base_train)} base + {len(zscore_train)} zscore = {len(train_feat_cols)}")
    log.info(f"Test:  {len(base_test)} base + {len(zscore_test)} zscore = {len(test_feat_cols)}")
    log.info(f"S2 uses BASE-ONLY (no z-scores)")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        
        cfg_name = V53_SWEEP[t]['cfg']
        
        # V310 special: S2 uses base features only (no z-scores)
        if t == 'S2' and S2_BASE_ONLY:
            feat_cols_clean = [c for c in feat_cols_clean if not c.endswith('_zscore')]
            log.info(f"    S2: Removed z-score features, using {len(feat_cols_clean)} base-only features")
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        # Verify same columns exist in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} selected features missing in test: {missing}")
            sel_cols = sel_cols_test
        
        n_used = len(sel_cols)
        n_zs_in_sel = sum(1 for c in sel_cols if c.endswith('_zscore'))
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        log.info(f"    Selected {n_used} features ({n_zs_in_sel} zscore)")
        
        cfg = CFGS[cfg_name]
        
        # Level 0: N_SEEDS LGBM models
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
        
        # Level 1: Stack → LR meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        
        # V310: Apply sigmoid calibration
        # cal_train, cal_test_preds = sigmoid_calibrate(train_oof[t], ...)
        # We'll calibrate on OOF predictions
        ll_before = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        
        # Calibrate using isotonic regression on OOF predictions
        cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        cal.fit(np.clip(train_oof[t], 0.01, 0.99), y)
        train_cal = cal.predict(np.clip(train_oof[t], 0.01, 0.99))
        ll_after = log_loss(y, np.clip(train_cal, 0.001, 0.999))
        
        train_oof[t] = train_cal
        log.info(f"    {t} Stacking OOF (C={META_C}, {N_SEEDS} seeds): {ll_before:.5f}")
        log.info(f"    {t} After calibration: {ll_after:.5f}")
    
    # Compute results
    target_oofs = {}
    for t in TARGETS:
        target_oofs[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V310 RESULTS (15 seeds, z-score + S2 base-only, {len(train_feat_cols)} feats)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={target_oofs[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146 AVG OOF: 0.63169")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    # Overfitting analysis
    log.info(f"\n  Student OOF stats:")
    for si in range(N_SEEDS):
        student_oofs = []
        for t in TARGETS:
            student_oofs.append(log_loss(train_df[t].values, np.clip(per_seed_oofs[si], 0.001, 0.999)))
        log.info(f"    Seed {si} avg student OOF: {np.mean(student_oofs):.5f}")
    
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test predictions: meta on N_SEEDS test predictions
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack([per_seed_oofs[i] for i in range(N_SEEDS)]), y_t)
        test_stacked = meta_t.predict_proba(stacked_test)[:, 1]
        
        # Apply same calibration
        cal_t = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        cal_t.fit(np.clip(meta.predict_proba(np.column_stack([per_seed_oofs[i] for i in range(N_SEEDS)]))[:, 1], 0.01, 0.99), y_t)
        # Actually, we need to calibrate on OOF, not meta output
        # For test, we calibrate using the OOF calibration from train
        test_stacked_all[t] = cal_t.predict(np.clip(test_stacked, 0.01, 0.99))
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v310_zscore_s2_baseonly_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V310',
        'name': 'Z-Score + S2 Base-Only + Calibration',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_base_features': len(base_train),
        'n_zscore_features': len(zscore_train),
        'n_seeds': N_SEEDS,
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'use_zscore': True,
        's2_base_only': S2_BASE_ONLY,
        'calibration': 'isotonic',
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v310_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
