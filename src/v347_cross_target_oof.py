"""
V347 — Cross-Target OOF Features

V339: self-OOF only → AVG OOF 0.61244
V347: self-OOF + cross-target OOF

Idea: targets are correlated. S1↔S2(0.382), S2↔S4(0.478), Q2↔Q3(0.340)
Cross-target OOF could capture this correlation as additional signal.

Architecture: Same V339 (15 seeds, GroupKFold 5, LR C=10)
Difference: For each target, add self-OOF + top-2 cross-target OOF features
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


def get_feature_cols(df, exclude_targets=True):
    exclude = set(META_COLS)
    if exclude_targets:
        exclude = exclude | set(TARGETS)
    return [c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


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


def run_pipeline_v347(train_df, test_df, cross_targets_map, add_all_oof=False):
    """
    Run pipeline for V347 with cross-target OOF features.
    
    cross_targets_map: dict of target -> [target1, target2, ...] (cross OOF features to add)
    add_all_oof: if True, add all 7 OOF features instead of just cross + self
    """
    log.info("Running V347 pipeline with cross-target OOF...")
    
    # Step 1: Build z-score features
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    train_work = train_df.copy()
    test_work = test_df.copy()
    
    for col in base_cols:
        vals_t = train_df[col].fillna(0).values.astype(np.float64)
        vals_te = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(vals_t)
        std = max(np.std(vals_t, ddof=0), 1e-8)
        train_work[f'{col}_zscore'] = (vals_t - mean) / std
        test_work[f'{col}_zscore'] = (vals_te - mean) / std
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_work)
    n_test = len(test_work)
    group = train_work['subject_id'].values
    
    # Step 2: Generate self-OOF for each target
    self_oof_train = {}
    self_oof_test = {}
    
    for target in TARGETS:
        feat_cols_clean = remove_leak(get_feature_cols(train_work), target)
        ranked = rank_features(train_work, feat_cols_clean, target)
        n_feat = V53_SWEEP[target]['n_feat']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_work)]
        
        y = train_work[target].values.astype(np.float64)
        oof_preds = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        cfg = CFGS[V53_SWEEP[target]['cfg']]
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
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
            oof_preds += np.clip(seed_oof, 0.001, 0.999) / N_SEEDS
            test_preds += seed_test / N_FOLDS
        
        self_oof_train[target] = np.clip(oof_preds, 0.01, 0.99)
        self_oof_test[target] = np.clip(test_preds, 0.01, 0.99)
    
    # Step 3: Add OOF features to data
    oof_feature_cols = list(self_oof_train.keys())
    for t in TARGETS:
        train_work[f'oof_{t}'] = self_oof_train[t]
        test_work[f'oof_{t}'] = self_oof_test[t]
    
    # Step 4: For each target, build feature set with self-OOF + cross-OOF
    result = {}
    for target in TARGETS:
        # Determine which OOF features to include
        cross_oofs = cross_targets_map.get(target, [])
        oof_cols_to_use = [f'oof_{target}'] + [f'oof_{ct}' for ct in cross_oofs]
        
        # Get base features (excluding OOF)
        base_feat_cols = remove_leak(
            [c for c in get_feature_cols(train_work) if not c.startswith('oof_')],
            target
        )
        
        # Rank base features
        ranked_base = rank_features(train_work, base_feat_cols, target)
        
        # Select top features, reserving slots for OOF
        n_oof = len(oof_cols_to_use)
        n_base = V53_SWEEP[target]['n_feat'] - n_oof
        if n_base < 5:
            n_base = 5
        
        sel_cols = ranked_base[:n_base] + oof_cols_to_use
        sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_work)]
        
        # Verify test has same features
        if set(sel_cols) != set(sel_cols_test):
            missing = set(sel_cols) - set(sel_cols_test)
            if missing:
                log.warning(f"  {target}: missing in test: {missing}")
                sel_cols_test = sel_cols  # fallback
        
        log.info(f"  {target}: n_feat={len(sel_cols)}, OOF={oof_cols_to_use}")
        log.info(f"    Base feats selected: {len([c for c in sel_cols if not c.startswith('oof_')])}")
        
        cfg = CFGS[V53_SWEEP[target]['cfg']]
        y = train_work[target].values.astype(np.float64)
        
        per_seed_oofs = []
        test_preds_arr = np.zeros((n_test, N_SEEDS))
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
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
            per_seed_oofs.append(seed_oof)
            test_preds_arr[:, si] = seed_test
        
        # Meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        oof_ll = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
        student_ll = log_loss(y, np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999))
        test_pred = meta.predict_proba(np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)]))[:, 1]
        
        result[target] = {'oof': oof_ll, 'student': student_ll, 'pred': np.clip(test_pred, 0.01, 0.99)}
        log.info(f"    {target}: student={student_ll:.5f}, meta={oof_ll:.5f}")
    
    return result, train_work, test_work


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V347 — Cross-Target OOF Features")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Check target correlations
    log.info("\nTarget correlations:")
    corr_matrix = train_df[TARGETS].corr()
    for t in TARGETS:
        others = [(t2, corr_matrix.loc[t, t2]) for t2 in TARGETS if t2 != t]
        others.sort(key=lambda x: -abs(x[1]))
        log.info(f"  {t}: top={others[0][0]}({others[0][1]:.3f}), 2nd={others[1][0]}({others[1][1]:.3f})")
    
    # Define cross-target map
    top_cross = {}
    for t in TARGETS:
        others = [(t2, corr_matrix.loc[t, t2]) for t2 in TARGETS if t2 != t]
        others.sort(key=lambda x: -abs(x[1]))
        top_cross[t] = [others[0][0], others[1][0]]
    
    log.info(f"\nCross-target map:")
    for t in TARGETS:
        log.info(f"  {t} → cross = {top_cross[t]}")
    
    # Strategy A: Self-OOF only (V339 baseline for comparison)
    log.info(f"\n{'='*70}")
    log.info("Strategy A: Self-OOF only (V339 baseline)")
    log.info(f"{'='*70}")
    result_A, _, _ = run_pipeline_v347(train_df, test_df, {})
    avg_A = np.mean([r['oof'] for r in result_A.values()])
    log.info(f"  AVG: {avg_A:.5f}")
    
    # Strategy B: Self-OOF + Top-2 Cross-Target OOF (V347)
    log.info(f"\n{'='*70}")
    log.info("Strategy B: Self-OOF + Top-2 Cross-Target OOF (V347)")
    log.info(f"{'='*70}")
    result_B, _, _ = run_pipeline_v347(train_df, test_df, top_cross)
    avg_B = np.mean([r['oof'] for r in result_B.values()])
    log.info(f"  AVG: {avg_B:.5f}")
    
    # Strategy C: Self-OOF + All-Other OOF (4 cross targets)
    log.info(f"\n{'='*70}")
    log.info("Strategy C: Self-OOF + ALL-OTHER OOF (7 OOF features)")
    log.info(f"{'='*70}")
    # For this, we need to reserve more slots for OOF, so reduce base features
    log.info("  Note: This reduces base feature slots significantly")
    
    # Strategy D: Self-OOF + weighted cross-target OOF (only targets with |corr| > 0.2)
    cross_strict = {}
    for t in TARGETS:
        others = [(t2, corr_matrix.loc[t, t2]) for t2 in TARGETS if t2 != t]
        others.sort(key=lambda x: -abs(x[1]))
        # Only include if |corr| > 0.25
        cross_strict[t] = [t2 for t2, corr in others[:2] if abs(corr) > 0.25]
    
    log.info(f"\n{'='*70}")
    log.info("Strategy D: Self-OOF + Strong Cross-Target OOF (|corr| > 0.25)")
    log.info(f"{'='*70}")
    result_D, _, _ = run_pipeline_v347(train_df, test_df, cross_strict)
    avg_D = np.mean([r['oof'] for r in result_D.values()])
    log.info(f"  AVG: {avg_D:.5f}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V347 COMPARISON")
    log.info(f"{'='*70}")
    log.info(f"  Strategy A (Self-OOF only):     AVG = {avg_A:.5f}  (ΔV339: {avg_A - 0.61244:+.5f})")
    log.info(f"  Strategy B (Top-2 cross):       AVG = {avg_B:.5f}  (ΔV339: {avg_B - 0.61244:+.5f})")
    log.info(f"  Strategy D (Strong cross):      AVG = {avg_D:.5f}  (ΔV339: {avg_D - 0.61244:+.5f})")
    log.info(f"  V308 AVG:                       0.62235")
    log.info(f"  V339 AVG:                       0.61244")
    
    best_avg = min(avg_A, avg_B, avg_D)
    best_strategy = 'A' if best_avg == avg_A else ('B' if best_avg == avg_B else 'D')
    best_result = result_A if best_strategy == 'A' else (result_B if best_strategy == 'B' else result_D)
    
    log.info(f"\n  → BEST: Strategy {best_strategy} (OOF: {best_avg:.5f})")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = best_result[t]['pred']
    
    sub_path = SUBMIT / f"submission_v347_{best_strategy}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V347',
        'name': f'Cross-Target OOF (Best: Strategy {best_strategy})',
        'avg_oof': round(float(best_avg), 5),
        'best_strategy': best_strategy,
        'avg_oof_A': round(float(avg_A), 5),
        'avg_oof_B': round(float(avg_B), 5),
        'avg_oof_D': round(float(avg_D), 5),
        'cross_targets_map': {t: top_cross[t] for t in TARGETS},
        'cross_strict_map': {t: cross_strict[t] for t in TARGETS},
        'per_target_oof': {t: round(float(best_result[t]['oof']), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(best_result[t]['student']), 5) for t in TARGETS},
        'delta_vs_v339': round(float(best_avg - 0.61244), 5),
        'delta_vs_v308': round(float(best_avg - 0.62235), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v347_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return best_avg, meta_data

if __name__ == '__main__':
    main()
