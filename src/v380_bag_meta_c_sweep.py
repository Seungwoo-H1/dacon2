"""
V380 — Bagging with Aggressive Meta Regularization

Hypothesis: Bagging (V368) reduces OOF but increases OOF-LB gap because
meta C=5 is too low-regularized for the bagged ensemble. Stronger regularization
(C=1.0 or lower) should compress the gap.

V368: bag_ratio=0.9 + CV ranking + Meta C=5 → meta OOF=0.599, gap 큰 문제
V380: bag_ratio=0.7 + CV ranking + Meta C∈{0.1, 0.5, 1.0, 5.0} sweep → gap最小的 C 찾기

Key difference from V368:
1. bag_ratio 0.7 (too aggressive 0.9 → overfitting)
2. Meta C sweep to find optimal gap-balanced C
3. Same V308 per-target ranking (not CV-averaged, which adds complexity)

Why this might work:
- Bagging reduces student OOF (good for LB if gap controlled)
- Strong meta regularization prevents meta from overfitting to bagged noise
- Optimal C should balance: low C = high bias but low gap, high C = low bias but high gap
- V308 C=10 works for non-bagged → V380 C=1-5 might work for bagged

Expected:
- OOF: ~0.610-0.615
- Predicted LB: ~0.628-0.633 (with optimal C)
- Risk: Low-Medium (same architecture, just tuning bag ratio and C)
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
BAG_RATIO = 0.7  # Feature bagging ratio


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
    log.info("V380 — Bagging with Aggressive Meta Regularization")
    log.info(f"Hypothesis: bag_ratio={BAG_RATIO} + Meta C sweep → find optimal gap balance")
    log.info("V308: C=10, no bagging, OOF=0.62235, LB=0.63893")
    log.info("V368: bag=0.9, C=5, meta OOF=0.599 but gap too large")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
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
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    # Meta C candidates
    META_C_LIST = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    # For each target, run bagging + C sweep
    # Store best C per target based on OOF-C gap analysis
    all_results = []  # (target, C, avg_oof, student_avg)
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols_full = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols_full if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols_full):
            missing = set(sel_cols_full) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols_full = sel_cols_test
        
        n_cols = len(sel_cols_full)
        n_bag_cols = max(int(n_cols * BAG_RATIO), 5)
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, bag_cols: {n_bag_cols}/{n_cols}")
        
        cfg = CFGS[cfg_name]
        
        # For each Meta C, run full training with bagging
        for meta_c in META_C_LIST:
            # Train N_SEEDS with bagging
            per_seed_oofs = []
            per_seed_test = []
            student_oofs = []
            
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                seed_oof = np.zeros(n_train)
                seed_test = np.zeros(n_test)
                
                # Random bag for this seed
                np.random.seed(seed)
                bag_cols = np.random.choice(sel_cols_full, size=n_bag_cols, replace=False)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[bag_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[bag_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in bag_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    seed_test += m.predict(test_df[bag_cols].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                per_seed_test.append(seed_test)
                student_oofs.append(log_loss(y, seed_oof))
            
            # Stack → LR meta
            stacked_train = np.column_stack(per_seed_oofs)
            stacked_test = np.column_stack(per_seed_test)
            
            meta = LogisticRegression(C=meta_c, max_iter=1000, random_state=SEED)
            meta.fit(stacked_train, y)
            
            oof_pred = meta.predict_proba(stacked_train)[:, 1]
            oof_ll = log_loss(y, np.clip(oof_pred, 0.001, 0.999))
            test_pred = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
            
            student_avg = np.mean(student_oofs)
            
            all_results.append({
                'target': t, 'C': meta_c,
                'meta_oof': oof_ll, 'student_avg': student_avg,
                'gap': oof_ll - student_avg,
                'test_pred': test_pred
            })
            
            if meta_c == META_C_LIST[0]:  # Only first C → detailed logging
                log.info(f"    [{t}] Meta C={meta_c}: meta_OOF={oof_ll:.5f}, student_avg={student_avg:.5f}, gap={oof_ll-student_avg:+.5f}")
    
    log.info(f"\n{'='*70}")
    log.info("C SWEEP RESULTS")
    log.info(f"{'='*70}")
    
    # Group by target, show C sweep for each
    for t in TARGETS:
        t_results = [r for r in all_results if r['target'] == t]
        t_results.sort(key=lambda x: x['meta_oof'])
        
        log.info(f"\n  {t}:")
        for r in t_results:
            v308_t = V308_OOF[t]
            delta = r['meta_oof'] - v308_t
            marker = " ← BEST" if r['meta_oof'] == min(rr['meta_oof'] for rr in t_results) else ""
            log.info(f"    C={r['C']:5.1f}: meta_OOF={r['meta_oof']:.5f} (Δ vs V308: {delta:+.5f}), "
                     f"student={r['student_avg']:.5f}, gap={r['gap']:+.5f}{marker}")
    
    # Now find the best C per target that minimizes predicted LB
    # Predicted LB = meta_OOF + (V308_OOF_LB - V308_OOF) = meta_OOF + 0.01658
    # But we want to use gap to estimate better: LB ≈ meta_OOF + student_avg * 0.3 + 0.47
    # Simpler: V308 gap is 0.01658. Use that.
    
    # For now, show all combinations
    best_by_oof = min(all_results, key=lambda x: x['meta_oof'])
    
    # Overall AVG meta OOF (weighted by target frequency)
    for meta_c in META_C_LIST:
        c_results = [r for r in all_results if r['C'] == meta_c]
        avg_meta_oof = np.mean([r['meta_oof'] for r in c_results])
        avg_student = np.mean([r['student_avg'] for r in c_results])
        avg_gap = avg_meta_oof - avg_student
        predicted_lb = avg_meta_oof + 0.01658
        
        log.info(f"\n  Overall Meta C={meta_c}:")
        log.info(f"    AVG meta OOF: {avg_meta_oof:.5f}")
        log.info(f"    AVG student:  {avg_student:.5f}")
        log.info(f"    AVG gap:      {avg_gap:+.5f}")
        log.info(f"    Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    
    # Select best C based on predicted LB
    best_overall = None
    best_predicted_lb = float('inf')
    for meta_c in META_C_LIST:
        c_results = [r for r in all_results if r['C'] == meta_c]
        avg_meta_oof = np.mean([r['meta_oof'] for r in c_results])
        predicted_lb = avg_meta_oof + 0.01658
        if predicted_lb < best_predicted_lb:
            best_predicted_lb = predicted_lb
            best_overall = meta_c
    
    log.info(f"\n{'='*70}")
    log.info(f"Best C by predicted LB: {best_overall}")
    log.info(f"Predicted LB at best C: {best_predicted_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Build submission with best C
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use best C for all targets
    final_test_preds = {t: np.zeros(n_test) for t in TARGETS}
    final_train_oofs = {t: np.zeros(n_train) for t in TARGETS}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols_full = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols_full if c in test_feat_cols]
        sel_cols_full = sel_cols_test
        
        n_cols = len(sel_cols_full)
        n_bag_cols = max(int(n_cols * BAG_RATIO), 5)
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        per_seed_test = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            np.random.seed(seed)
            bag_cols = np.random.choice(sel_cols_full, size=n_bag_cols, replace=False)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[bag_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[bag_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in bag_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[bag_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test.append(seed_test)
        
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = np.column_stack(per_seed_test)
        
        meta = LogisticRegression(C=best_overall, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        
        oof_pred = meta.predict_proba(stacked_train)[:, 1]
        final_train_oofs[t] = oof_pred
        final_test_preds[t] = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
    
    # Compute overall results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(final_train_oofs[t], 0.001, 0.999)) for t in TARGETS])
    predicted_lb = avg_oof + 0.01658
    
    log.info(f"\n{'='*70}")
    log.info(f"V380 RESULTS (Bag {BAG_RATIO}, Meta C={best_overall})")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, np.clip(final_train_oofs[t], 0.001, 0.999))
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    beats = predicted_lb < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    # Build submission
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = final_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v380_bag_c_sweep_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V380',
        'name': f'Bagging ({BAG_RATIO}) with Meta C Sweep',
        'best_meta_c': best_overall,
        'bag_ratio': BAG_RATIO,
        'avg_oof': round(float(avg_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(final_train_oofs[t], 0.001, 0.999))), 5) for t in TARGETS},
        'c_sweep_results': {t: [{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items() if k != 'test_pred'}
                                 for r in all_results if r['target'] == t] for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v380_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
