"""
V383 — Rank-Percentile Target Transformation

Hypothesis: V308 trains on raw binary labels (0/1). But the underlying
continuous variable is likely not well-calibrated with a 0.5 threshold.
Rank-percentile transformation converts binary → continuous uniform [0,1],
giving the model richer signal and better calibration.

Changes from V308:
1. Transform target values to rank-percentile (0-1 continuous)
2. Change objective from 'binary' to 'regression' with square/linear loss
3. Keep same architecture: 15 seeds × GroupKFold 5-fold → LR meta
4. Meta-learner: still logistic regression on OOF predictions
5. Clip predictions to [0,1] and round to binary for evaluation

Why this might work:
- Raw 0/1 labels throw away information about "how positive" a sample is
- Rank-percentile preserves ordering and gives continuous signal
- Regression loss might generalize better than binary cross-entropy
- Better calibration → student avg stays near 0.692
- Same 282 features → same OOF-LB gap dynamics as V308

Expected:
- OOF: ~0.615-0.625 (regression might not help classification directly)
- Student avg: ~0.692 (same features, same architecture)
- Predicted LB: depends on how much regression helps
- Risk: Medium (target transform changes the problem)
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
META_C = 10.0
METHODS = ['binary', 'regression']  # Compare binary vs regression


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


def to_rank_percentile(y_train, y_val=None):
    """Convert binary labels to rank-percentile (continuous [0,1])."""
    # Sort and assign percentile ranks
    sorted_indices = np.argsort(y_train)
    ranks = np.arange(len(y_train))
    percentiles = np.zeros(len(y_train))
    percentiles[sorted_indices] = (ranks + 0.5) / len(y_train)
    
    if y_val is not None:
        # For validation, use training rank distribution
        sorted_train = np.sort(y_train)
        # Map val values to percentiles based on their rank in combined distribution
        combined = np.concatenate([y_train, y_val])
        combined_sorted = np.argsort(combined)
        combined_ranks = np.arange(len(combined))
        combined_percentiles = np.zeros(len(combined))
        combined_percentiles[combined_sorted] = (combined_ranks + 0.5) / len(combined)
        val_percentiles = combined_percentiles[len(y_train):]
        return percentiles, val_percentiles
    
    return percentiles


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V383 — Rank-Percentile Target Transformation")
    log.info("Hypothesis: Rank-transform targets → better calibration")
    log.info("V308: binary logloss, OOF=0.62235, LB=0.63893")
    log.info("V383: regression on rank-percentile targets")
    log.info("=" * 70)
    
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
    
    for method in METHODS:
        log.info(f"\n{'='*70}")
        log.info(f"METHOD: {method.upper()}")
        log.info(f"{'='*70}")
        
        all_results = []
        all_student_oofs = []
        
        for t in TARGETS:
            y_raw = train_df[t].values.astype(np.float64)
            feat_cols_clean = remove_leak(train_feat_cols, t)
            n_feat = V53_SWEEP[t]['n_feat']
            cfg_name = V53_SWEEP[t]['cfg']
            
            # Rank-percentile transformation
            y_rank = to_rank_percentile(y_raw)
            
            # For binary method, use original labels
            if method == 'binary':
                y = y_raw
                y_train = y_raw
                obj = 'binary'
                metric = 'binary_logloss'
            else:
                y = y_train  # use rank-percentile for training
                obj = 'regression'
                metric = 'l2'
            
            ranked = rank_features(train_df, feat_cols_clean, t)
            sel_cols = ranked[:n_feat]
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            if len(sel_cols_test) != len(sel_cols):
                sel_cols = sel_cols_test
            
            cfg = CFGS[cfg_name]
            
            per_seed_oofs = []
            per_seed_test = []
            student_oofs = []
            
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
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1,
                              'objective': obj, 'metric': metric}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                per_seed_test.append(seed_test)
                
                # Compute student OOF using raw binary labels
                student_oofs.append(log_loss(y_raw, seed_oof))
            
            # Meta-learner
            stacked_train = np.column_stack(per_seed_oofs)
            stacked_test = np.column_stack(per_seed_test)
            
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked_train, y_raw)
            meta_oof = log_loss(y_raw, np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999))
            meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
            
            student_avg = np.mean(student_oofs)
            
            all_results.append({
                'target': t, 'method': method,
                'meta_oof': meta_oof, 'student_avg': student_avg
            })
            all_student_oofs.extend(student_oofs)
            
            if method == METHODS[0]:  # First method only: detailed logging
                log.info(f"    {t}: meta_OOF={meta_oof:.5f} (V308: {V308_OOF[t]:.5f}), "
                         f"student_avg={student_avg:.5f}")
        
        # Compare methods
        if method == METHODS[0]:
            # Compute overall for this method
            avg_meta_oof = np.mean([r['meta_oof'] for r in all_results])
            avg_student = np.mean(all_student_oofs)
            predicted_lb = avg_meta_oof + 0.01658
            
            all_results.append({
                'target': 'AVG', 'method': method,
                'meta_oof': avg_meta_oof, 'student_avg': avg_student
            })
            all_student_oofs.extend([avg_student])
            
            log.info(f"\n  {method.upper()} overall:")
            log.info(f"    AVG meta OOF: {avg_meta_oof:.5f} (V308: 0.62235, Δ: {avg_meta_oof-0.62235:+.5f})")
            log.info(f"    AVG student:  {avg_student:.5f} (V308: 0.69212, Δ: {avg_student-0.69212:+.5f})")
            log.info(f"    Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
        else:
            # Second method: just compare
            avg_meta_oof = np.mean([r['meta_oof'] for r in all_results])
            avg_student = np.mean(all_student_oofs)
            predicted_lb = avg_meta_oof + 0.01658
            
            log.info(f"\n  {method.upper()} overall:")
            log.info(f"    AVG meta OOF: {avg_meta_oof:.5f} (V308: 0.62235, Δ: {avg_meta_oof-0.62235:+.5f})")
            log.info(f"    AVG student:  {avg_student:.5f} (V308: 0.69212, Δ: {avg_student-0.69212:+.5f})")
            log.info(f"    Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    
    log.info(f"\n{'='*70}")
    log.info("METHOD COMPARISON SUMMARY")
    log.info(f"{'='*70}")
    for m in METHODS:
        results_m = [r for r in all_results if r['method'] == m and r['target'] == 'AVG']
        if results_m:
            r = results_m[0]
            marker = " ← BEST" if r['meta_oof'] == min(rr['meta_oof'] for rr in all_results if rr['target'] == 'AVG') else ""
            log.info(f"  {m.upper()}: meta_OOF={r['meta_oof']:.5f}, student_avg={r['student_avg']:.5f}{marker}")
    
    # Build submission with best method
    best_method = min(METHODS, key=lambda m: np.mean([r['meta_oof'] for r in all_results if r['method'] == m and r['target'] != 'AVG']))
    
    # Final run with best method (re-run everything)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_train_oofs = {}
    final_test_preds = {}
    final_student_oofs = []
    
    for t in TARGETS:
        y_raw = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        y_rank_full = to_rank_percentile(y_raw)
        if best_method == 'binary':
            y = y_raw
            obj = 'binary'
            metric = 'binary_logloss'
        else:
            y = y_train_full
            obj = 'regression'
            metric = 'l2'
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        per_seed_test = []
        
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
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1,
                          'objective': obj, 'metric': metric}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test.append(seed_test)
            final_student_oofs.append(log_loss(y_raw, seed_oof))
        
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = np.column_stack(per_seed_test)
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y_raw)
        final_train_oofs[t] = np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999)
        final_test_preds[t] = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
    
    avg_oof = np.mean([log_loss(train_df[t].values, final_train_oofs[t]) for t in TARGETS])
    student_avg = np.mean(final_student_oofs)
    predicted_lb = avg_oof + 0.01658
    
    log.info(f"\n{'='*70}")
    log.info(f"V383 RESULTS (best method: {best_method.upper()})")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, final_train_oofs[t])
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Student avg: {student_avg:.5f} (V308: 0.69212, Δ: {student_avg-0.69212:+.5f})")
    log.info(f"  Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    beats = predicted_lb < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = final_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v383_rank_transform_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V383',
        'name': f'Rank-Percentile Target Transform',
        'best_method': best_method,
        'avg_oof': round(float(avg_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'student_avg_oof': round(float(student_avg), 5),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, final_train_oofs[t])), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v383_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
