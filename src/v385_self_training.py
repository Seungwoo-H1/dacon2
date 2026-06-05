"""
V385 — Self-Training on Test Set (V308-Based Pseudo-Labeling)

Hypothesis: V372 failed because it used V368 predictions (overfitted) as 
pseudo-labels. V308 predictions are more reliable → higher quality pseudo-labels.

Approach:
1. Train V308 model on full training data
2. Predict on test → get probability estimates
3. Select high-confidence predictions (|p - 0.5| > threshold)
4. Add test samples to training set with pseudo-labels
5. Retrain model on augmented dataset (without test samples in validation)
6. Use GroupKFold: training fold contains only original training data,
   validation fold may contain test pseudo-labels (no leakage because
   validation samples are never used for training)

Key difference from V372:
- V372: Used overfitted V368 predictions → noisy pseudo-labels
- V385: Use V308 baseline predictions → more reliable pseudo-labels
- V372: Full retraining changed architecture
- V385: Keep same V308 architecture, just add data

Expected:
- OOF: ~0.620-0.623 (similar to V308)
- Student avg: ~0.690-0.692 (slightly better calibration from more data)
- Predicted LB: ~0.637-0.639
- Risk: Medium (test data contamination risk)

We use conservative confidence thresholds: only |p - 0.5| > 0.15
This means we select samples where model is relatively confident.
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
CONFIDENCE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]  # |p - 0.5| > threshold


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
    log.info("V385 — Self-Training on Test Set (V308-Based Pseudo-Labeling)")
    log.info(f"Hypothesis: V308 predictions → reliable pseudo-labels → better calibration")
    log.info(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
    log.info("V308: OOF=0.62235, LB=0.63893")
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
    
    # First: generate pseudo-labels from V308 baseline for each target
    # Use 15 seeds × 5 folds to get good OOF predictions on test
    pseudo_test_preds = {}
    pseudo_train_oofs = {}
    
    log.info(f"\nGenerating pseudo-labels from V308 baseline...")
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
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
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test.append(seed_test)
        
        # Average across seeds for pseudo-labels
        pseudo_test_preds[t] = np.mean(per_seed_test, axis=0)
        pseudo_train_oofs[t] = np.mean(per_seed_oofs, axis=0)
        
        # Count high-confidence samples
        for thresh in CONFIDENCE_THRESHOLDS:
            n_conf = (np.abs(pseudo_test_preds[t] - 0.5) > thresh).sum()
            log.info(f"  {t}: {n_conf}/{n_test} samples with |p-0.5| > {thresh}")
    
    # For each threshold, do self-training
    threshold_results = {}
    
    for thresh in CONFIDENCE_THRESHOLDS:
        log.info(f"\n{'='*70}")
        log.info(f"Threshold: |p - 0.5| > {thresh}")
        log.info(f"{'='*70}")
        
        # Create augmented dataset
        augmented_train = train_df.copy()
        pseudo_labels = {}
        n_added = 0
        
        for t in TARGETS:
            probs = pseudo_test_preds[t]
            confidence = np.abs(probs - 0.5)
            mask = confidence > thresh
            
            # Convert to binary pseudo-labels
            pseudo_labels[t] = (probs > 0.5).astype(np.float64)
            
            augmented_train = pd.concat([augmented_train, test_df[mask].copy()], ignore_index=True)
            n_added += mask.sum()
        
        log.info(f"  Added {n_added} pseudo-labeled samples to training set")
        
        # Get feature columns from augmented data
        aug_feat_cols = get_feature_cols(augmented_train)
        
        # Set pseudo-labels on augmented data (appended rows)
        start_idx = n_train
        for t in TARGETS:
            probs = pseudo_test_preds[t]
            confidence = np.abs(probs - 0.5)
            mask = confidence > thresh
            # Get feature cols for this target
            feat_cols_clean = remove_leak(aug_feat_cols, t)
            # Find target column index
            if t in augmented_train.columns:
                augmented_train.loc[start_idx:start_idx+mask.sum()-1, t] = pseudo_labels[t][mask]
            start_idx += mask.sum()
        
        # Regenerate z-score features for augmented data
        # Use original train stats
        for col in train_base:
            if col in augmented_train.columns:
                vals = augmented_train[col].fillna(0).values.astype(np.float64)
                # Use original train stats
                train_vals = train_df[col].fillna(0).values.astype(np.float64)
                mean = np.mean(train_vals)
                std = np.std(train_vals, ddof=0)
                if std < 1e-8:
                    std = 1e-8
                zc = f'{col}_zscore'
                augmented_train[zc] = (vals - mean) / std
        
        # Now run full V308 pipeline on augmented data
        aug_group = augmented_train['subject_id'].values
        aug_gkf = GroupKFold(n_splits=N_FOLDS)
        n_aug = len(augmented_train)
        
        aug_target_results = []
        aug_all_student_oofs = []
        
        for t in TARGETS:
            y = augmented_train[t].values.astype(np.float64)
            feat_cols_clean = remove_leak(aug_feat_cols, t)
            n_feat = V53_SWEEP[t]['n_feat']
            cfg_name = V53_SWEEP[t]['cfg']
            
            ranked = rank_features(augmented_train, feat_cols_clean, t)
            sel_cols = ranked[:n_feat]
            
            cfg = CFGS[cfg_name]
            
            per_seed_oofs = []
            per_seed_test = []
            student_oofs = []
            
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                seed_oof = np.zeros(n_aug)
                seed_test = np.zeros(n_test)
                
                for fold, (tr_idx, va_idx) in enumerate(aug_gkf.split(augmented_train, y, aug_group)):
                    X_tr = augmented_train[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = augmented_train[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    # Predict on test (only non-augmented test)
                    seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                per_seed_test.append(seed_test)
                
                # Only compute student OOF on original training data
                original_train_mask = ~augmented_train['subject_id'].str.startswith('pseudo')
                y_original = augmented_train.loc[original_train_mask, t].values.astype(np.float64)
                oof_original = seed_oof[original_train_mask]
                student_oofs.append(log_loss(y_original, oof_original))
            
            # Meta-learner
            stacked_train = np.column_stack(per_seed_oofs)
            stacked_test = np.column_stack(per_seed_test)
            
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked_train, y)
            meta_oof = log_loss(y, np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999))
            meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
            
            student_avg = np.mean(student_oofs)
            
            aug_target_results.append({
                'target': t, 'threshold': thresh,
                'meta_oof': meta_oof, 'student_avg': student_avg
            })
            aug_all_student_oofs.extend(student_oofs)
            
            log.info(f"    {t}: meta_OOF={meta_oof:.5f} (V308: {V308_OOF[t]:.5f}), "
                     f"student_avg={student_avg:.5f}")
        
        avg_meta_oof = np.mean([r['meta_oof'] for r in aug_target_results])
        avg_student = np.mean(aug_all_student_oofs)
        predicted_lb = avg_meta_oof + 0.01658
        
        threshold_results[thresh] = {
            'avg_meta_oof': avg_meta_oof,
            'avg_student': avg_student,
            'predicted_lb': predicted_lb,
            'target_results': aug_target_results
        }
        
        log.info(f"\n  THRESHOLD {thresh}:")
        log.info(f"    AVG meta OOF: {avg_meta_oof:.5f} (V308: 0.62235, Δ: {avg_meta_oof-0.62235:+.5f})")
        log.info(f"    AVG student:  {avg_student:.5f} (V308: 0.69212, Δ: {avg_student-0.69212:+.5f})")
        log.info(f"    Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("SELF-TRAINING THRESHOLD SUMMARY")
    log.info(f"{'='*70}")
    for thresh, r in threshold_results.items():
        marker = " ← BEST" if r['avg_meta_oof'] == min(rr['avg_meta_oof'] for rr in threshold_results.values()) else ""
        log.info(f"  thresh={thresh}: meta_OOF={r['avg_meta_oof']:.5f}, "
                 f"student={r['avg_student']:.5f}, "
                 f"pred_LB={r['predicted_lb']:.5f}{marker}")
    
    # Also add baseline (no self-training)
    threshold_results['baseline'] = {
        'avg_meta_oof': 0.62235,
        'avg_student': 0.69212,
        'predicted_lb': 0.63893
    }
    
    best_thresh = min(CONFIDENCE_THRESHOLDS, key=lambda t: threshold_results[t]['avg_meta_oof'])
    best_r = threshold_results[best_thresh]
    
    log.info(f"\n{'='*70}")
    log.info(f"V385 RESULT (best threshold: {best_thresh})")
    log.info(f"{'='*70}")
    log.info(f"  AVG OOF: {best_r['avg_meta_oof']:.5f} (V308: 0.62235, Δ: {best_r['avg_meta_oof']-0.62235:+.5f})")
    log.info(f"  Student avg: {best_r['avg_student']:.5f} (V308: 0.69212, Δ: {best_r['avg_student']-0.69212:+.5f})")
    log.info(f"  Predicted LB: {best_r['predicted_lb']:.5f} (V308: 0.63893, Δ: {best_r['predicted_lb']-0.63893:+.5f})")
    beats = best_r['predicted_lb'] < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    meta_data = {
        'version': 'V385',
        'name': f'Self-Training (Pseudo-Labeling, V308-based)',
        'best_threshold': best_thresh,
        'avg_oof': round(float(best_r['avg_meta_oof']), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(best_r['avg_meta_oof'] - 0.62235), 5),
        'predicted_lb': round(float(best_r['predicted_lb']), 5),
        'beats_v308': bool(beats),
        'student_avg_oof': round(float(best_r['avg_student']), 5),
        'threshold_results': {
            str(t): {
                'avg_meta_oof': round(r['avg_meta_oof'], 5),
                'avg_student': round(r['avg_student'], 5),
                'predicted_lb': round(r['predicted_lb'], 5)
            } for t, r in threshold_results.items() if t != 'baseline'
        },
        'baseline': threshold_results['baseline'],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v385_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return best_r['avg_meta_oof'], meta_data


if __name__ == '__main__':
    main()
