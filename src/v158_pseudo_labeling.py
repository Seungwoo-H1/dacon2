"""
V158 — Pseudo-Labeling from V146

Hypothesis: V146's structure is near-optimal but limited by 450 training samples.
Pseudo-label high-confidence test predictions and retrain with expanded dataset.

Approach:
1. Train V146 on train data → get seed-level + meta test predictions
2. Select high-confidence test predictions (|pred - 0.5| > threshold)
3. Use them as soft-labels (weight=0.5) in augmented training
4. Final = ensemble of original V146 + retrained model
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
N_SEEDS = 5
META_C = 10.0
CONFIDENCE_THRESHOLD = 0.55
PSEUDO_WEIGHT = 0.5


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
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def run_v146_pipeline(train_df, test_df, feat_cols, augment_X=None, augment_y=None, augment_w=None):
    """
    V146 stacking pipeline.
    Returns: (train_oof_dict, test_seed_preds_dict, test_meta_preds_dict)
    test_seed_preds: shape (n_test, N_SEEDS)
    test_meta_preds: shape (n_test,)
    """
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_seed = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    test_meta = {t: np.zeros(n_test) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        cfg = CFGS[cfg_name]
        
        log.info(f"    Selected {n_feat} features from {len(feat_cols_clean)}")
        
        # Build augmented training arrays
        if augment_X is not None and augment_X[t] is not None:
            aug_X = augment_X[t]  # shape (n_aug, n_feat)
            aug_y = augment_y[t]  # shape (n_aug,)
            aug_w = augment_w[t]  # shape (n_aug,)
            all_X = np.vstack([train_df[sel_cols].fillna(0).values.astype(np.float64), aug_X])
            all_y = np.concatenate([y, aug_y])
            all_w = augment_w[t] if augment_w and augment_w[t] is not None else None
            has_aug = True
        else:
            all_X = train_df[sel_cols].fillna(0).values.astype(np.float64)
            all_y = y
            all_w = None
            has_aug = False
        
        n_aug = len(aug_y) if has_aug else 0
        if has_aug:
            log.info(f"    Training with augmentation: {n_train} + {n_aug} = {n_train+n_aug} samples")
        
        per_seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            seed_oof = np.zeros(n_train)  # OOF only on original train
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                # For augmented data: always use all train + aug for prediction
                # But fold-based OOF: only evaluate on original train folds
                
                X_tr_all = np.vstack([all_X[tr_idx], all_X[n_train:]])  # train fold + all aug
                X_va = all_X[n_train + tr_idx] if has_aug else all_X[tr_idx]  # WRONG
                
                # Actually: let's do proper fold-based training
                # Fold train: original train fold + all augmented
                # Fold val: original train fold
                # Test: all test
                
                X_tr_fold = np.vstack([all_X[tr_idx], all_X[n_train:]])  # (n_tr+n_aug, n_feat)
                y_tr_fold = np.concatenate([y[tr_idx], aug_y]) if has_aug else y[tr_idx]
                
                w_tr_fold = np.concatenate([np.ones(len(tr_idx)), aug_w]) if (has_aug and all_w is not None) else None
                
                X_va_fold = all_X[va_idx]  # (n_va, n_feat)
                y_va = y[va_idx]
                
                spw = max(((y_tr_fold == 0).sum()) / max((y_tr_fold == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                
                if w_tr_fold is not None:
                    ds = lgb.Dataset(X_tr_fold, label=y_tr_fold, weight=w_tr_fold, feature_name=sn)
                else:
                    ds = lgb.Dataset(X_tr_fold, label=y_tr_fold, feature_name=sn)
                
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va_fold)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_seed[t][:, si] = seed_test
            
            log.info(f"    Seed {si} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        # Meta learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        test_stacked = np.column_stack([test_seed[t][:, i] for i in range(N_SEEDS)])
        test_meta[t] = meta.predict_proba(test_stacked)[:, 1]
    
    return train_oof, test_seed, test_meta


def pseudo_label_and_retrain(train_df, test_df, feat_cols, threshold=CONFIDENCE_THRESHOLD, pw=PSEUDO_WEIGHT):
    """V158: Pseudo-labeling + retrain + ensemble."""
    log.info("=== V158: Pseudo-Labeling ===")
    
    # Step 1: Train V146 on train data only
    log.info("Step 1: Training V146 base model...")
    train_oof, test_seed, test_meta = run_v146_pipeline(train_df, test_df, feat_cols)
    
    # Log V146 OOF
    v146_oof = {}
    for t in TARGETS:
        v146_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_v146_oof = np.mean(list(v146_oof.values()))
    log.info(f"\nV146 AVG OOF: {avg_v146_oof:.5f}")
    for t in TARGETS:
        log.info(f"  {t}: {v146_oof[t]:.5f}")
    
    # Step 2: Generate pseudo-labels from seed-level predictions (more robust)
    log.info("Step 2: Generating pseudo-labels...")
    augment_X = {}
    augment_y = {}
    augment_w = {}
    total_pseudo = 0
    
    for t in TARGETS:
        # Use meta predictions for pseudo-labeling (more stable)
        test_pred = test_meta[t].copy()
        confidence = np.abs(test_pred - 0.5)
        mask = confidence > threshold
        n_selected = int(mask.sum())
        log.info(f"  {t}: {n_selected}/{len(test_df)} selected (threshold={threshold})")
        
        if n_selected > 0:
            feat_cols_clean = remove_leak(feat_cols, t)
            ranked = rank_features(train_df, feat_cols_clean, t)
            sel = ranked[:V53_SWEEP[t]['n_feat']]
            
            augment_X[t] = test_df[sel].fillna(0).values.astype(np.float64)[mask]
            augment_y[t] = test_pred[mask]
            augment_w[t] = np.ones(n_selected) * pw
            total_pseudo += n_selected
        else:
            augment_X[t] = None
            augment_y[t] = None
            augment_w[t] = None
    
    log.info(f"Total pseudo-labels: {total_pseudo}")
    
    if total_pseudo == 0:
        log.info("No pseudo-labels! Using V146 only.")
        final_test_preds = test_meta
    else:
        # Step 3: Retrain with augmented data
        log.info("Step 3: Retraining with pseudo-labels...")
        retrain_oof, retrain_seed, retrain_meta = run_v146_pipeline(
            train_df, test_df, feat_cols,
            augment_X=augment_X, augment_y=augment_y, augment_w=augment_w
        )
        
        # Log retrained OOF
        retrain_oof_avg = np.mean([log_loss(train_df[t].values, np.clip(retrain_oof[t], 0.001, 0.999))
                                    for t in TARGETS])
        log.info(f"\nRetrained AVG OOF: {retrain_oof_avg:.5f}")
        for t in TARGETS:
            log.info(f"  {t}: {log_loss(train_df[t].values, np.clip(retrain_oof[t], 0.001, 0.999)):.5f}")
        
        # Step 4: Ensemble (V146 + Retrained)
        log.info("Step 4: Ensemble (V146 + Retrained)...")
        final_test_preds = {}
        for t in TARGETS:
            v146_m = test_meta[t]
            retrain_m = retrain_meta[t]
            final_test_preds[t] = (v146_m + retrain_m) / 2.0
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = final_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v158_pseudo_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved: {sub_path}")
    
    meta_data = {
        'version': 'V158',
        'name': 'Pseudo-Labeling from V146',
        'v146_avg_oof': round(float(avg_v146_oof), 5),
        'v146_per_target_oof': {t: round(float(v146_oof[t]), 5) for t in TARGETS},
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'pseudo_weight': PSEUDO_WEIGHT,
        'total_pseudo': total_pseudo,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v158_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_v146_oof, meta_data


# ================================================================
# MAIN
# ================================================================

t_start = time.time()
log.info("=" * 70)
log.info("V158 — Pseudo-Labeling from V146")
log.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}, Pseudo weight: {PSEUDO_WEIGHT}")
log.info("=" * 70)

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols = get_feature_cols(train_df)
log.info(f"Base features: {len(feat_cols)}")
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

avg_oof, meta = pseudo_label_and_retrain(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
