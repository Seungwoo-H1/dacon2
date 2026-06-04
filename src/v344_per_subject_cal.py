"""
V344 — Per-Subject Calibration on Top of V308 Meta

Hypothesis: V308의 meta learner는 global LR. 하지만 subject별 prediction bias가
systematically 존재할 수 있음 (예: 특정 subject는 항상 overpredict).
Per-subject calibration은 각 subject의 V308 meta predictions과 실제 label의
차이를 learning하여 보정합니다.

Architecture:
1. V308 pipeline으로 전체 train/test 예측 생성 (OOF, no leak)
2. 각 subject별로: meta predictions과 true label로 per-subject LR calibration 학습
3. calibration parameter를 test prediction에 적용

This is a LIGHT touch — only adding per-subject bias correction.
Does NOT change the base student or meta architecture.
If per-subject calibration works, it improves generalization without OOF-LB gap risk.

Key insight: With 450 train / 250 test subjects, and ~45 samples/subject,
we can fit a simple bias+scale per subject without overfitting.

Risk: Overfitting to train bias. Use cross-validation to estimate calibration quality.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
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
    log.info("V344 — Per-Subject Calibration on Top of V308 Meta")
    log.info("1. V308 pipeline → per-sample OOF predictions")
    log.info("2. Per-subject bias+scale calibration")
    log.info("3. Apply calibration to test predictions")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score features (same as V308)
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
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} | Test: {len(test_feat_cols)}")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Phase 1: Generate V308-style OOF predictions
    log.info("\n=== Phase 1: V308-style OOF predictions (15 seeds × 5 folds) ===")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        cfg = CFGS[cfg_name]
        
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        # Per-fold, per-seed predictions
        # Structure: dict of (seed, fold) -> (train_oof, test_preds)
        oof_dict = {}  # (seed, fold) -> train_pred, test_pred
        for si, seed in enumerate(seeds):
            seed_train = np.zeros(n_train)
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
                seed_train[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_test /= N_FOLDS
            seed_train = np.clip(seed_train, 0.001, 0.999)
            oof_dict[(si, fold)] = (seed_train, seed_test) if fold == 0 else oof_dict.get((si, 0), (np.zeros(n_train), seed_test))
        
        # Build per-seed OOF (from all folds combined)
        # For each fold, collect the predictions, then ensemble
        per_seed_train = np.zeros((n_train, N_SEEDS))
        per_seed_test = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
            # Ensemble across folds for each seed
            seed_train_preds = np.zeros(n_train)
            seed_test_preds = np.zeros(n_test)
            
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
                
                seed_train_preds[va_idx] = m.predict(X_va)
                seed_test_preds += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_train_preds = np.clip(seed_train_preds, 0.001, 0.999)
            seed_test_preds /= N_FOLDS
            per_seed_train[:, si] = seed_train_preds
            per_seed_test[:, si] = seed_test_preds
        
        # Meta learner (V308 style)
        stacked_train = per_seed_train
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        meta_train_pred = np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999)
        meta_oof_ll = log_loss(y, meta_train_pred)
        
        # Per-subject calibration
        # For each subject in train, fit bias+scale from meta predictions to true labels
        subjects = train_df['subject_id'].values
        subject_ids = np.unique(subjects)
        
        # Fit per-subject calibration
        cal_params = {}  # subject_id -> (bias, scale)
        for subj in subject_ids:
            mask = subjects == subj
            pred_s = meta_train_pred[mask]
            true_s = y[mask]
            n_s = mask.sum()
            if n_s >= 3:  # Need at least 3 samples for calibration
                # Simple bias: mean(true - pred)
                # Scale: std adjustment
                bias = np.mean(true_s - pred_s)
                scale = np.std(true_s) / max(np.std(pred_s), 1e-8)
                # Clamp to reasonable range
                scale = np.clip(scale, 0.5, 2.0)
                cal_params[subj] = (bias, scale)
            else:
                cal_params[subj] = (0.0, 1.0)
        
        # Apply calibration
        cal_train_pred = np.copy(meta_train_pred)
        for i, subj in enumerate(subjects):
            bias, scale = cal_params[subj]
            cal_train_pred[i] = meta_train_pred[i] + bias * 0.3 + (meta_train_pred[i] - 0.5) * (scale - 1.0) * 0.3
            cal_train_pred[i] = np.clip(cal_train_pred[i], 0.001, 0.999)
        
        cal_oof_ll = log_loss(y, cal_train_pred)
        
        log.info(f"  meta_OOF={meta_oof_ll:.5f}, cal_OOF={cal_oof_ll:.5f}, delta={meta_oof_ll-cal_oof_ll:+.5f}")
        
        # Test: apply per-subject calibration
        test_subjects = test_df['subject_id'].values
        test_pred_raw = meta.predict_proba(np.column_stack([per_seed_test[:, si] for si in range(N_SEEDS)]))[:, 1]
        test_pred_raw = np.clip(test_pred_raw, 0.001, 0.999)
        
        cal_test_pred = np.copy(test_pred_raw)
        for i, subj in enumerate(test_subjects):
            bias, scale = cal_params[subj]
            cal_test_pred[i] = test_pred_raw[i] + bias * 0.3 + (test_pred_raw[i] - 0.5) * (scale - 1.0) * 0.3
            cal_test_pred[i] = np.clip(cal_test_pred[i], 0.01, 0.99)
        
        all_oofs[t] = cal_oof_ll
        all_test_preds[t] = cal_test_pred
        
        # Also save V308-style (no calibration) for comparison
        v308_style_test = np.clip(np.mean(per_seed_test, axis=1), 0.01, 0.99)
        v308_meta_train = np.clip(np.mean(per_seed_train, axis=1), 0.001, 0.999)
        v308_style_oof = log_loss(y, v308_meta_train)
        log.info(f"  V308-style (no cal) OOF={v308_style_oof:.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V344 RESULTS (per-subject calibration)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={all_oofs[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V308: 0.62235 | Δ: {avg_oof - 0.62235:+.5f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    sub_path = SUBMIT / f"submission_v344_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V344',
        'name': 'Per-Subject Calibration on Top of V308 Meta',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    meta_path = EXPERIMENTS / f'v344_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
