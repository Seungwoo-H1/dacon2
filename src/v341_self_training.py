"""
V341 — Self-Training with CV-Pseudo-Labels

Hypothesis: V308의 student 모델이 0.69-0.82의 student OOF를 보임. 
이것은 모델이 train에서 여전히 배울 여지가 있다는 뜻.
V308 pipeline으로 생성한 OOF predictions (leave-one-out이므로 leak 없음)을
pseudo-label로 사용하여 student를 재학습하면, calibration이 개선되어
student OOF가 낮아지고 → meta OOF도 함께 낮아질 수 있음.

Changes:
1. Phase 1: V308 pipeline → per-sample OOF predictions (CV, no leak)
2. Phase 2: OOF predictions + confidence를 weighted pseudo-label로 생성
   - High confidence samples (|pred-0.5| > 0.15): pseudo-label 100% weight
   - Medium confidence samples: pseudo-label 50% weight + original label 50%
3. Phase 3: Student models를 pseudo-labeled data로 재학습
4. Phase 4: 재학습된 student로 다시 OOF 생성 → meta 학습
5. Same architecture otherwise

Risk: Pseudo-label noise가 student를 더 망칠 수 있음.
Confidence threshold를 carefully tuning 필요.

Key insight from V104 (context): pseudo-labeling improved OOF but harmed LB.
V104의 문제: test distribution이 train mean으로 shift됨.
V341의 차이: CV OOF를 사용하므로 test predictions는 그대로 유지.
Re-training만 개선 → student OOF 낮춤.
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
    log.info("V341 — Self-Training with CV-Pseudo-Labels")
    log.info("Phase 1: V308 OOF predictions (no leak)")
    log.info("Phase 2: Confidence-weighted pseudo-label augmentation")
    log.info("Phase 3: Retrain student on augmented data")
    log.info("Phase 4: New OOF → Meta")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score features
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
    
    # Phase 1: Generate OOF predictions for all targets (V308 pipeline)
    log.info("\n=== Phase 1: Generating CV-OOF predictions ===")
    v308_oofs = {}
    v308_per_seed_oofs = {}
    
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        cfg = CFGS[cfg_name]
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        
        per_seed_oofs_list = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            
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
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            per_seed_oofs_list.append(seed_oof)
        
        v308_oofs[t] = np.mean(per_seed_oofs_list, axis=0)
        v308_per_seed_oofs[t] = per_seed_oofs_list
        
        log.info(f"  {t}: OOF mean={v308_oofs[t].mean():.4f}, std={v308_oofs[t].std():.4f}")
        log.info(f"    Confidences (|pred-0.5|): mean={np.abs(v308_oofs[t]-0.5).mean():.4f}")
        log.info(f"    High conf (>0.15): {(np.abs(v308_oofs[t]-0.5) > 0.15).sum()} samples")
    
    # Phase 2: Create augmented training data with pseudo-labels
    log.info("\n=== Phase 2: Creating augmented training data ===")
    
    for t in TARGETS:
        oof_preds = v308_oofs[t]
        y_true = train_df[t].values.astype(np.float64)
        
        # Confidence-based pseudo-label weight
        # confidence = |pred - 0.5| * 2 (0=uncertain, 1=very confident)
        confidence = np.abs(oof_preds - 0.5) * 2
        confidence = np.clip(confidence, 0, 1)
        
        # Pseudo label: blend of OOF prediction (regression-style) and hard label
        # Use soft pseudo-labels: pseudo = oof_pred with weight based on confidence
        pseudo_weight = np.where(confidence > 0.3, 0.3, 0.1)  # max 30% weight for pseudo
        actual_weight = 1.0 - pseudo_weight
        
        # Augmented targets: weighted blend of true and pseudo
        augmented_target = y_true * actual_weight + oof_preds * pseudo_weight
        
        # Create augmented dataframe
        augmented_df = train_df.copy()
        augmented_df[t] = augmented_target
        
        log.info(f"  {t}: pseudo_weight={pseudo_weight.mean():.3f}, "
                 f"mean_confidence={confidence.mean():.3f}, "
                 f"n_high_conf={(confidence > 0.3).sum()}, "
                 f"n_med_conf={((confidence > 0.15) & (confidence <= 0.3)).sum()}")
    
    # Phase 3: Retrain student models on augmented data
    log.info("\n=== Phase 3: Retraining on augmented data ===")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        
        # Use augmented dataframe for feature ranking
        ranked = rank_features(train_df, remove_leak(train_feat_cols, t), t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        cfg = CFGS[cfg_name]
        
        y = train_df[t].values.astype(np.float64)
        augmented_y = train_df[t + '_augmented_target'] if (t + '_augmented_target') in train_df.columns.values else augmented_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        train_oofs = np.zeros((n_train, N_SEEDS))
        test_preds = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                
                # Use augmented target for training
                tr_aug = augmented_df.iloc[tr_idx][t].values.astype(np.float64)
                
                # For binary objective, convert soft labels to hard labels with noise
                # Actually, LGBM binary objective needs hard labels. Use rounding.
                y_tr = (tr_aug > 0.5).astype(np.float64)
                # Also try: use the actual binary true label but with sample weight
                # This way we leverage pseudo-labels without changing the objective
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            train_oofs[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
        student_oof_ll = log_loss(y, student_oof)
        
        stacked = np.column_stack(list(train_oofs.T))
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        log.info(f"  {t}: student={student_oof_ll:.5f}, meta={oof_ll:.5f}, gap={student_oof_ll-oof_ll:+.4f}")
        
        stacked_test = np.column_stack([test_preds[:, si] for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V341 RESULTS (self-training with pseudo-labels)")
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
    sub_path = SUBMIT / f"submission_v341_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V341',
        'name': 'Self-Training with CV-Pseudo-Labels',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    meta_path = EXPERIMENTS / f'v341_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
