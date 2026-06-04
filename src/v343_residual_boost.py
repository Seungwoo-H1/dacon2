"""
V343 — Residual Boosting: Stage 2 student with V308 OOF as features

Hypothesis: V308 student 모델들의 residual (true - pred)에는 predictable pattern이 있을 수 있음.
예를 들어 특정 subject 그룹에서 systematically 과대/과소 예측하는 경우.
V343은 V308 pipeline의 OOF predictions을 feature로 사용하여, residual을 예측하는
두 번째 stage student를 학습합니다.

Architecture:
Stage 1: V308 student models → OOF predictions per sample
Stage 2: residual = true - stage1_pred → train second-stage LGBM
         Input: original features + stage1 OOF predictions
         Output: residual prediction
Final: stage1_pred + stage2_residual_pred = calibrated prediction

Key insight: If V308 consistently underpredicts S1 for some subjects,
stage 2 can learn this pattern from the residual signal.
This is different from meta-learner because meta averages student preds.
Stage 2 specifically targets systematic bias in stage 1.

Risk: Residual contains noise, stage 2 overfits to noise.
Mitigation: Strong regularization (safety cfg), limited features, few seeds.
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
    log.info("V343 — Residual Boosting: Stage 2 student with V308 OOF as features")
    log.info("Stage 1: V308 pipeline → OOF predictions")
    log.info("Stage 2: residual prediction using OOF + base features")
    log.info("Final: stage1 + stage2_residual")
    log.info("=" * 70)
    
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
    
    # Phase 1: Generate stage 1 OOF predictions (V308 pipeline)
    log.info("\n=== Phase 1: Stage 1 OOF predictions (V308 pipeline) ===")
    stage1_oofs = {}
    stage1_per_seed_oofs = {}  # (n_train, N_SEEDS) per target
    stage1_test_per_seed = {}  # (n_test, N_SEEDS) per target
    
    for t in TARGETS:
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
        
        per_seed_list = []
        test_seed_list = []
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        for si, seed in enumerate(seeds):
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
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_list.append(seed_oof)
            test_seed_list.append(seed_test)
        
        stage1_oofs[t] = np.mean(per_seed_list, axis=0)
        stage1_per_seed_oofs[t] = np.column_stack(per_seed_list)
        stage1_test_per_seed[t] = np.column_stack(test_seed_list)
        log.info(f"  {t}: mean={stage1_oofs[t].mean():.4f}, std={stage1_oofs[t].std():.4f}")
    
    # Phase 2: Train stage 2 (residual prediction)
    log.info("\n=== Phase 2: Stage 2 residual prediction ===")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        
        y = train_df[t].values.astype(np.float64)
        stage1_pred = stage1_oofs[t]
        residual = y - stage1_pred  # residual
        
        # Stage 2 features: base features (top 20) + stage1 predictions
        # Get top 20 base features (no zscore, no temp, no interaction)
        base_feat_names = [c for c in train_feat_cols if '_zscore' not in c and 'temp_' not in c]
        top_base = base_feat_names[:20]
        
        # Build stage 2 feature table
        s2_train_cols = top_base.copy()
        s2_train_cols.extend(['stage1_pred', 'stage1_pred_sq', 'stage1_pred_abs'])
        for t2 in TARGETS:
            if t2 != t:
                s2_train_cols.extend([f'stage1_{t2}', f'stage1_{t2}_sq'])
        
        s2_train = pd.DataFrame()
        for c in top_base:
            if c in train_df.columns:
                s2_train[c] = train_df[c].fillna(0)
        s2_train['stage1_pred'] = stage1_pred
        s2_train['stage1_pred_sq'] = stage1_pred ** 2
        s2_train['stage1_pred_abs'] = np.abs(stage1_pred - 0.5)
        for t2 in TARGETS:
            if t2 != t:
                s2_train[f'stage1_{t2}'] = stage1_oofs[t2]
                s2_train[f'stage1_{t2}_sq'] = stage1_oofs[t2] ** 2
        
        s2_test = pd.DataFrame()
        for c in top_base:
            if c in test_df.columns:
                s2_test[c] = test_df[c].fillna(0)
        # Use training-stage1 OOF as proxy for test
        s2_test['stage1_pred'] = np.mean(stage1_oofs[t])  # fallback mean
        s2_test['stage1_pred_sq'] = s2_test['stage1_pred'] ** 2
        s2_test['stage1_pred_abs'] = 0.1
        for t2 in TARGETS:
            if t2 != t:
                s2_test[f'stage1_{t2}'] = np.mean(stage1_oofs[t2])
                s2_test[f'stage1_{t2}_sq'] = s2_test[f'stage1_{t2}'] ** 2
        # Override: use test stage 1 predictions directly
        s2_test['stage1_pred'] = np.mean(stage1_test_per_seed[t], axis=1)
        s2_test['stage1_pred_sq'] = s2_test['stage1_pred'] ** 2
        s2_test['stage1_pred_abs'] = np.abs(s2_test['stage1_pred'] - 0.5)
        for t2 in TARGETS:
            if t2 != t:
                s2_test[f'stage1_{t2}'] = np.mean(stage1_test_per_seed[t2], axis=1)
                s2_test[f'stage1_{t2}_sq'] = s2_test[f'stage1_{t2}'] ** 2
        
        # Rank features for stage 2 (use residual as proxy target for ranking)
        s2_feat_cols_clean = [c for c in s2_train.columns if c not in META_COLS | set(TARGETS) 
                              and np.issubdtype(s2_train[c].dtype, np.number)]
        # Rank by correlation with residual instead of target
        ranked_s2 = sorted(s2_feat_cols_clean, key=lambda c: abs(np.corrcoef(s2_train[c].fillna(0).values, residual)[0, 1]), reverse=True)
        n_feat_s2 = 15
        sel_s2 = ranked_s2[:n_feat_s2]
        sel_s2_test = [c for c in sel_s2 if c in s2_test.columns]
        
        log.info(f"  Stage 2 features: {len(sel_s2_test)} selected from {len(s2_feat_cols_clean)}")
        new_feats_in_s2 = [c for c in sel_s2_test if 'stage1' in c]
        log.info(f"    Stage1 features used: {new_feats_in_s2}")
        
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        stage2_oofs = np.zeros((n_train, N_SEEDS))
        stage2_test = np.zeros((n_test, N_SEEDS))
        
        # Residual prediction uses very conservative model
        stage2_cfg = {
            'num_leaves': 8, 'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 200,
            'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 15.0,
            'min_child_samples': 25
        }
        
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = s2_train.iloc[tr_idx][sel_s2].fillna(0).values.astype(np.float64)
                X_va = s2_train.iloc[va_idx][sel_s2].fillna(0).values.astype(np.float64)
                y_tr = residual[tr_idx]
                
                params = {**stage2_cfg, 'objective': 'regression', 'metric': 'l2',
                          'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_s2]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=stage2_cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(s2_test[sel_s2_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, -0.5, 0.5)  # Clip residual
            seed_test /= N_FOLDS
            stage2_oofs[:, si] = seed_oof
            stage2_test[:, si] = seed_test
        
        # Ensemble: average residual prediction
        stage2_pred_mean = np.mean(stage2_oofs, axis=1)
        stage2_test_pred_mean = np.mean(stage2_test, axis=1)
        
        # Combine: stage1 + stage2_residual
        final_train = np.clip(stage1_pred + stage2_pred_mean, 0.001, 0.999)
        final_test = np.clip(np.mean(stage1_test_per_seed[t], axis=1) + stage2_test_pred_mean, 0.01, 0.99)
        
        # Train meta learner on (stage1_per_seed, stage2_residual) ensemble
        stacked_combined = np.column_stack([
            stage1_per_seed_oofs[t],  # 15 stage1 seeds
            stage2_oofs,               # 15 stage2 residuals
        ])
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_combined, y)
        meta_oof = np.clip(meta.predict_proba(stacked_combined)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, meta_oof)
        all_oofs[t] = oof_ll
        
        # Test prediction via meta
        test_stage1 = stage1_test_per_seed[t]
        test_stage2 = stage2_test
        stacked_test = np.column_stack([test_stage1, test_stage2])
        meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.01, 0.99)
        all_test_preds[t] = meta_test
        
        log.info(f"  {t}: stage1_OOF={log_loss(y, stage1_pred):.5f}, "
                 f"stage2_residual_mean={stage2_pred_mean.mean():.4f}, "
                 f"meta_OOF={oof_ll:.5f}, "
                 f"stage1-vs-meta delta={log_loss(y, stage1_pred)-oof_ll:+.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V343 RESULTS (residual boosting)")
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
    sub_path = SUBMIT / f"submission_v343_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V343',
        'name': 'Residual Boosting (Stage 2 with V308 OOF features)',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    meta_path = EXPERIMENTS / f'v343_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
