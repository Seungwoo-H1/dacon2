"""
V339 — Pseudo-Label Refinement via CV Stacking

Hypothesis: V308의 student models가 test predictions에서 overconfident일 수 있음.
CV OOF predictions를 pseudo-label로 사용하여 student model을 재학습하면
train distribution에 더 잘 calibrated된 모델이 되어 OOF-LB gap이 감소.

Changes:
1. First pass: V308 pipeline으로 OOF predictions 생성
2. Second pass: OOF predictions를 augmented feature로 추가하여 model retrain
3. Feature: per-subject mean of OOF predictions (leave-one-out)
4. Same V308 architecture otherwise

Key insight: Using OOF predictions as features doesn't leak (they're out-of-fold).
But using test predictions as pseudo-labels DOES leak → use only OOF for training augmentation.
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


def generate_oof_features(train_df, test_df, TARGETS, SEED, N_FOLDS, N_SEEDS,
                           CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS):
    """Generate OOF predictions for all targets as additional features."""
    log.info("Generating OOF features...")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # For each target, generate OOF predictions
    oof_features_train = {}  # (n_train, n_targets)
    oof_features_test = {}   # (n_test, n_targets)
    
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    # Generate z-score features for test
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df = train_df.copy()
            train_df[zc] = (vals - mean) / std
            test_df = test_df.copy()
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    for target in TARGETS:
        feat_cols_clean = remove_leak(
            [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(train_df[c].dtype, np.number)],
            target
        )
        ranked = rank_features(train_df, feat_cols_clean, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        sel_cols = ranked[:n_feat]
        
        cfg = CFGS[cfg_name]
        y = train_df[target].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        oof_preds = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        for seed in seeds:
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
            oof_preds += seed_oof
            test_preds += seed_test
        
        oof_preds /= N_SEEDS
        test_preds /= N_SEEDS
        
        oof_features_train[target] = oof_preds
        oof_features_test[target] = np.clip(test_preds, 0.01, 0.99)
        
        log.info(f"  {target} OOF: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")
    
    return oof_features_train, oof_features_test


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V339 — OOF Feature Augmentation + V308 Pipeline")
    log.info("Generate OOF predictions per target → use as features in final model")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate OOF features first
    oof_train, oof_test = generate_oof_features(
        train_df, test_df, TARGETS, SEED, N_FOLDS, N_SEEDS,
        CFGS, V53_SWEEP, LEAK_S, LEAK_Q, META_COLS
    )
    
    # Add OOF features to data
    for t in TARGETS:
        train_df[f'oof_{t}'] = oof_train[t]
        test_df[f'oof_{t}'] = oof_test[t]
    
    # Z-score features
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and not c.startswith('oof_')
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
    
    log.info(f"Train: {len(train_feat_cols)} features (base+zscore+oof)")
    log.info(f"Test: {len(test_feat_cols)} features")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Feature ranking with OOF features included
    target_configs = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        # Don't rank OOF features — they're meta-features that might cause leakage
        feat_cols_no_oof = [c for c in feat_cols_clean if not c.startswith('oof_')]
        ranked = rank_features(train_df, feat_cols_no_oof, t)
        
        # Select top features + include oof_{t} feature
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        
        # Add OOF prediction of THIS target as an extra feature
        if f'oof_{t}' not in sel_cols:
            sel_cols.append(f'oof_{t}')
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        target_configs[t] = {
            'cfg': CFGS[cfg_name],
            'features': sel_cols,
            'features_test': sel_cols_test,
        }
        log.info(f"  {t}: cfg={cfg_name}, {len(sel_cols_test)} features (+oof_{t})")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        tc = target_configs[t]
        cfg = tc['cfg']
        feats = tc['features']
        feats_test = tc['features_test']
        
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} | features={len(feats)} | seeds={N_SEEDS}")
        
        y = train_df[t].values.astype(np.float64)
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
                X_tr = train_df[feats].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[feats].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in feats]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[feats_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            train_oofs[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
        
        stacked = np.column_stack(list(train_oofs.T))
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        log.info(f"  {t}: student={log_loss(y, student_oof):.5f}, meta={oof_ll:.5f}")
        
        stacked_test = np.column_stack([test_preds[:, si] for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V339 RESULTS (OOF feature augmentation)")
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
    
    sub_path = SUBMIT / f"submission_v339_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V339',
        'name': 'OOF Feature Augmentation + V308 Pipeline',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    
    meta_path = EXPERIMENTS / f'v339_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
