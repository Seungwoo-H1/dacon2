"""
V337 — Multi-Config Heterogeneous Ensemble with Proper Weighting

Hypothesis: V330-V336 실패의 공통점. 단일 feature set + 단일 cfg로 모든 target 학습.
V79가 이미 시도했지만 V145에서 heterogeneous model이 실패. 하지만 그건 XGBoost/CatBoost 때문.

V337: 같은 LGBM family이지만 다른 cfg로 여러 모델을 학습하고,
per-target으로 optimal model weights를 학습 (LR for weight optimization).

Changes:
1. 4 configs (wide, deep, v48, safety) × 4 seeds each = 16 students
2. Per-target weighted ensemble (weights learned via LR on OOF predictions)
3. Weight regularization (C=1) to prevent overfitting to weight optimization
4. Z-score features 유지

Key insight: Instead of bagging features (V333), bag hyperparameters.
Different hyperparameters → different bias-variance tradeoff → complementary errors.
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

CFG_LIST = [
    # (name, config)
    ('wide_high_lr',  {'num_leaves': 30, 'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': 500,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 5}),
    ('deep_strong',   {'num_leaves': 15, 'max_depth': 5, 'learning_rate': 0.01, 'n_estimators': 1500,
                        'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0, 'min_child_samples': 20}),
    ('v48_light',     {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
                        'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 2.0, 'reg_lambda': 8.0, 'min_child_samples': 8}),
    ('safety_deep',   {'num_leaves': 8, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2000,
                        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 15.0, 'min_child_samples': 25}),
]

SEED = 42
N_FOLDS = 5
N_SEEDS_PER_CFG = 4  # 각 cfg당 4 seeds
N_CONFIGS = len(CFG_LIST)
TOTAL_STUDENTS = N_CONFIGS * N_SEEDS_PER_CFG  # 16
META_C = 1.0  # weight regularization


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
    log.info("V337 — Multi-Config Heterogeneous Ensemble (16 students)")
    log.info("4 configs × 4 seeds = 16 students per target")
    log.info("Weighted ensemble via LR meta (C=1)")
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
    
    # Feature ranking per target (use all features, let configs differ in HP not features)
    target_configs = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        # Use same feature set for all configs (to isolate HP effect)
        n_feat = 20  # fixed moderate feature count
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        target_configs[t] = {
            'features': sel_cols,
            'features_test': sel_cols_test,
        }
        log.info(f"  {t}: {n_feat} features")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        tc = target_configs[t]
        feats = tc['features']
        feats_test = tc['features_test']
        
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} | {N_CONFIGS} configs × {N_SEEDS_PER_CFG} seeds = {TOTAL_STUDENTS} students")
        
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        # Generate predictions for all students
        all_student_oofs = []  # list of (n_train,) arrays
        all_student_test = []  # list of (n_test,) arrays
        
        student_idx = 0
        for cfg_name, cfg in CFG_LIST:
            for si in range(N_SEEDS_PER_CFG):
                seed = SEED + si * 13 + student_idx * 7
                
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
                all_student_oofs.append(seed_oof)
                all_student_test.append(seed_test)
                student_idx += 1
                
                if student_idx % 4 == 0:
                    log.info(f"    Config: {cfg_name}, seed {si}: OOF={log_loss(y, seed_oof):.5f}")
        
        # Stack all students
        stacked = np.column_stack(all_student_oofs)  # (n_train, TOTAL_STUDENTS)
        
        # Meta learner for weight optimization
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(stacked, y)
        
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        # Student ensemble (unweighted)
        student_oof = np.clip(np.mean(all_student_oofs, axis=0), 0.001, 0.999)
        student_oof_ll = log_loss(y, student_oof)
        
        log.info(f"  {t}: student={student_oof_ll:.5f}, meta={oof_ll:.5f}, gap={student_oof_ll-oof_ll:+.4f}")
        log.info(f"  {t}: meta weights (first 5): {meta.coef_[0][:5].round(4)}")
        
        # Test
        stacked_test = np.column_stack(all_student_test)
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V337 RESULTS ({TOTAL_STUDENTS} heterogeneous students)")
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
    
    sub_path = SUBMIT / f"submission_v337_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V337',
        'name': f'Multi-Config Heterogeneous Ensemble ({TOTAL_STUDENTS} students)',
        'avg_oof': round(float(avg_oof), 5),
        'n_students': TOTAL_STUDENTS,
        'n_configs': N_CONFIGS,
        'seeds_per_cfg': N_SEEDS_PER_CFG,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    
    meta_path = EXPERIMENTS / f'v337_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
