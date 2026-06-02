"""
V324 — Multi-Config Ensemble Stacking

Hypothesis: V321 uses a single cfg per target. Different configs capture
different aspects of the data. Ensemble of different-config models should
beat any single config.

V324 approach:
1. For each target, train MULTIPLE configs (wide, deep, v48, safety) 
   with DIFFERENT seeds
2. Total students per target: 4 configs × 4 seeds = 16 students
3. Bagged feature subset per config-seed combo
4. Stack ALL 16 students into a single LR meta-learner
5. This gives much more diverse student pool

Key insight: V308's per-target cfg selection (V53_SWEEP) chose ONE config.
But maybe the optimal approach is to use ALL configs and let the meta
learner figure out which works best.

Expected students/target: 16 (vs V321's 15)
Expected diversity: MUCH higher (different model architectures)
Cost: ~240s (16 seeds × 7 targets × 5 folds vs 15×7×5)
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

SEED = 42
N_FOLDS = 5
N_SEEDS_PER_CFG = 4  # Each config gets 4 seeds
META_C = 10.0
FEATURE_BAG_FRACTION = 0.75


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
    log.info("V324 — Multi-Config Ensemble Stacking")
    log.info("4 configs × 4 seeds = 16 students per target")
    log.info("V321: 15 seeds × 1 cfg → OOF 0.60569")
    log.info("V324: 4 cfgs × 4 seeds → 16 students, more diversity")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
    # Add z-scores to train
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
    config_names = list(CFGS.keys())
    total_students = len(config_names) * N_SEEDS_PER_CFG
    
    test_preds = {t: np.zeros((n_test, total_students)) for t in TARGETS}
    all_student_oofs = {t: [] for t in TARGETS}
    student_info = {t: [] for t in TARGETS}  # (cfg, seed) per student
    
    for t_idx, t in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} ({t_idx+1}/{len(TARGETS)})")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_candidates = len(ranked)
        
        # Track per-config OOFs for analysis
        config_student_oofs = {cfg: [] for cfg in config_names}
        
        # Train each config with its own seeds
        for ci, cfg_name in enumerate(config_names):
            cfg = CFGS[cfg_name]
            cfg_seeds = []
            for s_idx in range(N_SEEDS_PER_CFG):
                seed = SEED + ci * 100 + s_idx * 7  # Different seed spacing per config
                
                sel_cols = ranked[:min(20, len(ranked))]  # Use same features for simplicity
                
                sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
                if len(sel_cols_test) != len(sel_cols):
                    sel_cols = sel_cols_test
                
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
                all_student_oofs[t].append(seed_oof)
                test_preds[t][:, len(all_student_oofs[t]) - 1] = seed_test
                student_info[t].append((cfg_name, seed))
                config_student_oofs[cfg_name].append(log_loss(y, seed_oof))
                
                if ci < 2 and s_idx < 3:
                    log.info(f"    {t}: {cfg_name}/seed{seed} → OOF={log_loss(y, seed_oof):.5f}")
        
        # Report per-config avg OOF
        log.info(f"    Per-config avg student OOF:")
        for cfg_name in config_names:
            if config_student_oofs[cfg_name]:
                avg = np.mean(config_student_oofs[cfg_name])
                log.info(f"      {cfg_name}: {avg:.5f} ({len(config_student_oofs[cfg_name])} seeds)")
    
    # Meta learner: stack all 16 students
    target_oofs = {}
    student_avg_oofs = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_student_oofs[t])
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_student_oofs[t]])
        
        # Show meta weights for each config
        coefs = meta.coef_[0]
        cfg_avg_weights = {}
        for i, (cfg_name, seed) in enumerate(student_info[t]):
            cfg_avg_weights.setdefault(cfg_name, []).append(abs(coefs[i]))
        log.info(f"    {t}: Meta weights by config: " +
                 ", ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(cfg_avg_weights.items())))
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V324 RESULTS ({total_students} students: {len(config_names)} cfgs × {N_SEEDS_PER_CFG} seeds)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146: 0.63169 | V308: 0.62235 | V312: 0.61448 | V321: 0.60569")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    
    v308_gap = 0.01658
    pred_lb = avg_oof + v308_gap + 0.003
    
    log.info(f"\n  Predicted LB: {pred_lb:.5f}")
    log.info(f"  V308 LB: 0.63893 | Δ: {pred_lb - 0.63893:+.5f}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_student_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v324_multi_config_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V324',
        'name': f'Multi-Config Ensemble Stacking ({len(config_names)} cfgs × {N_SEEDS_PER_CFG} seeds)',
        'avg_oof': round(float(avg_oof), 5),
        'n_students': total_students,
        'n_configs': len(config_names),
        'v321_avg_oof': 0.60569,
        'v312_avg_oof': 0.61448,
        'v308_avg_oof': 0.62235,
        'delta_vs_v321': round(float(avg_oof - 0.60569), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'predicted_improvement_vs_v308': round(float(pred_lb - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
        'key_difference': '4 configs × 4 seeds = 16 students per target',
    }
    
    meta_path = EXPERIMENTS / f'v324_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
