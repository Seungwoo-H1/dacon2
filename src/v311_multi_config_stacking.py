"""
V311 — Multi-Config Ensemble Stacking

V308 uses one config per target (from V53_SWEEP). V311 uses ALL 4 configs
per target, creating 4x more student models (60 seeds × 4 configs = 240 students).

Hypothesis:
- Different configs capture different patterns
- Ensembling all configs increases diversity without overfitting
- LR meta-learner can learn which configs to weight more

Architecture:
- 4 configs × 15 seeds × 5 folds = 240 student models per target
- LR meta-learner (C=10) stacks all 60 predictions per sample
- Same z-score features as V308

Expected:
- OOF: 0.615-0.620 (better than V308's 0.622)
- LB: < 0.635 (OOF-LB gap similar to V308)
- Risk: Medium (240 students, more meta parameters)
- Cost: 4x training time (~4 hours)
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
N_SEEDS = 15
META_C = 10.0
N_CONFIGS = 4  # wide, deep, v48, safety

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}


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
    """Rank features by LGBM gain importance."""
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
    """Generate global z-score features for test set."""
    log.info("Generating test z-score features (global)...")
    
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
    log.info("V311 — Multi-Config Ensemble Stacking")
    log.info("Hypothesis: All 4 configs × 15 seeds × 5 folds = 240 students")
    log.info("V308: 1 config per target × 15 seeds, OOF=0.62235")
    log.info("V311: 4 configs per target × 15 seeds, OOF expected 0.615-0.620")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features
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
    zscore_train = [c for c in train_feat_cols if c.endswith('_zscore')]
    base_train = [c for c in train_feat_cols if not c.endswith('_zscore')]
    
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(base_train)} base + {len(zscore_train)} zscore = {len(train_feat_cols)}")
    log.info(f"Test:  {len(test_feat_cols)} features")
    log.info(f"Configs: {list(CFGS.keys())}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # V311: For each target, use ALL features (not per-target config selection)
    # Use ALL features for ranking, then take top-K
    config_names = list(CFGS.keys())  # ['wide', 'deep', 'v48', 'safety']
    
    # For each config, determine feature count from V53_SWEEP (average n_feat)
    # Actually, each config has different optimal n_feat per target
    # We'll use the V53_SWEEP config for feature ranking, but train ALL 4 configs
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS * N_CONFIGS)) for t in TARGETS}
    all_student_oofs = {t: [] for t in TARGETS}  # list of OOF arrays
    
    ci = 0  # config index across all students
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        
        # Use ALL features for ranking (same as V308 per-target)
        n_feat = V53_SWEEP[t]['n_feat']
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        n_used = len(sel_cols)
        n_zs = sum(1 for c in sel_cols if c.endswith('_zscore'))
        log.info(f"    Selected {n_used} features ({n_zs} zscore)")
        
        # V311: For each target, use ALL 4 configs
        # Each config has its own selected features for ranking
        # But we train all 4 configs, each with its own feature ranking
        for cfg_name in config_names:
            cfg = CFGS[cfg_name]
            log.info(f"\n    Config: {cfg_name} ({cfg['num_leaves']} leaves, depth={cfg['max_depth']}, lr={cfg['learning_rate']})")
            
            # Each config may have different optimal n_feat
            # Use V53_SWEEP to get config-specific n_feat
            cfg_n_feat = V53_SWEEP[t]['n_feat']  # same as default
            
            # Actually, let's use per-config n_feat sweep
            # wide: n_feat+2, deep: n_feat, v48: n_feat-2, safety: n_feat+4
            if cfg_name == 'wide':
                cfg_n_feat = min(cfg_n_feat + 4, 50)
            elif cfg_name == 'v48':
                cfg_n_feat = max(cfg_n_feat - 2, 8)
            elif cfg_name == 'safety':
                cfg_n_feat = min(cfg_n_feat + 4, 50)
            
            # Re-rank for this config with adjusted n_feat
            ranked_cfg = rank_features(train_df, feat_cols_clean, t)
            sel_cols_cfg = ranked_cfg[:cfg_n_feat]
            sel_cols_cfg_test = [c for c in sel_cols_cfg if c in test_feat_cols]
            if len(sel_cols_cfg_test) != len(sel_cols_cfg):
                missing = set(sel_cols_cfg) - set(sel_cols_cfg_test)
                log.warning(f"      [{cfg_name}] {t}: {len(missing)} missing in test")
                sel_cols_cfg = sel_cols_cfg_test
            
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                seed_oof = np.zeros(n_train)
                seed_test = np.zeros(n_test)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols_cfg].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols_cfg].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols_cfg]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    seed_test += m.predict(test_df[sel_cols_cfg].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                all_student_oofs[t].append(seed_oof)
                
                # Store in test_preds with correct index
                if ci < test_preds[t].shape[1]:
                    test_preds[t][:, ci] = seed_test
                
                if si < 3 or si == N_SEEDS - 1:
                    ll = log_loss(y, seed_oof)
                    log.info(f"      [{cfg_name}] Seed {si:2d} (s{seed}): OOF={ll:.5f}")
                
                ci += 1
        
        # Level 1: Stack ALL students → LR meta
        stacked = np.column_stack([all_student_oofs[t][i] for i in range(len(all_student_oofs[t]))])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"\n    {t} Stacking OOF (C={META_C}, {len(all_student_oofs[t])} students): {ll:.5f}")
        
        # Per-config average OOF
        for cfg_name in config_names:
            # Each config has N_SEEDS students, but they're interleaved by target
            # Actually all students for a config are consecutive
            cfg_idx = config_names.index(cfg_name)
            start_idx = sum(N_SEEDS for prev_cfg in config_names[:cfg_idx])
            end_idx = start_idx + N_SEEDS
            cfg_students = all_student_oofs[t][start_idx:end_idx]
            if len(cfg_students) == N_SEEDS:
                cfg_stacked = np.column_stack(cfg_students)
                cfg_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
                cfg_meta.fit(cfg_stacked, y)
                cfg_oof = log_loss(y, np.clip(cfg_meta.predict_proba(cfg_stacked)[:, 1], 0.001, 0.999))
                log.info(f"      {cfg_name} only: OOF={cfg_oof:.5f}")
    
    # Compute results
    target_oofs = {}
    for t in TARGETS:
        target_oofs[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V311 RESULTS (4 configs × 15 seeds = 60 students per target)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={target_oofs[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146 AVG OOF: 0.63169")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    # Student OOF stats
    log.info(f"\n  Student OOF stats (sample):")
    for i in range(min(5, len(all_student_oofs['Q1']))):
        s_oofs = [log_loss(train_df[t].values, np.clip(all_student_oofs[t][i], 0.001, 0.999)) for t in TARGETS]
        log.info(f"    Student {i}: avg={np.mean(s_oofs):.5f}")
    
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(len(test_preds[t][0]))])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack(all_student_oofs[t]), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v311_multi_config_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V311',
        'name': 'Multi-Config Ensemble Stacking (4 configs × 15 seeds)',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'n_configs': N_CONFIGS,
        'n_students_per_target': N_SEEDS * N_CONFIGS,
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'use_zscore': True,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v311_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
