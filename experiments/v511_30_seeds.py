#!/usr/bin/env python3
"""
V511 — V308 + 20 Seeds + Higher Meta C + Stratified Features

V510 result: Q targets gap barely changed with C=1 (0.104 vs 0.113 V308).
Meta C tuning has minimal effect on Q gaps.

Root cause analysis from V509:
- Q1 seed range: [0.764, 0.796] → 0.032 spread
- Q3 seed range: [0.727, 0.766] → 0.039 spread  
- S3 seed range: [0.622, 0.629] → 0.007 spread
- Q targets have HIGH seed variance → meta can't fix this

The fundamental issue: Q targets' top features are NOT stable.
Different seeds pick different features from the "flat" importance landscape.
Even with V53_SWEEP (same ranking for all seeds), the LGBM training itself
(various random seeds) produces different predictions.

New hypothesis: The problem is in the LGBM training variance, not the meta layer.
Solutions:
1. Fix seed variance by using the SAME training data split pattern → reduce seed noise
2. Use feature importance CONSENSUS: each seed uses different features but
   we select only features that appear in top-5 of ALL 15 seeds
3. Add more seeds (20→30) → average out noise → student predictions converge → gap ↓
4. Reduce feature dimensionality: use only top-5 features per target (from V509 analysis)

V511 tries approach 3: 30 seeds instead of 15.
More seeds → better average → lower student variance → lower gap.
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

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
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
N_SEEDS = 30  # DOUBLE the seeds!  hypothesis: more seeds → lower variance → gap ↓
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
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V511 — V308 + 30 Seeds (double!) + Consensus Features")
    log.info("Hypothesis: More seeds → lower student variance → gap ↓")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m, s = np.mean(tv), np.std(tv, ddof=0)
        if s < 1e-8: s = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m) / s
        test_df[zc] = (ev - m) / s
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features | Test: {len(test_feat_cols)}")
    log.info(f"Seeds: {N_SEEDS} (V308: 15)")
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    all_selected_features = {t: [] for t in TARGETS}  # Track which features each seed uses
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, n_sel: {len(sel_cols)}")
        
        # Level 0: N_SEEDS LGBM (30 seeds!)
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11  # Different spacing to avoid collisions
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
            test_preds[target][:, si] = seed_test
            
            if si < 5 or si % 5 == 0:
                ll = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={ll:.5f}")
        
        # Level 1: Stack → LR meta
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[target] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[target], 0.001, 0.999))
        log.info(f"    {target} Stacking OOF (C={META_C}, {N_SEEDS} seeds): {ll:.5f}")
        
        all_seed_oofs[target] = per_seed_oofs
    
    # Results
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(per_target_oof.values()))
    
    log.info(f"\n{'='*70}")
    log.info("V511 RESULTS (30 seeds)")
    log.info(f"{'='*70}")
    
    avg_gap = 0
    for t in TARGETS:
        t_y = train_df[t].values
        t_seeds = all_seed_oofs[t]
        t_student_lls = [log_loss(t_y, so) for so in t_seeds]
        t_meta_ll = per_target_oof[t]
        t_avg_student = np.mean(t_student_lls)
        t_gap = t_avg_student - t_meta_ll
        t_seed_std = np.std(t_student_lls)
        avg_gap += t_gap
        
        v308_gap = {
            'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
            'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
        }[t]
        
        status = "✅" if t_gap < v308_gap else "❌"
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} seed_std={t_seed_std:.5f} vs V308 {v308_gap:.3f} {status}")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V309: 0.070, Δ: {avg_gap-0.070:+.5f})")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta_t.fit(np.column_stack(all_seed_oofs[t]), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v511_30seeds_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    meta_data = {
        'version': 'V511',
        'name': f'V308 + {N_SEEDS} Seeds (doubled from 15)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_gap': round(float(avg_gap), 5),
        'n_seeds': N_SEEDS,
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_gap': round(float(avg_gap - 0.06977), 5),
        'per_target_gap': {t: round(float([log_loss(train_df[t].values, so) for so in all_seed_oofs[t]])[0] 
                   - per_target_oof[t], 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v511_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
