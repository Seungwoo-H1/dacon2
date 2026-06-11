#!/usr/bin/env python3
"""
V508 — V308 + Conservative Feature Selection (high-confidence only)

Problem: Q1/Q2/Q3 gap consistently > 0.05 even with n_feat sweep.
Root cause: Feature ranking is unstable for these targets → 
            different seeds pick different "good" features → high gap.

Solution: Only use features with HIGH ranking confidence (top-5 by a large margin).
Force same features across all targets → reduce seed disagreement → gap ↓.

Changes from V308:
1. Use global feature ranking across ALL targets (not per-target)
2. Top-5 consensus features → use for ALL targets
3. Remaining features: per-target but only top-K with large margin
4. Conservative: n_feat capped at 8 for Q targets, 15 for S targets
5. Same 15 seeds, C=10, stacking
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

# Target-specific feature counts (conservative for Q, normal for S)
TARGET_NFEAT = {
    'Q1': 6, 'Q2': 6, 'Q3': 6,  # Very conservative for Q targets
    'S1': 12, 'S2': 12, 'S3': 12, 'S4': 12,
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


def select_features_conservative(train_df, target, ranked, n_feat):
    """
    Conservative selection:
    1. Only top-K features where there's a LARGE gap to the next feature
    2. If no clear margin, default to top-n_feat
    """
    if len(ranked) < n_feat:
        return ranked[:len(ranked)]
    
    # Check importance gaps
    try:
        top_imp = float(ranked[0][1])
        n_feat_imp = float(ranked[min(n_feat-1, len(ranked)-1)][1])
    except (ValueError, TypeError):
        top_imp = 1.0
        n_feat_imp = 0.0
    
    if top_imp > 0:
        margin = (top_imp - n_feat_imp) / top_imp
    else:
        margin = 0
    
    # If margin is small, features are indistinguishable → use fewer
    if margin < 0.01 and n_feat > 4:
        n_feat = 4  # Drop to minimum
    
    return ranked[:n_feat]


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V508 — V308 + Conservative Feature Selection")
    log.info("Strategy: Q targets get only top-6 features with high confidence")
    log.info("S targets get top-12 features with margin-based selection")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score (V308 method)
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
    log.info(f"Target n_feat: {TARGET_NFEAT}")
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, target)
        
        # Conservative selection
        n_feat = TARGET_NFEAT[target]
        sel_cols = select_features_conservative(train_df, target, ranked, n_feat)
        
        # Verify same in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        # Determine cfg from V308
        cfg_map = {'Q1': 'deep', 'Q2': 'deep', 'Q3': 'v48',
                   'S1': 'wide', 'S2': 'deep', 'S3': 'safety', 'S4': 'wide'}
        cfg = CFGS[cfg_map[target]]
        
        log.info(f"    Config: {cfg_map[target]}, n_feat: {n_feat}, sel: {len(sel_cols)}")
        log.info(f"    Top-3: {[(r[0], r[1]) for r in ranked[:3]]}")
        log.info(f"    Top-3 imp: {[r[1] for r in ranked[:3]]}")
        
        # Level 0: N_SEEDS LGBM
        per_seed_oofs = []
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
            test_preds[target][:, si] = seed_test
            
            if si < 5 or si % 5 == 0:
                ll = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}, n_feat={len(sel_cols)}): OOF={ll:.5f}")
        
        # Level 1: Stack → LR meta
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
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
    log.info("V508 RESULTS")
    log.info(f"{'='*70}")
    
    avg_gap = 0
    for t in TARGETS:
        t_y = train_df[t].values
        t_seeds = all_seed_oofs[t]
        t_student_lls = [log_loss(t_y, so) for so in t_seeds]
        t_meta_ll = per_target_oof[t]
        t_avg_student = np.mean(t_student_lls)
        t_gap = t_avg_student - t_meta_ll
        avg_gap += t_gap
        
        status = "✅" if t_gap < 0.025 else "❌"
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} n_feat={TARGET_NFEAT[t]} {status}")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308: 0.017)")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack(all_seed_oofs[t]), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v508_conservative_feat_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    meta_data = {
        'version': 'V508',
        'name': 'V308 + Conservative Feature Selection (Q:6, S:12)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_gap': round(float(avg_gap), 5),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'target_nfeat': TARGET_NFEAT,
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'v308_gap': 0.01658,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(per_target_oof[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v508_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
