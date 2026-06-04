"""
V378 — Multi-Task Learning with Cross-Target Label Sharing

Hypothesis: V308 trains each target independently (15 seeds per target).
But Q1, Q2, Q3 are all "sleep quality" measures — they share signal.
S1, S2, S3, S4 are all "sleep metrics" — they share signal.

Instead of training 7 independent models, train a multi-task model that:
1. Shares feature ranking across related targets
2. Uses label smoothing with related targets' distributions
3. Reduces per-target overfitting by cross-target regularization

Approach:
- Group targets: Q-group (Q1,Q2,Q3), S-group (S1,S2,S3,S4)
- For each target, use feature ranking derived from GROUP-level label averaging
  (e.g., for Q1, rank features using (Q1+Q2+Q3)/3 as pseudo-label)
- This gives more stable feature ranking (750 positive samples vs 450)
- Same 15 seeds × GroupKFold 5-fold stacking with LR C=10
- Key: same V308 stacking architecture, only feature ranking differs

Why this should work:
- Feature ranking is done on 450 samples → noisy for correlated targets
- Group-averaged label has ~2x effective samples → more stable ranking
- Same pipeline architecture → same OOF-LB gap behavior as V308
- Only improvement: better feature selection via shared signal

Expected:
- OOF: ~0.618-0.622 (similar to V308, maybe slightly better)
- OOF-LB gap: same as V308 (~0.017) since pipeline architecture unchanged
- Predicted LB: ~0.635-0.639
- Risk: Low (same architecture, only ranking method differs)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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
Q_TARGETS = ['Q1','Q2','Q3']
S_TARGETS = ['S1','S2','S3','S4']
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


def rank_features_v308(feat_df, feat_cols, target, seed=SEED):
    """Standard V308 feature ranking: use target label directly."""
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


def rank_features_multi_task(feat_df, feat_cols, target, seed=SEED):
    """
    Multi-task feature ranking: use group-averaged label as pseudo-target.
    
    For Q1, rank using (Q1 + Q2 + Q3) / 3 as pseudo-label.
    For S1, rank using (S1 + S2 + S3 + S4) / 4 as pseudo-label.
    
    This gives more stable feature importance when targets are correlated.
    """
    if target in Q_TARGETS:
        group_targets = Q_TARGETS
    else:
        group_targets = S_TARGETS
    
    # Average labels across group
    group_labels = np.zeros(len(feat_df))
    for gt in group_targets:
        y_gt = feat_df[gt].values.astype(np.float64)
        group_labels += y_gt
    group_labels /= len(group_targets)
    
    # Round to binary (threshold at 0.5)
    y = (group_labels > 0.5).astype(np.float64)
    
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
    log.info("V378 — Multi-Task Learning with Cross-Target Label Sharing")
    log.info("Hypothesis: Group-averaged feature ranking → more stable selection")
    log.info(f"V308: per-target ranking, OOF=0.62235, LB=0.63893")
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
    
    # Add z-score columns to train
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
    
    # Feature columns
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Storage
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    per_seed_test = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_student_oofs = []
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    # Compare: V308 ranking vs Multi-task ranking for each target
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # V308 ranking
        ranked_v308 = rank_features_v308(train_df, feat_cols_clean, t)
        sel_v308 = ranked_v308[:n_feat]
        
        # Multi-task ranking
        ranked_mt = rank_features_multi_task(train_df, feat_cols_clean, t)
        sel_mt = ranked_mt[:n_feat]
        
        # Compare overlap
        overlap = len(set(sel_v308) & set(sel_mt))
        log.info(f"    V308 features: {len(sel_v308)}, MT features: {len(sel_mt)}")
        log.info(f"    Overlap: {overlap}/{n_feat} ({overlap/n_feat:.0%})")
        
        # Quick check: which ranking gives better OOF on first 5 seeds?
        cfg = CFGS[cfg_name]
        
        oofs_v308 = []
        oofs_mt = []
        for si in range(5):
            seed = SEED + si * 7
            seed_oof_v308 = np.zeros(n_train)
            seed_oof_mt = np.zeros(n_train)
            seed_test_v308 = np.zeros(n_test)
            seed_test_mt = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                # V308
                X_tr = train_df[sel_v308].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_v308].iloc[va_idx].fillna(0).values.astype(np.float64)
                # MT
                X_tr_mt = train_df[sel_mt].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va_mt = train_df[sel_mt].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in (sel_v308 if si < 2 else sel_mt)]
                
                # V308 model
                ds_v308 = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m_v308 = lgb.train(params, ds_v308, num_boost_round=cfg['n_estimators'])
                seed_oof_v308[va_idx] = m_v308.predict(X_va)
                seed_test_v308 += m_v308.predict(test_df[sel_v308].fillna(0).values.astype(np.float64))
                
                # MT model
                ds_mt = lgb.Dataset(X_tr_mt, label=y_tr, feature_name=sn)
                m_mt = lgb.train(params, ds_mt, num_boost_round=cfg['n_estimators'])
                seed_oof_mt[va_idx] = m_mt.predict(X_va_mt)
                seed_test_mt += m_mt.predict(test_df[sel_mt].fillna(0).values.astype(np.float64))
            
            oofs_v308.append(log_loss(y, np.clip(seed_oof_v308, 0.001, 0.999)))
            oofs_mt.append(log_loss(y, np.clip(seed_oof_mt, 0.001, 0.999)))
        
        avg_v308 = np.mean(oofs_v308)
        avg_mt = np.mean(oofs_mt)
        log.info(f"    5-seed avg OOF - V308 ranking: {avg_v308:.5f}, MT ranking: {avg_mt:.5f}")
        log.info(f"    Δ: {avg_mt - avg_v308:+.5f}")
        
        # Use the better ranking
        use_mt = avg_mt < avg_v308
        sel_cols = sel_mt if use_mt else sel_v308
        ranking_used = "multi_task" if use_mt else "v308"
        log.info(f"    -> Using {ranking_used} ranking for all seeds")
        
        # Verify same columns exist in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        
        # Full training with selected ranking
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
            per_seed_test[t][:, si] = seed_test
            
            s_oof = log_loss(y, seed_oof)
            all_student_oofs.append(s_oof)
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
        
        # Level 1: LR meta-learner (same as V308)
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = per_seed_test[t]
        
        student_mean = np.mean(per_seed_oofs)
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        
        meta_oof_pred = meta.predict_proba(stacked_train)[:, 1]
        meta_oof_ll = log_loss(y, np.clip(meta_oof_pred, 0.001, 0.999))
        meta_test_pred = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        train_oof[t] = meta_oof_pred
        test_preds[t] = meta_test_pred
        
        log.info(f"    {t} Meta OOF (C={META_C}, {N_SEEDS} seeds, {ranking_used}): {meta_oof_ll:.5f} "
                 f"(V308: {V308_OOF[t]:.5f}, Δ: {meta_oof_ll-V308_OOF[t]:+.5f})")
        log.info(f"    {t} Student mean OOF: {student_mean:.5f}")
    
    # Compute overall results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) for t in TARGETS])
    student_avg = np.mean(all_student_oofs)
    
    v308_gap = 0.01658
    predicted_lb = avg_oof + v308_gap
    
    log.info(f"\n{'='*70}")
    log.info(f"V378 RESULTS (Multi-Task Feature Ranking)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Student avg OOF: {student_avg:.5f}")
    log.info(f"  Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    beats = predicted_lb < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v378_multi_task_ranking_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V378',
        'name': 'Multi-Task Feature Ranking',
        'avg_oof': round(float(avg_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'student_avg_oof': round(float(student_avg), 5),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v378_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
