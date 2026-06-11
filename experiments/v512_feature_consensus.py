#!/usr/bin/env python3
"""
V512 — Feature Consensus Across Seeds

Hypothesis: Features that appear in top-5 importance across ALL 15 seeds
are robustly predictive. Using only these consensus features should:
1. Reduce seed variance (each seed uses same features)
2. Reduce overfitting (only most robust features)
3. Lower the student-meta gap

This is the FIRST fresh hypothesis in 3 hours of autonomous research.
None of the previous experiments implemented proper consensus.
V511 mentioned it but didn't implement.
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
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V512 — Feature Consensus Across 15 Seeds")
    log.info("Hypothesis: Features in top-5 of ALL seeds → robust, low gap")
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
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    consensus_features = {}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        
        # Phase 1: Find consensus features
        # Rank features with each seed, find top-5 overlap
        seed_top5 = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11
            ranked = rank_features(train_df, feat_cols_clean, target, seed=seed)
            seed_top5.append(set(ranked[:5]))
        
        # Consensus = features in top-5 of ALL seeds
        consensus = seed_top5[0]
        for st in seed_top5[1:]:
            consensus = consensus & st
        
        # If consensus < 2 features, relax to top-10
        if len(consensus) < 2:
            seed_top10 = []
            for si in range(N_SEEDS):
                seed = SEED + si * 11
                ranked = rank_features(train_df, feat_cols_clean, target, seed=seed)
                seed_top10.append(set(ranked[:10]))
            consensus = seed_top10[0]
            for st in seed_top10[1:]:
                consensus = consensus & st
        
        consensus_list = sorted(consensus)
        consensus_features[target] = consensus_list
        
        log.info(f"    Consensus features (top-5 all seeds): {len(consensus_list)} features")
        if len(consensus_list) > 0:
            log.info(f"    Features: {consensus_list[:10]}{'...' if len(consensus_list) > 10 else ''}")
        
        # Phase 2: Use consensus features + V53 sweep additional features
        # Get V53 sweep ranked features
        ranked_v53 = rank_features(train_df, feat_cols_clean, target, seed=SEED)
        v53_top_n = ranked_v53[:n_feat]
        
        # Consensus features that are NOT in V53 top-K
        consensus_plus = [f for f in consensus_list if f not in v53_top_n]
        
        # Combine: V53 top-K + consensus not already in top-K
        sel_cols = v53_top_n + consensus_plus
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        log.info(f"    Final selected features: {len(sel_cols)} (V53-K={n_feat} + consensus={len(consensus_plus)})")
        
        cfg = CFGS[cfg_name]
        
        # Level 0: 15 LGBM seeds
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11
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
            
            if si < 3 or si % 5 == 0:
                ll = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d} (s{seed}): OOF={ll:.5f}")
        
        # Level 1: Stack → LR meta
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[target] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[target], 0.001, 0.999))
        log.info(f"    {target} Stacking OOF (C={META_C}): {ll:.5f}")
        
        test_preds[target] = meta.predict_proba(
            np.column_stack([test_preds[target], 
                           np.zeros(N_SEEDS)])
        )[:, 1] if False else np.mean([
            log_loss(y, so) for so in per_seed_oofs  # placeholder
        ], axis=0)
        
        all_seed_oofs[target] = per_seed_oofs
    
    # Compute test predictions properly
    # Use each seed's test pred, average, then meta predict
    log.info("\n  Generating final test predictions...")
    
    # Re-run with proper test prediction collection
    per_seed_test = {t: [] for t in TARGETS}
    for t_idx, target in enumerate(TARGETS):
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        
        ranked_v53 = rank_features(train_df, feat_cols_clean, target, seed=SEED)
        v53_top_n = ranked_v53[:n_feat]
        
        consensus_list = consensus_features[target]
        consensus_plus = [f for f in consensus_list if f not in v53_top_n]
        sel_cols = v53_top_n + consensus_plus
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        
        test_seed_preds = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11
            seed_test = np.zeros(n_test)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            seed_test /= N_FOLDS
            test_seed_preds.append(seed_test)
        
        per_seed_test[target] = test_seed_preds
    
    # Meta predictions on test
    for target in TARGETS:
        stacked_test = np.column_stack(per_seed_test[target])
        meta_test = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        # Need to refit meta on the same features
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        ranked_v53 = rank_features(train_df, feat_cols_clean, target, seed=SEED)
        v53_top_n = ranked_v53[:n_feat]
        consensus_list = consensus_features[target]
        consensus_plus = [f for f in consensus_list if f not in v53_top_n]
        sel_cols = v53_top_n + consensus_plus
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        # Re-collect per-seed OOF for meta
        per_seed_oofs_final = []
        for si in range(N_SEEDS):
            seed = SEED + si * 11
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
            per_seed_oofs_final.append(seed_oof)
        
        stacked_train = np.column_stack(per_seed_oofs_final)
        meta_test = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta_test.fit(stacked_train, y)
        test_preds[target] = meta_test.predict_proba(stacked_test)[:, 1]
    
    # Results
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(per_target_oof.values()))
    
    log.info(f"\n{'='*70}")
    log.info("V512 RESULTS (Feature Consensus)")
    log.info(f"{'='*70}")
    
    avg_gap = 0
    per_target_gap = {}
    for t in TARGETS:
        t_y = train_df[t].values
        t_seeds = all_seed_oofs[t]
        t_student_lls = [log_loss(t_y, so) for so in t_seeds]
        t_meta_ll = per_target_oof[t]
        t_avg_student = np.mean(t_student_lls)
        t_gap = t_avg_student - t_meta_ll
        t_seed_std = np.std(t_student_lls)
        avg_gap += t_gap
        per_target_gap[t] = t_gap
        
        v308_gap = {
            'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
            'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
        }[t]
        
        status = "✅" if t_gap < v308_gap else "❌"
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} seed_std={t_seed_std:.5f} n_consensus={len(consensus_features[t])} vs V308 {v308_gap:.3f} {status}")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308 actual: 0.06977, Δ: {avg_gap-0.06977:+.5f})")
    
    # Target: avg_gap < 0.025
    if avg_gap < 0.025:
        log.info("  🎯 GAP TARGET HIT!")
    else:
        log.info(f"  ⚠️ Gap target not hit (need < 0.025)")
    
    # Generate submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_df = pd.DataFrame({
        'subject_id': test_df['subject_id'].values,
        'sleep_date': test_df['sleep_date'].values,
        'lifelog_date': test_df['lifelog_date'].values,
    })
    for t in TARGETS:
        sub_df[t] = test_preds[t]
    
    sub_path = SUBMIT / f'submission_v512_consensus_{timestamp}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n  ✅ Submission: {sub_path}")
    
    # Save experiment result
    result = {
        'version': 'V512',
        'name': 'Feature Consensus Across Seeds',
        'avg_oof': float(avg_oof),
        'avg_gap': float(avg_gap),
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': float(avg_oof - 0.62235),
        'delta_vs_v308_gap': float(avg_gap - 0.06977),
        'per_target_oof': {k: float(v) for k, v in per_target_oof.items()},
        'per_target_gap': {k: float(v) for k, v in per_target_gap.items()},
        'consensus_features': {k: len(v) for k, v in consensus_features.items()},
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'submission_file': str(sub_path),
        'timestamp': timestamp,
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v512_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"  📝 Result saved: {result_path}")
    
    log.info(f"\n  Total time: {time.time() - t_start:.1f}s")
    return result


if __name__ == '__main__':
    main()
