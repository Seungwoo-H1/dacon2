#!/usr/bin/env python3
"""
V509 — V308 + Multi-Ranking Consensus

Key finding from V507/V508:
- Q1/Q2/Q3/S2: ANY n_feat → gap > 0.05 (impossible to get < 0.025)
- S1/S3/S4: gap can be < 0.025

V308 uses n_feat=19(Q1), 14(Q2), 11(Q3), 21(S1), 19(S2), 23(S3), 20(S4).
These work in V308 → gap < 0.02. But in V507 sweep, even these n_feat give gap > 0.05.

Why? V308's feature importance ranking uses seed=42 (deterministic).
V507/V508 also use seed=42 for ranking. But the per-seed gap comes from
different seeds using DIFFERENT configs (deep/wide/v48/safety).

Wait — in V308, EACH target has ONE config but 15 seeds. The gap comes from
15 seeds with different random_state but SAME features. So gap < 0.02.

In V507, we also use same config per target. So why gap 0.05+?
Maybe it's the specific ranking + config combo that matters.

New hypothesis: The issue is that V507/V508 rank_features uses the FULL dataset,
while V308 may use a different method. Let me verify V308 uses the same rank_features.

Actually, V507 and V308 use identical ranking logic. The difference must be:
V507 uses train_df after z-score generation, V308 does too.

Wait — in V308's code, the feature ranking is done AFTER z-score generation,
and uses ALL 282 features. Same in V507.

The real difference: V308 uses GroupKFold on the ORIGINAL train_df.
V507 also uses GroupKFold on the ORIGINAL train_df.

So what's different? Let me check if V507 has a bug in how it runs the full pipeline
vs. V308's pipeline.

Actually, the simplest hypothesis: V308's specific n_feat values (19,14,11,21,19,23,20)
are a carefully tuned sweet spot. V507 sweep found that Q1 at n_feat=19 (V308's value)
gave gap=0.098. But V308 claims gap < 0.02 with the same setup.

This is contradictory. Either:
1. V308's gap is actually > 0.02 and was never properly measured
2. V507 has a subtle bug vs V308

Let me recreate V308 EXACTLY and measure its gap properly.
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
    """Rank features by LGBM gain importance — V308 method."""
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
    log.info("V509 — V308 EXACT REPRODUCTION + GAP VERIFICATION")
    log.info("Goal: Verify V308's gap is truly < 0.02 or if it was overstated")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score generation (V308 method)
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
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (rate={train_df[target].mean():.3f})")
        y = train_df[target].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, target)
        n_feat = V53_SWEEP[target]['n_feat']
        cfg_name = V53_SWEEP[target]['cfg']
        
        # Feature ranking (V308 method)
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:n_feat]
        
        # Verify same columns in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {target}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, n_sel: {len(sel_cols)}")
        log.info(f"    Selected features: {sel_cols}")
        
        # Level 0: N_SEEDS LGBM models
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
            
            ll = log_loss(y, seed_oof)
            log.info(f"    Seed {si:2d} (s{seed}, n_feat={len(sel_cols)}): OOF={ll:.5f}")
        
        # Level 1: Stack → LR meta (C=10)
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
    log.info("V509 RESULTS (V308 EXACT REPRODUCTION)")
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
        
        # Per-seed gap
        min_seed = min(t_student_lls)
        max_seed = max(t_student_lls)
        log.info(f"  {t}: OOF={t_meta_ll:.5f} gap={t_gap:.5f} student_range=[{min_seed:.5f},{max_seed:.5f}]")
    
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308 claimed: 0.62235)")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308 claimed: 0.017)")
    log.info(f"{'='*70}")
    
    # Now run the EXACT same pipeline with DIFFERENT features to test if V308 features are special
    # Try: random feature selection with same n_feat → measure gap
    log.info(f"\n{'='*70}")
    log.info("RANDOM FEATURES BENCHMARK (same n_feat, random selection)")
    log.info("If random also gives low gap, the architecture is inherently stable.")
    log.info("If random gives high gap, the specific features matter.")
    
    random_nfeat_results = []
    for trial in range(3):
        rng = np.random.RandomState(trial)
        log.info(f"\n  Random trial {trial}:")
        
        trial_oofs = []
        trial_gaps = []
        
        for t_idx, target in enumerate(TARGETS):
            y = train_df[target].values.astype(np.float64)
            feat_cols_clean = remove_leak(train_feat_cols, target)
            n_feat = V53_SWEEP[target]['n_feat']
            cfg = CFGS[V53_SWEEP[target]['cfg']]
            
            # Random selection
            perm = rng.permutation(len(feat_cols_clean))
            sel_cols = [feat_cols_clean[perm[i]] for i in range(min(n_feat, len(feat_cols_clean)))]
            
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            if len(sel_cols_test) != len(sel_cols):
                sel_cols = sel_cols_test
            
            per_seed_oofs_r = []
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
                per_seed_oofs_r.append(seed_oof)
            
            stacked_r = np.column_stack(per_seed_oofs_r)
            meta_r = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta_r.fit(stacked_r, y)
            
            oof_r = log_loss(y, np.clip(meta_r.predict_proba(stacked_r)[:, 1], 0.001, 0.999))
            student_lls_r = [log_loss(y, so) for so in per_seed_oofs_r]
            avg_student_r = np.mean(student_lls_r)
            gap_r = avg_student_r - oof_r
            
            trial_oofs.append(oof_r)
            trial_gaps.append(gap_r)
            log.info(f"    {target}: OOF={oof_r:.5f} gap={gap_r:.5f}")
        
        avg_oof_r = np.mean(trial_oofs)
        avg_gap_r = np.mean(trial_gaps)
        random_nfeat_results.append((avg_oof_r, avg_gap_r))
        log.info(f"  Random trial {trial}: AVG OOF={avg_oof_r:.5f} AVG GAP={avg_gap_r:.5f}")
    
    log.info(f"\n  Random features AVG gap: {np.mean([r[1] for r in random_nfeat_results]):.5f}")
    log.info(f"  V308 features AVG gap: {avg_gap:.5f}")
    log.info(f"  Gap ratio (random/V308): {np.mean([r[1] for r in random_nfeat_results]) / avg_gap:.2f}x")
    
    log.info(f"\n{'='*70}")
    log.info("CONCLUSION")
    log.info(f"{'='*70}")
    log.info(f"  If random gap ≈ V308 gap: The architecture inherently produces ~{avg_gap:.3f} gap")
    log.info(f"  If random gap >> V308 gap: V308's feature selection is critical")
    log.info(f"{'='*70}")
    
    # Build submission with V308 settings
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
    
    sub_path = SUBMIT / f"submission_v509_v308_repro_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    meta_data = {
        'version': 'V509',
        'name': 'V308 EXACT REPRODUCTION + RANDOM FEATURES BENCHMARK',
        'avg_oof': round(float(avg_oof), 5),
        'avg_gap': round(float(avg_gap), 5),
        'v308_claimed_avg_oof': 0.62235,
        'v308_claimed_gap': 0.01658,
        'random_features_gap': [round(float(r[1]), 5) for r in random_nfeat_results],
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'per_target_oof': {t: round(float(per_target_oof[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v509_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, avg_gap, meta_data


if __name__ == '__main__':
    main()
