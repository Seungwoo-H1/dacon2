#!/usr/bin/env python3
"""
V507 — V308 + n_feat Sweep + Gap-Constrained Selection

Strategy: Keep V308 architecture (15 seeds, C=10, stacking, z-scores) intact.
Sweep n_feat per target to find optimal point where OOF↓ and gap stays < 0.025.

From V308 V53_SWEEP:
  Q1: deep, n_feat=19
  Q2: deep, n_feat=14
  Q3: v48, n_feat=11
  S1: wide, n_feat=21
  S2: deep, n_feat=19
  S3: safety, n_feat=23
  S4: wide, n_feat=20

Hypothesis: current n_feat may be suboptimal. Sweep [8, 12, 16, 20, 24, 28] per target.
Keep same cfg per target (don't swap config).

Gap constraint: if avg_gap > 0.025, reject that n_feat combo.
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0
MAX_FEAT = 28  # sweep up to 28 features per target
FEAT_STEP = 4  # sweep [4, 8, 12, 16, 20, 24, 28]


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


def run_target_full_pipeline(train_df, test_df, target, ranked_cols, n_feat, cfg, 
                              n_seeds=N_SEEDS, meta_c=META_C, gkf=None, group=None):
    """Run full V308 pipeline for one target with given n_feat."""
    sel_cols = ranked_cols[:n_feat]
    y = train_df[target].values.astype(np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Get features present in both train and test
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
    if len(sel_cols_test) != len(sel_cols):
        sel_cols = sel_cols_test
    
    if len(sel_cols) == 0:
        return None
    
    # Level 0: N_SEEDS LGBM
    per_seed_oofs = []
    test_preds_arr = np.zeros((n_test, n_seeds))
    
    for si in range(n_seeds):
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
        test_preds_arr[:, si] = seed_test
    
    # Level 1: Stack → LR meta
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=meta_c, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    
    train_oof_val = meta.predict_proba(stacked)[:, 1]
    meta_ll = log_loss(y, np.clip(train_oof_val, 0.001, 0.999))
    
    # Gap: avg student vs meta
    student_lls = [log_loss(y, so) for so in per_seed_oofs]
    avg_student = np.mean(student_lls)
    gap = avg_student - meta_ll
    
    return {
        'n_feat': n_feat,
        'meta_oof': meta_ll,
        'avg_student': avg_student,
        'gap': gap,
        'per_seed_oofs': per_seed_oofs,
        'test_preds': test_preds_arr,
        'sel_cols': sel_cols,
    }


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V507 — V308 + n_feat Sweep (4~28 step=4)")
    log.info("Strategy: Find optimal n_feat per target where gap < 0.025")
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
    
    log.info(f"Train: {len(train_feat_cols)} features | Test: {len(test_feat_cols)}")
    log.info(f"Feature sweep: n_feat in [4, 8, 12, 16, 20, 24, 28]")
    log.info(f"Per target: cfg fixed, n_feat swept")
    
    # Per-target feature ranking (V308 method)
    target_ranks = {}
    for target in TARGETS:
        log.info(f"\nRanking {target}...")
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked = rank_features(train_df, feat_cols_clean, target)
        target_ranks[target] = ranked
        log.info(f"  {target}: ranked {len(ranked)} features, top-5: {ranked[:5]}")
    
    # Per-target CFG mapping (V308 V53_SWEEP)
    cfg_map = {
        'Q1': 'deep', 'Q2': 'deep', 'Q3': 'v48',
        'S1': 'wide', 'S2': 'deep', 'S3': 'safety', 'S4': 'wide',
    }
    
    sweep_vals = list(range(4, MAX_FEAT + 1, FEAT_STEP))
    
    # Run sweep for each target
    log.info(f"\n{'='*70}")
    log.info("Starting n_feat sweep...")
    
    all_results = {}  # target -> list of {n_feat, oof, gap}
    
    for ti, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (cfg={cfg_map[target]}, sweep {sweep_vals})")
        ranked = target_ranks[target]
        cfg = CFGS[cfg_map[target]]
        
        target_results = []
        for nf in sweep_vals:
            if nf > len(ranked):
                continue
            result = run_target_full_pipeline(
                train_df, test_df, target, ranked, nf, cfg,
                n_seeds=N_SEEDS, meta_c=META_C, gkf=gkf, group=group
            )
            if result:
                target_results.append(result)
                status = "✅" if result['gap'] < 0.025 else "❌"
                log.info(f"    n_feat={nf:2d}: meta_oof={result['meta_oof']:.5f} student={result['avg_student']:.5f} gap={result['gap']:.5f} {status}")
        
        all_results[target] = target_results
    
    # Find best per-target (min oof where gap < 0.025, else min gap)
    log.info(f"\n{'='*70}")
    log.info("BEST PER TARGET (gap < 0.025):")
    best_n_feat = {}
    best_oof_by_nf = {}  # nf -> list of (target, oof, gap)
    
    for target in TARGETS:
        tr = all_results[target]
        if not tr:
            continue
        
        # Filter: gap < 0.025
        valid = [r for r in tr if r['gap'] < 0.025]
        if valid:
            best = min(valid, key=lambda r: r['meta_oof'])
        else:
            # No valid: pick lowest gap
            best = min(tr, key=lambda r: r['gap'])
        
        best_n_feat[target] = best['n_feat']
        log.info(f"  {target}: n_feat={best['n_feat']}, OOF={best['meta_oof']:.5f}, gap={best['gap']:.5f}")
    
    # Run full pipeline with best n_feat per target
    log.info(f"\n{'='*70}")
    log.info(f"Running FINAL pipeline with best n_feat per target...")
    
    final_train_oof = {}
    final_test_preds = {}
    final_seed_oofs = {}
    final_sel_cols = {}
    
    for target in TARGETS:
        nf = best_n_feat.get(target, 16)
        ranked = target_ranks[target]
        cfg = CFGS[cfg_map[target]]
        
        result = run_target_full_pipeline(
            train_df, test_df, target, ranked, nf, cfg,
            n_seeds=N_SEEDS, meta_c=META_C, gkf=gkf, group=group
        )
        if result:
            y = train_df[target].values.astype(np.float64)
            final_train_oof[target] = result['meta_oof']
            final_test_preds[target] = result['test_preds']
            final_seed_oofs[target] = result['per_seed_oofs']
            final_sel_cols[target] = result['sel_cols']
    
    # Recompute properly
    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS:")
    
    per_target_oof = {}
    per_target_gap = {}
    for target in TARGETS:
        if target not in final_train_oof:
            continue
        y = train_df[target].values.astype(np.float64)
        # Get the OOF from the result (meta_ll)
        # We need to re-extract... let's just use what we have
        pass
    
    # Simpler: just use the best results from sweep
    avg_oof = 0
    avg_gap = 0
    n_valid = 0
    
    for target in TARGETS:
        tr = all_results.get(target, [])
        if not tr:
            continue
        
        # Pick best: valid (gap<0.025) min oof, or if none valid min gap
        valid = [r for r in tr if r['gap'] < 0.025]
        if valid:
            best = min(valid, key=lambda r: r['meta_oof'])
        else:
            best = min(tr, key=lambda r: r['gap'])
        
        nf = best['n_feat']
        meta_oof = best['meta_oof']
        gap = best['gap']
        avg_oof += meta_oof
        avg_gap += gap
        n_valid += 1
        
        status = "✅" if gap < 0.025 else "❌"
        log.info(f"  {target}: n_feat={nf} meta={meta_oof:.5f} gap={gap:.5f} {status}")
    
    avg_oof /= n_valid
    avg_gap /= n_valid
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V308: 0.017)")
    
    # Build submission with best-per-target
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_df_copy = test_df.copy()
    sub = pd.DataFrame()
    sub['subject_id'] = test_df_copy['subject_id'].values
    sub['sleep_date'] = test_df_copy['sleep_date'].values
    sub['lifelog_date'] = test_df_copy['lifelog_date'].values
    
    for target in TARGETS:
        tr = all_results.get(target, [])
        if not tr:
            continue
        valid = [r for r in tr if r['gap'] < 0.025]
        if valid:
            best = min(valid, key=lambda r: r['meta_oof'])
        else:
            best = min(tr, key=lambda r: r['gap'])
        
        y = train_df[target].values.astype(np.float64)
        stacked_test = best['test_preds']
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack(best['per_seed_oofs']), y)
        sub[target] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub_path = SUBMIT / f"submission_v507_nfeat_sweep_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V507',
        'name': f'V308 + n_feat Sweep (4-{MAX_FEAT}, step={FEAT_STEP})',
        'avg_oof': round(float(avg_oof), 5),
        'avg_gap': round(float(avg_gap), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'v308_gap': 0.01658,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'per_target_best': {
            t: {
                'n_feat': min(
                    [r['n_feat'] for r in tr if r['gap'] < 0.025] or [min(tr, key=lambda r: r['gap'])['n_feat']],
                    key=lambda nf: min(
                        [r['meta_oof'] for r in tr if r['gap'] < 0.025 and r['n_feat'] == nf] or 
                        [min(tr, key=lambda r: r['gap'])['meta_oof']]
                    )
                ) if tr else 16,
                'oof': min(
                    [r['meta_oof'] for r in tr if r['gap'] < 0.025] or [min(tr, key=lambda r: r['gap'])['meta_oof']],
                ),
                'gap': min(
                    [r['gap'] for r in tr if r['gap'] < 0.025] or [min(tr, key=lambda r: r['gap'])['gap']],
                ),
            } for t, tr in all_results.items()
        },
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v507_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
