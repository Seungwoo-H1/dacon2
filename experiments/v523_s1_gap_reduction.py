#!/usr/bin/env python3
"""
V523 — S1 Gap Reduction (correct implementation)

V522 results (2D meta):
- Q1: gap=0.028, Q2: gap=0.050, Q3: gap=0.007
- S1: gap=0.036 (target 0.020), S2: gap=0.023, S3: gap=0.007, S4: gap=0.028
- avg_gap=0.0255

S1 is the only bottleneck. Previous partial V523 showed:
- S1_xgb_strong_n10: avg_gap=0.02594, S1_gap=0.00251 (BEATS V522!)
- S1_xgb_narrow_n5: avg_gap=0.02603, S1_gap=0.00318

This time: Use V522 feature selections for Q1-Q3,S2-S4, only vary S1.
15 seeds per target (same as V522) for fair comparison.
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
import xgboost as xgb

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
    log.info("V523 — S1 GAP REDUCTION (corrected)")
    log.info("Only vary S1, keep Q1-Q3,S2-S4 at V522 best")
    log.info("15 seeds, 5-fold GroupKFold")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Build zscore features (same as V522)
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
    
    v308_gaps = {
        'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
        'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
    }
    
    # V522 fixed configs for non-S1 targets
    FIXED = {
        'Q1':  {'n_feat': 7,  'learner': 'xgb', 'n_est': 600, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        'Q2':  {'n_feat': 14, 'learner': 'xgb', 'n_est': 800, 'md': 5, 'lr': 0.03, 'ss': 0.8, 'cs': 0.7, 'ra': 1.0, 'rl': 3.0, 'mcw': 5},
        'Q3':  {'n_feat': 7,  'learner': 'xgb', 'n_est': 500, 'md': 4, 'lr': 0.05, 'ss': 0.8, 'cs': 0.7, 'ra': 5.0, 'rl': 10.0, 'mcw': 5},
        'S2':  {'n_feat': 7,  'learner': 'xgb', 'n_est': 300, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        'S3':  {'n_feat': 23, 'learner': 'lgbm', 'nl': 10, 'md': 3, 'lr': 0.02, 'n_est': 1000, 'ss': 0.6, 'cs': 0.6, 'ra': 3.0, 'rl': 10.0, 'mcs': 20},
        'S4':  {'n_feat': 20, 'learner': 'lgbm', 'nl': 30, 'md': 3, 'lr': 0.05, 'n_est': 300, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcs': 5},
    }
    
    # S1 candidates
    S1_CANDIDATES = [
        # XGB variants
        {'n_feat': 5,  'learner': 'xgb', 'n_est': 500, 'md': 3, 'lr': 0.03, 'ss': 0.8, 'cs': 0.8, 'ra': 5.0, 'rl': 10.0, 'mcw': 5},
        {'n_feat': 10, 'learner': 'xgb', 'n_est': 500, 'md': 3, 'lr': 0.03, 'ss': 0.8, 'cs': 0.8, 'ra': 5.0, 'rl': 10.0, 'mcw': 5},
        {'n_feat': 15, 'learner': 'xgb', 'n_est': 500, 'md': 3, 'lr': 0.03, 'ss': 0.8, 'cs': 0.8, 'ra': 5.0, 'rl': 10.0, 'mcw': 5},
        {'n_feat': 20, 'learner': 'xgb', 'n_est': 500, 'md': 3, 'lr': 0.03, 'ss': 0.8, 'cs': 0.8, 'ra': 5.0, 'rl': 10.0, 'mcw': 5},
        {'n_feat': 5,  'learner': 'xgb', 'n_est': 600, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        {'n_feat': 10, 'learner': 'xgb', 'n_est': 600, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        {'n_feat': 15, 'learner': 'xgb', 'n_est': 600, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        {'n_feat': 20, 'learner': 'xgb', 'n_est': 600, 'md': 4, 'lr': 0.04, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcw': 3},
        # LGBM variants (original V522)
        {'n_feat': 5,  'learner': 'lgbm', 'nl': 30, 'md': 3, 'lr': 0.05, 'n_est': 300, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcs': 5},
        {'n_feat': 10, 'learner': 'lgbm', 'nl': 30, 'md': 3, 'lr': 0.05, 'n_est': 300, 'ss': 0.8, 'cs': 0.8, 'ra': 2.0, 'rl': 5.0, 'mcs': 5},
    ]
    
    log.info(f"V522 baseline avg_gap=0.02550, S1_gap=0.036")
    log.info(f"Testing {len(S1_CANDIDATES)} S1 configs\n")
    
    results = []
    
    for ci, s1_cfg in enumerate(S1_CANDIDATES):
        s1_n = s1_cfg['n_feat']
        s1_lr = s1_cfg['learner']
        s1_desc = f"S1_{s1_lr}_n{s1_n}"
        
        log.info(f"[{ci+1}/{len(S1_CANDIDATES)}] {s1_desc}")
        
        # Pre-rank features for non-S1 targets
        all_ranked = {}
        for target in ['Q1', 'Q2', 'Q3', 'S2', 'S3', 'S4']:
            feat_cols_clean = remove_leak(train_feat_cols, target)
            all_ranked[target] = rank_features(train_df, feat_cols_clean, target)
        
        # Run all 7 targets
        all_seed_oofs = {t: [] for t in TARGETS}
        
        for target in TARGETS:
            y = train_df[target].values.astype(np.float64)
            
            if target == 'S1':
                feat_cols_clean = remove_leak(train_feat_cols, 'S1')
                sel_cols = all_ranked.get(target, rank_features(train_df, feat_cols_clean, target))[:s1_n]
                cfg = s1_cfg
            else:
                sel_cols = all_ranked[target][:FIXED[target]['n_feat']]
                cfg = FIXED[target]
            
            learner = cfg['learner']
            cfg_raw = {k: v for k, v in cfg.items() if k not in ('n_feat', 'learner')}
            n_est = cfg_raw['n_est']
            
            per_seed_oofs = []
            for si in range(N_SEEDS):
                seed = SEED + si * 11
                seed_oof = np.zeros(n_train)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    if learner == 'xgb':
                        params = {**cfg_raw, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
                        ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=sel_cols)
                        m = xgb.train(params, ds, num_boost_round=n_est)
                        pred_va = m.predict(xgb.DMatrix(X_va, feature_names=sel_cols))
                    else:
                        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        params = {**cfg_raw, 'scale_pos_weight': spw, 'random_state': seed,
                                 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                        sn = [sanitize_col(c) for c in sel_cols]
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                        m = lgb.train(params, ds, num_boost_round=n_est)
                        pred_va = m.predict(X_va)
                    
                    seed_oof[va_idx] = pred_va
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                per_seed_oofs.append(seed_oof)
            
            all_seed_oofs[target] = per_seed_oofs
        
        # Compute 2D meta gap for each target
        avg_gap = 0
        target_gaps = {}
        for t in TARGETS:
            oofs_arr = np.column_stack(all_seed_oofs[t])
            avg_pred = np.mean(oofs_arr, axis=1)
            std_pred = np.std(oofs_arr, axis=1)
            
            meta_2d = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
            meta_2d.fit(np.column_stack([avg_pred, std_pred]), train_df[t].values)
            train_oof = meta_2d.predict_proba(np.column_stack([avg_pred, std_pred]))[:, 1]
            
            meta_ll = log_loss(train_df[t].values, np.clip(train_oof, 0.001, 0.999))
            t_y = train_df[t].values
            t_student_lls = [log_loss(t_y, so) for so in all_seed_oofs[t]]
            avg_student = np.mean(t_student_lls)
            gap = avg_student - meta_ll
            target_gaps[t] = gap
            avg_gap += gap
        
        avg_gap /= len(TARGETS)
        
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        s1_marker = " 🔥" if target_gaps['S1'] < 0.036 else ""
        log.info(f"  avg_gap={avg_gap:.5f}, S1_gap={target_gaps['S1']:.5f} vs308={vs308}/7{s1_marker}")
        for t in TARGETS:
            vs = "✅" if target_gaps[t] < v308_gaps[t] else "❌"
            log.info(f"    {t}: gap={target_gaps[t]:.5f} V308={v308_gaps[t]:.3f} {vs}")
        
        if avg_gap < 0.025:
            log.info(f"  🎯🎯🎯 BELOW 0.025! 🎯🎯🎯")
        
        results.append({
            'key': s1_desc, 'avg_gap': avg_gap, 'target_gaps': target_gaps, 'vs308': vs308
        })
    
    results.sort(key=lambda x: x['avg_gap'])
    
    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS")
    log.info(f"{'='*70}")
    
    for r in results:
        vs308 = sum(1 for t in TARGETS if r['target_gaps'][t] < v308_gaps[t])
        marker = ""
        if r['avg_gap'] < 0.025: marker = " 🎯🎯🎯 BELOW 0.025"
        elif r['avg_gap'] < 0.030: marker = " ⭐"
        elif r['target_gaps']['S1'] < 0.036: marker = " 📉 S1 improved"
        log.info(f"  {r['key']}: avg_gap={r['avg_gap']:.5f}, S1={r['target_gaps']['S1']:.5f}, vs308={vs308}/7{marker}")
    
    best = results[0]
    log.info(f"\n✅ BEST: {best['key']} with avg_gap={best['avg_gap']:.5f}")
    
    if best['avg_gap'] < 0.025:
        log.info("🎯🎯🎯 GAP TARGET HIT! < 0.025! 🎯🎯🎯")
    
    result = {
        'version': 'V523',
        'name': 'S1 gap reduction',
        'results': results,
        'best_key': best['key'],
        'best_gap': float(best['avg_gap']),
        'best_target_gaps': best['target_gaps'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v523_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"📝 Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()
