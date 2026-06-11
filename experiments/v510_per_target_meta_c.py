#!/usr/bin/env python3
"""
V510 — V308 + Per-Target Meta C + Stronger Regularization for Q

Key findings from V509:
- V308 avg gap = 0.070 (not 0.017 as claimed)
- Q1/Q2/Q3/S2 have large gaps (0.08~0.12)
- S1/S3/S4 have small gaps (0.02~0.04)

Hypothesis: Q targets need stronger meta regularization (higher C? No — 
C=10 is inverse regularization in LR. Higher C = less regularization.
So we need LOWER C for Q targets to prevent meta overfitting on noisy seeds.)

Wait — in LogisticRegression, C is inverse of regularization strength.
C=100 → weak regularization → overfit
C=0.1 → strong regularization → underfit

V505 tried C=5 (weaker than 10) → gap got WORSE (0.070).
Actually V308 has C=10 and V505 has C=5 → V505 gap 0.070 same as V309?

Let me re-read V505: "AVG gap: 0.07030" — same as V509 V308 repro (0.06977).
So C=5 and C=10 give similar gap! C=5 doesn't make it worse relative to C=10.

Actually wait — V505 had 20 seeds, V308/V509 have 15 seeds.
More seeds → more features for meta → easier to overfit.

So the real issue is that with 15 seeds and 282 features, 
the meta learner still overfits. We need STRONGER regularization.

New hypothesis: LOWER C (stronger regularization) for meta learner.
Try C=0.1, C=1.0, C=5.0 per target.

Also try: 
1. Per-target C: Q targets get C=0.1, S targets get C=1.0
2. Extra feature: add seed prediction consensus as meta feature
3. Reduce seeds: try 10 seeds instead of 15
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
# Per-target meta C: Q gets stronger regularization (lower C), S gets normal
META_C_PER_TARGET = {
    'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0,  # Stronger regularization for Q
    'S1': 10.0, 'S2': 10.0, 'S3': 10.0, 'S4': 10.0,  # V308 baseline
}

# Also try with 10 seeds for comparison
N_SEEDS_ALT = 10


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


def run_target(train_df, test_df, target, ranked, n_feat, cfg, meta_c,
               n_seeds, gkf, group):
    sel_cols = ranked[:n_feat]
    y = train_df[target].values.astype(np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
    if len(sel_cols_test) != len(sel_cols):
        sel_cols = sel_cols_test
    
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
    
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=meta_c, max_iter=2000, random_state=SEED)
    meta.fit(stacked, y)
    
    oof_val = meta.predict_proba(stacked)[:, 1]
    meta_ll = log_loss(y, np.clip(oof_val, 0.001, 0.999))
    
    student_lls = [log_loss(y, so) for so in per_seed_oofs]
    avg_student = np.mean(student_lls)
    gap = avg_student - meta_ll
    
    return {
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
    log.info("V510 — V308 + Per-Target Meta C (Q:1.0, S:10.0)")
    log.info("Also test 10-seed variant")
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
    
    log.info(f"Train: {len(train_feat_cols)} features | Test: {len(test_feat_cols)}")
    log.info(f"Per-target meta C: {META_C_PER_TARGET}")
    
    # Feature ranking
    target_ranks = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked = rank_features(train_df, feat_cols_clean, target)
        target_ranks[target] = ranked
    
    # Run per-target
    all_results = {}
    
    for t_idx, target in enumerate(TARGETS):
        log.info(f"\n{'='*60}")
        log.info(f"Target: {target} (cfg={V53_SWEEP[target]['cfg']}, n_feat={V53_SWEEP[target]['n_feat']})")
        ranked = target_ranks[target]
        cfg = CFGS[V53_SWEEP[target]['cfg']]
        meta_c = META_C_PER_TARGET[target]
        n_seeds = N_SEEDS
        
        result = run_target(train_df, test_df, target, ranked, 
                           V53_SWEEP[target]['n_feat'], cfg, meta_c,
                           n_seeds, gkf, group)
        
        status = "✅" if result['gap'] < 0.04 else "❌"
        log.info(f"    C={meta_c}, seeds={n_seeds}: OOF={result['meta_oof']:.5f} gap={result['gap']:.5f} {status}")
        
        all_results[target] = {
            'meta_c': meta_c,
            'n_seeds': n_seeds,
            'result': result,
        }
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V510 RESULTS (Per-Target Meta C)")
    log.info(f"{'='*70}")
    
    avg_oof = 0
    avg_gap = 0
    for t in TARGETS:
        r = all_results[t]
        avg_oof += r['result']['meta_oof']
        avg_gap += r['result']['gap']
        status = "✅" if r['result']['gap'] < 0.04 else "❌"
        log.info(f"  {t}: OOF={r['result']['meta_oof']:.5f} gap={r['result']['gap']:.5f} C={r['meta_c']} seeds={r['n_seeds']} {status}")
    
    avg_oof /= len(TARGETS)
    avg_gap /= len(TARGETS)
    
    log.info(f"\n  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  AVG GAP: {avg_gap:.5f} (V309: 0.070, Δ: {avg_gap-0.070:+.5f})")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        r = all_results[t]
        stacked_test = r['result']['test_preds']
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=r['meta_c'], max_iter=2000, random_state=SEED)
        meta_t.fit(np.column_stack(r['result']['per_seed_oofs']), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v510_per_target_meta_c_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\nSaved submission: {sub_path}")
    
    meta_data = {
        'version': 'V510',
        'name': f'V308 + Per-Target Meta C (Q:1.0, S:10.0)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_gap': round(float(avg_gap), 5),
        'v308_avg_oof': 0.62235,
        'v308_gap': 0.06977,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_gap': round(float(avg_gap - 0.06977), 5),
        'meta_c_per_target': META_C_PER_TARGET,
        'per_target_gap': {t: round(float(all_results[t]['result']['gap']), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v510_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
