"""
V351 — Per-Target Feature Count Optimization

V339's feature counts are fixed per target (11-23).
Maybe some targets need MORE features (Q1: 19→30+), some need FEWER.

Hypothesis: Q targets need more features to capture complexity.
S targets are well-covered at 15-23.
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

# Same configs but with more features for Q targets
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

# V339 baseline feature counts
V339_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

# V351: More features for Q targets, same for S targets
V351_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 30},  # 19 → 30
    'Q2':  {'cfg': 'deep',   'n_feat': 22},  # 14 → 22
    'Q3':  {'cfg': 'v48',    'n_feat': 18},  # 11 → 18
    'S1':  {'cfg': 'wide',   'n_feat': 21},  # same
    'S2':  {'cfg': 'deep',   'n_feat': 19},  # same
    'S3':  {'cfg': 'safety', 'n_feat': 23},  # same
    'S4':  {'cfg': 'wide',   'n_feat': 20},  # same
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    exclude = set(META_COLS) | set(TARGETS)
    return [c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


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
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V351 — Per-Target Feature Count Optimization")
    log.info("Q targets: more features (30/22/18), S targets: same")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    group = train_df['subject_id'].values
    
    all_oofs = {}
    all_student_oofs = {}
    all_preds = {}
    
    # Run with V351 settings
    log.info(f"\nRunning V351 pipeline...")
    for t in TARGETS:
        log.info(f"\n  Target: {t}, config: {V351_SWEEP[t]['cfg']}, n_feat: {V351_SWEEP[t]['n_feat']}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(get_feature_cols(train_df), t)
        n_feat = V351_SWEEP[t]['n_feat']
        cfg_name = V351_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in get_feature_cols(test_df)]
        
        if len(sel_cols_test) != len(sel_cols):
            sel_cols_test = [c for c in sel_cols if c in test_df.columns]
        
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        test_preds_arr = np.zeros((n_test, N_SEEDS))
        
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
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds_arr[:, si] = seed_test
        
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        oof_ll = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
        all_oofs[t] = oof_ll
        
        student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        test_stacked = np.column_stack([test_preds_arr[:, i] for i in range(N_SEEDS)])
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        all_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        # Compare with V339
        v339_oofs = {
            'Q1': 0.63869, 'Q2': 0.61000, 'Q3': 0.62756,
            'S1': 0.58136, 'S2': 0.58955, 'S3': 0.60771, 'S4': 0.63217
        }
        v339_avg = 0.61244
        v308_avg = 0.62235
        
        log.info(f"    {t}: student={student_ll:.5f}, meta={oof_ll:.5f} (ΔV339: {oof_ll - v339_oofs[t]:+.5f})")
    
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    v339_avg = 0.61244
    
    log.info(f"\n{'='*70}")
    log.info(f"V351 RESULTS (Per-Target Feature Count Optimization)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'n_feat':>8} {'Student':>10} {'Meta':>10} {'ΔV308':>8} {'ΔV339':>8}")
    log.info(f"{'-'*55}")
    for t in TARGETS:
        v339_oofs = {
            'Q1': 0.63869, 'Q2': 0.61000, 'Q3': 0.62756,
            'S1': 0.58136, 'S2': 0.58955, 'S3': 0.60771, 'S4': 0.63217
        }
        log.info(f"{t:<6} {V351_SWEEP[t]['n_feat']:>8} {all_student_oofs[t]:>10.5f} {all_oofs[t]:>10.5f} {all_oofs[t]-v308_avg:>+8.5f} {all_oofs[t]-v339_oofs[t]:>+8.5f}")
    log.info(f"{'-'*55}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  Δ vs V339:       {avg_oof - v339_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_preds[t]
    
    sub_path = SUBMIT / f"submission_v351_featcount_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V351',
        'name': 'Per-Target Feature Count Optimization',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'v351_sweep': {t: V351_SWEEP[t]['n_feat'] for t in TARGETS},
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v339': round(float(avg_oof - v339_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v351_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
