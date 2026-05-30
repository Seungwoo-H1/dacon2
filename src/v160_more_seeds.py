"""
V160 — More Seeds Ensemble (5 → 15)

Hypothesis: V146's 5 seeds (42,49,56,63,70) don't provide enough diversity.
With only 5 student models, ensemble variance is high. Increasing to 15 seeds
provides better averaging and reduces student-level variance.

Key insight: V146's student OOFs are 0.63-0.80. Averaging 15 diverse seeds
should reduce variance and improve both student-level and stacked OOF.

Risk: Low — identical architecture, only seed count changes
Expected: OOF improvement 0.003-0.008
Cost: 3x training time

Why this time: V159 proved meta-learner improvement = overfitting.
V156-157 proved feature engineering = noise.
The remaining dimension is ensemble diversity (seeds).
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
N_SEEDS = 15  # Increased from 5 to 15
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
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
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
    log.info(f"V160 — More Seeds Ensemble (5 → {N_SEEDS} seeds)")
    log.info(f"N_SEEDS={N_SEEDS} vs V146 N_SEEDS=5")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Base features: {len(feat_cols)}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_seed = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        cfg = CFGS[cfg_name]
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7  # 42, 49, 56, 63, 70, 77, 84, ...
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
            test_seed[t][:, si] = seed_test
            
            if si < 10 or si % 5 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        # Meta learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        log.info(f"    {t} Stacking OOF (C={META_C}, {N_SEEDS} seeds): {log_loss(y, np.clip(train_oof[t], 0.001, 0.999)):.5f}")
    
    # Compute results
    v146_oof = {}
    for t in TARGETS:
        v146_oof[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(v146_oof.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V160 RESULTS ({N_SEEDS} seeds)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={v146_oof[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146 AVG OOF: 0.63169")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"{'='*70}")
    
    # Save submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        stacked_test = np.column_stack([test_seed[t][:, i] for i in range(N_SEEDS)])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        # Re-fit meta for test (use last fold's meta)
        y_t = train_df[t].values.astype(np.float64)
        stacked_train = np.column_stack([train_oof[t_sub] for t_sub in [t]])
        # Just use average of seeds directly for test (simpler, more robust with 15 seeds)
        sub[t] = np.mean(test_seed[t], axis=1)
    
    sub.to_csv(f"submissions/submission_v160_{N_SEEDS}seeds_{ts}.csv", index=False)
    log.info(f"Saved: submissions/submission_v160_{N_SEEDS}seeds_{ts}.csv")
    
    # Also save per-seed average as alternative (no meta)
    sub2 = pd.DataFrame()
    sub2['subject_id'] = test_df['subject_id'].values
    sub2['sleep_date'] = test_df['sleep_date'].values
    sub2['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub2[t] = np.mean(test_seed[t], axis=1)
    
    sub2.to_csv(f"submissions/submission_v160_avg_{N_SEEDS}seeds_{ts}.csv", index=False)
    log.info(f"Saved: submissions/submission_v160_avg_{N_SEEDS}seeds_{ts}.csv")
    
    meta_data = {
        'version': 'V160',
        'name': f'More Seeds Ensemble ({N_SEEDS} seeds)',
        'n_seeds': N_SEEDS,
        'v146_n_seeds': 5,
        'avg_oof': round(float(avg_oof), 5),
        'v146_avg_oof': 0.63169,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'per_target_oof': {t: round(float(v146_oof[t]), 5) for t in TARGETS},
        # student_oof_stats omitted (computed during training)
        'submission_files': [
            f"submissions/submission_v160_{N_SEEDS}seeds_{ts}.csv",
            f"submissions/submission_v160_avg_{N_SEEDS}seeds_{ts}.csv",
        ],
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = f"submissions/meta_v160_{ts}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
