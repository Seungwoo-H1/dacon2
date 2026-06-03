"""
V400 — Student Performance Boost: Per-Target Hyperparameter Tuning

Hypothesis: Current approach uses per-target CFG (wide/deep/v48/safety) + fixed n_feat,
but each target's optimal hyperparameters might be different.
Some targets (like S1, S2 with lower student OOF ~0.6) might benefit from
different learning rates / depths than Q targets (~0.7).

Method: Grid search per target:
- Try different n_estimators (200, 400, 600)
- Try different learning_rate (0.01, 0.02, 0.03, 0.05)
- Try different num_leaves (10, 15, 20, 30)
- Try different subsample (0.6, 0.7, 0.8)

For each (cfg, n_feat) combo in V53_SWEEP, do a finer search.

If this can reduce student avg OOF even by 0.02, we might hit OOF 0.53.
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

def build_v329_features(train_df, test_df):
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    date_col = 'sleep_date'
    
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_base) & set(test_base)
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in common_cols:
        vals = train_df[col].fillna(0).values.astype(np.float64)
        mean, std = np.mean(vals), np.std(vals, ddof=0)
        if std < 1e-8: std = 1e-8
        train_df[f'{col}_zscore'] = (vals - mean) / std
        test_df[f'{col}_zscore'] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    clean_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {date_col}
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    for col in clean_base:
        grp = train_df.groupby('subject_id')[col]
        for w in [3, 5]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(train_df.index)
            train_df[f'v329_rmean{w}_{col}'] = rm.values
        for w in [3, 5]:
            rs = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).reindex(train_df.index)
            train_df[f'v329_rstd{w}_{col}'] = rs.fillna(0).values
        for sn, sf in [('min', 'min'), ('max', 'max'), ('median', 'median')]:
            train_df[f'v329_{sn}_{col}'] = grp.transform(sf).values
        for q, qn in [(0.25, 'q25'), (0.75, 'q75')]:
            train_df[f'v329_{qn}_{col}'] = grp.quantile(q).reindex(train_df['subject_id']).values
        smean = grp.transform('mean')
        train_df[f'v329_ratio_{col}'] = train_df[col] / (smean + 1e-8)
        train_df[f'v329_dev_{col}'] = train_df[col] - train_df[col].mean()
        d1 = train_df[col].diff().fillna(0)
        d2 = d1.diff().fillna(0)
        train_df[f'v329_accel_{col}'] = d2.values
    
    for col in clean_base:
        grp = test_df.groupby('subject_id')[col]
        for w in [3, 5]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(test_df.index)
            test_df[f'v329_rmean{w}_{col}'] = rm.values
        for w in [3, 5]:
            rs = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).reindex(test_df.index)
            test_df[f'v329_rstd{w}_{col}'] = rs.fillna(0).values
        for sn, sf in [('min', 'min'), ('max', 'max'), ('median', 'median')]:
            test_df[f'v329_{sn}_{col}'] = grp.transform(sf).values
        for q, qn in [(0.25, 'q25'), (0.75, 'q75')]:
            test_df[f'v329_{qn}_{col}'] = grp.quantile(q).reindex(test_df['subject_id']).values
        smean = grp.transform('mean')
        test_df[f'v329_ratio_{col}'] = test_df[col] / (smean + 1e-8)
        test_df[f'v329_dev_{col}'] = test_df[col] - test_df[col].mean()
        d1 = test_df[col].diff().fillna(0)
        d2 = d1.diff().fillna(0)
        test_df[f'v329_accel_{col}'] = d2.values
    
    for col in clean_base[:50]:
        grp = train_df.groupby('subject_id')[col]
        subj_mean = grp.transform('mean')
        g_mean, g_std = train_df[col].mean(), train_df[col].std()
        if g_std < 1e-8: g_std = 1e-8
        train_df[f'v329_cross_z_{col}'] = (subj_mean - g_mean) / g_std
        grp_t = test_df.groupby('subject_id')[col]
        s_mean = grp_t.transform('mean')
        t_g_mean, t_g_std = test_df[col].mean(), test_df[col].std()
        if t_g_std < 1e-8: t_g_std = 1e-8
        test_df[f'v329_cross_z_{col}'] = (s_mean - t_g_mean) / t_g_std
    
    train_df['dow'] = train_df[date_col].dt.dayofweek
    train_df['dow_sin'] = np.sin(2*np.pi*train_df['dow']/7)
    train_df['dow_cos'] = np.cos(2*np.pi*train_df['dow']/7)
    test_df['dow'] = test_df[date_col].dt.dayofweek
    test_df['dow_sin'] = np.sin(2*np.pi*test_df['dow']/7)
    test_df['dow_cos'] = np.cos(2*np.pi*test_df['dow']/7)
    
    return train_df, test_df


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V400 — Student Performance: Per-Target HPO")
    log.info("Grid search on learning_rate, n_estimators, num_leaves per target")
    log.info("=" * 70)
    
    train_raw = pd.read_parquet(DATA / "features.parquet")
    test_raw = pd.read_parquet(DATA / "test_features.parquet")
    
    v329_train, v329_test = build_v329_features(train_raw.copy(), test_raw.copy())
    
    group = train_raw['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    n_train = len(v329_train)
    n_test = len(v329_test)
    
    v329_feat_cols = get_feature_cols(v329_train)
    
    # V53_SWEEP configs
    SWEEP = {
        'Q1':  {'n_feat': 19, 'base_lr': 0.02, 'base_leaves': 20, 'base_est': 1000, 'cfg': 'deep'},
        'Q2':  {'n_feat': 14, 'base_lr': 0.02, 'base_leaves': 20, 'base_est': 1000, 'cfg': 'deep'},
        'Q3':  {'n_feat': 11, 'base_lr': 0.03, 'base_leaves': 15, 'base_est': 500,  'cfg': 'v48'},
        'S1':  {'n_feat': 21, 'base_lr': 0.05, 'base_leaves': 30, 'base_est': 300,  'cfg': 'wide'},
        'S2':  {'n_feat': 19, 'base_lr': 0.02, 'base_leaves': 20, 'base_est': 1000, 'cfg': 'deep'},
        'S3':  {'n_feat': 23, 'base_lr': 0.02, 'base_leaves': 10, 'base_est': 1000, 'cfg': 'safety'},
        'S4':  {'n_feat': 20, 'base_lr': 0.05, 'base_leaves': 30, 'base_est': 300,  'cfg': 'wide'},
    }
    
    # HPO grid: try a few variants per target
    # Focus: lr × n_estimators (most impactful params)
    HPO_GRID = {
        'Q1':  {'lr': [0.01, 0.015, 0.02, 0.03], 'est': [500, 750, 1000], 'leaves': [15, 20, 25]},
        'Q2':  {'lr': [0.01, 0.015, 0.02, 0.03], 'est': [500, 750, 1000], 'leaves': [15, 20, 25]},
        'Q3':  {'lr': [0.02, 0.03, 0.04, 0.05], 'est': [300, 400, 500],    'leaves': [10, 15, 20]},
        'S1':  {'lr': [0.02, 0.03, 0.05, 0.07], 'est': [200, 300, 400],     'leaves': [20, 30, 40]},
        'S2':  {'lr': [0.01, 0.015, 0.02, 0.03], 'est': [500, 750, 1000],   'leaves': [15, 20, 25]},
        'S3':  {'lr': [0.01, 0.015, 0.02, 0.03], 'est': [500, 750, 1000],   'leaves': [8, 10, 12]},
        'S4':  {'lr': [0.02, 0.03, 0.05, 0.07], 'est': [200, 300, 400],     'leaves': [20, 30, 40]},
    }
    
    log.info("Running HPO on Q1 first to validate approach...")
    
    t = 'Q1'
    y = train_raw[t].values.astype(np.float64)
    feat_cols_clean = remove_leak(v329_feat_cols, t)
    n_feat = SWEEP[t]['n_feat']
    hpo = HPO_GRID[t]
    
    # Track best params per target
    best_params = {}
    best_avg_student_oof = float('inf')
    
    lr_vals = hpo['lr']
    est_vals = hpo['est']
    leaf_vals = hpo['leaves']
    
    for lr in lr_vals:
        for est in est_vals:
            for leaves in leaf_vals:
                # Train 15 seeds with this config, compute avg student OOF
                seed_oofs = []
                
                for si in range(N_SEEDS):
                    seed = SEED + si * 7
                    rng = np.random.RandomState(seed)
                    bag = rng.choice(feat_cols_clean, size=max(int(len(feat_cols_clean)*0.75), n_feat), replace=False)
                    bag_set = set(bag)
                    bag_feats = [f for f in feat_cols_clean if f in bag_set][:n_feat]
                    if len(bag_feats) < n_feat:
                        remaining = [f for f in feat_cols_clean if f not in bag_set][:n_feat - len(bag_feats)]
                        bag_feats.extend(remaining)
                    sel_cols = [c for c in bag_feats if c in v329_train.columns]
                    
                    oof = np.zeros(n_train)
                    
                    for fold, (tr_idx, va_idx) in enumerate(gkf.split(v329_train, y, group)):
                        X_tr = v329_train.iloc[tr_idx][sel_cols].fillna(0).values.astype(np.float64)
                        X_va = v329_train.iloc[va_idx][sel_cols].fillna(0).values.astype(np.float64)
                        y_tr = y[tr_idx]
                        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        params = {
                            'num_leaves': leaves, 'max_depth': -1,
                            'learning_rate': lr, 'n_estimators': est,
                            'subsample': 0.7, 'colsample_bytree': 0.7,
                            'scale_pos_weight': spw, 'random_state': seed,
                            'force_row_wise': True, 'n_jobs': 1, 'verbose': -1,
                            'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15,
                        }
                        sn = [sanitize_col(c) for c in sel_cols]
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                        m = lgb.train(params, ds, num_boost_round=est)
                        oof[va_idx] = m.predict(X_va)
                    
                    oof = np.clip(oof, 0.001, 0.999)
                    seed_oofs.append(log_loss(y, oof))
                
                avg_student_oof = np.mean(seed_oofs)
                
                if avg_student_oof < best_avg_student_oof:
                    best_avg_student_oof = avg_student_oof
                    best_params[t] = {'lr': lr, 'est': est, 'leaves': leaves}
                    log.info(f"  New best {t}: lr={lr} est={est} leaves={leaves} → avg_student_oof={avg_student_oof:.5f}")
    
    # Now run full experiment with best params
    log.info(f"\nBest params found:")
    for t in best_params:
        log.info(f"  {t}: {best_params[t]}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    log.info(f"Q1 student avg OOF baseline (current): ~0.69")
    log.info(f"Q1 student avg OOF best: {best_avg_student_oof:.5f}")
    log.info(f"Improvement: {best_avg_student_oof - 0.69:+.5f}")
    
    # Save meta
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_data = {
        'version': 'V400',
        'name': 'Student Performance: Per-Target HPO',
        'best_params': best_params,
        'student_avg_oof_Q1': round(float(best_avg_student_oof), 5),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v400_hpo_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")


if __name__ == '__main__':
    main()
