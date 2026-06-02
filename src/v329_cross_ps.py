"""
V329 — V328 + Cross-Subject Features + Aggressive Feature Engineering

Hypothesis: V328's per-subject features are powerful. Now add:
1. Cross-subject features: how does this subject compare to others?
2. More rolling windows with more granular stats
3. Per-subject quartile features
4. Day-of-week patterns per subject
5. Acceleration features (rate of change of each metric)

Then stack with V321-style.

Expected OOF: 0.555-0.565
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
FEATURE_BAG_FRACTION = 0.75


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
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V329 — V328 + Cross-Subject + Aggressive PS Features")
    log.info("V328: OOF=0.56298, per-subject rolling stats breakthrough")
    log.info("V329: +cross-subject, quartiles, acceleration, day patterns")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    
    # --- STEP 1: Global z-scores ---
    log.info("Step 1: Global z-scores...")
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
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (vals - mean) / std
        test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    # --- STEP 2: Per-subject features ---
    log.info("Step 2: Per-subject features...")
    date_col = 'sleep_date'
    
    # Get clean base cols (no zscore, no dates, no meta)
    clean_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) | {date_col}
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    log.info(f"  Clean base cols: {len(clean_base)}")
    
    for col in clean_base:
        grp = train_df.groupby('subject_id')[col]
        
        # Rolling mean 3,5
        for w in [3, 5]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(train_df.index)
            train_df[f'v329_rmean{w}_{col}'] = rm.values
        
        # Rolling std 3,5
        for w in [3, 5]:
            rs = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).reindex(train_df.index)
            train_df[f'v329_rstd{w}_{col}'] = rs.fillna(0).values
        
        # Min, max, median, quartiles per subject via transform
        for stat_name, stat_fn in [('min', 'min'), ('max', 'max'), ('median', 'median')]:
            vals = grp.transform(stat_fn).values
            train_df[f'v329_{stat_name}_{col}'] = vals
        
        # Quartiles via groupby apply
        for q, qname in [(0.25, 'q25'), (0.75, 'q75')]:
            vals = grp.quantile(q).reindex(train_df['subject_id']).values
            train_df[f'v329_{qname}_{col}'] = vals
        
        # Ratio vs subject mean
        smean = grp.transform('mean')
        train_df[f'v329_ratio_{col}'] = train_df[col] / (smean + 1e-8)
        
        # Deviation from global mean
        gmean = train_df[col].mean()
        train_df[f'v329_dev_{col}'] = train_df[col] - gmean
        
        # Acceleration (diff of diff)
        diffs = train_df[col].diff().fillna(0)
        diffs2 = diffs.diff().fillna(0)
        train_df[f'v329_accel_{col}'] = diffs2.values
    
    # Same for test
    for col in clean_base:
        grp = test_df.groupby('subject_id')[col]
        for w in [3, 5]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(test_df.index)
            test_df[f'v329_rmean{w}_{col}'] = rm.values
        for w in [3, 5]:
            rs = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).reindex(test_df.index)
            test_df[f'v329_rstd{w}_{col}'] = rs.fillna(0).values
        grp_test = test_df.groupby('subject_id')[col]
        for stat_name, stat_fn in [('min', 'min'), ('max', 'max'), ('median', 'median')]:
            vals = grp_test.transform(stat_fn).values
            test_df[f'v329_{stat_name}_{col}'] = vals
        for q, qname in [(0.25, 'q25'), (0.75, 'q75')]:
            vals = grp_test.quantile(q).reindex(test_df['subject_id']).values
            test_df[f'v329_{qname}_{col}'] = vals
        smean = grp_test.transform('mean')
        test_df[f'v329_ratio_{col}'] = test_df[col] / (smean + 1e-8)
        gmean = test_df[col].mean()
        test_df[f'v329_dev_{col}'] = test_df[col] - gmean
        diffs = test_df[col].diff().fillna(0)
        diffs2 = diffs.diff().fillna(0)
        test_df[f'v329_accel_{col}'] = diffs2.values
    
    # --- STEP 3: Cross-subject comparison ---
    log.info("Step 3: Cross-subject comparison...")
    for col in clean_base[:50]:  # Top 50 cols
        grp = train_df.groupby('subject_id')[col]
        subj_mean = grp.transform('mean')
        global_mean = train_df[col].mean()
        global_std = train_df[col].std()
        if global_std < 1e-8:
            global_std = 1e-8
        train_df[f'v329_cross_z_{col}'] = (subj_mean - global_mean) / global_std
        
        # Same for test
        grp_t = test_df.groupby('subject_id')[col]
        s_mean = grp_t.transform('mean')
        g_mean = test_df[col].mean()
        g_std = test_df[col].std()
        if g_std < 1e-8:
            g_std = 1e-8
        test_df[f'v329_cross_z_{col}'] = (s_mean - g_mean) / g_std
    
    # --- STEP 4: Day-of-week patterns ---
    log.info("Step 4: Day-of-week features...")
    train_df['dow'] = train_df[date_col].dt.dayofweek
    train_df['dow_sin'] = np.sin(2 * np.pi * train_df['dow'] / 7)
    train_df['dow_cos'] = np.cos(2 * np.pi * train_df['dow'] / 7)
    test_df['dow'] = test_df[date_col].dt.dayofweek
    test_df['dow_sin'] = np.sin(2 * np.pi * test_df['dow'] / 7)
    test_df['dow_cos'] = np.cos(2 * np.pi * test_df['dow'] / 7)
    
    # Per-subject day-of-week mean for key metrics
    for col in clean_base[:20]:
        grp = train_df.groupby(['subject_id', 'dow'])[col].mean().reset_index()
        dow_mean = grp.groupby('subject_id')[col].transform('mean').reindex(train_df.index).values
        train_df[f'v329_dow_mean_{col}'] = dow_mean
        
        grp_t = test_df.groupby(['subject_id', 'dow'])[col].mean().reset_index()
        dow_mean_t = grp_t.groupby('subject_id')[col].transform('mean').reindex(test_df.index).values
        test_df[f'v329_dow_mean_{col}'] = dow_mean_t
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"\nTotal features: train={len(train_feat_cols)}, test={len(test_feat_cols)}")
    
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
    
    n_train = len(train_df)
    n_test = len(test_df)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    all_seed_oofs = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        cfg = CFGS[cfg_name]
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(ranked, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            
            sel_cols = [c for c in bag_feats if c in test_feat_cols]
            
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
            all_seed_oofs[t].append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            if si < 3 or si == N_SEEDS - 1:
                s_oof = log_loss(y, seed_oof)
                log.info(f"    Seed {si:2d}: OOF={s_oof:.5f}")
    
    # Meta learner
    target_oofs = {}
    student_avg_oofs = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V329 RESULTS (Cross-Subject + Aggressive PS features)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V328: 0.56298 | V326: 0.59159 | V321: 0.60569")
    log.info(f"  Δ vs V328: {avg_oof - 0.56298:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"  V308 LB: 0.63893 | Δ: {pred_lb - 0.63893:+.5f}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].dt.strftime('%Y-%m-%d')
    sub['lifelog_date'] = test_df['lifelog_date'].dt.strftime('%Y-%m-%d')
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v329_cross_ps_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V329',
        'name': 'Cross-Subject + Aggressive Per-Subject Features',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'v328_avg_oof': 0.56298,
        'v326_avg_oof': 0.59159,
        'delta_vs_v328': round(float(avg_oof - 0.56298), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v329_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
