"""
V328 — V326 Enhanced: More Per-Subject Z-Scores + Interaction Tweaks

Hypothesis: V326 worked (OOF 0.592) thanks to per-subject z-scores.
But only 151 per-subject z-scores were added. With 443 total features,
we can add more per-subject features:

1. Per-subject rolling stats (mean over last N days, std over last N)
2. Per-subject trend features (slope over time)
3. Per-subject min/max/median (not just mean)
4. Domain-specific per-subject ratios
5. Cross-domain per-subject z-score interactions

Then run V321 stacking on the enriched feature set.

Expected OOF: 0.585-0.590 (more per-subject signal → better stacking)
Risk: MEDIUM (more features → more noise possible)
Cost: ~60s
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


def add_v328_features(df, is_train=True):
    """Add enhanced per-subject features on top of V326's features."""
    log.info("Adding V328 enhanced features...")
    df = df.copy()
    new_cols = []
    
    # Parse dates
    if 'sleep_date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['sleep_date']):
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    
    date_col = 'sleep_date' if 'sleep_date' in df.columns else 'date'
    
    # Base numeric columns (exclude already-added V326 features)
    base_cols = [c for c in df.columns if c not in META_COLS | set(TARGETS) | {date_col}
                 and not c.endswith('_zscore') and not c.startswith('roll_')
                 and not c.startswith('ps_zscore_') and not c.startswith('hr_')
                 and not c.startswith('light_') and not c.startswith('gps_')
                 and not c.startswith('wifi_') and not c.startswith('step_')
                 and not c.startswith('walk_') and not c.startswith('activity_')
                 and not c.startswith('usage_') and not c.startswith('total_')
                 and not c.endswith('_interaction')
                 and np.issubdtype(df[c].dtype, np.number)]
    
    log.info(f"  Base cols for per-subject: {len(base_cols)}")
    
    # 1. Per-subject rolling mean (3, 5, 10 day) - for key columns only
    for col in base_cols[:40]:  # Top 40 to limit
        for window in [3, 5]:
            rolled = df.groupby('subject_id')[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            rolled = rolled.reindex(df.index)
            cn = f'v328_rmean_{window}_{col}'
            df[cn] = rolled.values
            new_cols.append(cn)
    
    # 2. Per-subject rolling std
    for col in base_cols[:40]:
        for window in [3, 5]:
            rolled = df.groupby('subject_id')[col].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
            rolled = rolled.reindex(df.index)
            cn = f'v328_rstd_{window}_{col}'
            df[cn] = rolled.fillna(0).values
            new_cols.append(cn)
    
    # 3. Per-subject min/max/median
    for col in base_cols[:40]:
        for stat in ['min', 'max', 'median']:
            grouped = df.groupby('subject_id')[col]
            if stat == 'min':
                vals = grouped.transform('min')
            elif stat == 'max':
                vals = grouped.transform('max')
            else:
                vals = grouped.transform('median')
            cn = f'v328_{stat}_{col}'
            df[cn] = vals.values
            new_cols.append(cn)
    
    # 4. Per-subject ratio: value / subject_mean
    for col in base_cols[:40]:
        subj_mean = df.groupby('subject_id')[col].transform('mean')
        cn = f'v328_ratio_{col}'
        df[cn] = df[col] / (subj_mean + 1e-8)
        new_cols.append(cn)
    
    # 5. Per-subject deviation from global mean
    for col in base_cols[:40]:
        global_mean = df[col].mean()
        cn = f'v328_devi_{col}'
        df[cn] = df[col] - global_mean
        new_cols.append(cn)
    
    # 6. Per-subject activity ratio (sum of all motion features / total)
    motion_cols = [c for c in base_cols if any(x in c for x in ['pedo', 'activity', 'hr', 'light'])]
    if motion_cols:
        motion_sum = df[motion_cols].fillna(0).sum(axis=1)
        df['v328_motion_total'] = motion_sum
        new_cols.append('v328_motion_total')
    
    log.info(f"  Added {len(new_cols)} V328 features")
    return df, new_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V328 — V326 Enhanced: More Per-Subject Features")
    log.info("V326: OOF=0.59159, 443 features")
    log.info("V328: +per-subject rolling stats, min/max/median, ratios")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Add global z-scores (like V326)
    log.info("Generating global z-scores...")
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
    
    # Add interaction features (like V326)
    train_df, _ = add_v328_features(train_df)
    test_df, _ = add_v328_features(test_df)
    
    # Add per-subject z-scores (like V326)
    log.info("Generating per-subject z-scores...")
    for col in train_base:
        def calc_zscore(group):
            mean = group.mean()
            std = group.std(ddof=0)
            if std < 1e-8:
                std = 1e-8
            return (group - mean) / std
        
        train_df = train_df.copy()
        train_df[f'ps_zscore_{col}'] = train_df.groupby('subject_id')[col].transform(calc_zscore).values
        test_df = test_df.copy()
        test_df[f'ps_zscore_{col}'] = test_df.groupby('subject_id')[col].transform(calc_zscore).values
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"\nTotal features: train={len(train_feat_cols)}, test={len(test_feat_cols)}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
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
        candidate_feats = ranked
        
        cfg = CFGS[cfg_name]
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(candidate_feats) * FEATURE_BAG_FRACTION), n_feat)
            bag = rng.choice(candidate_feats, size=n_bag, replace=False)
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
    
    # LR meta-learner
    target_oofs = {}
    student_avg_oofs = {}
    meta_weights_info = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oofs[t] = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg_oofs[t] = np.mean([log_loss(y, p) for p in all_seed_oofs[t]])
        meta_weights_info[t] = meta.coef_[0]
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V328 RESULTS (V326 enhanced + more per-subject features)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        gap = student_avg_oofs[t] - target_oofs[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (student={student_avg_oofs[t]:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V326: 0.59159 | V321: 0.60569 | V312: 0.61448")
    log.info(f"  Δ vs V326: {avg_oof - 0.59159:+.5f}")
    log.info(f"  Δ vs V321: {avg_oof - 0.60569:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"  V308 LB: 0.63893 | Δ: {pred_lb - 0.63893:+.5f}")
    log.info(f"  Target LB: 0.500 (OOF <0.481 needed)")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        oof_matrix = np.column_stack(all_seed_oofs[t])
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(oof_matrix, y)
        sub[t] = meta_t.predict_proba(test_preds[t])[:, 1]
    
    sub_path = SUBMIT / f"submission_v328_enhanced_ps_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V328',
        'name': 'V326 Enhanced + More Per-Subject Features',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'v326_avg_oof': 0.59159,
        'v321_avg_oof': 0.60569,
        'v312_avg_oof': 0.61448,
        'delta_vs_v326': round(float(avg_oof - 0.59159), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'student_oof_avg': {t: round(float(student_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v328_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
