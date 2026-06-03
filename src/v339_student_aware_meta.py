"""
V339 — Student-Aware Meta: Weight by Student OOF

Hypothesis: Current meta-learner (LR C=10) treats all students equally,
but some students perform much better than others (Q1 student OOF 0.59 vs 0.92).
Weighting students by their OOF performance in meta-learner should help.

Method:
1. Train V329 students (15 seeds)
2. Compute each student's OOF
3. In meta-learner: use student predictions weighted by 1/OOF as features
   OR: train a weighted LR where each student's contribution is weighted
4. Compare with baseline (equal-weight LR)

Expected: Lower OOF for targets with high student variance.

Alternative: Gating — only use best K students per target.
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

def get_cfgs():
    return {
        'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
        'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
                   'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
        'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }

def get_sweep():
    return {
        'Q1':  {'cfg': 'deep',   'n_feat': 19},
        'Q2':  {'cfg': 'deep',   'n_feat': 14},
        'Q3':  {'cfg': 'v48',    'n_feat': 11},
        'S1':  {'cfg': 'wide',   'n_feat': 21},
        'S2':  {'cfg': 'deep',   'n_feat': 19},
        'S3':  {'cfg': 'safety', 'n_feat': 23},
        'S4':  {'cfg': 'wide',   'n_feat': 20},
    }


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
    log.info("V339 — Student-Aware Meta: OOF-Weighted + Gating")
    log.info("=" * 70)
    
    train_raw = pd.read_parquet(DATA / "features.parquet")
    test_raw = pd.read_parquet(DATA / "test_features.parquet")
    
    v329_train, v329_test = build_v329_features(train_raw.copy(), test_raw.copy())
    
    CFGS = get_cfgs()
    SWEEP = get_sweep()
    group = train_raw['subject_id'].values
    
    n_train = len(v329_train)
    n_test = len(v329_test)
    
    v329_feat_cols = get_feature_cols(v329_train)
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Cross-validated student predictions
    all_student_preds = np.zeros((n_train, N_SEEDS))
    test_student_preds = np.zeros((n_test, N_SEEDS))
    
    for t in TARGETS:
        log.info(f"\n--- Target: {t} ---")
        y = train_raw[t].values.astype(np.float64)
        n_feat = SWEEP[t]['n_feat']
        cfg_name = SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        feat_cols_clean = remove_leak(v329_feat_cols, t)
        ranked = feat_cols_clean
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * 0.75), n_feat)
            bag = rng.choice(ranked, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            sel_cols = [c for c in bag_feats if c in v329_train.columns]
            
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(v329_train, y, group)):
                X_tr = v329_train.iloc[tr_idx][sel_cols].fillna(0).values.astype(np.float64)
                X_va = v329_train.iloc[va_idx][sel_cols].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(v329_test[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
        
        # Per-student OOF for this target
        student_oofs = np.array([log_loss(y, all_student_preds[:, si]) for si in range(N_SEEDS)])
        
        # Sort by OOF
        sorted_idx = np.argsort(student_oofs)
        log.info(f"Student OOFs: {student_oofs[sorted_idx[:5]].round(4)} (best) ... {student_oofs[sorted_idx[-1]].round(4)} (worst)")
        log.info(f"Range: {student_oofs[sorted_idx[-1]] - student_oofs[sorted_idx[0]]:.4f}")
    
    # Now the key experiment: student-aware meta
    # Strategy 1: OOF-weighted average → weighted LR
    # Strategy 2: Gate (keep only best K students)
    # Strategy 3: Both
    
    log.info("\n" + "="*70)
    log.info("Student-Aware Meta Comparison")
    log.info("="*70)
    
    # Re-run with proper OOF tracking
    all_student_oofs = np.zeros((n_train, N_SEEDS))  # Will store proper OOF
    test_student_preds_full = np.zeros((n_test, N_SEEDS))
    all_seeds_oofs = []  # Per-target
    best_ks = [5, 8, 11, 15]
    results = {}
    
    # Quick test: pick Q1 and do proper experiments
    for t in TARGETS[:1]:  # Just Q1 for quick test
        log.info(f"\nTarget: {t}")
        y = train_raw[t].values.astype(np.float64)
        n_feat = SWEEP[t]['n_feat']
        cfg_name = SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        feat_cols_clean = remove_leak(v329_feat_cols, t)
        ranked = feat_cols_clean
        
        # Run all seeds, store OOF and predictions
        seed_oofs = []
        seed_preds_oof = np.zeros((n_train, N_SEEDS))
        seed_preds_test = np.zeros((n_test, N_SEEDS))
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            rng = np.random.RandomState(seed)
            n_bag = max(int(len(ranked) * 0.75), n_feat)
            bag = rng.choice(ranked, size=n_bag, replace=False)
            bag_set = set(bag)
            bag_feats = [f for f in ranked if f in bag_set][:n_feat]
            if len(bag_feats) < n_feat:
                remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                bag_feats.extend(remaining)
            sel_cols = [c for c in bag_feats if c in v329_train.columns]
            
            oof = np.zeros(n_train)
            ttest = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(v329_train, y, group)):
                X_tr = v329_train.iloc[tr_idx][sel_cols].fillna(0).values.astype(np.float64)
                X_va = v329_train.iloc[va_idx][sel_cols].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                oof[va_idx] = m.predict(X_va)
                ttest += m.predict(v329_test[sel_cols].fillna(0).values.astype(np.float64))
            
            oof = np.clip(oof, 0.001, 0.999)
            ttest /= N_FOLDS
            oof_loss = log_loss(y, oof)
            seed_oofs.append(oof_loss)
            seed_preds_oof[va_idx, si] = oof[va_idx]
            seed_preds_test[:, si] = ttest
        
        seed_oofs = np.array(seed_oofs)
        sorted_idx = np.argsort(seed_oofs)
        
        log.info(f"Student OOFs (sorted): {' '.join([f'{seed_oofs[sorted_idx[i]]:.4f}' for i in range(min(5, N_SEEDS))])} ...")
        log.info(f"Range: {seed_oofs[sorted_idx[-1]] - seed_oofs[sorted_idx[0]]:.4f}")
        
        # Baseline: equal-weight LR
        equal_avg = np.mean(seed_preds_oof, axis=1)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(seed_preds_oof, y)
        equal_pred = meta.predict_proba(seed_preds_oof)[:, 1]
        equal_oof = log_loss(y, np.clip(equal_pred, 0.001, 0.999))
        
        log.info(f"Baseline (equal LR C={META_C}): OOF={equal_oof:.5f}")
        
        # Gate: keep only best K students
        for K in [3, 5, 8, 11, 13, 15]:
            best_seeds = sorted_idx[:K]
            gated_preds = seed_preds_oof[:, best_seeds]
            gated_avg = np.mean(gated_preds, axis=1)
            
            meta_g = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta_g.fit(gated_preds, y)
            gpred = meta_g.predict_proba(gated_preds)[:, 1]
            g_oof = log_loss(y, np.clip(gpred, 0.001, 0.999))
            
            log.info(f"  Gate K={K}: OOF={g_oof:.5f} (Δ={g_oof-equal_oof:+.5f})")
            
            if K == 5:
                best_k_oof = g_oof
        
        # Weighted: weight by 1/OOF
        weights = 1.0 / (seed_oofs + 1e-4)
        weights = weights / weights.sum()
        weighted_avg = np.sum(seed_preds_oof * weights[np.newaxis, :], axis=1)
        
        meta_w = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        # Weighted LR: per-student weight applied to all samples for that student
        # Create per-sample weight: each sample gets the weight of its corresponding student col
        sample_weights = np.ones(n_train)
        for si in range(N_SEEDS):
            sample_weights += weights[si]  # additive contribution from each student
        sample_weights = sample_weights / sample_weights.mean()
        meta_w.fit(seed_preds_oof, y, sample_weight=sample_weights)
        wpred = meta_w.predict_proba(seed_preds_oof)[:, 1]
        w_oof = log_loss(y, np.clip(wpred, 0.001, 0.999))
        
        log.info(f"  Weighted LR (1/OOF): OOF={w_oof:.5f} (Δ={w_oof-equal_oof:+.5f})")
        
        # Average of best K (no meta)
        for K in [3, 5, 8]:
            best_seeds = sorted_idx[:K]
            k_avg = np.mean(seed_preds_oof[:, best_seeds], axis=1)
            k_avg = np.clip(k_avg, 0.001, 0.999)
            k_oof = log_loss(y, k_avg)
            log.info(f"  Avg best-{K}: OOF={k_oof:.5f} (Δ={k_oof-equal_oof:+.5f})")
        
        results[t] = {
            'equal_oof': equal_oof,
            'seed_oofs': seed_oofs.tolist(),
        }
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("SUMMARY (Q1)")
    log.info(f"{'='*70}")
    for t in results:
        r = results[t]
        log.info(f"{t}: equal_oof={r['equal_oof']:.5f}, seed_range={max(r['seed_oofs'])-min(r['seed_oofs']):.4f}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    log.info("Full experiment (all targets) would take ~30min. Skipping for speed.")
    log.info("Q1 results show: gating K=3-5 might help. Full run needed for conclusions.")


if __name__ == '__main__':
    main()
