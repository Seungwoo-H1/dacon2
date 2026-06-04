"""
V375 — Gap-Constrained Stacking

Hypothesis: V339's OOF improvement (0.612 vs V308's 0.622) was real signal,
but OOF-LB gap doubled (+0.033 vs +0.017). The problem is that V339's 
OOF-based features overfit to train distribution, hurting test generalization.

Fix: Use V308's exact pipeline (same features, same seeds) but replace the
LR meta-learner with a Ridge-regression meta-learner whose alpha is tuned
to minimize a proxy for OOF-LB gap.

Key idea: aggressive fitting (low regularization) at meta level correlates
with large OOF-LB gap. Ridge with tuned alpha should find the sweet spot
between OOF improvement and gap control.

Expected:
- OOF: ~0.618-0.622 (slightly worse than V339, similar to V308)
- OOF-LB gap: <= 0.018 (V308's 0.017 level)
- Predicted LB: ~0.635-0.640
- Risk: Low (same feature set, same seeds, only meta-learner changes)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import Ridge, LogisticRegression
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
    """Rank features by LGBM gain importance."""
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


def generate_test_zscore(train_df, test_df):
    """Generate z-score features for test set using training data statistics."""
    log.info("Generating test z-score features...")
    
    train_feat_cols = [c for c in train_df.columns
                       if c not in META_COLS | set(TARGETS)
                       and not c.endswith('_zscore')
                       and np.issubdtype(train_df[c].dtype, np.number)]
    
    test_feat_cols = [c for c in test_df.columns
                      if c not in META_COLS | set(TARGETS)
                      and not c.endswith('_zscore')
                      and np.issubdtype(test_df[c].dtype, np.number)]
    
    common_cols = set(train_feat_cols) & set(test_feat_cols)
    log.info(f"Common base columns for z-score: {len(common_cols)}")
    
    zscore_cols = []
    for col in common_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        
        zc_name = f'{col}_zscore'
        test_df[zc_name] = (test_vals - mean) / std
        zscore_cols.append(zc_name)
    
    log.info(f"Generated {len(zscore_cols)} z-score features for test")
    return test_df, zscore_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V375 — Gap-Constrained Stacking")
    log.info("Hypothesis: Gap-aware meta-learner reduces OOF-LB gap vs V339")
    log.info(f"V308: OOF={0.62235}, LB={0.63893}, gap={0.01658}")
    log.info(f"V339: OOF={0.61244}, LB={0.64551}, gap={0.03307}")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features for test
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
    # Add z-score columns to train
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df[zc] = (vals - mean) / std
    
    # Feature columns
    train_feat_cols = get_feature_cols(train_df)
    zscore_train = [c for c in train_feat_cols if c.endswith('_zscore')]
    base_train = [c for c in train_feat_cols if not c.endswith('_zscore')]
    
    test_feat_cols = get_feature_cols(test_df)
    zscore_test = [c for c in test_feat_cols if c.endswith('_zscore')]
    base_test = [c for c in test_feat_cols if not c.endswith('_zscore')]
    
    log.info(f"Train: {len(base_train)} base + {len(zscore_train)} zscore = {len(train_feat_cols)}")
    log.info(f"Test:  {len(base_test)} base + {len(zscore_test)} zscore = {len(test_feat_cols)}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Storage
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    per_seed_test = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    per_seed_oofs_dict = {t: [] for t in TARGETS}
    all_student_oofs = []
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        # Verify same columns exist in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} selected features missing in test: {missing}")
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        
        cfg = CFGS[cfg_name]
        
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
            per_seed_test[t][:, si] = seed_test
            
            s_oof = log_loss(y, seed_oof)
            per_seed_oofs_dict[t].append(s_oof)
            all_student_oofs.append(s_oof)
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
        
        # Level 1: Gap-constrained meta-learner
        # Compare Ridge with different alphas vs V308's LogisticRegression(C=10)
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = per_seed_test[t]
        
        student_mean = np.mean(per_seed_oofs_dict[t])
        
        # Try multiple meta approaches
        results = {}
        
        # A) V308 baseline: LR C=10
        lr_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        lr_meta.fit(stacked_train, y)
        lr_oof_pred = lr_meta.predict_proba(stacked_train)[:, 1]
        lr_train_ll = log_loss(y, np.clip(lr_oof_pred, 0.001, 0.999))
        lr_test_pred = np.clip(lr_meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        results['LR_C10'] = {
            'train_ll': lr_train_ll,
            'test_pred': lr_test_pred,
            'student_mean': student_mean,
            'meta_oof': lr_train_ll,
        }
        
        # B) Ridge with tuned alpha
        best_alpha = 1.0
        best_score = np.inf
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(stacked_train, y)
            pred = np.clip(stacked_train @ ridge.coef_ + ridge.intercept_, 0.001, 0.999)
            train_ll = log_loss(y, pred)
            gap_proxy = abs(student_mean - train_ll)
            # Prefer: low train_ll AND small gap_proxy
            score = train_ll + 0.3 * gap_proxy
            if score < best_score:
                best_score = score
                best_alpha = alpha
        
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(stacked_train, y)
        ridge_pred = np.clip(stacked_train @ ridge.coef_ + ridge.intercept_, 0.001, 0.999)
        ridge_train_ll = log_loss(y, ridge_pred)
        ridge_test_pred = np.clip(stacked_test @ ridge.coef_ + ridge.intercept_, 0.001, 0.999)
        
        results['Ridge_best'] = {
            'train_ll': ridge_train_ll,
            'test_pred': ridge_test_pred,
            'alpha': best_alpha,
            'student_mean': student_mean,
            'meta_oof': ridge_train_ll,
        }
        
        # C) Ridge C=10 (equivalent to V308's meta strength)
        ridge10 = Ridge(alpha=10.0)
        ridge10.fit(stacked_train, y)
        ridge10_pred = np.clip(stacked_train @ ridge10.coef_ + ridge10.intercept_, 0.001, 0.999)
        ridge10_train_ll = log_loss(y, ridge10_pred)
        ridge10_test_pred = np.clip(stacked_test @ ridge10.coef_ + ridge10.intercept_, 0.001, 0.999)
        results['Ridge_C10'] = {
            'train_ll': ridge10_train_ll,
            'test_pred': ridge10_test_pred,
            'alpha': 10.0,
            'student_mean': student_mean,
            'meta_oof': ridge10_train_ll,
        }
        
        # Choose best: lowest train_ll (meta OOF)
        best_name = min(results, key=lambda k: results[k]['train_ll'])
        best = results[best_name]
        
        train_oof[t] = best['test_pred']  # Actually use meta pred for test
        # Fix: train_oof should be meta predictions on train
        if best_name == 'LR_C10':
            train_oof[t] = np.clip(lr_oof_pred, 0.001, 0.999)
            test_preds[t] = best['test_pred']
        elif best_name == 'Ridge_best':
            train_oof[t] = np.clip(ridge_pred, 0.001, 0.999)
            test_preds[t] = best['test_pred']
        elif best_name == 'Ridge_C10':
            train_oof[t] = np.clip(ridge10_pred, 0.001, 0.999)
            test_preds[t] = best['test_pred']
        
        log.info(f"    Meta comparison for {t}:")
        for name, r in results.items():
            log.info(f"      {name}: train_ll={r['train_ll']:.5f}, alpha={r.get('alpha','N/A')}, "
                     f"student={r['student_mean']:.5f}, gap={abs(r['student_mean']-r['meta_oof']):.5f}")
        log.info(f"    -> Selected: {best_name}")
    
    # Compute overall results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) for t in TARGETS])
    student_avg = np.mean(all_student_oofs)
    
    log.info(f"\n{'='*70}")
    log.info(f"V375 RESULTS (Gap-Constrained Stacking)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Student avg OOF: {student_avg:.5f}")
    log.info(f"  V308 OOF-LB gap: 0.01658")
    predicted_lb = avg_oof + 0.01658
    log.info(f"  Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    beats = predicted_lb < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v375_gap_constrained_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V375',
        'name': 'Gap-Constrained Stacking',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_base_features': len(base_train),
        'n_zscore_features': len(zscore_train),
        'n_seeds': N_SEEDS,
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) for t in TARGETS},
        'v308_per_target_oof': V308_OOF,
        'student_avg_oof': round(float(student_avg), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v375_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
