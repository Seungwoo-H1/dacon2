"""
V379 — Ensemble-Weighted Stacking with Outlier Rejection

Hypothesis: V308 equally averages 15 seed predictions at student level.
But some seeds are clearly noise (e.g., one seed might have OOF=0.93
while others are 0.50-0.60 for S1). Including bad seeds at student level
degrades the meta-learner's ability to generalize.

Key insight from V308 S1 data: some seeds get OOF 0.93-0.98, others 0.50-0.55.
The meta-learner needs clean signals from consistent seeds, not diluted by
outlier seeds.

Approach:
1. For each target, compute per-seed OOF
2. Reject top-worst seeds (top 2-3 seeds with highest OOF)
3. Only use remaining seeds for stacking → cleaner signal
4. Keep V308 architecture: GroupKFold 5-fold, LR C=10 meta
5. Test prediction: average of retained seeds only

V308 S1 example:
- Seed 14 (s140): OOF ~0.94 (terrible!)
- Others: OOF 0.50-0.60
- If we reject the worst 2-3 seeds, meta-learner gets cleaner signal

This is DIFFERENT from:
- V313 (30 seeds): more seeds → more noise, not less
- V368 (bagging): feature bagging → different approach
- V361 (multi-model): adding models → more noise

V379: REMOVING bad models → cleaner signal → better generalization

Expected:
- OOF: ~0.615-0.620 (selecting best 12/15 seeds)
- Predicted LB: depends on gap behavior
- Risk: Medium (rejecting seeds changes ensemble diversity)
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
N_REJECT = 2  # Reject worst 2 seeds per target


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


def generate_test_zscore(train_df, test_df):
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
    log.info("V379 — Ensemble-Weighted Stacking with Outlier Rejection")
    log.info(f"Hypothesis: Reject worst {N_REJECT} seeds → cleaner signal")
    log.info(f"V308: 15 seeds, OOF=0.62235, LB=0.63893")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
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
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    all_student_oofs = []
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    # Compare: V308 (all 15 seeds) vs V379 (15 - N_REJECT seeds)
    # We need to run the full training to compare
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        
        cfg = CFGS[cfg_name]
        
        # Train all 15 seeds
        per_seed_oofs = []
        per_seed_test = []
        
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
            per_seed_test.append(seed_test)
            
            s_oof = log_loss(y, seed_oof)
            all_student_oofs.append(s_oof)
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
        
        # Analyze seed quality
        seed_oof_values = [log_loss(y, p) for p in per_seed_oofs]
        avg_seed_oof = np.mean(seed_oof_values)
        std_seed_oof = np.std(seed_oof_values)
        min_seed_oof = min(seed_oof_values)
        max_seed_oof = max(seed_oof_values)
        
        log.info(f"    Seed OOF stats: min={min_seed_oof:.4f}, max={max_seed_oof:.4f}, "
                 f"mean={avg_seed_oof:.4f}, std={std_seed_oof:.4f}")
        log.info(f"    Range (max-min): {max_seed_oof-min_seed_oof:.4f}")
        
        # Identify and reject worst seeds
        worst_indices = np.argsort(seed_oof_values)[-N_REJECT:]
        retained_indices = [i for i in range(N_SEEDS) if i not in worst_indices]
        
        rejected_oofs = [f"{seed_oof_values[i]:.4f}" for i in worst_indices]
        log.info(f"    Rejected seeds: {list(worst_indices)} (OOFs: {rejected_oofs})")
        log.info(f"    Retained: {len(retained_indices)}/{N_SEEDS} seeds")
        
        # V308 baseline: all 15 seeds
        stacked_all = np.column_stack(per_seed_oofs)
        stacked_test_all = np.column_stack(per_seed_test)
        
        meta_all = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_all.fit(stacked_all, y)
        oof_pred_all = meta_all.predict_proba(stacked_all)[:, 1]
        oof_all = log_loss(y, np.clip(oof_pred_all, 0.001, 0.999))
        test_pred_all = np.clip(meta_all.predict_proba(stacked_test_all)[:, 1], 0.001, 0.999)
        
        # V379: retained seeds only
        stacked_retained = np.column_stack([per_seed_oofs[i] for i in retained_indices])
        stacked_test_retained = np.column_stack([per_seed_test[i] for i in retained_indices])
        
        meta_retained = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_retained.fit(stacked_retained, y)
        oof_pred_retained = meta_retained.predict_proba(stacked_retained)[:, 1]
        oof_retained = log_loss(y, np.clip(oof_pred_retained, 0.001, 0.999))
        test_pred_retained = np.clip(meta_retained.predict_proba(stacked_test_retained)[:, 1], 0.001, 0.999)
        
        # Also try: inverse-OOF weighted ensemble of retained seeds
        retained_oof_vals = [seed_oof_values[i] for i in retained_indices]
        inv_weights = [1.0 / max(v, 0.5) for v in retained_oof_vals]
        total_w = sum(inv_weights)
        inv_weights = [w / total_w for w in inv_weights]
        
        stacked_weighted = np.column_stack([per_seed_oofs[i] for i in retained_indices])
        stacked_test_weighted = np.column_stack([per_seed_test[i] for i in retained_indices])
        
        weighted_pred = np.zeros(n_train)
        weighted_test = np.zeros(n_test)
        for wi, (i, w) in enumerate(zip(retained_indices, inv_weights)):
            weighted_pred += w * per_seed_oofs[i]
            weighted_test += w * per_seed_test[i]
        weighted_pred_oof = log_loss(y, np.clip(weighted_pred, 0.001, 0.999))
        
        # Compare all three
        results = {
            'v308_all': {'oof': oof_all, 'test': test_pred_all},
            'v379_retained': {'oof': oof_retained, 'test': test_pred_retained},
            'v379_weighted': {'oof': weighted_pred_oof, 'test': weighted_test},
        }
        
        best_name = min(results, key=lambda k: results[k]['oof'])
        best = results[best_name]
        
        train_oof[t] = np.clip(best['test'], 0.001, 0.999)
        # Fix: train_oof should be predictions on train
        if best_name == 'v308_all':
            train_oof[t] = np.clip(oof_pred_all, 0.001, 0.999)
        elif best_name == 'v379_retained':
            train_oof[t] = np.clip(oof_pred_retained, 0.001, 0.999)
        else:
            train_oof[t] = np.clip(weighted_pred, 0.001, 0.999)
        test_preds[t] = best['test']
        
        log.info(f"    {t} Comparison:")
        for name, r in results.items():
            log.info(f"      {name}: OOF={r['oof']:.5f}")
        log.info(f"    -> Selected: {best_name}")
    
    # Compute overall results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) for t in TARGETS])
    student_avg = np.mean(all_student_oofs)
    
    v308_gap = 0.01658
    predicted_lb = avg_oof + v308_gap
    
    log.info(f"\n{'='*70}")
    log.info(f"V379 RESULTS (Outlier Rejection)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Student avg OOF: {student_avg:.5f}")
    log.info(f"  Predicted LB: {predicted_lb:.5f} (V308: 0.63893, Δ: {predicted_lb-0.63893:+.5f})")
    beats = predicted_lb < 0.63893
    log.info(f"  Beats V308: {beats}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v379_outlier_rejection_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V379',
        'name': f'Ensemble with Outlier Rejection (reject {N_REJECT} worst)',
        'avg_oof': round(float(avg_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'student_avg_oof': round(float(student_avg), 5),
        'n_seeds_retained': N_SEEDS - N_REJECT,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v379_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
