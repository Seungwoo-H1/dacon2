"""
V382 — Label Smoothing Sweep on V308 Base

Hypothesis: V308's student avg OOF (0.692) is limited by overly confident
predictions. Label smoothing forces the model to produce less extreme predictions,
improving calibration and potentially lowering student OOF without changing
meta-learner architecture.

Changes from V308:
1. Add label_smoothing parameter to LGBM training (0.05, 0.1, 0.15)
2. Same V308 architecture: 15 seeds × GroupKFold 5-fold → LR C=10
3. Same features, same configs, same V53 sweep

Expected:
- OOF: ~0.620-0.625 (slight degradation due to softer targets)
- Student avg: 0.685-0.695 (slight improvement or same)
- Predicted LB: ~0.637-0.642 (depends on gap behavior)
- Risk: Low (minimal change, same architecture)

If label smoothing improves calibration, the OOF-LB gap should narrow.
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
LS_VALUES = [0.0, 0.05, 0.1, 0.15]  # label smoothing values


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
    log.info("V382 — Label Smoothing Sweep")
    log.info(f"Hypothesis: Label smoothing → better calibration, same architecture")
    log.info(f"LS values: {LS_VALUES}")
    log.info("V308: OOF=0.62235, LB=0.63893")
    log.info("=" * 70)
    
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
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    # For each target, run label smoothing sweep on 5 seeds (quick)
    # Then full 15 seeds for best LS
    ls_results = {}  # (target, ls) -> (avg_meta_oof, student_avg)
    
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
        
        cfg = CFGS[cfg_name]
        
        # Quick 5-seed sweep for each LS value
        log.info(f"    Quick 5-seed sweep:")
        for ls_val in LS_VALUES:
            per_seed_oofs = []
            student_oofs = []
            
            for si in range(5):
                seed = SEED + si * 7
                seed_oof = np.zeros(n_train)
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    if ls_val > 0:
                        params['label_smoothing'] = ls_val
                    
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    seed_oof[va_idx] = m.predict(X_va)
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                per_seed_oofs.append(seed_oof)
                student_oofs.append(log_loss(y, seed_oof))
            
            student_avg = np.mean(student_oofs)
            log_loss_v308 = log_loss(y, np.mean(per_seed_oofs, axis=0))
            ls_results[(t, ls_val)] = {'student_avg': student_avg, 'equal_avg_oof': log_loss_v308}
            log.info(f"      LS={ls_val:.2f}: student_avg={student_avg:.5f}, equal_avg_OOF={log_loss_v308:.5f}")
    
    # Find best LS per target
    log.info(f"\n{'='*60}")
    log.info("BEST LS per target:")
    best_ls_per_target = {}
    for t in TARGETS:
        best_ls = None
        best_score = float('inf')
        for ls_val in LS_VALUES:
            score = ls_results[(t, ls_val)]['student_avg']
            if score < best_score:
                best_score = score
                best_ls = ls_val
        best_ls_per_target[t] = best_ls
        log.info(f"  {t}: LS={best_ls:.2f} (student_avg={ls_results[(t, best_ls)]['student_avg']:.5f})")
    
    # Use the same LS for all targets (global best)
    global_best_ls = min(LS_VALUES, key=lambda ls: np.mean([ls_results[(t, ls)]['student_avg'] for t in TARGETS]))
    log.info(f"\n  Global best LS (avg student): {global_best_ls:.2f}")
    
    # Also try: per-target best LS (might overfit but worth checking)
    log.info(f"  Per-target best LS:")
    for t in TARGETS:
        log.info(f"    {t}: {best_ls_per_target[t]:.2f}")
    
    # Full training with global best LS
    log.info(f"\n{'='*70}")
    log.info(f"Full training with LS={global_best_ls:.2f}")
    log.info(f"{'='*70}")
    
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
            sel_cols = sel_cols_test
        
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        per_seed_test = []
        student_oofs_list = []
        
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
                if global_best_ls > 0:
                    params['label_smoothing'] = global_best_ls
                
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test.append(seed_test)
            student_oofs_list.append(log_loss(y, seed_oof))
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d}: OOF={log_loss(y, seed_oof):.5f}")
        
        # Compare: equal avg vs LR meta
        equal_avg = np.mean(per_seed_oofs, axis=0)
        equal_avg_ll = log_loss(y, equal_avg)
        
        # Meta
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = np.column_stack(per_seed_test)
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        meta_oof = log_loss(y, np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999))
        meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        student_avg = np.mean(student_oofs_list)
        
        log.info(f"    LS={global_best_ls:.2f}: equal_avg_OOF={equal_avg_ll:.5f}, meta_OOF={meta_oof:.5f}")
        log.info(f"    Student avg: {student_avg:.5f} (V308: 0.69212)")
        log.info(f"    Meta OOF Δ vs V308: {meta_oof - V308_OOF[t]:+.5f}")
    
    # Now also try per-target best LS if it's different
    # Use per-target best LS from the quick sweep for the full run
    # But only if it gives better student avg
    
    # Actually, let's also run the same with per-target best LS
    # For simplicity, just use global best
    
    # Compute overall
    # Re-run full with global best LS to get final results
    # (Already did above loop but didn't store final predictions)
    # Let me restructure...
    
    # Actually the loop above already trained with global_best_ls but didn't store test preds
    # Need to re-do. But the quick sweep already gave us the key insight.
    # Let me just save the key findings.
    
    ls_summary = {}
    for ls_val in LS_VALUES:
        avg_student = np.mean([ls_results[(t, ls_val)]['student_avg'] for t in TARGETS])
        avg_equal = np.mean([ls_results[(t, ls_val)]['equal_avg_oof'] for t in TARGETS])
        ls_summary[ls_val] = {'student_avg': avg_student, 'equal_avg_oof': avg_equal}
    
    log.info(f"\n{'='*70}")
    log.info("LABEL SMOOTHING SUMMARY")
    log.info(f"{'='*70}")
    for ls_val in LS_VALUES:
        s = ls_summary[ls_val]
        marker = " ← BEST" if s['student_avg'] == min(ls_summary[l]['student_avg'] for l in LS_VALUES) else ""
        log.info(f"  LS={ls_val:.2f}: equal_avg_OOF={s['equal_avg_oof']:.5f}, student_avg={s['student_avg']:.5f}{marker}")
    
    # V308 baseline (LS=0)
    v308_student = 0.69212
    v308_oof = 0.62235
    
    best_ls = min(LS_VALUES, key=lambda l: ls_summary[l]['student_avg'])
    best_student = ls_summary[best_ls]['student_avg']
    
    # If best LS improves student avg and doesn't hurt equal_avg_oof too much
    student_improvement = v308_student - best_student
    oof_degradation = ls_summary[best_ls]['equal_avg_oof'] - v308_oof
    
    log.info(f"\n{'='*70}")
    log.info(f"V382 CONCLUSION")
    log.info(f"{'='*70}")
    log.info(f"  Best LS: {best_ls}")
    log.info(f"  Student improvement: {student_improvement:+.5f}")
    log.info(f"  Equal avg OOF degradation: {oof_degradation:+.5f}")
    log.info(f"  If gap stays at V308 level (+0.017), predicted LB = {ls_summary[best_ls]['equal_avg_oof']+0.017:.5f}")
    beats = ls_summary[best_ls]['equal_avg_oof'] + 0.017 < 0.63893
    log.info(f"  Beats V308 (gap=0.017): {beats}")
    log.info(f"{'='*70}")
    
    # Don't submit if not clearly beating V308
    if beats:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Re-run full with best LS
        final_test_preds = {}
        final_train_oofs = {}
        all_student_oofs = []
        
        for t in TARGETS:
            y = train_df[t].values.astype(np.float64)
            feat_cols_clean = remove_leak(train_feat_cols, t)
            n_feat = V53_SWEEP[t]['n_feat']
            cfg_name = V53_SWEEP[t]['cfg']
            
            ranked = rank_features(train_df, feat_cols_clean, t)
            sel_cols = ranked[:n_feat]
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            sel_cols = sel_cols_test
            
            cfg = CFGS[cfg_name]
            
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
                    params['label_smoothing'] = best_ls
                    
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    seed_oof[va_idx] = m.predict(X_va)
                    seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
                
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                seed_test /= N_FOLDS
                per_seed_oofs.append(seed_oof)
                per_seed_test.append(seed_test)
                all_student_oofs.append(log_loss(y, seed_oof))
            
            stacked_train = np.column_stack(per_seed_oofs)
            stacked_test = np.column_stack(per_seed_test)
            
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked_train, y)
            final_train_oofs[t] = np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999)
            final_test_preds[t] = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        avg_oof = np.mean([log_loss(train_df[t].values, final_train_oofs[t]) for t in TARGETS])
        student_avg = np.mean(all_student_oofs)
        predicted_lb = avg_oof + 0.01658
        
        sub = pd.DataFrame()
        sub['subject_id'] = test_df['subject_id'].values
        sub['sleep_date'] = test_df['sleep_date'].values
        sub['lifelog_date'] = test_df['lifelog_date'].values
        for t in TARGETS:
            sub[t] = final_test_preds[t]
        
        sub_path = SUBMIT / f"submission_v382_label_smoothing_{ts}.csv"
        sub.to_csv(sub_path, index=False)
        log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V382',
        'name': f'Label Smoothing Sweep',
        'ls_values_tested': LS_VALUES,
        'best_ls': best_ls,
        'best_student_avg': round(float(ls_summary[best_ls]['student_avg']), 5),
        'best_equal_avg_oof': round(float(ls_summary[best_ls]['equal_avg_oof']), 5),
        'v308_student_avg': 0.69212,
        'v308_oof': 0.62235,
        'v308_lb': 0.63893,
        'student_improvement': round(float(student_improvement), 5),
        'oof_degradation': round(float(oof_degradation), 5),
        'predicted_lb_at_gap017': round(float(ls_summary[best_ls]['equal_avg_oof']+0.017), 5),
        'beats_v308': bool(beats),
        'ls_summary': {str(k): {'student_avg': round(v['student_avg'], 5), 'equal_avg_oof': round(v['equal_avg_oof'], 5)}
                       for k, v in ls_summary.items()},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v382_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return None, meta_data


if __name__ == '__main__':
    main()
