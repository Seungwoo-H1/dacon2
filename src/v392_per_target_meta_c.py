"""
V392 — Per-Target Meta C Optimization

Hypothesis: V308 uses C=10 for ALL targets. But Q targets have high student OOF
(~0.75-0.79) indicating underfitting/need more regularization, while S targets
have low student OOF (~0.58-0.64) indicating good calibration.

V392: Optimize LR C per target for the meta-learner.
- Q targets: try C=0.1, 0.5, 1.0, 5.0 (stronger regularization)
- S targets: try C=1, 5, 10, 50 (weaker regularization)

Expected:
- Meta OOF: ~0.620-0.623 (similar)
- Student: ~0.685-0.692 (similar)
- Predicted LB: ~0.635-0.640
- Risk: Low (target-specific meta C is fine-tuning, not structural change)

Changes from V308:
1. Meta C: per-target instead of global C=10
2. Everything else identical
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
    log.info("V392 — Per-Target Meta C Optimization")
    log.info("Hypothesis: Per-target meta C → better calibrated for each target")
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
    
    # Per-target meta C candidates
    META_C_CANDIDATES = {
        'Q1': [0.1, 0.5, 1.0, 5.0, 10.0],
        'Q2': [0.1, 0.5, 1.0, 5.0, 10.0],
        'Q3': [0.1, 0.5, 1.0, 5.0, 10.0],
        'S1': [1.0, 5.0, 10.0, 50.0, 100.0],
        'S2': [1.0, 5.0, 10.0, 50.0, 100.0],
        'S3': [1.0, 5.0, 10.0, 50.0, 100.0],
        'S4': [1.0, 5.0, 10.0, 50.0, 100.0],
    }
    
    all_results = []
    all_student_oofs = []
    all_per_seed_test = {}
    all_per_seed_oofs = {}
    best_meta_cs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        
        # V308-style ranking
        y_rank = train_df[t].values.astype(np.float64)
        X_rank = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        spw_rank = max(((y_rank == 0).sum()) / max((y_rank == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw_rank, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn_rank = [sanitize_col(c) for c in feat_cols_clean]
        ds_rank = lgb.Dataset(X_rank, label=y_rank, feature_name=sn_rank)
        m_rank = lgb.train(params_rank, ds_rank, num_boost_round=50)
        imp_rank = m_rank.feature_importance(importance_type='gain')
        ranked_all = sorted(zip(feat_cols_clean, imp_rank), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked_all[:n_feat]]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        # Train models per fold
        per_seed_oofs = []
        per_seed_test = []
        student_oofs = []
        
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
            student_oofs.append(log_loss(y, seed_oof))
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d}: OOF={log_loss(y, seed_oof):.5f}")
        
        # Per-target meta C optimization
        best_C = META_C
        best_OOF = float('inf')
        
        for c_val in META_C_CANDIDATES[t]:
            stacked_train = np.column_stack(per_seed_oofs)
            meta = LogisticRegression(C=c_val, max_iter=1000, random_state=SEED)
            meta.fit(stacked_train, y)
            oof_pred = np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999)
            oof = log_loss(y, oof_pred)
            
            if oof < best_OOF:
                best_OOF = oof
                best_C = c_val
        
        log.info(f"    Target {t}: best meta C = {best_C} (OOF = {best_OOF:.5f})")
        best_meta_cs[t] = best_C
        
        # Retrain with best C
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = np.column_stack(per_seed_test)
        meta = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        meta_oof = best_OOF
        meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        equal_avg = np.mean(per_seed_oofs, axis=0)
        equal_avg_ll = log_loss(y, equal_avg)
        
        student_avg = np.mean(student_oofs)
        
        all_results.append({
            'target': t, 'meta_oof': meta_oof, 'equal_avg_ll': equal_avg_ll,
            'student_avg': student_avg, 'best_meta_C': best_C
        })
        all_student_oofs.extend(student_oofs)
        all_per_seed_test[t] = per_seed_test
        all_per_seed_oofs[t] = per_seed_oofs
        
        log.info(f"    V392: meta_OOF={meta_oof:.5f} (V308: {V308_OOF[t]:.5f}, Δ: {meta_oof-V308_OOF[t]:+.5f})")
        log.info(f"    V392: best_C={best_C} (V308 C=10)")
        log.info(f"    V392: equal_OOF={equal_avg_ll:.5f} (Δ: {equal_avg_ll-V308_OOF[t]:+.5f})")
        log.info(f"    V392: Student avg={student_avg:.5f} (V308: 0.69212, Δ: {student_avg-0.69212:+.5f})")
    
    # Summary
    avg_meta_oof = np.mean([r['meta_oof'] for r in all_results])
    avg_equal_oof = np.mean([r['equal_avg_ll'] for r in all_results])
    avg_student = np.mean(all_student_oofs)
    predicted_lb_meta = avg_meta_oof + 0.01658
    predicted_lb_equal = avg_equal_oof + 0.01658
    
    log.info(f"\n{'='*70}")
    log.info("V392 RESULTS")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        r = next(rr for rr in all_results if rr['target'] == t)
        v308_t = V308_OOF[t]
        log.info(f"  {t}: meta={r['meta_oof']:.5f} (Δ: {r['meta_oof']-v308_t:+.5f}), "
                 f"best_C={r['best_meta_C']}")
    
    log.info(f"\n  AVG meta OOF: {avg_meta_oof:.5f} (V308: 0.62235, Δ: {avg_meta_oof-0.62235:+.5f})")
    log.info(f"  AVG equal OOF: {avg_equal_oof:.5f} (V308: 0.62235, Δ: {avg_equal_oof-0.62235:+.5f})")
    log.info(f"  AVG student: {avg_student:.5f} (V308: 0.69212, Δ: {avg_student-0.69212:+.5f})")
    log.info(f"  Predicted LB (meta): {predicted_lb_meta:.5f} (V308: 0.63893, Δ: {predicted_lb_meta-0.63893:+.5f})")
    
    meta_beats = predicted_lb_meta < 0.63893
    equal_beats = predicted_lb_equal < 0.63893
    
    log.info(f"\n  Beats V308 (meta): {meta_beats}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    meta_data = {
        'version': 'V392',
        'name': 'Per-Target Meta C Optimization',
        'avg_meta_oof': round(float(avg_meta_oof), 5),
        'avg_equal_oof': round(float(avg_equal_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta_oof - 0.62235), 5),
        'predicted_lb_meta': round(float(predicted_lb_meta), 5),
        'meta_beats_v308': bool(meta_beats),
        'student_avg_oof': round(float(avg_student), 5),
        'best_meta_Cs': {t: best_meta_cs[t] for t in TARGETS},
        'per_target': {
            r['target']: {
                'meta_oof': round(r['meta_oof'], 5),
                'equal_avg_oof': round(r['equal_avg_ll'], 5),
                'student_avg': round(r['student_avg'], 5),
                'best_meta_C': r['best_meta_C']
            } for r in all_results
        },
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v392_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    if meta_beats or equal_beats:
        log.info(f"\nV392 BEATS V308! Creating submission...")
        sub = pd.DataFrame()
        sub['subject_id'] = test_df['subject_id'].values
        sub['sleep_date'] = test_df['sleep_date'].values
        sub['lifelog_date'] = test_df['lifelog_date'].values
        
        for t in TARGETS:
            r = next(rr for rr in all_results if rr['target'] == t)
            stacked_test = np.column_stack(all_per_seed_test[t])
            stacked_train = np.column_stack(all_per_seed_oofs[t])
            meta_t = LogisticRegression(C=r['best_meta_C'], max_iter=1000, random_state=SEED)
            meta_t.fit(stacked_train, train_df[t].values.astype(np.float64))
            sub[t] = np.clip(meta_t.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        sub_path = SUBMIT / f"submission_v392_per_target_meta_c_{ts}.csv"
        sub.to_csv(sub_path, index=False)
        log.info(f"Saved submission: {sub_path}")
        meta_data['submission_file'] = str(sub_path)
    else:
        log.info(f"\nV392 does NOT beat V308. No submission.")
    
    return avg_meta_oof, meta_data


if __name__ == '__main__':
    main()
