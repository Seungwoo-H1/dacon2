"""
V391 — Hyperparameter Diversity Seeds

Hypothesis: V308 uses 15 seeds with same hyperparameters, different random_state.
This creates models with similar architecture → similar predictions → similar
student OOF → limited ensemble benefit.

V391: Use fewer seeds (5) but with MORE DIVERSE hyperparameters.
Each seed gets a completely different config (wide/deep/v48/safety + different lr/leaves/etc).
This creates true diversity → some seeds will be much better calibrated →
the meta-learner can learn from the best ones.

Key insight from V361/V386/V387:
- V361 (LGBM+RF+ET): failed → mixed model families hurt
- V386 (multi-config): failed → diversity without gap control
- V391: controlled diversity with SAME model family (LGBM only) but different hyperparams

Config diversity (5 seeds, 5 configs):
- Seed 0: wide (shallow, few leaves, high lr)
- Seed 1: deep (deeper, more leaves, low lr)
- Seed 2: v48 (balanced)
- Seed 3: safety (very regularized)
- Seed 4: ultra-wide (many leaves, high lr, high subsample)

Each config gets 1 seed, trained per fold. 5 predictions → 5 features for meta.
With only 5 meta-features, LR C=10 has less risk of overfitting.

Expected:
- OOF: ~0.620-0.625 (similar to V308)
- Student: ~0.685-0.695 (similar)
- Predicted LB: ~0.635-0.642
- Risk: Medium (fewer seeds → higher variance, but better diversity)

Changes from V308:
1. 15 seeds → 5 seeds, 5 different configs
2. 5 meta-features instead of 15 (less overfitting risk for meta)
3. Same V53 feature selection, GroupKFold 5-fold
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

# 5 highly diverse configs (all LGBM)
DIV_CONFIGS = [
    {'name': 'wide',     'num_leaves': 63, 'max_depth': -1, 'learning_rate': 0.1, 'n_estimators': 200,
     'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'min_child_samples': 3},
    {'name': 'deep',     'num_leaves': 15, 'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 2000,
     'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 25},
    {'name': 'balanced', 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500,
     'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    {'name': 'conservative', 'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 2000,
     'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 30},
    {'name': 'aggressive', 'num_leaves': 127, 'max_depth': -1, 'learning_rate': 0.2, 'n_estimators': 100,
     'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.5, 'min_child_samples': 1},
]

V53_SWEEP = {
    'Q1':  {'cfg': 1,   'n_feat': 19},
    'Q2':  {'cfg': 1,   'n_feat': 14},
    'Q3':  {'cfg': 2,   'n_feat': 11},
    'S1':  {'cfg': 0,   'n_feat': 21},
    'S2':  {'cfg': 1,   'n_feat': 19},
    'S3':  {'cfg': 3,   'n_feat': 23},
    'S4':  {'cfg': 0,   'n_feat': 20},
}

# Actually let's use ALL 5 configs, each gets 1 seed per fold
# This is different from V308 which uses the same config for all seeds

SEED = 42
N_FOLDS = 5
N_SEEDS = 5  # Reduced to 5, but each is a different config
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
    log.info("V391 — Hyperparameter Diversity Seeds (5 configs)")
    log.info("Hypothesis: Diverse hyperparams → better calibrated ensemble")
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
    log.info(f"N_SEEDS={N_SEEDS} (diverse configs)")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    all_results = []
    all_student_oofs = []
    all_per_seed_test = {}
    all_per_seed_oofs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        
        # Use V308-style feature ranking (same for all seeds)
        n_feat = 15  # Use a moderate number to balance all configs
        log.info(f"    V391: {N_SEEDS} diverse configs")
        
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
        
        log.info(f"    Selected {len(sel_cols)} features")
        
        # Train 5 diverse configs
        per_seed_oofs = []
        per_seed_test = []
        student_oofs = []
        
        for ci, cfg in enumerate(DIV_CONFIGS):
            seed = SEED + ci * 13  # Different seeds per config
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
            student_oofs.append(s_oof)
            
            log.info(f"    Config {DIV_CONFIGS[ci]['name']}: OOF={s_oof:.5f}")
        
        # Meta-learner with 5 features
        stacked_train = np.column_stack(per_seed_oofs)
        stacked_test = np.column_stack(per_seed_test)
        
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_train, y)
        meta_oof = log_loss(y, np.clip(meta.predict_proba(stacked_train)[:, 1], 0.001, 0.999))
        meta_test = np.clip(meta.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
        
        equal_avg = np.mean(per_seed_oofs, axis=0)
        equal_avg_ll = log_loss(y, equal_avg)
        
        student_avg = np.mean(student_oofs)
        
        all_results.append({
            'target': t, 'meta_oof': meta_oof, 'equal_avg_ll': equal_avg_ll,
            'student_avg': student_avg
        })
        all_student_oofs.extend(student_oofs)
        all_per_seed_test[t] = per_seed_test
        all_per_seed_oofs[t] = per_seed_oofs
        
        log.info(f"    V391: meta_OOF={meta_oof:.5f} (V308: {V308_OOF[t]:.5f}, Δ: {meta_oof-V308_OOF[t]:+.5f})")
        log.info(f"    V391: equal_OOF={equal_avg_ll:.5f} (Δ: {equal_avg_ll-V308_OOF[t]:+.5f})")
        log.info(f"    V391: Student avg={student_avg:.5f} (V308: 0.69212, Δ: {student_avg-0.69212:+.5f})")
    
    # Summary
    avg_meta_oof = np.mean([r['meta_oof'] for r in all_results])
    avg_equal_oof = np.mean([r['equal_avg_ll'] for r in all_results])
    avg_student = np.mean(all_student_oofs)
    predicted_lb_meta = avg_meta_oof + 0.01658
    predicted_lb_equal = avg_equal_oof + 0.01658
    
    log.info(f"\n{'='*70}")
    log.info("V391 RESULTS")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        r = next(rr for rr in all_results if rr['target'] == t)
        v308_t = V308_OOF[t]
        log.info(f"  {t}: meta={r['meta_oof']:.5f} (Δ: {r['meta_oof']-v308_t:+.5f}), "
                 f"equal={r['equal_avg_ll']:.5f} (Δ: {r['equal_avg_ll']-v308_t:+.5f})")
    
    log.info(f"\n  AVG meta OOF: {avg_meta_oof:.5f} (V308: 0.62235, Δ: {avg_meta_oof-0.62235:+.5f})")
    log.info(f"  AVG equal OOF: {avg_equal_oof:.5f} (V308: 0.62235, Δ: {avg_equal_oof-0.62235:+.5f})")
    log.info(f"  AVG student: {avg_student:.5f} (V308: 0.69212, Δ: {avg_student-0.69212:+.5f})")
    log.info(f"  Predicted LB (meta): {predicted_lb_meta:.5f} (V308: 0.63893, Δ: {predicted_lb_meta-0.63893:+.5f})")
    log.info(f"  Predicted LB (equal): {predicted_lb_equal:.5f} (V308: 0.63893, Δ: {predicted_lb_equal-0.63893:+.5f})")
    
    meta_beats = predicted_lb_meta < 0.63893
    equal_beats = predicted_lb_equal < 0.63893
    
    log.info(f"\n  Beats V308 (meta): {meta_beats}")
    log.info(f"  Beats V308 (equal): {equal_beats}")
    log.info(f"{'='*70}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    meta_data = {
        'version': 'V391',
        'name': 'Hyperparameter Diversity Seeds (5 diverse configs)',
        'avg_meta_oof': round(float(avg_meta_oof), 5),
        'avg_equal_oof': round(float(avg_equal_oof), 5),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_meta_oof - 0.62235), 5),
        'delta_vs_v308_equal': round(float(avg_equal_oof - 0.62235), 5),
        'predicted_lb_meta': round(float(predicted_lb_meta), 5),
        'predicted_lb_equal': round(float(predicted_lb_equal), 5),
        'meta_beats_v308': bool(meta_beats),
        'equal_beats_v308': bool(equal_beats),
        'student_avg_oof': round(float(avg_student), 5),
        'n_seeds': N_SEEDS,
        'per_target': {
            r['target']: {
                'meta_oof': round(r['meta_oof'], 5),
                'equal_avg_oof': round(r['equal_avg_ll'], 5),
                'student_avg': round(r['student_avg'], 5)
            } for r in all_results
        },
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v391_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    if meta_beats or equal_beats:
        log.info(f"\nV391 BEATS V308! Creating submission...")
        sub = pd.DataFrame()
        sub['subject_id'] = test_df['subject_id'].values
        sub['sleep_date'] = test_df['sleep_date'].values
        sub['lifelog_date'] = test_df['lifelog_date'].values
        
        for t in TARGETS:
            stacked_test = np.column_stack(all_per_seed_test[t])
            stacked_train = np.column_stack(all_per_seed_oofs[t])
            meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta_t.fit(stacked_train, train_df[t].values.astype(np.float64))
            test_pred = np.clip(meta_t.predict_proba(stacked_test)[:, 1], 0.001, 0.999)
            sub[t] = test_pred
        
        sub_path = SUBMIT / f"submission_v391_diverse_configs_{ts}.csv"
        sub.to_csv(sub_path, index=False)
        log.info(f"Saved submission: {sub_path}")
        meta_data['submission_file'] = str(sub_path)
    else:
        log.info(f"\nV391 does NOT beat V308. No submission.")
    
    return avg_meta_oof, meta_data


if __name__ == '__main__':
    main()
