"""
V377 — Per-Target Isolated Pipeline with Target-Conditional Ensemble

Hypothesis: V308 applies identical pipeline (15 seeds, same feature ranking,
same V53 config) across all 7 targets. But Q1-OOF=0.67 vs S1-OOF=0.58 shows
targets have very different difficulty levels. A one-size-fits-all pipeline
cannot be optimal for all.

Approach:
1. Each target gets its OWN pipeline optimized independently
   - Q targets: different configs, more aggressive (they need more help)
   - S targets: conservative, stability-focused
   - Per-target seed count: 5 × 3 configs (15 total, same cost as V308)
2. Independent feature ranking per target (already done in V308)
3. Independent V53 config selection per target (already done)
4. Key innovation: Target-Conditional Ensemble Weights
   - Instead of single LR meta, use per-target weight optimization on holdout
   - CV-based weight tuning per target

Why this differs from V369 (which failed):
- V369 split features into Q/S groups → signal dilution
- V377 keeps SAME full feature set, just different model configs per target
- No target-conditional features, only target-conditional modeling

Expected:
- OOF: ~0.615-0.620 (improved over V308's 0.622)
- Predicted LB: ~0.632-0.638 (competitive with V308)
- Risk: Medium (different configs per target could introduce variance)
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

# V308 baseline configs (for reference)
CFGS_V308 = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

# V377: Per-target configs
# Q targets need more diverse ensemble (they're harder)
# S targets need stability (they're easier)
CFGS_V377 = {
    # Q targets: 5 seeds × 3 diverse configs = 15 models per Q target
    'Q1': {'configs': ['deep', 'v48', 'wide'],    'seeds_per': 5, 'n_feat': 19},
    'Q2': {'configs': ['deep', 'v48', 'wide'],    'seeds_per': 5, 'n_feat': 14},
    'Q3': {'configs': ['v48', 'wide', 'deep'],    'seeds_per': 5, 'n_feat': 11},
    # S targets: 5 seeds × 2 configs + 5 safety = 15 models per S target
    'S1': {'configs': ['wide', 'wide'],           'seeds_per': 8, 'n_feat': 21},  # wide dominates S targets
    'S2': {'configs': ['deep', 'v48'],            'seeds_per': 8, 'n_feat': 19},
    'S3': {'configs': ['safety', 'v48'],          'seeds_per': 8, 'n_feat': 23},
    'S4': {'configs': ['wide', 'wide'],           'seeds_per': 8, 'n_feat': 20},
}

# All configs
ALL_CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

SEED = 42
N_FOLDS = 5


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


def train_and_predict(target, sel_cols, cfg, train_df, test_df, group, y, gkf, seed, t_start):
    """Train one model (one config, one seed) and return OOF + test predictions."""
    n_train = len(train_df)
    n_test = len(test_df)
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
    
    return np.clip(seed_oof, 0.001, 0.999), seed_test / N_FOLDS


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V377 — Per-Target Isolated Pipeline")
    log.info("Hypothesis: Per-target optimized configs beat one-size-fits-all")
    log.info(f"V308: 15 seeds uniform, OOF=0.62235, LB=0.63893")
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
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    # Storage
    train_oof = {t: np.zeros(n_train) for t in TARGETS}
    test_preds = {t: np.zeros(n_test) for t in TARGETS}
    per_seed_test = {t: [] for t in TARGETS}  # list of (oof_pred, test_pred)
    per_seed_oofs = {t: [] for t in TARGETS}
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
        
        # Per-target config from V377
        t_cfg = CFGS_V377[t]
        target_configs = t_cfg['configs']
        seeds_per = t_cfg['seeds_per']
        n_feat = t_cfg['n_feat']
        
        total_models = len(target_configs) * seeds_per
        log.info(f"    Per-target pipeline: {len(target_configs)} configs × {seeds_per} seeds = {total_models} models")
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        
        # Verify same columns exist in test
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            missing = set(sel_cols) - set(sel_cols_test)
            log.warning(f"    {t}: {len(missing)} features missing in test")
            sel_cols = sel_cols_test
        
        log.info(f"    Selected {len(sel_cols)} features")
        
        # Train all models for this target
        model_results = []  # list of (oof_pred, test_pred)
        oof_seed_counter = 0
        
        for ci, cfg_name in enumerate(target_configs):
            cfg = ALL_CFGS[cfg_name]
            log.info(f"    Config: {cfg_name}, n_est={cfg['n_estimators']}")
            
            for si in range(seeds_per):
                seed = SEED + ci * 100 + si * 7
                seed_oof, seed_test = train_and_predict(
                    t, sel_cols, cfg, train_df, test_df, group, y, gkf, seed, t_start
                )
                model_results.append((seed_oof, seed_test))
                s_oof = log_loss(y, seed_oof)
                per_seed_oofs[t].append(s_oof)
                all_student_oofs.append(s_oof)
                oof_seed_counter += 1
                
                if oof_seed_counter <= 3 or oof_seed_counter % 5 == 0:
                    log.info(f"    Model #{oof_seed_counter:2d} ({cfg_name}, s{seed}): OOF={s_oof:.5f}")
        
        per_seed_test[t] = [mr[1] for mr in model_results]
        
        # Level 1: Ensemble weights optimization
        # Simple approach: equal weight (baseline)
        # Better: optimize weights on a small holdout or using CV
        
        n_models = len(model_results)
        
        # Method A: Equal weight
        equal_pred = np.zeros(n_train)
        equal_test = np.zeros(n_test)
        for seed_oof, seed_test in model_results:
            equal_pred += seed_oof
            equal_test += seed_test
        
        equal_pred /= n_models
        equal_test /= n_models
        equal_oof = log_loss(y, np.clip(equal_pred, 0.001, 0.999))
        
        # Method B: Weight optimization via CV (leave-one-fold-out within the 5-folds)
        # We'll use the OOF predictions to find optimal weights
        # Use gradient-free: try different weight combinations
        
        # Simple greedy: each model gets weight = 1/OOF_score
        # Better: just use equal weight for stability
        # Since we only have 450 samples and up to 15 models, complex weighting overfits
        
        # Let's try a middle ground: weight by inverse OOF of each individual model
        inv_oof_weights = []
        for seed_oof, _ in model_results:
            m_oof = log_loss(y, seed_oof)
            inv_oof_weights.append(1.0 / max(m_oof, 0.5))
        
        # Normalize weights
        inv_w = np.array(inv_oof_weights) / np.sum(inv_oof_weights)
        
        inv_pred = np.zeros(n_train)
        inv_test = np.zeros(n_test)
        for (seed_oof, seed_test), w in zip(model_results, inv_oof_weights):
            inv_pred += w * seed_oof
            inv_test += w * seed_test
        
        inv_oof = log_loss(y, np.clip(inv_pred, 0.001, 0.999))
        
        # Compare and choose
        if inv_oof < equal_oof:
            log.info(f"    {t}: InvOOF-weighted OOF={inv_oof:.5f} vs Equal OOF={equal_oof:.5f} → InvOOF wins")
            train_oof[t] = np.clip(inv_pred, 0.001, 0.999)
            test_preds[t] = np.clip(inv_test, 0.001, 0.999)
            chosen = "inverse_oof_weighted"
        else:
            log.info(f"    {t}: Equal OOF={equal_oof:.5f} vs InvOOF-weighted OOF={inv_oof:.5f} → Equal wins")
            train_oof[t] = np.clip(equal_pred, 0.001, 0.999)
            test_preds[t] = np.clip(equal_test, 0.001, 0.999)
            chosen = "equal_weight"
        
        student_mean = np.mean(per_seed_oofs[t])
        log.info(f"    {t} Ensemble OOF: {log_loss(y, train_oof[t]):.5f} ({chosen})")
        log.info(f"    {t} Student mean OOF: {student_mean:.5f}")
        log.info(f"    {t} Models: {n_models}")
    
    # Compute overall results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) for t in TARGETS])
    student_avg = np.mean(all_student_oofs)
    
    v308_gap = 0.01658
    
    log.info(f"\n{'='*70}")
    log.info(f"V377 RESULTS (Per-Target Isolated Pipeline)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        oof_t = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
        v308_t = V308_OOF[t]
        log.info(f"  {t}: OOF={oof_t:.5f} (V308: {v308_t:.5f}, Δ: {oof_t-v308_t:+.5f})")
    log.info(f"  AVG OOF: {avg_oof:.5f} (V308: 0.62235, Δ: {avg_oof-0.62235:+.5f})")
    log.info(f"  Student avg OOF: {student_avg:.5f} (V308: ~0.692)")
    
    # Per-target config summary
    log.info(f"  Per-target model counts:")
    for t in TARGETS:
        n = len(per_seed_oofs[t])
        sm = np.mean(per_seed_oofs[t]) if per_seed_oofs[t] else 0
        log.info(f"    {t}: {n} models, student OOF={sm:.5f}")
    
    predicted_lb = avg_oof + v308_gap
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
    
    sub_path = SUBMIT / f"submission_v377_per_target_isolated_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V377',
        'name': 'Per-Target Isolated Pipeline',
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'delta_vs_v308_oof': round(float(avg_oof - 0.62235), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'beats_v308': bool(beats),
        'student_avg_oof': round(float(student_avg), 5),
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) for t in TARGETS},
        'v308_per_target_oof': V308_OOF,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v377_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
