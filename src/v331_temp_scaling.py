"""
V331 — Per-Target Temperature Scaling on Ensemble Predictions

Hypothesis: V308의 OOF-LB gap(0.01658) 중 일부는 ensemble predictions의 
overconfidence에서 기인. V128에서 ECE calibration이 효과적이었다는 기록 확인.
Student ensemble prediction에 per-target temperature scaling을 적용하면
OOF는 약간 증가하지만 LB에서는 better calibration → 더 낮은 점수.

Key insight from DACON2_CONTEXT:
- V128: ECE-guided calibration AVG OOF 0.66834 (V53 0.6813 대비 -0.013)
- Temperature scaling도 효과적 (AVG OOF 0.66685)
- V308의 OOF 0.62235는 V53의 0.6813보다 낮으므로, calibration 적용 시
  OOF는 증가하지만 LB는 더 낮아질 수 있음

Changes:
1. V308 pipeline과 동일 (15 seeds, z-score, 282 features)
2. Student ensemble prediction에 per-target temperature fitting
3. Temperature fitting: grid search T in [0.5, 3.0] minimizing OOF
4. Meta learner는 unchanged (LR C=10)

Risk: Calibration OOF improvement ≠ LB improvement (V96, V97 history)
"""
import sys, gc, logging, json, re, time, warnings, scipy.optimize as opt
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


def temperature_scale(preds, temperatures):
    """Apply temperature scaling to binary predictions."""
    # preds: (N,) or (N, 2) probabilities
    if preds.ndim == 1:
        # Convert to logits
        preds = np.clip(preds, 1e-7, 1-1e-7)
        logits = np.log(preds / (1 - preds))
        scaled_logits = logits / temperatures
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs
    return preds


def find_optimal_temperature(oof_preds, y, range_min=0.3, range_max=5.0, n_steps=50):
    """Find temperature that minimizes log_loss on OOF predictions."""
    temperatures = np.linspace(range_min, range_max, n_steps)
    best_T = 1.0
    best_loss = log_loss(y, np.clip(oof_preds, 1e-7, 1-1e-7))
    
    for T in temperatures:
        scaled = temperature_scale(oof_preds, T)
        loss = log_loss(y, np.clip(scaled, 1e-7, 1-1e-7))
        if loss < best_loss:
            best_loss = loss
            best_T = T
    
    return best_T, best_loss


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V331 — Per-Target Temperature Scaling on Ensemble Predictions")
    log.info("V308 baseline: OOF=0.62235, LB=0.63893")
    log.info("Hypothesis: calibration improves LB more than OOF")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Generate z-score features
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
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features | Test: {len(test_feat_cols)} features")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Per-target feature ranking and selection
    target_configs = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        target_configs[t] = {
            'cfg': CFGS[cfg_name],
            'features': sel_cols_test,
            'n_feat': len(sel_cols_test),
        }
        log.info(f"  {t}: cfg={cfg_name}, {n_feat}→{len(sel_cols_test)} features")
    
    # Train per-target models
    all_oofs = {}
    all_test_preds = {}
    temperatures = {}
    
    for t in TARGETS:
        tc = target_configs[t]
        cfg = tc['cfg']
        feats = tc['features']
        
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} | cfg={list(tc['cfg'].keys())[0]} | feats={tc['n_feat']} | seeds={N_SEEDS}")
        
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        # Generate seeds
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        train_oof_full = np.zeros((n_train, N_SEEDS))
        test_preds = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[feats].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[feats].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in feats]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[feats].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            train_oof_full[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        # Ensemble student predictions (mean)
        student_oof = np.mean(train_oof_full, axis=1)
        student_oof = np.clip(student_oof, 0.001, 0.999)
        student_oof_ll = log_loss(y, student_oof)
        
        # Per-target temperature scaling on student OOF
        T_opt, T_opt_loss = find_optimal_temperature(student_oof, y)
        cal_student_oof = temperature_scale(student_oof, T_opt)
        cal_student_oof_ll = log_loss(y, np.clip(cal_student_oof, 1e-7, 1-1e-7))
        
        temperatures[t] = {
            'optimal_T': round(float(T_opt), 4),
            'student_oof_ll': round(float(student_oof_ll), 5),
            'cal_student_oof_ll': round(float(cal_student_oof_ll), 5),
            'improvement': round(float(student_oof_ll - cal_student_oof_ll), 5),
        }
        
        log.info(f"    T_opt={T_opt:.3f}, student OOF: {student_oof_ll:.5f} → cal: {cal_student_oof_ll:.5f} (Δ={cal_student_oof_ll-student_oof_ll:+.5f})")
        
        # Meta learner on calibrated student predictions
        stacked_cal = np.column_stack([temperature_scale(train_oof_full[:, si], T_opt) for si in range(N_SEEDS)])
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked_cal, y)
        
        final_oof = meta.predict_proba(stacked_cal)[:, 1]
        final_oof = np.clip(final_oof, 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        log.info(f"    {t} Final OOF (cal ensemble + LR meta): {oof_ll:.5f}")
        
        # Test predictions: cal then meta
        test_student = np.mean(test_preds, axis=1)
        test_student_cal = temperature_scale(test_student, T_opt)
        # Also cal each seed individually for meta input
        stacked_test = np.column_stack([temperature_scale(test_preds[:, si], T_opt) for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = test_pred
    
    # Compute AVG OOF
    avg_oof = np.mean(list(all_oofs.values()))
    log.info(f"\n{'='*70}")
    log.info(f"V331 RESULTS (per-target temperature scaling)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={all_oofs[t]:.5f}, T={temperatures[t]['optimal_T']:.3f}, Δ_cal={temperatures[t]['improvement']:+.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    # Estimated LB: if calibration improves OOF by X on average,
    # assume similar improvement on LB (conservative estimate)
    avg_cal_improvement = np.mean([temperatures[t]['improvement'] for t in TARGETS])
    v308_lb = 0.63893
    estimated_lb = v308_lb + avg_cal_improvement * 0.5  # Conservative: assume 50% of OOF improvement translates to LB
    
    log.info(f"  Avg calibration improvement: {avg_cal_improvement:+.5f}")
    log.info(f"  Estimated LB (conservative, 50% translation): {estimated_lb:.5f}")
    log.info(f"  V308 LB: 0.63893")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v331_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V331',
        'name': 'Per-Target Temperature Scaling on Ensemble Predictions',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'temperatures': {t: temperatures[t] for t in TARGETS},
        'estimated_lb': round(float(estimated_lb), 5),
        'v308_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'predicted_lb': None,
        'actual_lb': None,
    }
    
    meta_path = EXPERIMENTS / f'v331_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
