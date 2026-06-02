"""
V318 — Forced Strong Regularization on S4 (and Q1/Q3 if needed)

Problem: V316/Q1/Q3/S4에서 aggressive cfg가 best로 선택되지만 gap이 큼.
V317에서 gap balancing은 OOF를 올려서 predicted LB를 악화시킴.

V318 approach:
1. Q1/Q3/S4는 cfg search SKIP하고 reg_heavy(num_leaves=12, max_depth=3, lr=0.02,
   n_est=200, reg_alpha=10, reg_lambda=20, subsample=0.9, colsample=0.9)로 고정
2. Q2/S1/S2/S3는 V316 best cfg 유지
3. Meta C=10 유지
4. This eliminates the OOF↑ from gap balancing — force stability on gap targets

Rationale: Q1/Q3/S4의 gap이 0.04+인 것은 cfg의 문제라기보다
해당 target이 inherently noisy. Reg-heavy가 noise를 잘 걸러낼 것.

Expected OOF: 0.620-0.623 (slightly worse than V316, but stable gaps)
Expected LB: < 0.638 (if gaps stay under V308 levels)
Risk: Low-Medium (forced cfg on gap targets, simple approach)
Cost: ~150s

Architecture: Same V308 + forced reg on Q1/Q3/S4
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

# V316 best cfgs for stable targets
V316_BEST = {
    'Q1': 'aggressive', 'Q2': 'safety', 'Q3': 'safety',
    'S1': 'aggressive', 'S2': 'safety', 'S3': 'wide', 'S4': 'aggressive',
}

# CFGs
STUDENT_CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    'shallow_fast': {'num_leaves': 15, 'max_depth': 2, 'learning_rate': 0.08, 'n_estimators': 200,
                     'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'balanced':      {'num_leaves': 25, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 400,
                     'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 1.5, 'reg_lambda': 4.0, 'min_child_samples': 8},
    'aggressive':    {'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.01, 'n_estimators': 2000,
                     'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 3},
    'reg_heavy':     {'num_leaves': 12, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 200,
                      'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_samples': 20},
    'deep_balanced': {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 600,
                      'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 2.0, 'reg_lambda': 6.0, 'min_child_samples': 12},
}

N_FEAT = {
    'wide': 21, 'deep': 19, 'v48': 11, 'safety': 23,
    'shallow_fast': 25, 'balanced': 20, 'aggressive': 15,
    'reg_heavy': 22, 'deep_balanced': 20,
}

# V317 targets to force reg_heavy
FORCED_REG_TARGETS = {'Q1', 'Q3', 'S4'}
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
    log.info("Generating test z-score features (global)...")
    train_feat_cols = [c for c in train_df.columns
                       if c not in META_COLS | set(TARGETS)
                       and not c.endswith('_zscore')
                       and np.issubdtype(train_df[c].dtype, np.number)]
    test_feat_cols = [c for c in test_df.columns
                      if c not in META_COLS | set(TARGETS)
                      and not c.endswith('_zscore')
                      and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_feat_cols) & set(test_feat_cols)
    log.info(f"Common base columns: {len(common_cols)}")
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
    log.info(f"Generated {len(zscore_cols)} z-score features")
    return test_df, zscore_cols


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V318 — Forced Strong Regularization on Q1/Q3/S4")
    log.info("Q1/Q3/S4 → reg_heavy (reg_alpha=10, reg_lambda=20)")
    log.info("Q2/S1/S2/S3 → V316 best cfg")
    log.info("Goal: stable gaps + competitive OOF → LB < 0.63893")
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
    test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    student_oof_avg = {t: [] for t in TARGETS}
    per_seed_train_oofs = {t: [] for t in TARGETS}
    cfg_per_target = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        # Decide cfg
        if t in FORCED_REG_TARGETS:
            cfg_name = 'reg_heavy'
            log.info(f"    FORCED: {cfg_name} (reg_heavy)")
        else:
            cfg_name = V316_BEST[t]
            log.info(f"    V316 best: {cfg_name}")
        
        n_feat = N_FEAT[cfg_name]
        cfg = STUDENT_CFGS[cfg_name]
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        log.info(f"    n_feat={n_feat}, cfg={cfg}")
        
        all_seed_oofs = []
        all_seed_tests = []
        seed_student_oofs = []
        
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
            all_seed_oofs.append(seed_oof)
            all_seed_tests.append(seed_test)
            
            s_oof = log_loss(y, seed_oof)
            seed_student_oofs.append(s_oof)
            
            if si < 5 or si % 3 == 0 or si == N_SEEDS - 1:
                log.info(f"    Seed {si:2d} (s{seed}): OOF={s_oof:.5f}")
        
        # Meta learner
        stacked = np.column_stack(all_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        
        for i in range(N_SEEDS):
            test_preds[t][:, i] = all_seed_tests[i]
        student_oof_avg[t] = seed_student_oofs
        per_seed_train_oofs[t] = all_seed_oofs
        cfg_per_target[t] = cfg_name
        
        log.info(f"    Student avg OOF: {np.mean(seed_student_oofs):.5f}")
    
    # Compute overall results
    target_oofs = {}
    for t in TARGETS:
        target_oofs[t] = log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V318 RESULTS (forced reg on Q1/Q3/S4)")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        s_avg = np.mean(student_oof_avg[t])
        gap = s_avg - target_oofs[t]
        cfg_used = cfg_per_target[t]
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (cfg={cfg_used}, student={s_avg:.5f}, gap={gap:.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V146: 0.63169 | V308: 0.62235 | V316: 0.61800 | V317: 0.61859")
    log.info(f"  Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ vs V316: {avg_oof - 0.61800:+.5f}")
    log.info(f"  Δ vs V317: {avg_oof - 0.61859:+.5f}")
    
    # Gap analysis
    log.info(f"\n  Gap analysis:")
    for t in TARGETS:
        s_avg = np.mean(student_oof_avg[t])
        gap = s_avg - target_oofs[t]
        v316_gap = {
            'Q1': 0.053, 'Q2': 0.010, 'Q3': 0.041, 'S1': 0.027,
            'S2': 0.032, 'S3': 0.029, 'S4': 0.042,
        }[t]
        improvement = v316_gap - gap
        status = "✅" if gap < 0.020 else ("⚠️" if gap < 0.035 else "❌")
        log.info(f"    {t}: gap={gap:.4f} (V316: {v316_gap:.3f}, Δ={improvement:+.4f}) {status}")
    
    global_student_avg = np.mean([np.mean(student_oof_avg[t]) for t in TARGETS])
    log.info(f"  Global Student avg: {global_student_avg:.5f}")
    
    max_gap = max(np.mean(student_oof_avg[t]) - target_oofs[t] for t in TARGETS)
    min_gap = min(np.mean(student_oof_avg[t]) - target_oofs[t] for t in TARGETS)
    gap_std = np.std([np.mean(student_oof_avg[t]) - target_oofs[t] for t in TARGETS])
    log.info(f"  Max gap: {max_gap:.4f}")
    log.info(f"  Min gap: {min_gap:.4f}")
    log.info(f"  Gap std: {gap_std:.4f}")
    
    # OOF-LB Gap Estimation
    v308_gap = 0.01658
    avg_gap = np.mean([np.mean(student_oof_avg[t]) - target_oofs[t] for t in TARGETS])
    
    # Conservative: use max gap
    if max_gap < 0.025:
        est_gap = avg_gap * 0.6  # very confident
        confidence = "HIGH"
    elif max_gap < 0.035:
        est_gap = avg_gap * 0.7
        confidence = "HIGH-MED"
    elif max_gap < 0.045:
        est_gap = max_gap * 0.5
        confidence = "MEDIUM"
    else:
        est_gap = max_gap * 0.6
        confidence = "LOW"
    
    pred_lb = avg_oof + est_gap
    
    log.info(f"\n  OOF-LB Gap Estimation:")
    log.info(f"    V308 gap: +{v308_gap:.5f} (actual)")
    log.info(f"    V318 avg gap: {avg_gap:.4f}, max gap: {max_gap:.4f}")
    log.info(f"    Estimated V318 gap: +{est_gap:.5f} (confidence: {confidence})")
    log.info(f"    Predicted LB: {pred_lb:.5f}")
    log.info(f"    V308 LB: 0.63893")
    log.info(f"    Δ vs V308 LB: {pred_lb - 0.63893:+.5f}")
    
    log.info(f"{'='*70}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_stacked_all = {}
    for t in TARGETS:
        stacked_test = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        y_t = train_df[t].values.astype(np.float64)
        meta_t = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta_t.fit(np.column_stack([per_seed_train_oofs[t][i] for i in range(N_SEEDS)]), y_t)
        test_stacked_all[t] = meta_t.predict_proba(stacked_test)[:, 1]
    
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_stacked_all[t]
    
    sub_path = SUBMIT / f"submission_v318_forced_reg_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V318',
        'name': f"Forced reg_heavy on Q1/Q3/S4 + {N_SEEDS} Seeds, C={META_C}",
        'avg_oof': round(float(avg_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'forced_reg_targets': list(FORCED_REG_TARGETS),
        'v146_avg_oof': 0.63169,
        'v308_avg_oof': 0.62235,
        'v316_avg_oof': 0.61800,
        'v317_avg_oof': 0.61859,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v316': round(float(avg_oof - 0.61800), 5),
        'delta_vs_v317': round(float(avg_oof - 0.61859), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'per_target_cfg': cfg_per_target,
        'student_oof_avg': {t: round(float(np.mean(student_oof_avg[t])), 5) for t in TARGETS},
        'per_target_gap': {t: round(float(np.mean(student_oof_avg[t]) - target_oofs[t]), 5) for t in TARGETS},
        'global_student_avg_oof': round(float(global_student_avg), 5),
        'max_gap': round(float(max_gap), 5),
        'avg_gap': round(float(avg_gap), 5),
        'gap_std': round(float(gap_std), 5),
        'estimated_gap': round(float(est_gap), 5),
        'gap_confidence': confidence,
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'predicted_improvement_vs_v308': round(float(pred_lb - 0.63893), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v318_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
