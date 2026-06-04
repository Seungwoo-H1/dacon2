"""
V332 — Per-Target Optimal Feature Count Sweep

Hypothesis: V308의 V53 sweep n_feat values가 V308의 z-score enriched context에서
optimal하지 않을 수 있음. 특히 calibration이 bad인 targets(Q1, Q3)는 feature count가
过多하여 overconfidence를 유발할 수 있음.

각 target별 n_feat를 sweep하여 OOF-LB gap이最小인 config 찾기.

Changes:
1. n_feat sweep: 5, 7, 10, 13, 15, 19, 23, 27, 31
2. Per-target OOF 평가 후 best n_feat 선택
3. 같은 cfg, 같은 seeds(15) 유지

Risk: Overfitting to CV only — need OOF-LB gap analysis
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


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V332 — Per-Target Optimal Feature Count Sweep")
    log.info("Testing n_feat: 5,7,10,13,15,19,23,27,31")
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
    
    # For each target, sweep n_feat
    sweep_results = {}
    n_feat_options = [5, 7, 10, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        n_train_local = len(train_df)
        n_test_local = len(test_df)
        
        n_feat_results = []
        for nf in n_feat_options:
            if nf > len(ranked):
                continue
            sel_cols = ranked[:nf]
            sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
            if len(sel_cols_test) != nf:
                continue
            
            # Train
            train_oofs = np.zeros((n_train_local, N_SEEDS))
            test_preds = np.zeros((n_test_local, N_SEEDS))
            
            for si, seed in enumerate(seeds):
                seed_oof = np.zeros(n_train_local)
                seed_test = np.zeros(n_test_local)
                
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
                train_oofs[:, si] = seed_oof
                test_preds[:, si] = seed_test
            
            # Ensemble + meta
            student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
            stacked = np.column_stack(list(train_oofs.T))  # (450, 15)
            meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
            meta.fit(stacked, y)
            final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
            oof_ll = log_loss(y, final_oof)
            
            n_feat_results.append({
                'n_feat': nf,
                'oof': oof_ll,
                'student_oof': log_loss(y, student_oof),
            })
            
            log.info(f"  {t} n_feat={nf}: OOF={oof_ll:.5f}, student={log_loss(y, student_oof):.5f}")
        
        # Find best n_feat
        best = min(n_feat_results, key=lambda x: x['oof'])
        sweep_results[t] = best
        
        log.info(f"  {t} BEST: n_feat={best['n_feat']}, OOF={best['oof']:.5f}")
    
    # Now build final submission with best n_feat for each target
    log.info(f"\n{'='*70}")
    log.info("Building final submission with optimized n_feat per target")
    log.info(f"{'='*70}")
    
    all_oofs = {}
    all_test_preds = {}
    best_nfeats = {}
    
    for t in TARGETS:
        best_nf = sweep_results[t]['n_feat']
        best_nfeats[t] = best_nf
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        sel_cols = ranked[:best_nf]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        
        n_train = len(train_df)
        n_test = len(test_df)
        
        train_oofs = np.zeros((n_train, N_SEEDS))
        test_preds = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
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
            train_oofs[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
        stacked = np.column_stack(list(train_oofs.T))  # (450, 15)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        # Test
        stacked_test = np.column_stack([test_preds[:, si] for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = test_pred
        
        log.info(f"  {t}: best_nfeat={best_nf}, OOF={oof_ll:.5f}")
    
    avg_oof = np.mean(list(all_oofs.values()))
    log.info(f"\n{'='*70}")
    log.info(f"V332 RESULTS (optimized n_feat per target)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: best_nfeat={best_nfeats[t]}, OOF={all_oofs[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  Δ vs V308: {avg_oof - 0.62235:+.5f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v332_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V332',
        'name': 'Per-Target Optimal Feature Count Sweep',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'best_nfeats': {t: best_nfeats[t] for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    
    meta_path = EXPERIMENTS / f'v332_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data

if __name__ == '__main__':
    main()
