"""
V387 — V308 + Bagged Model Ensemble (Prediction Averaging)

Hypothesis: V308 (no bagging, meta OOF=0.622, student=0.692)과 bagged model
(V367-like, bag ratio 0.6, meta OOF=0.602, student=0.647)의 예측을 average하면
서로 다른 bias를 보정하여 더 robust한 predictions 얻음.

Key insight from V367/V368:
- Bagged models: student OOF 0.647 (V308 0.692 대비 매우 낮음)
- Bagged meta OOF: 0.602 (V308 0.622 대비 -0.020)
- Bagging이 student calibration을 크게 개선

V387: V308 predictions과 bagged predictions을 target별로 average.
- Bag ratio: 0.6 (V367 최적)
- V308과 bagged model은 동일한 features (282개), 동일한 feature ranking
- 차이: bagging (feature sampling) + 더 많은 seeds

Expected:
- Student avg: ~0.670 (V308 0.692, bagged 0.647 average)
- Meta OOF: ~0.612 (V308 0.622, bagged 0.602 average)
- Predicted LB: ~0.630-0.635 (V308 0.639 개선)
- Risk: Medium (bagged model의 OOF-LB gap이 클 수 있음)

Bagging strategy:
- Feature bagging: 각 seed에서 feature의 60%만 사용
- More seeds (30): V308 15 seeds + bagged 15 seeds
- Bagging은 GroupKFold 내에서 per-fold 수행
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
N_SEEDS = 15  # V308 seeds
N_BAG_SEEDS = 15  # Bagged seeds
META_C = 10.0
BAG_RATIO = 0.6  # V367 최적


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


def train_bagged_model(train_df, test_df, sel_cols, y, group, cfg, n_train, n_test,
                       n_seeds, bag_ratio, gkf, prefix=""):
    """Train bagged model with feature sampling."""
    all_sel_cols = sel_cols
    n_feat = len(all_sel_cols)
    n_bag_feat = max(int(n_feat * bag_ratio), 5)
    
    per_seed_oofs = []
    per_seed_test = []
    student_oofs = []
    
    for si in range(n_seeds):
        seed = SEED + si * 7 + 100  # offset to avoid overlap with V308 seeds
        
        # Feature bagging: randomly select 60% of features
        rng = np.random.RandomState(seed)
        bag_indices = rng.choice(n_feat, size=n_bag_feat, replace=False)
        bag_cols = [all_sel_cols[i] for i in bag_indices]
        
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[bag_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[bag_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in bag_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[bag_cols].fillna(0).values.astype(np.float64))
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        per_seed_test.append(seed_test)
        student_oofs.append(log_loss(y, seed_oof))
        
        if si < 5 or si % 3 == 0:
            log.info(f"    {prefix}Seed {si:2d} (bag {n_bag_feat}/{n_feat}): "
                     f"OOF={log_loss(y, seed_oof):.5f}")
    
    # Equal average (bagged doesn't need stacking - equal avg is fine)
    equal_avg = np.mean(per_seed_oofs, axis=0)
    equal_avg_ll = log_loss(y, equal_avg)
    
    # Also try stacking
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    meta_oof = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
    
    student_avg = np.mean(student_oofs)
    
    return {
        'per_seed_oofs': per_seed_oofs,
        'per_seed_test': per_seed_test,
        'equal_avg_train': equal_avg,
        'equal_avg_ll': equal_avg_ll,
        'meta_oof': meta_oof,
        'student_avg': student_avg,
        'equal_avg_test': np.mean(per_seed_test, axis=0),
    }


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V387 — V308 + Bagged Model Ensemble (Prediction Averaging)")
    log.info("Hypothesis: V308 + bagged model prediction average → better calibration")
    log.info("V308: OOF=0.62235, LB=0.63893, student=0.692")
    log.info("V367 bag 0.6: OOF=0.60219, student=0.647")
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
    
    # Results storage
    v308_results = {}
    bagged_results = {}
    ensemble_results = {}
    all_student_oofs = []
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        # ── V308 (standard stacking) ──
        log.info(f"    Training V308 (standard stacking)...")
        
        v308_seed_oofs = []
        v308_seed_test = []
        v308_student_oofs = []
        
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
            v308_seed_oofs.append(seed_oof)
            v308_seed_test.append(seed_test)
            v308_student_oofs.append(log_loss(y, seed_oof))
        
        # V308 meta-learner
        v308_stacked_train = np.column_stack(v308_seed_oofs)
        v308_stacked_test = np.column_stack(v308_seed_test)
        v308_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        v308_meta.fit(v308_stacked_train, y)
        v308_meta_oof = log_loss(y, np.clip(v308_meta.predict_proba(v308_stacked_train)[:, 1], 0.001, 0.999))
        v308_meta_test = np.clip(v308_meta.predict_proba(v308_stacked_test)[:, 1], 0.001, 0.999)
        
        v308_equal_avg = np.mean(v308_seed_oofs, axis=0)
        v308_equal_ll = log_loss(y, v308_equal_avg)
        v308_student_avg = np.mean(v308_student_oofs)
        
        log.info(f"    V308: meta_OOF={v308_meta_oof:.5f}, equal_OOF={v308_equal_ll:.5f}, "
                 f"student_avg={v308_student_avg:.5f}")
        
        # ── Bagged Model ──
        log.info(f"    Training bagged model (bag_ratio={BAG_RATIO}, {N_BAG_SEEDS} seeds)...")
        
        bagged = train_bagged_model(
            train_df, test_df, sel_cols, y, group, cfg,
            n_train, n_test, N_BAG_SEEDS, BAG_RATIO, gkf, prefix="Bag-"
        )
        
        log.info(f"    Bagged: equal_avg_OOF={bagged['equal_avg_ll']:.5f}, "
                 f"student_avg={bagged['student_avg']:.5f}")
        
        # ── Ensemble (Average V308 + Bagged predictions) ──
        # Use meta prediction from V308, equal_avg from bagged
        # Average them
        
        # V308 meta test prediction
        v308_final = v308_meta_test
        
        # Bagged: try both equal_avg_train (for OOF) and equal_avg_test (for prediction)
        bagged_equal_train = bagged['equal_avg_train']
        bagged_equal_final = bagged['equal_avg_test']
        bagged_equal_ll = log_loss(y, bagged_equal_train)  # recalculate for training data
        
        # Bagged meta test
        bagged_stacked_train = np.column_stack(bagged['per_seed_oofs'])
        bagged_stacked_test = np.column_stack(bagged['per_seed_test'])
        bagged_meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        bagged_meta.fit(bagged_stacked_train, y)
        bagged_meta_final = np.clip(bagged_meta.predict_proba(bagged_stacked_test)[:, 1], 0.001, 0.999)
        
        # Ensemble OOF (use training predictions)
        ens_meta_equal_train = 0.5 * v308_meta.predict_proba(v308_stacked_train)[:, 1] + 0.5 * bagged_equal_train
        ens_meta_meta_train = 0.5 * v308_meta.predict_proba(v308_stacked_train)[:, 1] + 0.5 * bagged_meta.predict_proba(bagged_stacked_train)[:, 1]
        
        ens_meta_equal_oof = log_loss(y, np.clip(ens_meta_equal_train, 0.001, 0.999))
        ens_meta_meta_oof = log_loss(y, np.clip(ens_meta_meta_train, 0.001, 0.999))
        
        # Ensemble test predictions
        ens_meta_equal_test = 0.5 * v308_meta_test + 0.5 * bagged_equal_final
        ens_meta_meta_test = 0.5 * v308_meta_test + 0.5 * bagged_meta_final
        
        # Test different weights (using OOF for selection)
        best_w = 0.5
        best_w_oof = ens_meta_equal_oof
        for w_bag_i in np.arange(0.1, 0.91, 0.05):
            w_v308_i = 1.0 - w_bag_i
            ens_i = w_v308_i * ens_meta_equal_train + (1 - w_v308_i) * bagged_equal_train
            oof_i = log_loss(y, np.clip(ens_i, 0.001, 0.999))
            if oof_i < best_w_oof:
                best_w_oof = oof_i
                best_w = 1 - w_v308_i  # w_bag
        
        # Find test predictions with best weight
        ens_final = best_w * bagged_equal_final + (1 - best_w) * v308_meta_test
        
        ens_student = (v308_student_avg + bagged['student_avg']) / 2.0
        
        log.info(f"    {t}: Best ensemble weight: w_bag={best_w:.2f}, OOF={best_w_oof:.5f}")
        log.info(f"    {t}: V308 student={v308_student_avg:.5f}, Bagged student={bagged['student_avg']:.5f}")
        log.info(f"    {t}: Ensemble student={ens_student:.5f}")
        
        v308_results[t] = {
            'meta_oof': v308_meta_oof, 'equal_ll': v308_equal_ll, 'student_avg': v308_student_avg,
            'meta_test': v308_meta_test, 'equal_avg': v308_equal_avg,
            'seed_oofs': v308_seed_oofs, 'seed_test': v308_seed_test,
        }
        bagged_results[t] = bagged
        ensemble_results[t] = {
            'best_ens': 'meta_equal', 'best_w': best_w,
            'best_w_oof': best_w_oof, 'ens_student': ens_student,
            'ens_test': ens_final,
        }
        all_student_oofs.extend(v308_student_oofs)
        all_student_oofs.extend(bagged['per_seed_oofs'])
    
    # Summary
    v308_avg_meta = np.mean([v308_results[t]['meta_oof'] for t in TARGETS])
    v308_avg_equal = np.mean([v308_results[t]['equal_ll'] for t in TARGETS])
    v308_avg_student = np.mean([v308_results[t]['student_avg'] for t in TARGETS])
    
    bagged_avg_equal = np.mean([bagged_results[t]['equal_avg_ll'] for t in TARGETS])
    bagged_avg_meta = np.mean([bagged_results[t]['meta_oof'] for t in TARGETS])
    bagged_avg_student = np.mean([bagged_results[t]['student_avg'] for t in TARGETS])
    
    ens_avg_oof = np.mean([ensemble_results[t]['best_w_oof'] for t in TARGETS])
    ens_avg_student = np.mean([ensemble_results[t]['ens_student'] for t in TARGETS])
    
    v308_pred_lb = v308_avg_meta + 0.01658  # V308 gap
    ens_pred_lb = ens_avg_oof + 0.01658  # optimistic gap assumption
    
    log.info(f"\n{'='*70}")
    log.info("V387 RESULTS")
    log.info(f"{'='*70}")
    
    for t in TARGETS:
        v = v308_results[t]
        b = bagged_results[t]
        e = ensemble_results[t]
        v308_t = V308_OOF[t]
        log.info(f"  {t}: V308_meta={v['meta_oof']:.5f} (Δ: {v['meta_oof']-v308_t:+.5f}), "
                 f"Bagged_equal={b['equal_avg_ll']:.5f}, "
                 f"Ensemble={e['best_w_oof']:.5f} (w_bag={e['best_w']:.2f})")
    
    log.info(f"\n  V308:  AVG meta={v308_avg_meta:.5f}, equal={v308_avg_equal:.5f}, "
             f"student={v308_avg_student:.5f}")
    log.info(f"  Bagged: AVG equal={bagged_avg_equal:.5f}, meta={bagged_avg_meta:.5f}, "
             f"student={bagged_avg_student:.5f}")
    log.info(f"  Ensemble: AVG OOF={ens_avg_oof:.5f}, student={ens_avg_student:.5f}")
    
    log.info(f"\n  Δ vs V308:")
    log.info(f"    V308 avg_meta: {v308_avg_meta:.5f}")
    log.info(f"    Bagged avg_equal: {bagged_avg_equal:.5f} (Δ: {bagged_avg_equal-v308_avg_meta:+.5f})")
    log.info(f"    Ensemble avg: {ens_avg_oof:.5f} (Δ: {ens_avg_oof-v308_avg_meta:+.5f})")
    log.info(f"    Bagged student: {bagged_avg_student:.5f} (Δ: {bagged_avg_student-v308_avg_student:+.5f})")
    log.info(f"    Ensemble student: {ens_avg_student:.5f} (Δ: {ens_avg_student-v308_avg_student:+.5f})")
    
    log.info(f"\n  Predicted LB (V308 gap):")
    log.info(f"    V308:  {v308_pred_lb:.5f}")
    log.info(f"    Bagged: {bagged_avg_equal + 0.01658:.5f}")
    log.info(f"    Ensemble: {ens_pred_lb:.5f}")
    
    ens_beats = ens_pred_lb < 0.63893
    log.info(f"\n  Beats V308: {ens_beats}")
    log.info(f"{'='*70}")
    
    # Save meta
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    meta_data = {
        'version': 'V387',
        'name': 'V308 + Bagged Model Ensemble',
        'v308_avg_meta_oof': round(float(v308_avg_meta), 5),
        'v308_avg_equal_oof': round(float(v308_avg_equal), 5),
        'v308_avg_student': round(float(v308_avg_student), 5),
        'bagged_avg_equal_oof': round(float(bagged_avg_equal), 5),
        'bagged_avg_meta_oof': round(float(bagged_avg_meta), 5),
        'bagged_avg_student': round(float(bagged_avg_student), 5),
        'ens_avg_oof': round(float(ens_avg_oof), 5),
        'ens_avg_student': round(float(ens_avg_student), 5),
        'predicted_lb': round(float(ens_pred_lb), 5),
        'beats_v308': bool(ens_beats),
        'per_target': {
            t: {
                'v308_meta_oof': round(v308_results[t]['meta_oof'], 5),
                'v308_equal_ll': round(v308_results[t]['equal_ll'], 5),
                'v308_student_avg': round(v308_results[t]['student_avg'], 5),
                'bagged_equal_ll': round(bagged_results[t]['equal_avg_ll'], 5),
                'bagged_student_avg': round(bagged_results[t]['student_avg'], 5),
                'ensemble_oof': round(ensemble_results[t]['best_w_oof'], 5),
                'ensemble_w': round(ensemble_results[t]['best_w'], 3),
                'ensemble_student': round(ensemble_results[t]['ens_student'], 5),
            } for t in TARGETS
        },
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v387_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    if ens_beats:
        log.info(f"\nV387 BEATS V308! Creating submission...")
        # Create submission
        sub = pd.DataFrame()
        sub['subject_id'] = test_df['subject_id'].values
        sub['sleep_date'] = test_df['sleep_date'].values
        sub['lifelog_date'] = test_df['lifelog_date'].values
        
        for t in TARGETS:
            v = v308_results[t]
            b = bagged_results[t]
            e = ensemble_results[t]
            w_bag = e['best_w']
            w_v308 = 1.0 - w_bag
            
            # Use best ensemble: V308 meta + Bagged equal avg
            ens_pred = w_v308 * v['meta_test'] + w_bag * b['equal_avg']
            sub[t] = np.clip(ens_pred, 0.001, 0.999)
        
        sub_path = SUBMIT / f"submission_v387_ensemble_{ts}.csv"
        sub.to_csv(sub_path, index=False)
        log.info(f"Saved submission: {sub_path}")
        meta_data['submission_file'] = str(sub_path)
    else:
        log.info(f"\nV386 does NOT beat V308. No submission.")
    
    return ens_avg_oof, meta_data


if __name__ == '__main__':
    main()
