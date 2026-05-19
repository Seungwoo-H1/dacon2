"""
V144 — Double-Blind Ensemble Stacking

Hypothesis: V140's single stacking is a local optimum.
V141-V143 tried modifying stacking and failed.
Instead: run TWO independent stacking pipelines with DIFFERENT feature sets,
then ensemble their predictions.

Key insight from V142: fold drift weighting broke OOF-LB correlation.
V144 avoids weighting entirely — uses purely orthogonal approaches.

Pipeline A: V140 style — top-K stability-ranked features per target
Pipeline B: All leak-free features — broader coverage, different signal

Each pipeline: 3 seeds × GroupKFold 5-fold → LR meta-learner (V140 structure)
Final: Equal-weight ensemble of Pipeline A + Pipeline B predictions

This preserves V140's stable stacking while adding orthogonal diversity.
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
N_SEEDS = 3
META_C = 0.1


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
    """Rank features by LGBM importance on this target."""
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def run_stacking_pipeline(train_df, test_df, feat_cols, target, selected_cols, n_seeds=3):
    """
    Standard V140 stacking: n_seeds × GroupKFold 5-fold → LR meta-learner.
    Uses only the specified selected_cols (for Pipeline A vs B).
    Returns OOF predictions, test predictions, and per-seed OOFs.
    """
    y = train_df[target].values.astype(np.float64)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    per_seed_oofs = []
    per_seed_test = []
    
    for si in range(n_seeds):
        seed = SEED + si * 7
        seed_oof = np.zeros(len(train_df))
        seed_test = np.zeros(len(test_df))
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[selected_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[selected_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[target]['cfg']
            cfg = CFGS[cfg_name]
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in selected_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[selected_cols].fillna(0).values.astype(np.float64))
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        per_seed_test.append(seed_test)
    
    # Level 1: LR meta-learner
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    
    oof_pred = meta.predict_proba(stacked)[:, 1]
    test_stacked = np.column_stack(per_seed_test)
    test_pred = meta.predict_proba(test_stacked)[:, 1]
    
    return oof_pred, test_pred


def compute_feature_set_a(feat_cols):
    """
    Pipeline A: V140 style — top-K stability-ranked features per target.
    Same as V140.
    """
    return feat_cols  # ranking happens per-target in rank_features()


def compute_feature_set_b(feat_cols):
    """
    Pipeline B: All leak-free features, ranked by importance.
    Instead of top-K, use MORE features — different signal coverage.
    """
    # Use more features than V140 (2× the top-K)
    return feat_cols  # will select more in run_stacking_pipeline_b


def run_stacking_pipeline_b(train_df, test_df, feat_cols, target, n_seeds=3, use_all_feats=False):
    """
    Pipeline B: Same stacking structure as V140, but uses MORE features.
    V140 uses top-K=11~23. Pipeline B uses top-50~70.
    Different feature subsets = different model = orthogonal diversity.
    """
    y = train_df[target].values.astype(np.float64)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    fc_leaked = remove_leak(feat_cols, target)
    ranked = rank_features(train_df, fc_leaked, target)
    
    # Use top-K but much wider than V140
    v140_n_feat = V53_SWEEP[target]['n_feat']
    n_feat_b = min(v140_n_feat * 4, len(fc_leaked))  # 4× wider
    
    selected_cols = ranked[:n_feat_b]
    
    per_seed_oofs = []
    per_seed_test = []
    
    for si in range(n_seeds):
        seed = SEED + si * 7
        seed_oof = np.zeros(len(train_df))
        seed_test = np.zeros(len(test_df))
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[selected_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[selected_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[target]['cfg']
            cfg = CFGS[cfg_name]
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in selected_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            seed_oof[va_idx] = m.predict(X_va)
            seed_test += m.predict(test_df[selected_cols].fillna(0).values.astype(np.float64))
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        per_seed_test.append(seed_test)
    
    # Level 1: LR meta-learner
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)
    
    oof_pred = meta.predict_proba(stacked)[:, 1]
    test_stacked = np.column_stack(per_seed_test)
    test_pred = meta.predict_proba(test_stacked)[:, 1]
    
    return oof_pred, test_pred, n_feat_b


if __name__ == '__main__':
    t_start = time.time()
    log.info("=" * 70)
    log.info("V144 — Double-Blind Ensemble Stacking")
    log.info("Pipeline A: V140 top-K features")
    log.info("Pipeline B: 4× wider feature set")
    log.info("Final: Equal-weight ensemble")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
    
    # Pipeline A: V140 style (top-K features, same as V140)
    log.info("\n[1/3] Pipeline A: V140 top-K features")
    train_oof_a = {}
    test_preds_a = {}
    for t in TARGETS:
        fc_leaked = remove_leak(feat_cols, t)
        ranked = rank_features(train_df, fc_leaked, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel = ranked[:n_feat]
        oof, test_p = run_stacking_pipeline(train_df, test_df, feat_cols, t, sel)
        train_oof_a[t] = oof
        test_preds_a[t] = test_p
        ll = log_loss(train_df[t].values, np.clip(oof, 0.001, 0.999))
        log.info(f"  {t}: OOF={ll:.5f} (n_feat={n_feat})")
    
    avg_oof_a = np.mean([log_loss(train_df[t].values, np.clip(train_oof_a[t], 0.001, 0.999)) for t in TARGETS])
    log.info(f"  Pipeline A AVG OOF: {avg_oof_a:.5f}")
    
    # Pipeline B: 4× wider features
    log.info("\n[2/3] Pipeline B: 4× wider feature set")
    train_oof_b = {}
    test_preds_b = {}
    for t in TARGETS:
        v140_n_feat = V53_SWEEP[t]['n_feat']
        n_feat_b = min(v140_n_feat * 4, len(remove_leak(feat_cols, t)))
        oof, test_p, actual_n = run_stacking_pipeline_b(train_df, test_df, feat_cols, t, n_seeds=N_SEEDS)
        train_oof_b[t] = oof
        test_preds_b[t] = test_p
        ll = log_loss(train_df[t].values, np.clip(oof, 0.001, 0.999))
        log.info(f"  {t}: OOF={ll:.5f} (n_feat={actual_n})")
    
    avg_oof_b = np.mean([log_loss(train_df[t].values, np.clip(train_oof_b[t], 0.001, 0.999)) for t in TARGETS])
    log.info(f"  Pipeline B AVG OOF: {avg_oof_b:.5f}")
    
    # Ensemble: equal-weight average of A and B
    log.info("\n[3/3] Ensemble: Pipeline A + Pipeline B")
    train_oof_ens = {}
    test_preds_ens = {}
    for t in TARGETS:
        # Equal-weight average (both [0,1] scale)
        ens_a = np.clip(train_oof_a[t], 0.001, 0.999)
        ens_b = np.clip(train_oof_b[t], 0.001, 0.999)
        train_oof_ens[t] = (ens_a + ens_b) / 2.0
        test_preds_ens[t] = (np.clip(test_preds_a[t], 0.001, 0.999) + np.clip(test_preds_b[t], 0.001, 0.999)) / 2.0
        
        ll = log_loss(train_df[t].values, np.clip(train_oof_ens[t], 0.001, 0.999))
        log.info(f"  {t}: OOF={ll:.5f}")
    
    avg_oof_ens = np.mean([log_loss(train_df[t].values, np.clip(train_oof_ens[t], 0.001, 0.999)) for t in TARGETS])
    log.info(f"\n  Ensemble AVG OOF: {avg_oof_ens:.5f}")
    log.info(f"  V140 AVG OOF: 0.64110")
    log.info(f"  Δ vs V140: {avg_oof_ens - 0.64110:+.5f}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds_ens[t]
    
    sub_path = SUBMIT / f"submission_v144_ensemble_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    
    # Meta
    meta_data = {
        'version': 'V144',
        'name': 'Double-Blind Ensemble Stacking (Pipeline A + Pipeline B)',
        'avg_oof_pipeline_a': round(float(avg_oof_a), 5),
        'avg_oof_pipeline_b': round(float(avg_oof_b), 5),
        'avg_oof_ensemble': round(float(avg_oof_ens), 5),
        'meta_C': META_C,
        'n_seeds': N_SEEDS,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof_ens[t], 0.001, 0.999))), 5) for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v144_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"  Saved: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
