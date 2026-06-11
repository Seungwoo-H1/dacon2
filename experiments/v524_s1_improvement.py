#!/usr/bin/env python3
"""
V524 — Reproduce V522 exactly, then add S1 variants

V522 baseline: avg_gap=0.02550, S1_gap=0.036

Goal: Verify we can reproduce V522 first, then test S1 improvements.
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
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
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


def train_target_models(target, train_df, test_df, sel_cols, group, learner, n_est, **model_params):
    """Train all seeds for a target and return OOF and test predictions."""
    y = train_df[target].values.astype(np.float64)
    n_train = len(train_df)
    n_test = len(test_df)
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    per_seed_oofs = []
    per_seed_tests = []
    
    for si in range(N_SEEDS):
        seed = SEED + si * 11
        seed_oof = np.zeros(n_train)
        seed_test = np.zeros(n_test)
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
            X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            X_test = test_df[sel_cols].fillna(0).values.astype(np.float64)
            
            if learner == 'xgb':
                params = {**model_params, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
                ds_train = xgb.DMatrix(X_tr, label=y_tr, feature_names=sel_cols)
                m = xgb.train(params, ds_train, num_boost_round=n_est)
                pred_va = m.predict(xgb.DMatrix(X_va, feature_names=sel_cols))
                pred_test = m.predict(xgb.DMatrix(X_test, feature_names=sel_cols))
            else:
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**model_params, 'scale_pos_weight': spw, 'random_state': seed,
                         'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=n_est)
                pred_va = m.predict(X_va)
                pred_test = m.predict(X_test)
            
            seed_oof[va_idx] = pred_va
            seed_test += pred_test
        
        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
        per_seed_tests.append(seed_test)
    
    return per_seed_oofs, per_seed_tests


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V524 — Reproduce V522, then improve S1")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Z-score features
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m, s = np.mean(tv), np.std(tv, ddof=0)
        if s < 1e-8: s = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m) / s
        test_df[zc] = (ev - m) / s
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    n_train = len(train_df)
    n_test = len(test_df)
    
    log.info(f"Features: {len(train_feat_cols)} train, {len(test_feat_cols)} test")
    
    v308_gaps = {
        'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124,
        'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039
    }
    
    # V522 configs
    XGB_CFGS = {
        'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
        'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
        'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
        's_wide':    {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    }
    LGBM_CFGS = {
        'wide':    {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'safety':  {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }
    
    V522_CONFIGS = {
        'Q1':  {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 'q_narrow',  'n_est': 600},
        'Q2':  {'n_feat': 14, 'learner': 'xgb', 'cfg_name': 'q_deep',    'n_est': 800},
        'Q3':  {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 'q_strong',  'n_est': 500},
        'S1':  {'n_feat': 10, 'learner': 'lgbm', 'cfg_name': 'wide',     'n_est': 300},
        'S2':  {'n_feat': 7,  'learner': 'xgb', 'cfg_name': 's_wide',    'n_est': 300},
        'S3':  {'n_feat': 23, 'learner': 'lgbm', 'cfg_name': 'safety',   'n_est': 1000},
        'S4':  {'n_feat': 20, 'learner': 'lgbm', 'cfg_name': 'wide',     'n_est': 300},
    }
    
    # Step 1: Reproduce V522
    log.info("\n--- Step 1: Reproduce V522 ---")
    
    all_seed_oofs_v522 = {}
    test_preds_v522 = {}
    group = train_df['subject_id'].values
    
    for target in TARGETS:
        cfg = V522_CONFIGS[target]
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:cfg['n_feat']]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        learner = cfg['learner']
        cfg_raw = {'n_estimators': cfg['n_est']}
        
        if learner == 'xgb':
            xgb_cfg = XGB_CFGS[cfg['cfg_name']]
            cfg_raw.update(xgb_cfg)
        else:
            lgbm_cfg = LGBM_CFGS[cfg['cfg_name']]
            cfg_raw.update(lgbm_cfg)
        
        oofs, tests = train_target_models(target, train_df, test_df, sel_cols, group, learner, cfg['n_est'], **cfg_raw)
        all_seed_oofs_v522[target] = oofs
        test_preds_v522[target] = tests
        
        if target == 'S1':
            t_y = train_df[target].values
            t_student_lls = [log_loss(t_y, so) for so in oofs]
            avg_student = np.mean(t_student_lls)
            
            oofs_arr = np.column_stack(oofs)
            avg_pred = np.mean(oofs_arr, axis=1)
            std_pred = np.std(oofs_arr, axis=1)
            meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
            meta.fit(np.column_stack([avg_pred, std_pred]), t_y)
            train_oof = meta.predict_proba(np.column_stack([avg_pred, std_pred]))[:, 1]
            meta_ll = log_loss(t_y, np.clip(train_oof, 0.001, 0.999))
            gap = avg_student - meta_ll
            log.info(f"V522 S1: student={avg_student:.5f}, meta={meta_ll:.5f}, gap={gap:.5f}")
    
    # Compute V522 avg gap (2D meta)
    v522_gaps = {}
    v522_avg_gap = 0
    for t in TARGETS:
        oofs_arr = np.column_stack(all_seed_oofs_v522[t])
        avg_pred = np.mean(oofs_arr, axis=1)
        std_pred = np.std(oofs_arr, axis=1)
        meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
        meta.fit(np.column_stack([avg_pred, std_pred]), train_df[t].values)
        train_oof = meta.predict_proba(np.column_stack([avg_pred, std_pred]))[:, 1]
        meta_ll = log_loss(train_df[t].values, np.clip(train_oof, 0.001, 0.999))
        t_y = train_df[t].values
        t_student_lls = [log_loss(t_y, so) for so in all_seed_oofs_v522[t]]
        avg_student = np.mean(t_student_lls)
        gap = avg_student - meta_ll
        v522_gaps[t] = gap
        v522_avg_gap += gap
        log.info(f"  V522 {t}: gap={gap:.5f} vs308={v308_gaps[t]:.3f} {'✅' if gap < v308_gaps[t] else '❌'}")
    v522_avg_gap /= 7
    log.info(f"\nV522 REPRO: avg_gap={v522_avg_gap:.5f} (target: 0.02550)")
    
    # Step 2: V522 baseline is established. Now test S1 variants.
    log.info("\n--- Step 2: Test S1 variants (keep others at V522) ---")
    
    # Pre-compute all targets EXCEPT S1
    all_seed_oofs_fixed = {}
    for target in ['Q1', 'Q2', 'Q3', 'S2', 'S3', 'S4']:
        cfg = V522_CONFIGS[target]
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked = rank_features(train_df, feat_cols_clean, target)
        sel_cols = ranked[:cfg['n_feat']]
        learner = cfg['learner']
        cfg_raw = {'n_estimators': cfg['n_est']}
        if learner == 'xgb':
            cfg_raw.update(XGB_CFGS[cfg['cfg_name']])
        else:
            cfg_raw.update(LGBM_CFGS[cfg['cfg_name']])
        oofs, _ = train_target_models(target, train_df, test_df, sel_cols, group, learner, cfg['n_est'], **cfg_raw)
        all_seed_oofs_fixed[target] = oofs
    
    # S1 candidates
    S1_CANDIDATES = [
        {'n_feat': 5,  'learner': 'xgb', 'cfg': XGB_CFGS['q_strong'], 'n_est': 500, 'desc': 'xgb_strong_n5'},
        {'n_feat': 10, 'learner': 'xgb', 'cfg': XGB_CFGS['q_strong'], 'n_est': 500, 'desc': 'xgb_strong_n10'},
        {'n_feat': 15, 'learner': 'xgb', 'cfg': XGB_CFGS['q_strong'], 'n_est': 500, 'desc': 'xgb_strong_n15'},
        {'n_feat': 20, 'learner': 'xgb', 'cfg': XGB_CFGS['q_strong'], 'n_est': 500, 'desc': 'xgb_strong_n20'},
        {'n_feat': 5,  'learner': 'xgb', 'cfg': XGB_CFGS['q_narrow'], 'n_est': 600, 'desc': 'xgb_narrow_n5'},
        {'n_feat': 10, 'learner': 'xgb', 'cfg': XGB_CFGS['q_narrow'], 'n_est': 600, 'desc': 'xgb_narrow_n10'},
        {'n_feat': 15, 'learner': 'xgb', 'cfg': XGB_CFGS['q_narrow'], 'n_est': 600, 'desc': 'xgb_narrow_n15'},
        {'n_feat': 20, 'learner': 'xgb', 'cfg': XGB_CFGS['q_narrow'], 'n_est': 600, 'desc': 'xgb_narrow_n20'},
        {'n_feat': 10, 'learner': 'lgbm', 'cfg': LGBM_CFGS['wide'], 'n_est': 300, 'desc': 'lgbm_wide_n10'},
    ]
    
    results = []
    for ci, s1_cfg in enumerate(S1_CANDIDATES):
        log.info(f"\n[{ci+1}/{len(S1_CANDIDATES)}] {s1_cfg['desc']} (n_feat={s1_cfg['n_feat']})")
        
        s1_clean = remove_leak(train_feat_cols, 'S1')
        s1_ranked = rank_features(train_df, s1_clean, 'S1')
        s1_sel = s1_ranked[:s1_cfg['n_feat']]
        
        oofs, _ = train_target_models('S1', train_df, test_df, s1_sel, group, s1_cfg['learner'], s1_cfg['n_est'], **s1_cfg['cfg'])
        all_seed_oofs_fixed['S1'] = oofs
        
        # Compute 2D meta for ALL targets
        avg_gap = 0
        target_gaps = {}
        for t in TARGETS:
            oofs_arr = np.column_stack(all_seed_oofs_fixed[t])
            avg_pred = np.mean(oofs_arr, axis=1)
            std_pred = np.std(oofs_arr, axis=1)
            meta = LogisticRegression(C=META_C, max_iter=2000, random_state=SEED)
            meta.fit(np.column_stack([avg_pred, std_pred]), train_df[t].values)
            train_oof = meta.predict_proba(np.column_stack([avg_pred, std_pred]))[:, 1]
            meta_ll = log_loss(train_df[t].values, np.clip(train_oof, 0.001, 0.999))
            t_y = train_df[t].values
            t_student_lls = [log_loss(t_y, so) for so in all_seed_oofs_fixed[t]]
            avg_student = np.mean(t_student_lls)
            gap = avg_student - meta_ll
            target_gaps[t] = gap
            avg_gap += gap
        avg_gap /= 7
        
        vs308 = sum(1 for t in TARGETS if target_gaps[t] < v308_gaps[t])
        log.info(f"  avg_gap={avg_gap:.5f}, S1_gap={target_gaps['S1']:.5f}, vs308={vs308}/7")
        for t in TARGETS:
            vs = "✅" if target_gaps[t] < v308_gaps[t] else "❌"
            log.info(f"    {t}: gap={target_gaps[t]:.5f} V308={v308_gaps[t]:.3f} {vs}")
        
        if avg_gap < 0.025:
            log.info(f"  🎯🎯🎯 BELOW 0.025! 🎯🎯🎯")
        
        results.append({
            'key': s1_cfg['desc'], 'avg_gap': avg_gap, 'target_gaps': target_gaps, 'vs308': vs308
        })
    
    # Summary
    results.sort(key=lambda x: x['avg_gap'])
    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS")
    log.info(f"{'='*70}")
    for r in results:
        marker = ""
        if r['avg_gap'] < 0.025: marker = " 🎯🎯🎯"
        elif r['avg_gap'] < 0.030: marker = " ⭐"
        elif r['target_gaps']['S1'] < 0.036: marker = " 📉"
        log.info(f"  {r['key']}: avg_gap={r['avg_gap']:.5f}, S1={r['target_gaps']['S1']:.5f} V308=0.020 vs308={r['vs308']}/7{marker}")
    
    best = results[0]
    log.info(f"\n✅ BEST: {best['key']} with avg_gap={best['avg_gap']:.5f}")
    
    result = {
        'version': 'V524',
        'v522_avg_gap': float(v522_avg_gap),
        'results': results,
        'best_key': best['key'],
        'best_gap': float(best['avg_gap']),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_time_s': round(time.time() - t_start, 1),
    }
    
    result_path = EXPERIMENTS / f'v524_{result["timestamp"]}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"📝 Result saved: {result_path}")
    log.info(f"\nTotal time: {time.time() - t_start:.1f}s")
    return result

if __name__ == '__main__':
    main()
