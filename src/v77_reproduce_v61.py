"""
V77 — Reproduce V61's CV exactly

V61 hardcodes avg_cv_v60=0.5830 but we need to actually compute CV.
V61 used: 3-fold GroupKFold, 30 seeds, single CatBoost, per-target n_feat
  Q1:19, Q2:14, Q3:5, S1:21, S2:19, S3:21, S4:20

The key insight: V61 used 3-fold GroupKFold for CV but the code shows
test-prediction-only split. The CV must have been computed separately.

Let's do proper 3-fold GroupKFold OOF CV for each target with:
- Same feature set (leakage-clean, n_feat per V61 config)
- CatBoost with V61 params
- 30 seeds

We'll also try:
- n_feat sweep around V61's choices
- CatBoost + XGBoost single
- CatBoost + LGBM + XGBoost single
- Ensemble of top models
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import catboost as cb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout
)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min',
    'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count'}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def remove_leak(cols, target):
    leak = set()
    if target.startswith('S'):
        leak = LEAK_S
    elif target.startswith('Q'):
        leak = LEAK_Q
    return [c for c in cols if c not in leak]


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]


CB_CONFIGS = [
    {'name': 'cb_v61', 'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_v61_l2=1', 'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 1.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_v61_l2=5', 'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_deep', 'iter': 1500, 'lr': 0.02, 'depth': 7, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_wide', 'iter': 800, 'lr': 0.05, 'depth': 6, 'l2': 2.0, 'bagging': 0.6, 'rs': 0.5},
    {'name': 'cb_safe', 'iter': 1200, 'lr': 0.025, 'depth': 6, 'l2': 4.0, 'bagging': 0.4, 'rs': 2.0},
]

LGBM_CONFIGS = [
    {'name': 'lgb_v61', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'lgb_wide', 'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
]

N_SEEDS = 30
N_FOLDS = 3


def train_cb_predict(X_tr, y_tr, X_va, cfg, seed, spw):
    params = {
        'iterations': cfg['iter'], 'learning_rate': cfg['lr'], 'depth': cfg['depth'],
        'loss_function': 'Logloss', 'eval_metric': 'Logloss',
        'random_seed': seed, 'verbose': 0, 'task_type': 'CPU',
        'bagging_temperature': cfg['bagging'], 'l2_leaf_reg': cfg['l2'],
        'random_strength': cfg['rs'], 'scale_pos_weight': spw,
        'max_ctr_complexity': 1,
    }
    m = cb.CatBoostClassifier(**params)
    m.fit(X_tr, y_tr, verbose=0)
    pred = np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
    del m
    return pred


def train_lgbm_predict(X_tr, y_tr, X_va, feat_names, cfg, seed, spw):
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
        'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, params={'verbose': '-1'})
    m = lgb.train(params, ds_tr, num_boost_round=cfg['ne'])
    pred = np.clip(m.predict(X_va), 0.0001, 0.9999)
    del m, ds_tr
    return pred


def compute_oof(model_type, X, y, feat_names, cfg, n_seeds, n_folds, groups):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    cnt = 0
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        
        fold_oof = np.zeros(len(va_idx))
        for s in range(n_seeds):
            if model_type == 'catboost':
                fold_oof += train_cb_predict(X_tr, y_tr, X_va, cfg, s, spw_f)
            else:
                fold_oof += train_lgbm_predict(X_tr, y_tr, X_va, feat_names, cfg, s, spw_f)
        oof[va_idx] += fold_oof / n_seeds
        cnt += 1
        del fold_oof
    
    return oof / cnt


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V77 — V61 reproduction + n_feat sweep + model comparison")
    log.info("=" * 70)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)}")
    
    # V61 per-target configs
    V61_NFEAT = {
        'Q1': 19, 'Q2': 14, 'Q3': 5, 'S1': 21, 'S2': 19, 'S3': 21, 'S4': 20,
    }
    
    all_results = {}
    all_predictions = {}
    
    for target in TARGETS:
        t_t = time.time()
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate: {y.mean():.3f}, leak-cols: {len(leak_cols)})")
        
        X = train[leak_cols].fillna(0).values.astype(np.float64)
        
        # Feature ranking
        sn = [sanitize(c) for c in leak_cols]
        ds_rank = lgb.Dataset(X, label=y, feature_name=sn)
        mr = lgb.train(
            {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
             'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
             'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
             'reg_alpha': 0.5, 'reg_lambda': 2.0,
             'scale_pos_weight': max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1),
             'random_state': 42, 'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1},
            ds_rank, num_boost_round=100
        )
        imp = mr.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        del mr, ds_rank
        gc.collect()
        
        # V61 n_feat and ±3 around it
        target_nfeat = V61_NFEAT[target]
        nfeats = list(range(max(5, target_nfeat - 3), target_nfeat + 4))
        
        target_results = {}
        
        # Test each model + config + n_feat combo
        for nf in nfeats:
            sel_cols = [r[0] for r in ranked[:nf]]
            sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]
            X_sel = X[:, sel_idx]
            
            for cfg in CB_CONFIGS:
                oof = compute_oof('catboost', X_sel, y, sel_sn, cfg, N_SEEDS, N_FOLDS, groups)
                cv = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15))
                log.info(f"    CB {cfg['name']:15s} n_feat={nf:2d}: cv={cv:.4f}")
                target_results.setdefault(f'CB_{cfg["name"]}', []).append((nf, cv))
                del oof
                gc.collect()
            
            for cfg in LGBM_CONFIGS:
                oof = compute_oof('lgbm', X_sel, y, sel_sn, cfg, N_SEEDS, N_FOLDS, groups)
                cv = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15))
                log.info(f"    LGB {cfg['name']:15s} n_feat={nf:2d}: cv={cv:.4f}")
                target_results.setdefault(f'LGB_{cfg["name"]}', []).append((nf, cv))
                del oof
                gc.collect()
            
            del X_sel
            gc.collect()
        
        # Find best per-model
        model_best = {}
        for model_key, results_list in target_results.items():
            best_nf, best_cv = min(results_list, key=lambda x: x[1])
            model_best[model_key] = {'n_feat': best_nf, 'cv': best_cv}
        
        # Find global best
        global_best = min(model_best.items(), key=lambda x: x[1]['cv'])
        log.info(f"\n  ✅ Best {target}: {global_best[0]} cv={global_best[1]['cv']:.4f} (n_feat={global_best[1]['n_feat']})")
        log.info(f"  Model details:")
        for mk in sorted(model_best.keys(), key=lambda k: model_best[k]['cv']):
            mb = model_best[mk]
            marker = " ← BEST" if mk == global_best[0] else ""
            log.info(f"    {mk}: cv={mb['cv']:.4f} n_feat={mb['n_feat']}{marker}")
        log.info(f"     [{time.time() - t_t:.0f}s]")
        
        all_results[target] = {
            'global_best': global_best[0],
            'global_cv': global_best[1]['cv'],
            'global_nfeat': global_best[1]['n_feat'],
            'model_best': {k: v for k, v in model_best.items()},
        }
        
        # Train best model on all data
        best_model_type = global_best[0].split('_')[0]
        best_model_name = global_best[0].split('_')[1]
        best_nf = global_best[1]['n_feat']
        
        if best_model_type == 'CB':
            cfg = next(c for c in CB_CONFIGS if c['name'] == best_model_name)
            X_all = X[:, [leak_cols.index(c) for c in [r[0] for r in ranked[:best_nf]]]]
            X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, [leak_cols.index(c) for c in [r[0] for r in ranked[:best_nf]]]]
            test_preds = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_preds += train_cb(X_all, y, feat_names=None, cfg=cfg, seed=s, spw=spw_f)
            test_preds /= N_SEEDS
            
        elif best_model_type == 'LGB':
            cfg = next(c for c in LGBM_CONFIGS if c['name'] == best_model_name)
            X_all = X[:, [leak_cols.index(c) for c in [r[0] for r in ranked[:best_nf]]]]
            X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, [leak_cols.index(c) for c in [r[0] for r in ranked[:best_nf]]]]
            sel_sn_test = [sanitize(r[0]) for r in ranked[:best_nf]]
            test_preds = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_preds += train_lgbm(X_all, y, feat_names=sel_sn_test, cfg=cfg, seed=s, spw=spw_f)
            test_preds /= N_SEEDS
        
        all_predictions[target] = np.clip(test_preds, 0.0001, 0.9999)
        log.info(f"  {target} test_mean: {test_preds.mean():.4f}")
        
        del X_all, X_test
        gc.collect()
    
    # Summary
    avg_cv = np.mean([v['global_cv'] for v in all_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V77 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = all_results[t]
        log.info(f"  {t}: cv={r['global_cv']:.4f} ({r['global_best']}, n_feat={r['global_nfeat']})")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Target: 0.5000 | Current: {avg_cv:.4f} | Gap: {avg_cv - 0.5:.4f}")
    log.info(f"  V61 avg: 0.5830 | Gap to V61: {avg_cv - 0.5830:.4f}")
    log.info(f"  Total time: {time.time() - t_start:.0f}s")
    
    # Submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = all_predictions[t]
    sub_path = SUBMIT / f"submission_v77_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")
    
    meta = {
        'version': 'V77_v61_reproduce',
        'name': 'CatBoost/LGBM 30 seeds × 3-fold, n_feat sweep around V61',
        'avg_cv': float(avg_cv),
        'results': all_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v77_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
