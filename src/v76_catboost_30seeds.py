"""
V76 — CatBoost exact V61 reproduction + 30 seeds

V61 params (verified from source):
- iterations=1000, lr=0.03, depth=6
- bagging_temperature=0.5, l2_leaf_reg=3.0, random_strength=1.0
- boosting_type: NOT specified (defaults to Plain)
- 3-fold GroupKFold, 5 seeds originally → 30 seeds here
- No subsample, no colsample
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import catboost as cb

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


# EXACTLY V61 params + variations
CB_CONFIGS = [
    {'name': 'cb_v61_exact',   'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_v61_nobag',   'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'bagging': 1.0, 'rs': 1.0},
    {'name': 'cb_v61_l2=5',    'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_v61_l2=1',    'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 1.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_v61_rs=3',    'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'bagging': 0.5, 'rs': 3.0},
    {'name': 'cb_deep',        'iter': 1500, 'lr': 0.02, 'depth': 7, 'l2': 5.0, 'bagging': 0.5, 'rs': 1.0},
    {'name': 'cb_wide',        'iter': 800,  'lr': 0.05, 'depth': 6, 'l2': 2.0, 'bagging': 0.6, 'rs': 0.5},
]

N_SEEDS = 30
N_FOLDS = 3


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V76 — CatBoost exact V61 params + 30 seeds × 3-fold")
    log.info("  V61: iterations=1000, lr=0.03, depth=6, l2=3, bagging=0.5, rs=1.0")
    log.info("  NO boosting_type='Ordered' (use default Plain)")
    log.info("=" * 70)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)}, Subjects: {np.unique(groups).shape[0]}")
    
    all_results = {}
    all_predictions = {}
    
    for target in TARGETS:
        t_t = time.time()
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate: {y.mean():.3f}, leak-cols: {len(leak_cols)})")
        
        X = train[leak_cols].fillna(0).values.astype(np.float64)
        spw_global = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        # Feature ranking
        sn = [sanitize(c) for c in leak_cols]
        import lightgbm as lgb
        ds_rank = lgb.Dataset(X, label=y, feature_name=sn)
        mr = lgb.train(
            {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
             'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
             'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
             'reg_alpha': 0.5, 'reg_lambda': 2.0, 'scale_pos_weight': spw_global,
             'random_state': 42, 'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1},
            ds_rank, num_boost_round=100
        )
        imp = mr.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        del mr, ds_rank
        gc.collect()
        
        best_overall_cv = float('inf')
        best_overall_info = None
        
        for nf in [5, 10, 15, 20, 25, 30]:
            sel_cols = [r[0] for r in ranked[:nf]]
            sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]
            X_sel = X[:, sel_idx]
            
            for cfg in CB_CONFIGS:
                oof_cb = np.zeros(len(y))
                cnt = 0
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_sel, y, groups)):
                    X_tr, X_va = X_sel[tr_idx], X_sel[va_idx]
                    y_tr = y[tr_idx]
                    spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    
                    fold_preds = []
                    for s in range(N_SEEDS):
                        params = {
                            'iterations': cfg['iter'],
                            'learning_rate': cfg['lr'],
                            'depth': cfg['depth'],
                            'loss_function': 'Logloss',
                            'eval_metric': 'Logloss',
                            'random_seed': s,
                            'verbose': 0,
                            'task_type': 'CPU',
                            'bagging_temperature': cfg['bagging'],
                            'l2_leaf_reg': cfg['l2'],
                            'random_strength': cfg['rs'],
                            'scale_pos_weight': spw_f,
                            'max_ctr_complexity': 1,
                            # NO boosting_type — use default Plain
                        }
                        m = cb.CatBoostClassifier(**params)
                        m.fit(X_tr, y_tr, verbose=0)
                        pred_va = np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
                        fold_preds.append(pred_va)
                        del m
                    
                    # Average predictions across seeds for this fold
                    avg_fold_pred = np.mean(fold_preds, axis=0)
                    oof_cb[va_idx] += avg_fold_pred
                    cnt += 1
                    del fold_preds
                
                oof_cb_avg = oof_cb / cnt
                cv = log_loss(y, np.clip(oof_cb_avg, 1e-15, 1 - 1e-15))
                
                log.info(f"    {cfg['name']:15s} n_feat={nf:2d}: cv={cv:.4f}")
                del oof_cb
                gc.collect()
                
                if cv < best_overall_cv:
                    best_overall_cv = cv
                    best_overall_info = {
                        'model': 'catboost',
                        'cfg': cfg['name'],
                        'n_feat': nf,
                        'cv': cv,
                    }
            
            del X_sel
            gc.collect()
        
        log.info(f"\n  ✅ Best {target}: cv={best_overall_cv:.4f} ({best_overall_info})")
        log.info(f"     [{time.time() - t_t:.0f}s]")
        
        all_results[target] = best_overall_info
        
        # Final training
        nf = best_overall_info['n_feat']
        sel_cols = [r[0] for r in ranked[:nf]]
        sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
        sel_idx = [leak_cols.index(c) for c in sel_cols]
        X_all = X[:, sel_idx]
        X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        
        cfg = next(c for c in CB_CONFIGS if c['name'] == best_overall_info['cfg'])
        test_preds = np.zeros(len(X_test))
        for s in range(N_SEEDS):
            spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            params = {
                'iterations': cfg['iter'], 'learning_rate': cfg['lr'],
                'depth': cfg['depth'], 'loss_function': 'Logloss',
                'eval_metric': 'Logloss', 'random_seed': s,
                'verbose': 0, 'task_type': 'CPU',
                'bagging_temperature': cfg['bagging'],
                'l2_leaf_reg': cfg['l2'], 'random_strength': cfg['rs'],
                'scale_pos_weight': spw_f, 'max_ctr_complexity': 1,
            }
            m = cb.CatBoostClassifier(**params)
            m.fit(X_all, y, verbose=0)
            test_preds += np.clip(m.predict_proba(X_test)[:, 1], 0.0001, 0.9999)
            del m
        test_preds /= N_SEEDS
        
        all_predictions[target] = np.clip(test_preds, 0.0001, 0.9999)
        log.info(f"  {target} test_mean: {test_preds.mean():.4f}")
        
        del X_all, X_test
        gc.collect()
    
    # Summary
    avg_cv = np.mean([v['cv'] for v in all_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V76 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = all_results[t]
        log.info(f"  {t}: cv={r['cv']:.4f} ({r['cfg']}, n_feat={r['n_feat']})")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Target: 0.5000 | Current: {avg_cv:.4f} | Gap: {avg_cv - 0.5:.4f}")
    log.info(f"  V61 avg: 0.5830 | Gap to V61: {avg_cv - 0.5830:.4f}")
    log.info(f"  Total time: {time.time() - t_start:.0f}s")
    
    # Submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = all_predictions[t]
    sub_path = SUBMIT / f"submission_v76_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")
    
    meta = {
        'version': 'V76_catboost_v61_exact',
        'name': 'CatBoost exact V61 params + 30 seeds × 3-fold',
        'n_seeds': N_SEEDS,
        'n_folds': N_FOLDS,
        'avg_cv': float(avg_cv),
        'results': all_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v76_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
