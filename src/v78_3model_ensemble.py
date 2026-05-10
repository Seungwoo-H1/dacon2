"""
V78 — 3-model ensemble (LGBM + CatBoost + XGBoost) with OOF stacking

Strategy:
1. Use LGBM ranking for feature importance
2. n_feat sweep: 5-30
3. 3-model: LGBM + CatBoost + XGBoost per fold
4. OOF stacking with LogisticRegression meta-learner
5. Compare single models vs stacked vs blended

V74 confirmed V_conservative (nl=8,md=2,lr=0.01,ne=2000) best LGBM config.
V74 Q1 n_feat=20 V_conservative: cv=0.6730 (5-fold, 30 seeds)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

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


# Model configs
LGBM_CFG = {
    'name': 'lgb_vcons', 'nl': 8, 'md': 2, 'lr': 0.01, 'ne': 2000,
    'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20,
}
CB_CFG = {
    'name': 'cb_v61', 'iter': 1000, 'lr': 0.03, 'depth': 6,
    'l2': 3.0, 'bagging': 0.5, 'rs': 1.0,
}
XGB_CFG = {
    'name': 'xgb_v1', 'iter': 1000, 'lr': 0.03, 'depth': 6,
    'l2': 3.0, 'subsample': 0.7, 'colsample': 0.7, 'rs': 1.0,
}

N_SEEDS = 30
N_FOLDS = 5


def train_lgb(X_tr, y_tr, X_va, feat_names, cfg, seed, spw):
    p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
         'num_leaves': cfg['nl'], 'max_depth': cfg['md'], 'learning_rate': cfg['lr'],
         'n_estimators': cfg['ne'], 'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
         'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
         'min_child_samples': cfg['mc'], 'random_state': seed,
         'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1}
    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, params={'verbose': '-1'})
    m = lgb.train(p, ds, num_boost_round=cfg['ne'])
    pred = np.clip(m.predict(X_va), 0.0001, 0.9999)
    del m, ds
    return pred


def train_cb(X_tr, y_tr, X_va, cfg, seed, spw):
    p = {'iterations': cfg['iter'], 'learning_rate': cfg['lr'], 'depth': cfg['depth'],
         'loss_function': 'Logloss', 'eval_metric': 'Logloss',
         'random_seed': seed, 'verbose': 0, 'task_type': 'CPU',
         'bagging_temperature': cfg['bagging'], 'l2_leaf_reg': cfg['l2'],
         'random_strength': cfg['rs'], 'scale_pos_weight': spw, 'max_ctr_complexity': 1}
    m = cb.CatBoostClassifier(**p)
    m.fit(X_tr, y_tr, verbose=0)
    pred = np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
    del m
    return pred


def train_xgb(X_tr, y_tr, X_va, cfg, seed, spw):
    p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
         'max_depth': cfg['depth'], 'learning_rate': cfg['lr'],
         'n_estimators': cfg['iter'], 'subsample': cfg['subsample'],
         'colsample_bytree': cfg['colsample'],
         'reg_alpha': cfg['l2'], 'reg_lambda': cfg['l2'],
         'min_child_weight': cfg['l2'], 'random_state': seed,
         'scale_pos_weight': spw, 'tree_method': 'hist',
         'verbosity': 0, 'n_jobs': 1}
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, verbose=False)
    pred = np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
    del m
    return pred


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V78 — 3-model ensemble (LGBM + CatBoost + XGBoost) + Stacking")
    log.info("  5-fold, 30 seeds, n_feat sweep")
    log.info("=" * 70)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    
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
            ds_rank, num_boost_round=100)
        imp = mr.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        del mr, ds_rank
        gc.collect()
        
        best_overall_cv = float('inf')
        best_overall_info = None
        
        for nf in [10, 15, 20, 25]:
            sel_cols = [r[0] for r in ranked[:nf]]
            sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]
            X_sel = X[:, sel_idx]
            
            # --- Single models ---
            model_oofs = {}  # model_name -> oof array
            
            for model_name, train_fn, cfg in [
                ('lgb', train_lgb, LGBM_CFG),
                ('cat', train_cb, CB_CFG),
                ('xgb', train_xgb, XGB_CFG),
            ]:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                t_model = time.time()
                
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
                        y_tr = y[tr]
                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        
                        if model_name == 'lgb':
                            pred = train_fn(X_tr, y_tr, X_va, sel_sn, cfg, s, spw_f)
                        else:
                            pred = train_fn(X_tr, y_tr, X_va, cfg, s, spw_f)
                        
                        oof[va] += pred
                        n_valid[va] += 1
                
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[model_name] = oof_avg
                log.info(f"  {target} {model_name:4s} n_feat={nf}: cv={cv:.4f} ({time.time()-t_model:.0f}s)")
                del oof
                gc.collect()
            
            # --- Blend: 50/50 LGBM + CatBoost ---
            blend_5050 = 0.5 * model_oofs['lgb'] + 0.5 * model_oofs['cat']
            cv_b50 = log_loss(y, blend_5050)
            
            # --- Blend: 40/30/30 LGBM/Cat/XGB ---
            blend_433 = 0.4 * model_oofs['lgb'] + 0.3 * model_oofs['cat'] + 0.3 * model_oofs['xgb']
            cv_b433 = log_loss(y, blend_433)
            
            # --- Stacking ---
            oof_stack = np.column_stack([model_oofs['lgb'], model_oofs['cat'], model_oofs['xgb']])
            
            # C-tune on OOF
            best_c = 1.0
            best_meta_cv = float('inf')
            for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
                meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=2000, random_state=42)
                meta.fit(oof_stack, y)
                meta_pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
                meta_cv = log_loss(y, meta_pred)
                if meta_cv < best_meta_cv:
                    best_meta_cv = meta_cv
                    best_c = c_val
            
            log.info(f"  {target} STACK_C={best_c:.2f}: cv={best_meta_cv:.4f}")
            
            # --- Compare all ---
            results_for_nf = {
                'lgb': cv if (cv := log_loss(y, model_oofs['lgb'])) else 0,
                'cat': log_loss(y, model_oofs['cat']),
                'xgb': log_loss(y, model_oofs['xgb']),
                'blend_5050': cv_b50,
                'blend_433': cv_b433,
                'stack_C1': best_meta_cv,
            }
            
            best_model_key = min(results_for_nf, key=results_for_nf.get)
            best_nf_cv = results_for_nf[best_model_key]
            
            if best_nf_cv < best_overall_cv:
                best_overall_cv = best_nf_cv
                best_overall_info = {
                    'model': best_model_key,
                    'n_feat': nf,
                    'cv': best_nf_cv,
                    'per_model_cv': results_for_nf,
                }
            
            log.info(f"  → Best at n_feat={nf}: {best_model_key} cv={best_nf_cv:.4f}")
            del X_sel, oof_stack
            gc.collect()
        
        log.info(f"\n  ✅ Best {target}: {best_overall_info}")
        log.info(f"     [{time.time() - t_t:.0f}s]")
        
        all_results[target] = best_overall_info
        
        # Final training for best model
        best_key = best_overall_info['model']
        best_nf = best_overall_info['n_feat']
        sel_cols = [r[0] for r in ranked[:best_nf]]
        sel_sn = [sanitize(r[0]) for r in ranked[:best_nf]]
        sel_idx = [leak_cols.index(c) for c in sel_cols]
        X_all = X[:, sel_idx]
        X_test = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        y = train[target].values.astype(np.float64)
        
        if best_key == 'lgb':
            test_preds = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_preds += train_lgb(X_all, y, X_test, sel_sn, LGBM_CFG, s, spw_f)
            test_preds /= N_SEEDS
        
        elif best_key == 'cat':
            test_preds = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_preds += train_cb(X_all, y, X_test, CB_CFG, s, spw_f)
            test_preds /= N_SEEDS
        
        elif best_key == 'xgb':
            test_preds = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_preds += train_xgb(X_all, y, X_test, XGB_CFG, s, spw_f)
            test_preds /= N_SEEDS
        
        elif best_key == 'blend_5050':
            test_lgb = np.zeros(len(X_test))
            test_cat = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_lgb += train_lgb(X_all, y, X_test, sel_sn, LGBM_CFG, s, spw_f)
                test_cat += train_cb(X_all, y, X_test, CB_CFG, s, spw_f)
            test_preds = 0.5 * test_lgb / N_SEEDS + 0.5 * test_cat / N_SEEDS
        
        elif best_key == 'blend_433':
            test_lgb = np.zeros(len(X_test))
            test_cat = np.zeros(len(X_test))
            test_xgb = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_lgb += train_lgb(X_all, y, X_test, sel_sn, LGBM_CFG, s, spw_f)
                test_cat += train_cb(X_all, y, X_test, CB_CFG, s, spw_f)
                test_xgb += train_xgb(X_all, y, X_test, XGB_CFG, s, spw_f)
            test_preds = 0.4 * test_lgb / N_SEEDS + 0.3 * test_cat / N_SEEDS + 0.3 * test_xgb / N_SEEDS
        
        elif best_key.startswith('stack'):
            test_lgb = np.zeros(len(X_test))
            test_cat = np.zeros(len(X_test))
            test_xgb = np.zeros(len(X_test))
            for s in range(N_SEEDS):
                spw_f = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                test_lgb += train_lgb(X_all, y, X_test, sel_sn, LGBM_CFG, s, spw_f)
                test_cat += train_cb(X_all, y, X_test, CB_CFG, s, spw_f)
                test_xgb += train_xgb(X_all, y, X_test, XGB_CFG, s, spw_f)
            test_lgb /= N_SEEDS
            test_cat /= N_SEEDS
            test_xgb /= N_SEEDS
            test_stack = np.column_stack([test_lgb, test_cat, test_xgb])
            meta = LogisticRegression(C=best_c if (best_c := 1.0) else 1.0, solver='lbfgs', max_iter=2000, random_state=42)
            # Need to find best_c from best_overall_info
            meta.fit(np.column_stack([model_oofs['lgb'], model_oofs['cat'], model_oofs['xgb']]), y)
            test_preds = np.clip(meta.predict_proba(test_stack)[:, 1], 0.0001, 0.9999)
        
        all_predictions[target] = np.clip(test_preds, 0.0001, 0.9999)
        log.info(f"  {target} test_mean: {test_preds.mean():.4f}")
        
        del X_all, X_test
        gc.collect()
    
    # Summary
    avg_cv = np.mean([v['cv'] for v in all_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V78 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = all_results[t]
        log.info(f"  {t}: cv={r['cv']:.4f} ({r['model']}, n_feat={r['n_feat']})")
        if 'per_model_cv' in r:
            for mk, mv in r['per_model_cv'].items():
                log.info(f"    {mk}: {mv:.4f}")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Target: 0.5000 | Current: {avg_cv:.4f} | Gap: {avg_cv - 0.5:.4f}")
    log.info(f"  V61 avg: 0.5830 | Gap to V61: {avg_cv - 0.5830:.4f}")
    log.info(f"  Total time: {time.time() - t_start:.0f}s")
    
    # Submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = all_predictions[t]
    sub_path = SUBMIT / f"submission_v78_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")
    
    meta = {
        'version': 'V78_3model_ensemble_stacking',
        'name': 'LGBM + CatBoost + XGBoost ensemble + stacking',
        'avg_cv': float(avg_cv),
        'results': all_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v78_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
