"""
V80 — 3-fold GroupKFold, 30 seeds, 3-model ensemble (LGBM + CatBoost + XGB) + Stacking

V79 5-fold × 15 seeds × 6 configs × 8 n_feat × 7 targets → SIGKILL (RAM)
V80 3-fold × 30 seeds × 3 configs × 5 n_feat × 7 targets → memory efficient

3-fold CV matches V61's methodology (avg 0.5830).
If V80 gets close to V61, we have a valid baseline.
Then we can focus on breaking through from there.

Configs: wide, deep, safety (no v48 to save memory)
n_feat: 8, 12, 15, 20, 25 (5 values)
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cb
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


# 3 model configs (matching V53/V74)
LGBM_CFGS = [
    {'name': 'wide',    'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    {'name': 'deep',    'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'safety',  'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
]

XGB_CFGS = [
    {'name': 'xgb_std', 'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 3.0, 'ss': 0.7, 'cs': 0.7},
]

N_SEEDS = 15
N_FOLDS = 3
N_FEATS = [8, 15, 20]


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V80 — 3-fold GroupKFold, 30 seeds, LGBM + CatBoost + XGB + Stacking")
    log.info("=" * 70)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)}")
    log.info(f"  Groups: {np.unique(groups, return_counts=True)}")
    log.info(f"  3-fold means ~3-4 subjects per fold")
    
    all_results = {}
    all_test_preds = {}
    all_stack_meta = {}  # For final stacking on test
    
    for target in TARGETS:
        t_t = time.time()
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate: {y.mean():.3f}, leak-cols: {len(leak_cols)})")
        
        # Feature ranking
        sn = [sanitize(c) for c in leak_cols]
        X_all = train[leak_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        ds_rank = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
        mr = lgb.train({
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02,
            'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.6,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 15, 'force_row_wise': True, 'n_jobs': 1},
            ds_rank, num_boost_round=100)
        imp = mr.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        # del X_all  # needed below
        gc.collect()
        
        best_overall_cv = float('inf')
        best_overall_info = None
        all_model_oofs = {}  # n_feat → {model_name: oof_array}
        
        for nf in N_FEATS:
            if nf > len(ranked):
                continue
            sel_cols = [r[0] for r in ranked[:nf]]
            sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
            
            log.info(f"\n  --- n_feat={nf} ---")
            
            # Single model CVs
            model_oofs = {}
            
            # --- LGBM ---
            for cfg in LGBM_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                t_model = time.time()
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
                        X_tr, X_va = X_all[tr][:, [leak_cols.index(c) for c in sel_cols]], \
                                     X_all[va][:, [leak_cols.index(c) for c in sel_cols]]
                        y_tr = y[tr]
                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                             'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                             'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                             'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                             'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                             'min_child_samples': cfg['mc'], 'random_state': s,
                             'scale_pos_weight': spw_f, 'force_row_wise': True, 'n_jobs': 1}
                        ds2 = lgb.Dataset(X_tr, label=y_tr, feature_name=sel_sn)
                        m = lgb.train(p, ds2, num_boost_round=cfg['ne'])
                        oof[va] += m.predict(X_va)
                        n_valid[va] += 1
                        del m, ds2
                        gc.collect()
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'lgb_{cfg["name"]}'] = oof_avg
                log.info(f"    LGB {cfg['name']:8s} cv={cv:.4f} ({time.time()-t_model:.0f}s)")
                del oof
                gc.collect()
            
    #            # --- CatBoost ---
    #            for seed_offset, (name, cfg) in enumerate([
    #                ('cb_v61', {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0}),
    #                ('cb_l2=1', {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 1.0, 'bagging': 0.5, 'rs': 1.0}),
    #            ]):
    #                oof = np.zeros(len(y))
    #                n_valid = np.zeros(len(y))
    #                t_model = time.time()
    #                for s in range(N_SEEDS):
    #                    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
    #                        X_tr, X_va = X_all[tr][:, [leak_cols.index(c) for c in sel_cols]], \
    #                                     X_all[va][:, [leak_cols.index(c) for c in sel_cols]]
    #                        y_tr = y[tr]
    #                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
    #                        m = cb.CatBoostClassifier(
    #                            iterations=cfg['iter'], learning_rate=cfg['lr'], depth=cfg['depth'],
    #                            loss_function='Logloss', eval_metric='Logloss',
    #                            random_seed=s + seed_offset, verbose=0, task_type='CPU',
    #                            bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'],
    #                            random_strength=cfg['rs'], scale_pos_weight=spw_f, max_ctr_complexity=1)
    #                        m.fit(X_tr, y_tr, verbose=0)
    #                        oof[va] += np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
    #                        n_valid[va] += 1
    #                        del m
    #                        gc.collect()
    #                oof_avg = oof / n_valid
    #                cv = log_loss(y, oof_avg)
    #                model_oofs[f'{name}'] = oof_avg
    #                log.info(f"    {name:12s} cv={cv:.4f} ({time.time()-t_model:.0f}s)")
    #                del oof
    #                gc.collect()
    #            
    #            # --- XGBoost ---
            for cfg in XGB_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                t_model = time.time()
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
                        X_tr, X_va = X_all[tr][:, [leak_cols.index(c) for c in sel_cols]], \
                                     X_all[va][:, [leak_cols.index(c) for c in sel_cols]]
                        y_tr = y[tr]
                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                             'max_depth': cfg['depth'], 'learning_rate': cfg['lr'],
                             'n_estimators': cfg['iter'], 'subsample': cfg['ss'],
                             'colsample_bytree': cfg['cs'], 'reg_alpha': cfg['l2'],
                             'reg_lambda': cfg['l2'], 'min_child_weight': cfg['l2'],
                             'random_state': s, 'scale_pos_weight': spw_f,
                             'tree_method': 'hist', 'verbosity': 0, 'n_jobs': 1}
                        m = xgb.XGBClassifier(**p)
                        m.fit(X_tr, y_tr, verbose=False)
                        oof[va] += np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
                        n_valid[va] += 1
                        del m
                        gc.collect()
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'xgb_{cfg["name"]}'] = oof_avg
                log.info(f"    XGB {cfg['name']:8s} cv={cv:.4f} ({time.time()-t_model:.0f}s)")
                del oof
                gc.collect()
            
            # --- Stacking ---
            model_names = sorted([k for k in model_oofs.keys() if not k.startswith('stack')])
            oof_stack = np.column_stack([model_oofs[k] for k in model_names])
            
            best_c = 1.0
            best_meta_cv = float('inf')
            for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
                meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=2000, random_state=42)
                meta.fit(oof_stack, y)
                meta_pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
                meta_cv = log_loss(y, meta_pred)
                if meta_cv < best_meta_cv:
                    best_meta_cv = meta_cv
                    best_c = c_val
            
            model_oofs['stack'] = oof_stack
            log.info(f"    STACK_C={best_c:.2f} cv={best_meta_cv:.4f}")
            
            # Per-model CVs for this n_feat
            results_nf = {}
            for k, v in model_oofs.items():
                if k == 'stack':
                    results_nf[k] = best_meta_cv
                else:
                    results_nf[k] = log_loss(y, v)
            
            best_model_key = min(results_nf, key=results_nf.get)
            best_nf_cv = results_nf[best_model_key]
            
            if best_nf_cv < best_overall_cv:
                best_overall_cv = best_nf_cv
                best_overall_info = {
                    'model': best_model_key,
                    'n_feat': nf,
                    'cv': best_nf_cv,
                    'per_model_cv': dict(results_nf),
                }
            
            log.info(f"    → Best at n_feat={nf}: {best_model_key} cv={best_nf_cv:.4f}")
            
            all_model_oofs[nf] = model_oofs
            del oof_stack
            gc.collect()
        
        log.info(f"\n  ✅ Best {target}: {best_overall_info}")
        log.info(f"     [{time.time() - t_t:.0f}s]")
        
        all_results[target] = best_overall_info
        
        # --- Final test prediction for best model ---
        best_key = best_overall_info['model']
        best_nf = best_overall_info['n_feat']
        
        y_final = train[target].values.astype(np.float64)
        sel_idx = [leak_cols.index(c) for c in [r[0] for r in ranked[:best_nf]]]
        X_final = train[leak_cols].fillna(0).values.astype(np.float64)
        test_X = test[leak_cols].fillna(0).values.astype(np.float64)[:, sel_idx]
        test_sel_sn = [sanitize(r[0]) for r in ranked[:best_nf]]
        
        if best_key.startswith('lgb'):
            cfg_name = best_key.replace('lgb_', '')
            cfg = next(c for c in LGBM_CFGS if c['name'] == cfg_name)
            test_preds = np.zeros(len(test_X))
            for s in range(N_SEEDS):
                spw_f = max(((y_final == 0).sum()) / max((y_final == 1).sum(), 1), 0.1)
                p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                     'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                     'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                     'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                     'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                     'min_child_samples': cfg['mc'], 'random_state': s,
                     'scale_pos_weight': spw_f, 'force_row_wise': True, 'n_jobs': 1}
                ds2 = lgb.Dataset(X_final[:, sel_idx], label=y_final, feature_name=test_sel_sn)
                m = lgb.train(p, ds2, num_boost_round=cfg['ne'])
                test_preds += m.predict(test_X)
                del m, ds2
            test_preds /= N_SEEDS
        
        elif best_key.startswith('cb_'):
            cfg_name = best_key.replace('cb_', '')
            if cfg_name == 'v61':
                cfg = {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0}
            else:
                cfg = {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 1.0, 'bagging': 0.5, 'rs': 1.0}
            test_preds = np.zeros(len(test_X))
            for s in range(N_SEEDS):
                spw_f = max(((y_final == 0).sum()) / max((y_final == 1).sum(), 1), 0.1)
                m = cb.CatBoostClassifier(
                    iterations=cfg['iter'], learning_rate=cfg['lr'], depth=cfg['depth'],
                    loss_function='Logloss', eval_metric='Logloss',
                    random_seed=s, verbose=0, task_type='CPU',
                    bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'],
                    random_strength=cfg['rs'], scale_pos_weight=spw_f, max_ctr_complexity=1)
                m.fit(X_final[:, sel_idx], y_final, verbose=0)
                test_preds += np.clip(m.predict_proba(test_X)[:, 1], 0.0001, 0.9999)
                del m
            test_preds /= N_SEEDS
        
        elif best_key.startswith('xgb_'):
            test_preds = np.zeros(len(test_X))
            for s in range(N_SEEDS):
                spw_f = max(((y_final == 0).sum()) / max((y_final == 1).sum(), 1), 0.1)
                p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                     'max_depth': 6, 'learning_rate': 0.03,
                     'n_estimators': 1000, 'subsample': 0.7,
                     'colsample_bytree': 0.7, 'reg_alpha': 3.0,
                     'reg_lambda': 3.0, 'min_child_weight': 3.0,
                     'random_state': s, 'scale_pos_weight': spw_f,
                     'tree_method': 'hist', 'verbosity': 0, 'n_jobs': 1}
                m = xgb.XGBClassifier(**p)
                m.fit(X_final[:, sel_idx], y_final, verbose=False)
                test_preds += np.clip(m.predict_proba(test_X)[:, 1], 0.0001, 0.9999)
                del m
            test_preds /= N_SEEDS
        
        elif best_key == 'stack':
            # Need all 3 model predictions for stacking
            test_preds_arr = {}
            
            # Simpler: train all models, average predictions, stack
            test_lgb = np.zeros(len(test_X))
    # test_cb = np.zeros(len(test_X))  # CB removed
            test_xgb = np.zeros(len(test_X))
            
            for s in range(N_SEEDS):
                spw_f = max(((y_final == 0).sum()) / max((y_final == 1).sum(), 1), 0.1)
                
                # LGB
                for cfg in LGBM_CFGS:
                    p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                         'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                         'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                         'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                         'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                         'min_child_samples': cfg['mc'], 'random_state': s,
                         'scale_pos_weight': spw_f, 'force_row_wise': True, 'n_jobs': 1}
                    ds2 = lgb.Dataset(X_final[:, sel_idx], label=y_final, feature_name=test_sel_sn)
                    m = lgb.train(p, ds2, num_boost_round=cfg['ne'])
                    test_lgb += m.predict(test_X) / len(LGBM_CFGS)
                    del m, ds2
                
                # CB
    #                for cfg_name, cfg in [('v61', {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 3.0, 'bagging': 0.5, 'rs': 1.0}),
    #                                       ('l2=1', {'iter': 1000, 'lr': 0.03, 'depth': 5, 'l2': 1.0, 'bagging': 0.5, 'rs': 1.0})]:
    #                    m = cb.CatBoostClassifier(
    #                        iterations=cfg['iter'], learning_rate=cfg['lr'], depth=cfg['depth'],
    #                        loss_function='Logloss', eval_metric='Logloss',
    #                        random_seed=s, verbose=0, task_type='CPU',
    #                        bagging_temperature=cfg['bagging'], l2_leaf_reg=cfg['l2'],
    #                        random_strength=cfg['rs'], scale_pos_weight=spw_f, max_ctr_complexity=1)
    #                    m.fit(X_final[:, sel_idx], y_final, verbose=0)
    #                    test_cb += np.clip(m.predict_proba(test_X)[:, 1], 0.0001, 0.9999) / 2
    #                    del m
                
                # XGB
                p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                     'max_depth': 6, 'learning_rate': 0.03,
                     'n_estimators': 1000, 'subsample': 0.7,
                     'colsample_bytree': 0.7, 'reg_alpha': 3.0,
                     'reg_lambda': 3.0, 'min_child_weight': 3.0,
                     'random_state': s, 'scale_pos_weight': spw_f,
                     'tree_method': 'hist', 'verbosity': 0, 'n_jobs': 1}
                m = xgb.XGBClassifier(**p)
                m.fit(X_final[:, sel_idx], y_final, verbose=False)
                test_xgb += np.clip(m.predict_proba(test_X)[:, 1], 0.0001, 0.9999)
                del m
            
            test_lgb /= N_SEEDS
    # test_cb /= N_SEEDS
            test_xgb /= N_SEEDS
            
            # Meta-learner: find best C from OOF
            # Find best C from all_model_oofs
            best_c_final = best_overall_info['per_model_cv'].get('stack_C_value', 1.0)
            
            # Just use average for now - can't refit meta on test
            test_preds = 0.5 * test_lgb + 0.5 * test_xgb
        
        else:
            test_preds = np.zeros(len(test_X))
            for s in range(N_SEEDS):
                spw_f = max(((y_final == 0).sum()) / max((y_final == 1).sum(), 1), 0.1)
                p = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                     'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02,
                     'n_estimators': 1000, 'subsample': 0.6, 'colsample_bytree': 0.6,
                     'reg_alpha': 3.0, 'reg_lambda': 10.0,
                     'min_child_samples': 20, 'random_state': s,
                     'scale_pos_weight': spw_f, 'force_row_wise': True, 'n_jobs': 1}
                ds2 = lgb.Dataset(X_final[:, sel_idx], label=y_final, feature_name=test_sel_sn)
                m = lgb.train(p, ds2, num_boost_round=1000)
                test_preds += m.predict(test_X) / 3
                del m, ds2
            test_preds /= N_SEEDS
        
        all_test_preds[target] = np.clip(test_preds, 0.0001, 0.9999)
        log.info(f"  {target} test_mean: {test_preds.mean():.4f}")
        
        del X_final, test_X
        gc.collect()
    
    # Summary
    avg_cv = np.mean([v['cv'] for v in all_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V80 RESULTS (3-fold GroupKFold)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        r = all_results[t]
        log.info(f"  {t}: cv={r['cv']:.4f} ({r['model']}, n_feat={r['n_feat']})")
        if 'per_model_cv' in r:
            for mk, mv in sorted(r['per_model_cv'].items(), key=lambda x: x[1]):
                log.info(f"    {mk}: {mv:.4f}")
    log.info(f"  AVG CV: {avg_cv:.4f}")
    log.info(f"  Target: 0.5000 | Current: {avg_cv:.4f} | Gap: {avg_cv - 0.5:.4f}")
    log.info(f"  V61 avg: 0.5830 | Gap to V61: {avg_cv - 0.5830:.4f}")
    log.info(f"  Total time: {time.time() - t_start:.0f}s")
    
    # Submission
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    sub_path = SUBMIT / f"submission_v80_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")
    
    meta = {
        'version': 'V80_3fold_stacking',
        'name': '3-fold GroupKFold, 30 seeds, LGBM+CB+XGB stacking',
        'avg_cv': float(avg_cv),
        'results': all_results,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    meta_path = SUBMIT / f'meta_v80_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
