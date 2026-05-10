"""
V79 — LGBM + XGB Stacking Ensemble on V60 Features (15 seeds, 5-fold)

Uses features_clean_v60.parquet directly (already has personalization zscore).
LGBM 4 configs × XGB 2 configs × 6 n_feat values × 15 seeds × 5 folds.
Per-target optimal model selection + stacking.

Goal: Break past 0.65 barrier toward 0.50
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


# Configs (from V53/V74)
LGBM_CFGS = [
    {'name': 'wide',    'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    {'name': 'deep',    'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'v48',     'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'safety',  'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
]

XGB_CFGS = [
    {'name': 'xgb_std', 'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'ss': 0.7, 'cs': 0.7},
    {'name': 'xgb_deep','iter': 1500, 'lr': 0.02, 'depth': 7, 'l2': 5.0, 'ss': 0.6, 'cs': 0.6},
]

N_SEEDS = 15
N_FOLDS = 5
# Wider n_feat sweep
N_FEATS = [8, 10, 12, 15, 17, 20, 23, 30]


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


def train_xgb(X_tr, y_tr, X_va, cfg, seed, spw):
    p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
         'max_depth': cfg['depth'], 'learning_rate': cfg['lr'],
         'n_estimators': cfg['iter'], 'subsample': cfg['ss'],
         'colsample_bytree': cfg['cs'], 'reg_alpha': cfg['l2'],
         'reg_lambda': cfg['l2'], 'min_child_weight': cfg['l2'],
         'random_state': seed, 'scale_pos_weight': spw,
         'tree_method': 'hist', 'verbosity': 0, 'n_jobs': 1}
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, verbose=False)
    pred = np.clip(m.predict_proba(X_va)[:, 1], 0.0001, 0.9999)
    del m
    return pred


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V79 — LGBM + XGB Stacking on V60 Features (15 seeds)")
    log.info("=" * 70)
    
    train = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test = test[list(train.columns)]
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)}")
    
    all_results = {}
    all_predictions = {}
    
    for target in TARGETS:
        t_t = time.time()
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate: {y.mean():.3f}, leak-cols: {len(leak_cols)})")
        
        # Feature ranking (100 trees LGBM)
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
        del mr, ds_rank
        gc.collect()
        
        best_overall_cv = float('inf')
        best_overall_info = None
        
        for nf in N_FEATS:
            if nf > len(ranked):
                continue
            sel_cols = [r[0] for r in ranked[:nf]]
            sel_sn = [sanitize(r[0]) for r in ranked[:nf]]
            sel_idx = [leak_cols.index(c) for c in sel_cols]
            
            model_oofs = {}
            
            # LGBM models
            for cfg in LGBM_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
                        X_tr = X_all[tr][:, sel_idx]
                        X_va = X_all[va][:, sel_idx]
                        y_tr = y[tr]
                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        pred = train_lgb(X_tr, y_tr, X_va, sel_sn, cfg, s, spw_f)
                        oof[va] += pred
                        n_valid[va] += 1
                        del pred
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'lgb_{cfg["name"]}'] = oof_avg
                log.info(f"  {target} LGB {cfg['name']:8s} n_feat={nf:2d}: cv={cv:.4f}")
                del oof
                gc.collect()
            
            # XGB models
            for cfg in XGB_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
                        X_tr = X_all[tr][:, sel_idx]
                        X_va = X_all[va][:, sel_idx]
                        y_tr = y[tr]
                        spw_f = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        pred = train_xgb(X_tr, y_tr, X_va, cfg, s, spw_f)
                        oof[va] += pred
                        n_valid[va] += 1
                        del pred
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'xgb_{cfg["name"]}'] = oof_avg
                log.info(f"  {target} XGB {cfg['name']:8s} n_feat={nf:2d}: cv={cv:.4f}")
                del oof
                gc.collect()
            
            # Stacking
            model_names = sorted(model_oofs.keys())
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
            
            model_oofs[f'stack_C{best_c}'] = oof_stack
            log.info(f"  {target} STACK_C={best_c:.2f}: cv={best_meta_cv:.4f}")
            
            # Find best
            results_for_nf = {}
            for k, v in model_oofs.items():
                if v.ndim == 1:
                    results_for_nf[k] = log_loss(y, v)
                else:
                    # stacking OOF matrix - use precomputed meta_cv
                    pass
            # Add stack CV from meta optimization
            results_for_nf[f'stack_C{best_c}'] = best_meta_cv
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
            del oof_stack
            gc.collect()
        
        log.info(f"\n  ✅ Best {target}: {best_overall_info}")
        log.info(f"     [{time.time() - t_t:.0f}s]")
        
        all_results[target] = best_overall_info
    
    # Summary
    avg_cv = np.mean([v['cv'] for v in all_results.values()])
    log.info(f"\n{'='*70}")
    log.info(f"V79 RESULTS")
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
    
    meta = {
        'version': 'V79_lgbm_xgb_stacking',
        'name': 'LGBM 4 + XGB 2 configs, stacking, 15 seeds, V60 features',
        'avg_cv': float(avg_cv),
        'results': all_results,
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    log.info(f"  Meta: {json.dumps(meta, indent=2)}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
