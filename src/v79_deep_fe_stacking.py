"""
V79 — Multi-Target Deep Feature Engineering + Stacking Ensemble

Key changes from V78:
1. Remove CatBoost (memory hog) → only LGBM + XGBoost
2. Reduce seeds: 30 → 15 (diminishing returns at 30)
3. Add personalization features: subject-level stats → zscore
4. Wider n_feat sweep: 5-40
5. Per-subject feature + interaction features
6. 5-fold, 2-model stacking

Target: < 0.50 CV
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
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]


def add_personalization_features(df, group_col='subject_id'):
    """Add subject-level aggregation features + zscore."""
    result = df.copy()
    numeric_cols = [c for c in result.columns if np.issubdtype(result[c].dtype, np.number)
                    and c not in META_COLS | set(TARGETS)]
    
    groups = result.groupby(group_col)[numeric_cols]
    
    subj_stats = {}
    for func_name, agg_func in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
        agg = groups.agg(agg_func).add_prefix(f'subj_{func_name}_')
        subj_stats[f'subj_{func_name}'] = agg
        # Merge back by subject_id
        merged = agg.reset_index()
        for col in agg.columns:
            result[col] = merged.set_index(group_col).loc[result[group_col].values, col].values
    
    subj_count = groups.size().rename('subj_count')
    result['subj_count'] = result[group_col].map(subj_count).values
    
    # Z-score personalization features (within-subject deviation)
    # Skip columns that already have _zscore suffix
    # Use sanitized name to avoid duplicate column names from special chars
    zscore_cols = []
    for col in numeric_cols:
        if col in result.columns and '_zscore' not in col and result[col].std() > 0:
            col_series = result[col].values
            grp_mean = result.groupby(group_col)[col].transform('mean').values
            grp_std = result.groupby(group_col)[col].transform('std').fillna(0).values
            zscore = (col_series - grp_mean) / (grp_std + 1e-8)
            zcol = sanitize(col) + '_zscore'
            # Avoid duplicate column names
            while zcol in result.columns:
                zcol = sanitize(col) + '_p' + '_zscore'
            result[zcol] = zscore
            zscore_cols.append(zcol)
    
    # Clean: fill nans from std=0
    result = result.fillna(0)
    
    total_new = len(subj_stats) * len(numeric_cols) + len(zscore_cols)
    log.info(f"  Personalization: added {total_new} new features ({len(numeric_cols)} cols × {len(subj_stats)} agg + {len(zscore_cols)} zscore)")
    
    return result


# LGBM configs
LGBM_CFGS = [
    {'name': 'vcons', 'nl': 8, 'md': 2, 'lr': 0.01, 'ne': 2000, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
    {'name': 'vdeep', 'nl': 25, 'md': 5, 'lr': 0.015, 'ne': 1500, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    {'name': 'vwide', 'nl': 40, 'md': 3, 'lr': 0.04, 'ne': 500, 'ss': 0.8, 'cb': 0.8, 'ra': 1.0, 'rl': 3.0, 'mc': 5},
    {'name': 'vstd',  'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 1000, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
]

XGB_CFGS = [
    {'name': 'xgb_std', 'iter': 1000, 'lr': 0.03, 'depth': 6, 'l2': 3.0, 'ss': 0.7, 'cs': 0.7},
    {'name': 'xgb_deep', 'iter': 1500, 'lr': 0.02, 'depth': 7, 'l2': 5.0, 'ss': 0.6, 'cs': 0.6},
]

N_SEEDS = 15
N_FOLDS = 5
N_FEATS = [10, 15, 20, 25, 30, 40]


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V79 — Multi-Target Deep FE + Stacking (LGBM + XGB, 15 seeds)")
    log.info("=" * 70)
    
    # Load data with personalization features
    train_raw = pd.read_parquet(DATA / "features_clean_v60.parquet")
    test_raw = pd.read_parquet(DATA / "test_features_clean_v60.parquet")
    test_raw = test_raw[list(train_raw.columns)]
    
    # Add personalization features
    log.info("Adding personalization features...")
    train = add_personalization_features(train_raw)
    test = add_personalization_features(test_raw)
    
    feat_cols = get_feature_cols(train)
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    log.info(f"  Features: {len(feat_cols)}")
    
    # Pre-compute common columns between train and test (after personalization)
    common_cols = list(set(train.columns) & set(test.columns))
    
    all_results = {}
    all_predictions = {}
    
    for target in TARGETS:
        t_t = time.time()
        y = train[target].values.astype(np.float64)
        leak_cols = remove_leak(feat_cols, target)
        
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate: {y.mean():.3f}, leak-cols: {len(leak_cols)})")
        
        # Use only common columns between train and test
        common_leak = [c for c in leak_cols if c in common_cols]
        if len(common_leak) < len(leak_cols):
            log.info(f"    Dropping {len(leak_cols)-len(common_leak)} non-common cols: {[c for c in leak_cols if c not in common_cols][:5]}...")
        X = train[common_leak].fillna(0).values.astype(np.float64)
        test_X = test[common_leak].fillna(0).values.astype(np.float64)
        
        # Feature ranking (fast LGBM)
        sn = [sanitize(c) for c in common_leak]
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
        ranked = sorted(zip(common_leak, imp), key=lambda x: -x[1])
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
            X_sel = X[:, sel_idx]
            test_X_sel = test_X[:, sel_idx]
            
            model_oofs = {}
            
            # --- LGBM ---
            for cfg in LGBM_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
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
                        del m, ds2
                        gc.collect()
                
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'lgb_{cfg["name"]}'] = oof_avg
                log.info(f"  {target} LGB {cfg['name']:10s} n_feat={nf:2d}: cv={cv:.4f}")
                del oof
                gc.collect()
            
            # --- XGBoost ---
            for cfg in XGB_CFGS:
                oof = np.zeros(len(y))
                n_valid = np.zeros(len(y))
                
                for s in range(N_SEEDS):
                    for fold, (tr, va) in enumerate(gkf.split(X_sel, y, groups)):
                        X_tr, X_va = X_sel[tr], X_sel[va]
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
                        del m
                        gc.collect()
                
                oof_avg = oof / n_valid
                cv = log_loss(y, oof_avg)
                model_oofs[f'xgb_{cfg["name"]}'] = oof_avg
                log.info(f"  {target} XGB {cfg['name']:10s} n_feat={nf:2d}: cv={cv:.4f}")
                del oof
                gc.collect()
            
            if len(model_oofs) >= 2:
                # --- Stacking ---
                oof_stack = np.column_stack(list(model_oofs.values()))
                
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
                
                log.info(f"  {target} STACK_C={best_c:.2f}: cv={best_meta_cv:.4f}")
                model_oofs[f'stack_C{best_c}'] = oof_stack  # dummy, will refit later
            
            # Find best for this n_feat
            results_for_nf = {k: log_loss(y, v) for k, v in model_oofs.items()}
            # Re-compute stack cv for display
            if len(model_oofs) >= 2:
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
            del X_sel, test_X_sel
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
        'version': 'V79_deep_fe_stacking',
        'name': 'Personalization FE + LGBM + XGB stacking, 15 seeds',
        'avg_cv': float(avg_cv),
        'results': all_results,
        'timestamp': datetime.now().isoformat(),
        'total_time': f"{time.time() - t_start:.0f}s",
    }
    log.info(f"  Meta: {meta}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
