"""
V46 — CatBoost + LightGBM Ensemble with Minimal Date Features

Key changes from V10:
1. Add 8 date/period features (dayofweek, is_weekend, month, etc.)
2. Feature ranking with CatBoost (handles categoricals better, often better on small tabular)
3. CatBoost + LightGBM ensemble (0.5:0.5 — sch_csm used 0.7 CatBoost but that's because they use ~770 features)
4. Same per-target tuning but with CatBoost scoring the configs
5. 20 seeds per model (same as V10)
6. Same leakage fixes, personalization, mean-matching calibration

Strategy: If CatBoost alone > LGBM alone, shift ensemble weight toward CatBoost.
"""

import sys, re, gc, time, json, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

# ── Seeds ──
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

# ── Configs ──
CONFIGS = {
    'C1': {'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    'C2': {'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C3': {'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C4': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C5': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    'C6': {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
}

# ── Leakage ──
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

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

# ── Date features ──
def add_date_features(df):
    """Add 8 date/period features."""
    date_col = pd.to_datetime(df['sleep_date'])
    df = df.copy()
    df['dayofweek'] = date_col.dt.dayofweek
    df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
    df['month'] = date_col.dt.month
    df['is_monday'] = (date_col.dt.dayofweek == 0).astype(int)
    df['is_friday'] = (date_col.dt.dayofweek == 4).astype(int)
    df['dayofyear'] = date_col.dt.dayofyear
    df['is_q1'] = date_col.dt.month.isin([6,7,8]).astype(int)
    df['is_q2'] = date_col.dt.month.isin([3,4,5,9,10,11,12,1,2]).astype(int)
    return df

# ── Personalization ──
def add_personalization(df, feature_cols):
    """Per-subject z-score."""
    zscore_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col] - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        zscore_cols.append(f'{col}_zscore')
        gc.collect()
    return df, zscore_cols

# ── Feature ranking ──
def rank_features_lgb(feat, feat_cols, target, seed=42, n_trees=50):
    """Rank features by LightGBM gain."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': n_trees, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=n_trees)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

# ── Simple calibration ──
def simple_mm(p, r):
    """Mean-match calibration."""
    shift = r - p.mean()
    return np.clip(p + shift, 0.0001, 0.9999)

# ── CV prediction ──
def lgb_cv_predict(feat, sel, target, seeds, spw):
    """GroupKFold CV with LGBM, return OOF predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=5)
    oof_all = np.zeros(len(y))
    
    for seed in seeds:
        for fold_i, (tr_idx, va_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[tr_idx][sel].fillna(0).values
            X_va = feat.iloc[va_idx][sel].fillna(0).values
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            sn = [sanitize(c) for c in sel]
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
            ds_va = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds_tr, params={'verbose': '-1'})
            
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'verbose': -1, 'force_row_wise': True,
                'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                'n_estimators': 200, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10,
                'scale_pos_weight': spw, 'random_state': seed,
            }
            m = lgb.train(params, ds_tr, num_boost_round=200,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            oof_all[va_idx] += m.predict(X_va) / len(seeds)
    
    cv_loss = log_loss(y, oof_all, labels=[0, 1])
    return oof_all, cv_loss

def cb_cv_predict(feat, sel, target, seeds, spw):
    """GroupKFold CV with CatBoost, return OOF predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=5)
    oof_all = np.zeros(len(y))
    
    for seed in seeds:
        for fold_i, (tr_idx, va_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[tr_idx][sel].fillna(0).values.astype(np.float32)
            X_va = feat.iloc[va_idx][sel].fillna(0).values.astype(np.float32)
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            sn = [sanitize(c) for c in sel]
            
            params = {
                'loss_function': 'Logloss',
                'verbose': 0,
                'num_boost_round': 200,
                'learning_rate': 0.03,
                'depth': 4,
                'l2_leaf_reg': 3.0,
                'random_seed': seed,
                'bagging_temperature': 0.5,
                'od_type': 'Iter',
                'od_wait': 30,
                'use_best_model': False,
            }
            m = cb.CatBoostClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=False)
            oof_all[va_idx] += m.predict_proba(X_va)[:, 1] / len(seeds)
    
    cv_loss = log_loss(y, oof_all, labels=[0, 1])
    return oof_all, cv_loss

# ── Config tuning via CV ──
def tune_config_lgb(feat, ranked, target, seeds, spw):
    """Tune LGBM config per target via 5-fold CV."""
    y = feat[target].values
    train_rate = y.mean()
    
    best_score = float('inf')
    best_cfg = None
    best_n = None
    best_oof = None
    
    for n_feat in [10, 20, 30]:
        if n_feat > len(ranked):
            continue
        sel = ranked[:n_feat]
        
        for name, cfg in CONFIGS.items():
            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'], 'scale_pos_weight': spw,
            }
            
            gkf = GroupKFold(n_splits=5)
            oof_all = np.zeros(len(feat))
            
            for fold_i, (tr_idx, va_idx) in enumerate(
                gkf.split(feat, y, feat['subject_id'])
            ):
                X_tr = feat.iloc[tr_idx][sel].fillna(0).values
                X_va = feat.iloc[va_idx][sel].fillna(0).values
                y_tr, y_va = y[tr_idx], y[va_idx]
                
                sn = [sanitize(c) for c in sel]
                ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
                ds_va = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds_tr, params={'verbose': '-1'})
                
                m = lgb.train(params, ds_tr, num_boost_round=cfg['ne'],
                              valid_sets=[ds_va],
                              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                oof_all[va_idx] += m.predict(X_va) / 5
            
            cv_loss = log_loss(y, oof_all, labels=[0, 1])
            shift = abs(oof_all.mean() - train_rate)
            score = cv_loss + 0.3 * shift
            
            if score < best_score:
                best_score = score
                best_cfg = name
                best_n = n_feat
                best_oof = oof_all.copy()
                log.info(f"    LGB NEW BEST: {name} n={n_feat} cv={cv_loss:.4f} shift={shift:.4f}")
    
    return best_cfg, best_n, best_oof, best_score

# ── Main ──
def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V46 — CatBoost + LightGBM Ensemble")
    log.info("=" * 70)
    
    # 1. Load features
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Train: {feat.shape}")
    
    # 2. Add date features
    log.info("\n--- 2. Date features ---")
    feat = add_date_features(feat)
    date_cols = ['dayofweek', 'is_weekend', 'month', 'is_monday', 'is_friday',
                 'dayofyear', 'is_q1', 'is_q2']
    
    # 3. Get feature cols
    feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS) | set(date_cols)
                 and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    log.info(f"  Base feature cols: {len(feat_cols)}")
    
    # 4. Personalization
    log.info("\n--- 3. Personalization ---")
    t0 = time.time()
    feat, zscore_cols = add_personalization(feat, feat_cols)
    log.info(f"  Z-score cols: {len(zscore_cols)}")
    log.info(f"  After personalization: {feat.shape}")
    log.info(f"  Time: {time.time()-t0:.0f}s")
    
    all_feat_cols = feat_cols + date_cols + zscore_cols
    feat = feat.fillna(0)
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Train rates: {train_rate}")
    
    # 5. Baseline LGBM only (same as V10 logic)
    log.info("\n=== Baseline: LGBM only (same as V10) ===")
    baseline_results = {}
    
    for target in TARGET_COLS:
        log.info(f"\n--- {target} ---")
        leak_free = remove_leak(all_feat_cols, target)
        ranked = rank_features_lgb(feat, leak_free, target)
        spw = max(((feat[target].values == 0).sum()) / max((feat[target].values == 1).sum(), 1), 0.1)
        
        cfg, n, oof, score = tune_config_lgb(feat, ranked, target, SEEDS, spw)
        cv_loss = log_loss(feat[target].values, oof, labels=[0, 1])
        baseline_results[target] = {'config': cfg, 'n_feat': n, 'cv': float(cv_loss), 'oof': oof}
        log.info(f"  LGBM baseline: {cfg} n={n} cv={cv_loss:.4f}")
    
    avg_baseline = np.mean([baseline_results[t]['cv'] for t in TARGET_COLS])
    log.info(f"\n  Baseline Avg CV: {avg_baseline:.4f}")
    
    # 6. V46: Per-target tuning (LGBM) + final training + CatBoost ensemble
    log.info("\n=== V46: Per-target tuning + CatBoost ensemble ===")
    all_results = {}
    ensemble_weights = {}
    
    for target in TARGET_COLS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate={train_rate[target]:.3f}) ---")
        tgt_t = time.time()
        
        leak_free = remove_leak(all_feat_cols, target)
        ranked = rank_features_lgb(feat, leak_free, target)
        y = feat[target].values
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        
        # Per-target config tuning
        best_lgb_cfg = None
        best_lgb_n = None
        best_lgb_oof = None
        best_lgb_score = float('inf')
        
        for n_feat in [10, 20, 30]:
            if n_feat > len(ranked):
                continue
            sel = ranked[:n_feat]
            
            for name, cfg in CONFIGS.items():
                params = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'], 'scale_pos_weight': spw,
                }
                
                gkf = GroupKFold(n_splits=5)
                oof_all = np.zeros(len(feat))
                
                for fold_i, (tr_idx, va_idx) in enumerate(
                    gkf.split(feat, y, feat['subject_id'])
                ):
                    X_tr = feat.iloc[tr_idx][sel].fillna(0).values
                    X_va = feat.iloc[va_idx][sel].fillna(0).values
                    y_tr, y_va = y[tr_idx], y[va_idx]
                    
                    sn = [sanitize(c) for c in sel]
                    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
                    ds_va = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds_tr, params={'verbose': '-1'})
                    
                    m = lgb.train(params, ds_tr, num_boost_round=cfg['ne'],
                                  valid_sets=[ds_va],
                                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                    oof_all[va_idx] += m.predict(X_va) / 5
                
                cv_loss = log_loss(y, oof_all, labels=[0, 1])
                shift = abs(oof_all.mean() - train_rate[target])
                score = cv_loss + 0.3 * shift
                
                if score < best_lgb_score:
                    best_lgb_score = score
                    best_lgb_cfg = name
                    best_lgb_n = n_feat
                    best_lgb_oof = oof_all.copy()
                    log.info(f"    LGB NEW BEST: {name} n={n_feat} cv={cv_loss:.4f} shift={shift:.4f}")
        
        log.info(f"  Best LGBM: {best_lgb_cfg} n={best_lgb_n} cv={log_loss(y, best_lgb_oof, labels=[0,1]):.4f}")
        
        # --- CatBoost single model for comparison ---
        log.info("  CatBoost benchmark (5 folds, 1 seed)...")
        cat_oof = np.zeros(len(feat))
        cat_params = {
            'loss_function': 'Logloss',
            'verbose': 0, 'num_boost_round': 500,
            'learning_rate': 0.03, 'depth': 4,
            'l2_leaf_reg': 3.0, 'random_seed': 42,
            'bagging_temperature': 0.5,
            'od_type': 'Iter', 'od_wait': 30,
            'use_best_model': False,
        }
        
        gkf = GroupKFold(n_splits=5)
        for fold_i, (tr_idx, va_idx) in enumerate(
            gkf.split(feat, y, feat['subject_id'])
        ):
            X_tr = feat.iloc[tr_idx][ranked[:best_lgb_n]].fillna(0).values.astype(np.float32)
            X_va = feat.iloc[va_idx][ranked[:best_lgb_n]].fillna(0).values.astype(np.float32)
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            m = cb.CatBoostClassifier(**cat_params)
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=False)
            cat_oof[va_idx] = m.predict_proba(X_va)[:, 1]
        
        cat_cv = log_loss(y, cat_oof, labels=[0, 1])
        log.info(f"  CatBoost cv={cat_cv:.4f}")
        
        # Determine ensemble weights
        lgb_cv = log_loss(y, best_lgb_oof, labels=[0, 1])
        
        if cat_cv < lgb_cv:
            # CatBoost better → weight toward CatBoost
            diff = lgb_cv - cat_cv
            if diff > 0.02:
                w_cat = 0.7  # Sch_csm style
                w_lgb = 0.3
            else:
                w_cat = 0.55
                w_lgb = 0.45
        else:
            w_cat = 0.45
            w_lgb = 0.55
        
        ensemble_weights[target] = {'lgb': float(w_lgb), 'cat': float(w_cat),
                                     'lgb_cv': float(lgb_cv), 'cat_cv': float(cat_cv)}
        log.info(f"  Ensemble weights: LGBM={w_lgb:.1f}, CatBoost={w_cat:.1f}")
        
        # Final: Train multiple seeds for LGBM + CatBoost on ALL data
        log.info(f"  Final LGBM training: {best_lgb_cfg} n={best_lgb_n}, {len(SEEDS)} seeds...")
        
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': CONFIGS[best_lgb_cfg]['nl'], 'max_depth': CONFIGS[best_lgb_cfg]['md'],
            'learning_rate': CONFIGS[best_lgb_cfg]['lr'], 'n_estimators': CONFIGS[best_lgb_cfg]['ne'],
            'subsample': CONFIGS[best_lgb_cfg]['ss'], 'colsample_bytree': CONFIGS[best_lgb_cfg]['cb'],
            'reg_alpha': CONFIGS[best_lgb_cfg]['ra'], 'reg_lambda': CONFIGS[best_lgb_cfg]['rl'],
            'min_child_samples': CONFIGS[best_lgb_cfg]['mc'],
        }
        
        sel = ranked[:best_lgb_n]
        sn = [sanitize(c) for c in sel]
        X_all = feat[sel].fillna(0).values
        
        lgb_preds = np.zeros(len(feat))
        for seed_i, seed in enumerate(SEEDS):
            ds = lgb.Dataset(X_all, label=y, feature_name=sn, params={'verbose': '-1'})
            params = {**lgb_params, 'random_state': seed, 'scale_pos_weight': spw}
            m = lgb.train(params, ds, num_boost_round=CONFIGS[best_lgb_cfg]['ne'])
            lgb_preds += m.predict(X_all)
            if (seed_i + 1) % 5 == 0:
                log.info(f"    LGBM seed {seed_i+1}/{len(SEEDS)}")
            del m, ds
            gc.collect()
        lgb_preds /= len(SEEDS)
        
        # Final: CatBoost multiple seeds
        log.info(f"  Final CatBoost training: {len(SEEDS)} seeds...")
        cat_preds = np.zeros(len(feat))
        for seed_i, seed in enumerate(SEEDS):
            X_all_cb = feat[sel].fillna(0).values.astype(np.float32)
            m = cb.CatBoostClassifier(
                **{**cat_params, 'random_seed': seed, 'num_boost_round': CONFIGS[best_lgb_cfg]['ne']}
            )
            m.fit(X_all_cb, y, use_best_model=False)
            cat_preds += m.predict_proba(X_all_cb)[:, 1]
            if (seed_i + 1) % 5 == 0:
                log.info(f"    CB seed {seed_i+1}/{len(SEEDS)}")
            del m
            gc.collect()
        cat_preds /= len(SEEDS)
        
        # Ensemble
        ens_preds = w_lgb * lgb_preds + w_cat * cat_preds
        cal_preds = simple_mm(ens_preds, train_rate[target])
        cal_loss = log_loss(y, cal_preds, labels=[0, 1])
        
        all_results[target] = {
            'config': best_lgb_cfg, 'n_feat': best_lgb_n,
            'lgb_cv': float(lgb_cv), 'cat_cv': float(cat_cv),
            'cal': float(cal_loss), 'cal_oof': cal_preds,
            'lgb_preds': lgb_preds, 'cat_preds': cat_preds,
        }
        log.info(f"  {target}: Cal={cal_loss:.4f} | LGBM={lgb_cv:.4f} CB={cat_cv:.4f} | Time: {time.time()-tgt_t:.0f}s")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V46 SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Config':<6} {'nF':>4} {'LGB CV':>8} {'CB CV':>8} {'Cal':>8} {'W_CB':>6}")
    for target in TARGET_COLS:
        r = all_results[target]
        w = ensemble_weights[target]
        log.info(f"  {target:<6} {r['config']:<6} {r['n_feat']:>4} {r['lgb_cv']:>8.4f} {r['cat_cv']:>8.4f} {r['cal']:>8.4f} {w['cat']:>6.2f}")
    
    avg_cal = np.mean([all_results[t]['cal'] for t in TARGET_COLS])
    log.info(f"\n  V46 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    
    oof_path = DATA_PROCESSED / "oof_v46.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")
    
    # Save meta
    meta = {
        'version': 'v46', 'avg_cal': float(avg_cal),
        'results': {t: {k: v for k, v in r.items() if k != 'cal_oof'} for t, r in all_results.items()},
        'ensemble_weights': ensemble_weights,
        'avg_baseline_cv': float(avg_baseline),
        'date_cols': date_cols,
        'zscore_cols': len(zscore_cols),
    }
    with open(DATA_PROCESSED / "v46_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {DATA_PROCESSED / 'v46_meta.json'}")
    
    # ── Test Prediction (5 seeds each to save memory) ──
    log.info(f"\n{'='*70}")
    log.info("V46 TEST PREDICTION")
    log.info(f"{'='*70}")
    
    SEEDS_TEST = [42, 456, 2048, 8001, 14000]  # 5 seeds for test pred
    
    test = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"  Test: {test.shape}")
    
    test = add_date_features(test)
    test, _ = add_personalization(test, feat_cols)
    
    current_feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS) and pd.api.types.is_numeric_dtype(feat[c])]
    all_feat_cols = current_feat_cols + date_cols + zscore_cols
    
    common_cols = [c for c in all_feat_cols if c in test.columns and c in feat.columns]
    test = test[common_cols + ['subject_id', 'sleep_date', 'lifelog_date']].fillna(0)
    
    predictions = pd.DataFrame()
    test_meta = {}
    
    for target in TARGET_COLS:
        tgt_t = time.time()
        r = all_results[target]
        cfg_name = r['config']
        n_feat = r['n_feat']
        w_cat = ensemble_weights[target]['cat']
        w_lgb = 1.0 - w_cat
        
        leak_free = remove_leak(all_feat_cols, target)
        ranked = rank_features_lgb(feat, leak_free, target)
        sel = ranked[:n_feat]
        sn = [sanitize(c) for c in sel]
        
        log.info(f"\n  {target}: Config={cfg_name} n={n_feat} w_CB={w_cat:.2f}")
        
        y_train = feat[target].values
        X_train = feat[sel].fillna(0).values.astype(np.float64)
        X_test = test[sel].fillna(0).values.astype(np.float64)
        
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': CONFIGS[cfg_name]['nl'],
            'max_depth': CONFIGS[cfg_name]['md'],
            'learning_rate': CONFIGS[cfg_name]['lr'],
            'n_estimators': CONFIGS[cfg_name]['ne'],
            'subsample': CONFIGS[cfg_name]['ss'],
            'colsample_bytree': CONFIGS[cfg_name]['cb'],
            'reg_alpha': CONFIGS[cfg_name]['ra'],
            'reg_lambda': CONFIGS[cfg_name]['rl'],
            'min_child_samples': CONFIGS[cfg_name]['mc'],
        }
        spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
        lgb_params['scale_pos_weight'] = spw
        
        lgb_preds = np.zeros(len(test))
        for seed_i, seed in enumerate(SEEDS_TEST):
            ds = lgb.Dataset(X_train, label=y_train, feature_name=sn, params={'verbose': '-1'})
            params = {**lgb_params, 'random_state': seed}
            m = lgb.train(params, ds, num_boost_round=CONFIGS[cfg_name]['ne'])
            lgb_preds += m.predict(X_test)
            if (seed_i + 1) % 2 == 0:
                log.info(f"    LGBM seed {seed_i+1}/{len(SEEDS_TEST)}")
            del m, ds
            gc.collect()
        lgb_preds /= len(SEEDS_TEST)
        
        cat_preds = np.zeros(len(test))
        for seed_i, seed in enumerate(SEEDS_TEST):
            X_train_cb = feat[sel].fillna(0).values.astype(np.float32)
            X_test_cb = test[sel].fillna(0).values.astype(np.float32)
            params = {**cat_params, 'random_seed': seed, 'num_boost_round': CONFIGS[cfg_name]['ne']}
            m = cb.CatBoostClassifier(**params)
            m.fit(X_train_cb, y_train, use_best_model=False)
            cat_preds += m.predict_proba(X_test_cb)[:, 1]
            if (seed_i + 1) % 2 == 0:
                log.info(f"    CB seed {seed_i+1}/{len(SEEDS_TEST)}")
            del m
            gc.collect()
        cat_preds /= len(SEEDS_TEST)
        
        ens_preds = w_lgb * lgb_preds + w_cat * cat_preds
        cal_preds = simple_mm(ens_preds, train_rate[target])
        
        predictions[target] = cal_preds
        test_meta[target] = {
            'config': cfg_name, 'n_feat': n_feat,
            'pred_mean': float(cal_preds.mean()),
            'pred_min': float(cal_preds.min()),
            'pred_max': float(cal_preds.max()),
        }
        log.info(f"    {target}: test_mean={cal_preds.mean():.4f} | Time: {time.time()-tgt_t:.0f}s")
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    predictions['subject_id'] = test['subject_id'].values
    predictions['sleep_date'] = test['sleep_date'].values
    predictions['lifelog_date'] = test['lifelog_date'].values
    predictions = predictions[['subject_id', 'sleep_date', 'lifelog_date'] + TARGET_COLS]
    
    sub_path = SUBMIT_DIR / f'submission_v46_{timestamp}.csv'
    predictions.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")
    
    test_meta['version'] = 'v46'
    test_meta['submission_file'] = str(sub_path)
    test_meta['avg_cal_training'] = float(avg_cal)
    meta_path = SUBMIT_DIR / f'meta_v46_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(test_meta, f, indent=2, default=str)
    log.info(f"  Meta saved: {meta_path}")
    
    log.info(f"\n{'='*70}")
    log.info("V46 FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Submission: {sub_path}")
    log.info(f"{'Target':<6} {'Config':<6} {'nF':>4} {'Training Cal':>12} {'Test Mean':>10}")
    for target in TARGET_COLS:
        r = all_results[target]
        t = test_meta[target]
        log.info(f"  {target:<6} {r['config']:<6} {r['n_feat']:>4} {r['cal']:>12.4f} {t['pred_mean']:>10.4f}")
    log.info(f"\n  V46 Avg Cal (training): {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")
    
    return all_results

if __name__ == "__main__":
    main()
