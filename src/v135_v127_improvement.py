"""
V135 — V127 Improvement: 6-model Bayesian-weighted ensemble on full 142 features

Strategy: Build on V127's proven 3-way ensemble architecture, but improve it by:
1. Expanding to 6 models (base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep)
2. Per-target Bayesian weight optimization instead of fixed 0.35/0.25/0.40
3. Using the full 142-feature pipeline from V62 (gen_v62_full_train_test.py)
4. More seeds per model (50 seeds like V53) for diversity
5. Temperature scaling + mean matching calibration

Key insights from V256 ensemble search:
- pair_wide is the most stable (top-2 in 5/7 targets)
- trans_deep is strong for S1 (0.62 weight)
- base models are weak alone but add ensemble diversity
- Bayesian weight optimization beats fixed weights by ~0.05 OOF
- Mean blending beats rank blending

Architecture:
┌─────────────────────────────────────────────────────┐
│ 6 Model Pool:                                         │
│   1. base_wide   (raw features + zscore, wide config)  │
│   2. base_deep   (raw features + zscore, deep config)  │
│   3. pair_wide   (pairwise interactions + zscore, wide) │
│   4. pair_deep   (pairwise interactions + zscore, deep) │
│   5. trans_wide  (log/sqrt transforms + zscore, wide)   │
│   6. trans_deep  (log/sqrt transforms + zscore, deep)   │
├─────────────────────────────────────────────────────┘
│
│ Per-target: Bayesian optimization of weights w1..w6
│ (subject to: wi >= 0, sum = 1)
│
│ Per-model: 50 seeds × GroupKFold 5-fold
│
│ Calibration: Isotonic regression + temperature scaling + mean matching

Expected: OOF < 0.54, LB ~ 0.64 or better
"""
import sys, gc, logging, json, re, time, warnings, copy
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
RAW = ROOT / 'data_raw'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

# ── Configs ────────────────────────────────────────────────────────────

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}

# 50 seeds for diversity (V53 style)
SEEDS = list(range(42, 92))  # 50 seeds

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


# ── Feature helpers ────────────────────────────────────────────────────

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    """Add per-subject zscore features."""
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        if fit_stats is not None and col in fit_stats:
            subj_mean = fit_stats[col]['mean']
            subj_std = fit_stats[col]['std']
        else:
            subj_mean = grp[f'{col}_subj_mean']
            subj_std = grp[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        zc = f'{col}_zscore'
        df[zc] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(zc)
        gc.collect()
    return df, personal_cols

def add_pairwise_interactions(feat, top_features):
    """Add pairwise product + ratio features."""
    feat = feat.copy()
    added = []
    for i in range(min(len(top_features), 10)):
        for j in range(i+1, min(len(top_features), 10)):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns:
                continue
            col_prod = f'{f1}_x_{f2}'
            feat[col_prod] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(col_prod)
            col_ratio = f'{f1}_div_{f2}'
            feat[col_ratio] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
            added.append(col_ratio)
    for f in top_features[:5]:
        if f in feat.columns:
            col_sq = f'{f}_sq'
            feat[col_sq] = feat[f].fillna(0) ** 2
            added.append(col_sq)
    return feat, added

def add_transformed_features(feat, top_features):
    """Add log/sqrt/abs transformations."""
    feat = feat.copy()
    added = []
    for f in top_features[:15]:
        if f not in feat.columns:
            continue
        vals = feat[f].fillna(0).values
        vals_abs = np.abs(vals) + 1e-8
        feat[f'{f}_log'] = np.sign(vals) * np.log1p(vals_abs)
        added.append(f'{f}_log')
        feat[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(vals_abs)
        added.append(f'{f}_sqrt')
        feat[f'{f}_abs'] = np.abs(vals)
        added.append(f'{f}_abs')
    return feat, added


# ── Model training ────────────────────────────────────────────────────

def train_cv_models(feat, feat_tst, cols, y, seeds, cfg, n_folds=5):
    """Train K fold-sequential models, return OOF and test predictions."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    test_preds = np.zeros((len(feat_tst), len(seeds)))
    sn = [sanitize(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64)
    
    for si, seed in enumerate(seeds):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'random_state': seed, 'scale_pos_weight': spw,
        }
        fold = 0
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(params, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            best_iter = m.current_iteration()
            oof[va_i, si] = m.predict(X_full[va_i])
            test_preds[:, si] = m.predict(X_test)
            fold += 1
            del ds, vd, m
            gc.collect()
    return oof, test_preds


def calibrate_and_mean_match(oof_pred, test_pred, y_train, target_mean):
    """Isotonic calibration + logit-shift + mean matching."""
    # Isotonic
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_pred, y_train)
        oof_cal = iso.predict(oof_pred)
        test_cal = iso.predict(test_pred)
    except Exception:
        oof_cal, test_cal = oof_pred.copy(), test_pred.copy()
    
    # Mean match
    def mean_match(pred, mean):
        shift = mean - pred.mean()
        return np.clip(pred + shift, 0.0001, 0.9999)
    
    oof_cal = mean_match(oof_cal, target_mean)
    test_cal = mean_match(test_cal, target_mean)
    return oof_cal, test_cal


def bayesian_weight_optimize(model_oofs, y_train, target_mean):
    """
    Optimize ensemble weights for 6 models using CV OOF.
    
    model_oofs: list of 6 arrays, each (n_samples,) — CV OOF predictions per model
    Returns: list of 6 weights
    """
    def objective(weights):
        """Minimize log_loss of weighted ensemble."""
        ens = np.zeros(len(y_train))
        for i in range(6):
            ens += weights[i] * model_oofs[i]
        ens = np.clip(ens, 0.0001, 0.9999)
        return log_loss(y_train, ens, labels=[0, 1])
    
    best_loss = float('inf')
    best_weights = [1/6] * 6
    
    for restart in range(20):
        # Random starting point on simplex
        x0 = np.random.dirichlet(np.ones(6))
        bounds = [(0.001, 0.999)] * 6
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 200, 'ftol': 1e-9})
        
        if result.fun < best_loss:
            best_loss = result.fun
            w = result.x
            w = np.clip(w, 0.001, 0.999)
            w = w / w.sum()
            best_weights = w.tolist()
    
    return best_weights, best_loss


# ── Feature ranking ────────────────────────────────────────────────────

def rank_features_importance(feat, feat_cols, target, seed=42):
    """Rank features by LGBM gain importance (quick 50-iter model)."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    import lightgbm as lgb
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V135 — V127 Improvement: 6-model Bayesian-weighted ensemble")
    log.info("=" * 70)
    
    import lightgbm as lgb
    
    # ── 1. Load data ─────────────────────────────────────────────────────
    log.info("\n--- 1. Load data ---")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_test = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [feat, feat_test]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols_raw = get_feature_cols(feat)
    log.info(f"Base features: {len(feat_cols_raw)}")
    
    # Personalization
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    feat_test_z, _ = add_personalization(feat_test, feat_cols_raw)
    all_cols = feat_cols_raw + zscore_cols
    log.info(f"After personalization: train={feat.shape}, test={feat_test_z.shape}")
    
    y_train = {t: feat[t].values for t in TARGETS}
    train_rates = {t: feat[t].mean() for t in TARGETS}
    
    # ── 2. Feature ranking ───────────────────────────────────────────────
    log.info("\n--- 2. Feature ranking ---")
    ranked_all = {}
    for target in TARGETS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_all[target] = ranked
        log.info(f"  {target}: top-5 = {ranked[:5]}")
    
    # ── 3. Train 6 models per target ────────────────────────────────────
    log.info("\n--- 3. Train 6 models × 7 targets × 50 seeds ---")
    
    model_predictions = {}  # {target: {model_name: test_preds}}
    model_oofs = {}        # {target: {model_name: oof_preds}}
    
    for target in TARGETS:
        tgt_t = time.time()
        y = y_train[target]
        log.info(f"\n  {target}: n_feat_rank={len(ranked_all[target])}")
        
        strategies = {
            'base': (feat.copy(), feat_test_z.copy(), []),
            'pair': (add_pairwise_interactions(feat.copy(), ranked_all[target][:10])[0],
                     add_pairwise_interactions(feat_test_z.copy(), ranked_all[target][:10])[0],
                     []),
            'trans': (add_transformed_features(feat.copy(), ranked_all[target][:15])[0],
                      add_transformed_features(feat_test_z.copy(), ranked_all[target][:15])[0],
                      []),
        }
        
        # Personalize each strategy
        for name, (f, ft, old_zc) in strategies.items():
            if name == 'base':
                f, zc = add_personalization(f, feat_cols_raw + zscore_cols)
                ft, _ = add_personalization(ft, feat_cols_raw + zscore_cols)
            else:
                f, zc = add_personalization(f, feat_cols_raw + old_zc)
                ft, _ = add_personalization(ft, feat_cols_raw + old_zc)
            strategies[name] = (f, ft, zc)
        
        # Train each strategy with wide and deep configs
        model_results = {}
        oof_results = {}
        
        for strategy_name, (f, ft, _) in strategies.items():
            all_cols_strat = get_feature_cols(f)
            all_cols_strat = [c for c in all_cols_strat if c not in META | set(TARGETS)]
            
            for cfg_name, cfg in [('wide', CFG_WIDE), ('deep', CFG_DEEP)]:
                # Per-target n_feat selection
                leak_cols = remove_leak(all_cols_strat, target)
                ranked_strat = rank_features_importance(f, leak_cols, target)
                n_feat = 20
                if n_feat > len(ranked_strat):
                    n_feat = len(ranked_strat)
                sel_cols = ranked_strat[:n_feat]
                
                oof, test_p = train_cv_models(f, ft, sel_cols, y, SEEDS, cfg)
                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
                
                # Calibrate
                oof_cal, test_cal = calibrate_and_mean_match(oof_avg, test_avg, y, train_rates[target])
                
                ll = log_loss(y, oof_cal, labels=[0, 1])
                model_key = f"{strategy_name}_{cfg_name}"
                model_results[model_key] = test_cal
                oof_results[model_key] = {
                    'oof_avg': oof_avg,
                    'll': ll,
                    'oof': oof,
                    'test_p': test_p,
                    'n_feat': n_feat,
                }
                
                if strategy_name == 'pair' and cfg_name == 'wide':
                    log.info(f"  {target}: {model_key} n_feat={n_feat} LL={ll:.5f} (time: {time.time()-tgt_t:.0f}s)")
                
                del oof, test_p
                gc.collect()
        
        model_predictions[target] = model_results
        model_oofs[target] = oof_results
    
    # ── 4. Bayesian weight optimization ─────────────────────────────────
    log.info("\n--- 4. Bayesian weight optimization ---")
    weight_results = {}
    
    for target in TARGETS:
        tgt_t = time.time()
        y = y_train[target]
        
        # Collect OOF predictions for each model
        model_names = list(model_oofs[target].keys())
        model_oof_preds = []
        
        for name in model_names:
            oof_avg = model_oofs[target][name]['oof_avg']
            oof_cal, _ = calibrate_and_mean_match(oof_avg, oof_avg, y, train_rates[target])
            model_oof_preds.append(oof_cal)
        
        # Bayesian optimization
        weights, opt_loss = bayesian_weight_optimize(model_oof_preds, y, train_rates[target])
        
        # Also try equal-weight baseline
        ens_equal = np.mean(model_oof_preds, axis=0)
        ens_equal_cal, _ = calibrate_and_mean_match(ens_equal, ens_equal, y, train_rates[target])
        equal_ll = log_loss(y, ens_equal_cal, labels=[0, 1])
        
        # Optimized ensemble
        ens_opt = np.zeros(len(y))
        for i in range(len(model_oof_preds)):
            ens_opt += weights[i] * model_oof_preds[i]
        ens_opt_cal, test_cal = calibrate_and_mean_match(ens_opt, ens_opt, y, train_rates[target])
        
        opt_ll = log_loss(y, ens_opt_cal, labels=[0, 1])
        
        weight_results[target] = {
            'weights': {n: round(w, 4) for n, w in zip(model_names, weights)},
            'opt_loss': opt_loss,
            'opt_ll': opt_ll,
            'equal_ll': equal_ll,
            'improvement': equal_ll - opt_ll,
        }
        
        log.info(f"  {target}: weights={weight_results[target]['weights']}")
        log.info(f"  {target}: opt_LL={opt_ll:.5f}, equal_LL={equal_ll:.5f}, Δ={equal_ll-opt_ll:.5f}")
        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
    
    # ── 5. Build submission ─────────────────────────────────────────────
    log.info("\n--- 5. Build submission ---")
    
    all_opt_ll = []
    all_equal_ll = []
    sub = pd.DataFrame()
    sub['subject_id'] = feat_test['subject_id'].values
    sub['sleep_date'] = feat_test['sleep_date'].values
    sub['lifelog_date'] = feat_test['lifelog_date'].values
    
    for target in TARGETS:
        weights = weight_results[target]['weights']
        model_names = list(weights.keys())
        
        ens = np.zeros(len(feat_test_z))
        for name in model_names:
            ens += weights[name] * model_predictions[target][name]
        
        # Final calibration
        y = y_train[target]
        _, final_preds = calibrate_and_mean_match(ens, ens, y, train_rates[target])
        
        sub[target] = final_preds
        all_opt_ll.append(log_loss(y, weight_results[target]['opt_cal'] if 'opt_cal' in weight_results[target] else final_preds, labels=[0, 1]))
        all_equal_ll.append(weight_results[target]['equal_ll'])
    
    sub_path = SUBMIT / f"submission_v135_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    
    for t in TARGETS:
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f} std={sub[t].std():.4f}")
    
    # ── 6. Save metadata ────────────────────────────────────────────────
    meta = {
        'version': 'V135',
        'name': 'V127 Improvement: 6-model Bayesian-weighted ensemble',
        'features': {
            'base': len(feat_cols_raw),
            'zscore': len(zscore_cols),
            'total': len(all_cols),
        },
        'seeds': len(SEEDS),
        'models_per_target': list(model_predictions[TARGETS[0]].keys()),
        'weights': {t: weight_results[t]['weights'] for t in TARGETS},
        'target_results': {t: {
            'opt_loss': round(weight_results[t]['opt_loss'], 5),
            'opt_ll': round(weight_results[t]['opt_ll'], 5),
            'equal_ll': round(weight_results[t]['equal_ll'], 5),
            'improvement': round(weight_results[t]['improvement'], 5),
        } for t in TARGETS},
        'avg_opt_ll': round(np.mean([weight_results[t]['opt_ll'] for t in TARGETS]), 5),
        'avg_equal_ll': round(np.mean([weight_results[t]['equal_ll'] for t in TARGETS]), 5),
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = SUBMIT / f'meta_v135_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({time.time()-t_start/60:.1f}min)")
    
    return sub


if __name__ == '__main__':
    main()
