"""
V136 — V127 Improvement: 3-model Bayesian-weighted ensemble (lightweight)

From V256 analysis:
  pair_wide is #1 most stable (top-2 in 5/7 targets)
  pair_deep #2 for S1/S3
  trans_wide #3 for Q2/Q3/S4

Strategy: Use 3 best models (pair_wide, pair_deep, trans_wide) with
per-target Bayesian weight optimization. 20 seeds each (down from 50)
for speed. Same pipeline as V127 (V115_base + pairwise + transformed).

Expected time: ~20-30 min

Architecture:
┌─────────────────────────────────────────────────────┐
│ 3 Model Pool (per target):                          │
│   1. pair_wide   (pairwise features, wide config)     │
│   2. pair_deep   (pairwise features, deep config)     │
│   3. trans_wide  (transformed features, wide config)  │
├─────────────────────────────────────────────────────┘
│
│ Per-target: Bayesian weight optimization (20 restarts)
│ Each model: 20 seeds × GroupKFold 5-fold
│
│ Calibration: Isotonic regression + mean matching

Compare against V127 baseline: OOF 0.53731, LB 0.64763
"""
import sys, gc, logging, json, re, time, warnings
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

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}

SEEDS = list(range(42, 62))  # 20 seeds for speed

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

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

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
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

def train_cv_models(feat, feat_tst, cols, y, seeds, cfg, n_folds=5):
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
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(params, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            test_preds[:, si] = m.predict(X_test)
            del ds, vd, m
            gc.collect()
    return oof, test_preds

def calibrate_and_mean_match(oof_pred, test_pred, y_train, target_mean):
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_pred, y_train)
        oof_cal = iso.predict(oof_pred)
        test_cal = iso.predict(test_pred)
    except Exception:
        oof_cal, test_cal = oof_pred.copy(), test_pred.copy()
    
    def mean_match(pred, mean):
        shift = mean - pred.mean()
        return np.clip(pred + shift, 0.0001, 0.9999)
    
    oof_cal = mean_match(oof_cal, target_mean)
    test_cal = mean_match(test_cal, target_mean)
    return oof_cal, test_cal

def bayesian_weight_optimize(model_oofs, y_train, target_mean):
    def objective(weights):
        ens = np.zeros(len(y_train))
        for i in range(3):
            ens += weights[i] * model_oofs[i]
        ens = np.clip(ens, 0.0001, 0.9999)
        return log_loss(y_train, ens, labels=[0, 1])
    
    best_loss = float('inf')
    best_weights = [1/3] * 3
    
    for restart in range(30):
        x0 = np.random.dirichlet(np.ones(3))
        bounds = [(0.001, 0.999)] * 3
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-10})
        
        if result.fun < best_loss:
            best_loss = result.fun
            w = result.x
            w = np.clip(w, 0.001, 0.999)
            w = w / w.sum()
            best_weights = w.tolist()
    
    return best_weights, best_loss

def rank_features_importance(feat, feat_cols, target, seed=42):
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
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]


# ================================================================
# MAIN
# ================================================================

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V136 — V127 Improvement: 3-model Bayesian-weighted ensemble")
    log.info("Models: pair_wide, pair_deep, trans_wide")
    log.info("Seeds: 20 per model")
    log.info("=" * 70)
    
    # ── 1. Load data ────────────────────────────────────────────────
    log.info("\n--- 1. Load data ---")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_test = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [feat, feat_test]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols_raw = get_feature_cols(feat)
    log.info(f"Base features: {len(feat_cols_raw)}")
    
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    feat_test_z, _ = add_personalization(feat_test, feat_cols_raw)
    all_cols = feat_cols_raw + zscore_cols
    log.info(f"After personalization: train={feat.shape}, test={feat_test_z.shape}")
    
    y_train = {t: feat[t].values for t in TARGETS}
    train_rates = {t: feat[t].mean() for t in TARGETS}
    
    # ── 2. Feature ranking ──────────────────────────────────────────
    log.info("\n--- 2. Feature ranking ---")
    ranked_all = {}
    for target in TARGETS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_all[target] = ranked
        log.info(f"  {target}: top-5 = {ranked[:5]}")
    
    # ── 3. Train 3 models per target ────────────────────────────────
    log.info("\n--- 3. Train 3 models × 7 targets × 20 seeds ---")
    
    model_predictions = {}  # {target: {model_name: test_preds}}
    model_oofs = {}         # {target: {model_name: (oof_cal, ll)}}
    for target in TARGETS:
        model_predictions[target] = {}
        model_oofs[target] = {}
    
    # V127-like: 3 strategies with 3 config combos
    # V115_base = wide config on raw features
    # V123_pair = pairwise + deep
    # V121_p+t = pairwise + wide (V121 uses wide, V123 uses pair)
    # 
    # Based on V256: pair_wide, pair_deep, trans_wide are top 3
    model_specs = [
        ('pair_wide', 'pair', 'wide', 10),
        ('pair_deep', 'pair', 'deep', 10),
        ('trans_wide', 'trans', 'wide', 15),
    ]
    
    for target in TARGETS:
        tgt_t = time.time()
        y = y_train[target]
        log.info(f"\n  {target}:")
        
        # Build feature sets for each strategy
        feat_pair, _ = add_pairwise_interactions(feat.copy(), ranked_all[target][:10])
        feat_test_pair, _ = add_pairwise_interactions(feat_test_z.copy(), ranked_all[target][:10])
        feat_trans, _ = add_transformed_features(feat.copy(), ranked_all[target][:15])
        feat_test_trans, _ = add_transformed_features(feat_test_z.copy(), ranked_all[target][:15])
        
        feat_base = feat.copy()
        feat_test_base = feat_test_z.copy()
        
        feat_strategies = {
            'pair': (feat_pair, feat_test_pair),
            'trans': (feat_trans, feat_test_trans),
            'base': (feat_base, feat_test_base),
        }
        
        # Personalize non-base strategies
        for name, (f, ft) in feat_strategies.items():
            if name != 'base':
                f, _ = add_personalization(f, feat_cols_raw + zscore_cols)
                ft, _ = add_personalization(ft, feat_cols_raw + zscore_cols)
            feat_strategies[name] = (f, ft)
        
        # Train each model spec
        for model_name, strategy_name, cfg_name, n_top in model_specs:
            f, ft = feat_strategies[strategy_name]
            
            all_cols_strat = get_feature_cols(f)
            all_cols_strat = [c for c in all_cols_strat if c not in META | set(TARGETS)]
            
            leak_cols = remove_leak(all_cols_strat, target)
            ranked_strat = rank_features_importance(f, leak_cols, target)
            n_feat = 20
            if n_feat > len(ranked_strat):
                n_feat = len(ranked_strat)
            sel_cols = ranked_strat[:n_feat]
            
            cfg = CFG_WIDE if cfg_name == 'wide' else CFG_DEEP
            oof, test_p = train_cv_models(f, ft, sel_cols, y, SEEDS, cfg)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            oof_cal, test_cal = calibrate_and_mean_match(oof_avg, test_avg, y, train_rates[target])
            ll = log_loss(y, oof_cal, labels=[0, 1])
            
            model_predictions[target][model_name] = test_cal
            model_oofs[target][model_name] = {
                'oof_cal': oof_cal,
                'test_cal': test_cal,
                'll': ll,
                'oof_raw': oof_avg,
                'n_feat': n_feat,
            }
            
            log.info(f"    {model_name:12s} n_feat={n_feat} LL={ll:.5f} (time: {time.time()-tgt_t:.0f}s)")
            
            del oof, test_p
            gc.collect()
    
    # ── 4. Bayesian weight optimization ─────────────────────────────
    log.info("\n--- 4. Bayesian weight optimization ---")
    log.info("  Also comparing: V127 fixed weights (0.35/0.25/0.40), equal weight")
    
    weight_results = {}
    
    # Map V127 components: V121=pair_wide, V123=pair_deep, V115=base_wide
    # But our 3 models are: pair_wide, pair_deep, trans_wide
    v127_weight_map = {'pair_wide': 0.35, 'pair_deep': 0.25, 'trans_wide': 0.40}
    
    for target in TARGETS:
        tgt_t = time.time()
        y = y_train[target]
        
        model_names = list(model_oofs[target].keys())
        model_oof_preds = [model_oofs[target][n]['oof_cal'] for n in model_names]
        
        # 1) Bayesian optimized
        weights_opt, opt_loss = bayesian_weight_optimize(model_oof_preds, y, train_rates[target])
        
        ens_opt = np.zeros(len(y))
        for i in range(3):
            ens_opt += weights_opt[i] * model_oof_preds[i]
        ens_opt_cal, _ = calibrate_and_mean_match(ens_opt, ens_opt, y, train_rates[target])
        opt_ll = log_loss(y, ens_opt_cal, labels=[0, 1])
        
        # 2) Equal weight
        ens_equal = np.mean(model_oof_preds, axis=0)
        ens_equal_cal, _ = calibrate_and_mean_match(ens_equal, ens_equal, y, train_rates[target])
        equal_ll = log_loss(y, ens_equal_cal, labels=[0, 1])
        
        # 3) V127 fixed weights (for comparison)
        ens_v127 = np.zeros(len(y))
        for i, n in enumerate(model_names):
            ens_v127 += v127_weight_map[n] * model_oof_preds[i]
        ens_v127_cal, _ = calibrate_and_mean_match(ens_v127, ens_v127, y, train_rates[target])
        v127_ll = log_loss(y, ens_v127_cal, labels=[0, 1])
        
        weight_results[target] = {
            'weights_opt': {n: round(w, 4) for n, w in zip(model_names, weights_opt)},
            'weights_v127': {n: v127_weight_map[n] for n in model_names},
            'opt_ll': opt_ll,
            'equal_ll': equal_ll,
            'v127_ll': v127_ll,
            'opt_vs_v127': v127_ll - opt_ll,
            'equal_vs_v127': v127_ll - equal_ll,
        }
        
        log.info(f"\n  {target}:")
        log.info(f"    V127 fixed weights: {weight_results[target]['weights_v127']}  LL={v127_ll:.5f}")
        log.info(f"    Bayesian optimized: {weight_results[target]['weights_opt']}  LL={opt_ll:.5f}")
        log.info(f"    Equal weight:       LL={equal_ll:.5f}")
        log.info(f"    Δ(opt vs V127): {v127_ll - opt_ll:+.5f}")
        log.info(f"    Δ(equal vs V127): {v127_ll - equal_ll:+.5f}")
        log.info(f"  time: {time.time()-tgt_t:.0f}s")
    
    # ── 5. Build submission (optimized weights) ─────────────────────
    log.info("\n--- 5. Build submission ---")
    
    sub = pd.DataFrame()
    sub['subject_id'] = feat_test_z['subject_id'].values
    sub['sleep_date'] = feat_test_z['sleep_date'].values
    sub['lifelog_date'] = feat_test_z['lifelog_date'].values
    
    for target in TARGETS:
        weights = weight_results[target]['weights_opt']
        model_names = list(weights.keys())
        
        ens = np.zeros(len(feat_test_z))
        for name in model_names:
            ens += weights[name] * model_predictions[target][name]
        
        y = y_train[target]
        _, final_preds = calibrate_and_mean_match(ens, ens, y, train_rates[target])
        sub[target] = final_preds
        
        log.info(f"    {target}: min={sub[target].min():.4f} max={sub[target].max():.4f} mean={sub[target].mean():.4f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT / f"submission_v136_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\n  Saved: {sub_path}")
    
    # ── 6. Save metadata ────────────────────────────────────────────
    avg_opt_ll = np.mean([weight_results[t]['opt_ll'] for t in TARGETS])
    avg_v127_ll = np.mean([weight_results[t]['v127_ll'] for t in TARGETS])
    avg_equal_ll = np.mean([weight_results[t]['equal_ll'] for t in TARGETS])
    
    meta = {
        'version': 'V136',
        'name': 'V127 Improvement: 3-model Bayesian-weighted ensemble',
        'models': ['pair_wide', 'pair_deep', 'trans_wide'],
        'seeds': len(SEEDS),
        'weights_per_target': {t: weight_results[t] for t in TARGETS},
        'avg_cal_ll': {
            'optimized': round(avg_opt_ll, 5),
            'v127_fixed': round(avg_v127_ll, 5),
            'equal_weight': round(avg_equal_ll, 5),
        },
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    meta_path = SUBMIT / f'meta_v136_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({time.time()-t_start/60:.1f}min)")
    
    return sub


if __name__ == '__main__':
    main()
