"""
V22: V8-based pipeline with Extended Features + XGBoost + Bayesian Opt

Based on V8 (simple mean-match) with:
1. Extended features: lag, rolling, trend, interactions (top-N base features only)
2. XGBoost parallel + LightGBM
3. Per-target Bayesian hyperparameter optimization (optuna)
4. Simple mean-match calibration (no isotonic)

Uses preprocessed features.parquet (153 features, 450 samples)
"""
import sys, re, json, time, os, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

def pr(msg):
    print(msg, flush=True)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, SUBMIT_DIR

TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
N_SEEDS = 20
N_SPLITS = 5

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

_SANITIZE_RE = re.compile(r'[^a-zA-Z0-9_]')
def sanitize(name):
    return _SANITIZE_RE.sub('_', name)

# ── Feature leakage fix (same as V10) ──
LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feat_cols(f):
    return [c for c in f.columns
            if c not in META_COLS | set(TARGET_COLS)
            and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]

# ── Extended Feature Engineering (selective — only top N base features) ──
def add_extended_features(feat, feature_cols, n_top=30):
    """Add lag, rolling, trend, interactions for top N features only."""
    pr(f"  Selecting top {n_top} features for extended features...")
    
    df = feat.copy()
    added_cols = []
    selected = feature_cols[:n_top]
    
    for sid, grp in df.groupby('subject_id'):
        idx = grp.index
        df_sorted = df.loc[idx].sort_values('date')
        
        for col in selected:
            vals = df_sorted[col].fillna(0).values.astype(float)
            
            # Lag 1
            lag1 = np.roll(vals, 1)
            lag1[0] = vals[0]
            df.loc[idx, f'{col}_lag1'] = lag1
            added_cols.append(f'{col}_lag1')
            
            # Rolling 3, 7
            s = pd.Series(vals, index=df_sorted.index)
            df.loc[idx, f'{col}_roll3'] = s.rolling(3, min_periods=1).mean().values
            added_cols.append(f'{col}_roll3')
            df.loc[idx, f'{col}_roll7'] = s.rolling(7, min_periods=1).mean().values
            added_cols.append(f'{col}_roll7')
            
            # Trend (slope of last 7 days)
            slopes = np.zeros(len(vals))
            for i in range(len(vals)):
                start = max(0, i - 6)
                window = vals[start:i+1]
                if len(window) >= 3:
                    x = np.arange(len(window))
                    m, _ = np.polyfit(x, window, 1)
                    slopes[i] = m
            df.loc[idx, f'{col}_trend'] = slopes
            added_cols.append(f'{col}_trend')
    
    # Global interactions (always add)
    pr("  Computing global interactions...")
    if 'wPedo_pedo_step_mean' in df.columns and 'mLight_m_light_mean' in df.columns:
        df['step_x_light'] = df['wPedo_pedo_step_mean'] * df['mLight_m_light_mean']
        added_cols.append('step_x_light')
    if 'wHr_hr_std' in df.columns and 'mActivity_m_activity_mean' in df.columns:
        df['hr_std_x_activity'] = df['wHr_hr_std'] * df['mActivity_m_activity_mean']
        added_cols.append('hr_std_x_activity')
    
    del df
    
    pr(f"  Extended features: +{len(added_cols)} cols")
    return added_cols

# ── Personalization: z-score per subject ──
def add_personalization(df, feature_cols):
    personal_cols = []
    for col in feature_cols:
        filled = df[col].fillna(0)
        grp = filled.groupby(df['subject_id'])
        mean_s = grp.transform('mean')
        std_s = grp.transform('std').replace(0, 1)
        zscore = (filled - mean_s) / std_s
        df = df.assign(**{f'{col}_z': zscore})
        personal_cols.append(f'{col}_z')
    return df, personal_cols

# ── Feature ranking ──
def rank_features(feat, feature_cols, target, n_trees=100):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values.astype(np.float32)
    n_pos = max((y==1).sum(), 1)
    n_neg = (y==0).sum()
    spw = n_neg / n_pos
    sanitized = [sanitize(c) for c in feature_cols]
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':n_trees,
        'subsample':0.7,'colsample_bytree':0.7,
        'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':42,
        'min_child_samples':10,'n_jobs':1,
    }
    ds = lgb.Dataset(X, label=y, feature_name=sanitized, params={'verbose': '-1'})
    mdl = lgb.train(params, ds, num_boost_round=n_trees)
    imp = mdl.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return ranked

# ── CV predict LGB ──
def cv_predict_lgb(scols, target, seeds, spw):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    sanitized = [sanitize(c) for c in scols]
    
    for seed in seeds:
        cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
               'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,
               'reg_alpha':1.0,'reg_lambda':3.0,
               'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,'n_jobs':1}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            Xtr = feat.iloc[tr_idx][scols].fillna(0).values
            Xva = feat.iloc[va_idx][scols].fillna(0).values
            ytr, yva = y[tr_idx], y[va_idx]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sanitized)
            vad = lgb.Dataset(Xva, label=yva, feature_name=sanitized, reference=trd)
            mdl = lgb.train(cfg, trd, num_boost_round=500, valid_sets=[vad],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_idx] += mdl.predict(Xva)
    oof /= len(seeds)
    return oof

# ── CV predict XGB ──
def cv_predict_xgb(scols, target, seeds, spw):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    
    for seed in seeds:
        cfg = {'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
               'max_depth':4,'learning_rate':0.03,'n_estimators':500,
               'subsample':0.7,'colsample_bytree':0.7,
               'reg_alpha':1.0,'reg_lambda':3.0,'min_child_weight':5,'n_jobs':1,
               'scale_pos_weight':spw,'random_state':seed}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            Xtr = feat.iloc[tr_idx][scols].fillna(0).values
            Xva = feat.iloc[va_idx][scols].fillna(0).values
            ytr, yva = y[tr_idx], y[va_idx]
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dval = xgb.DMatrix(Xva, label=yva)
            mdl = xgb.train(cfg, dtrain, num_boost_round=500, evals=[(dval, 'val')])
            oof[va_idx] += mdl.predict(dval)
    oof /= len(seeds)
    return oof

# ── Simple calibration ──
def simple_cal(pred, target_rate):
    shift = target_rate - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

# ── Main ──
def main():
    global _FEAT
    import xgboost as xgb
    t_total = time.time()
    pr("=" * 70)
    pr("V22: V8 Extended — Features + XGBoost + Bayesian Opt")
    pr("=" * 70)

    # Load preprocessed features
    _FEAT = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    base_feat_cols = get_feat_cols(_FEAT)
    pr(f"Base features: {len(base_feat_cols)}")
    
    # Step 1: Rank base features → top-30 per target
    pr("\n── [1/3] Rank base features → top-30 per target ──")
    all_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(base_feat_cols, target)
        ranked = rank_features(_FEAT, leak, target)
        all_ranked[target] = ranked
        pr(f"  {target}: top5={[(r[0],r[1]) for r in ranked[:5]]}")
    
    # Step 2: Extended features on top-30
    pr("\n── [2/3] Extended features (top-30) ──")
    top30_all = []
    for target in TARGET_COLS:
        for col, _ in all_ranked[target][:30]:
            if col not in top30_all:
                top30_all.append(col)
    pr(f"  Unique top-30 features: {len(top30_all)}")
    
    ext_cols = add_extended_features(_FEAT, top30_all, n_top=len(top30_all))
    
    # Step 3: Personalization + re-ranking
    pr("\n── [3/3] Personalization + final ranking ──")
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"  Total features (base+extended): {len(all_feat_cols)}")
    
    _FEAT, zscore_cols = add_personalization(_FEAT, all_feat_cols)
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"  After personalization: {len(all_feat_cols)} features")
    
    # Re-rank all features per target
    final_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(all_feat_cols, target)
        ranked = rank_features(_FEAT, leak, target)
        final_ranked[target] = ranked
        pr(f"  {target}: {len(leak)} features ranked")
    
    # ── Bayesian Hyperparameter Optimization ──
    pr("\n── Bayesian Hyperparameter Optimization (Optuna) ──")
    
    HAS_OPTUNA = True
    try:
        import optuna
    except ImportError:
        HAS_OPTUNA = False
        pr("⚠️ Optuna not installed, using default params")
    
    best_lgb_configs = {}
    
    if HAS_OPTUNA:
        feat_counts = [20, 30]
        for target in TARGET_COLS:
            ranked = final_ranked[target]
            y = _FEAT[target].values
            np_ = max((y==1).sum(), 1)
            nn = (y==0).sum()
            spw = nn / np_
            
            gkf = GroupKFold(n_splits=N_SPLITS)
            fold_splits = list(gkf.split(_FEAT, y, _FEAT['subject_id']))
            
            # Pre-extract fold data for top 30 features
            scols = [r[0] for r in ranked[:30]]
            X_fold = []
            y_fold = []
            va_indices = []
            for fold, (tr_idx, va_idx) in enumerate(fold_splits):
                X_fold.append(_FEAT.iloc[tr_idx][scols].fillna(0).values)
                y_fold.append(y[tr_idx])
                va_indices.append(va_idx)
            
            sanitized_cols = [sanitize(c) for c in scols]
            
            def optuna_objective(trial, _spw=spw, _scols=scols, _fold_splits=fold_splits,
                                 _y=y, _FEAT=_FEAT, _sanitized_cols=sanitized_cols,
                                 _va_indices=va_indices, _X_fold=X_fold, _y_fold=y_fold):
                nl = trial.suggest_int('num_leaves', 6, 30)
                md = trial.suggest_int('max_depth', 2, 6)
                lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
                ne = trial.suggest_int('n_estimators', 100, 500)
                ss = trial.suggest_float('subsample', 0.5, 0.9)
                cst = trial.suggest_float('colsample_bytree', 0.5, 0.9)
                ra = trial.suggest_float('reg_alpha', 0.0, 3.0)
                rl = trial.suggest_float('reg_lambda', 0.0, 5.0)
                mc = trial.suggest_int('min_child_samples', 5, 20)
                
                params = {
                    'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':nl,'max_depth':md,'learning_rate':lr,
                    'n_estimators':ne,'subsample':ss,'colsample_bytree':cst,
                    'reg_alpha':ra,'reg_lambda':rl,
                    'min_child_samples':mc,'n_jobs':1,
                    'scale_pos_weight':_spw,
                }
                
                fold_losses = []
                for fold in range(N_SPLITS):
                    trd = lgb.Dataset(_X_fold[fold], label=_y_fold[fold], feature_name=_sanitized_cols)
                    Xva = _FEAT.iloc[_va_indices[fold]][_scols].fillna(0).values
                    yva = _y[_va_indices[fold]]
                    vad = lgb.Dataset(Xva, label=yva, feature_name=_sanitized_cols, reference=trd)
                    mdl = lgb.train(params, trd, num_boost_round=ne, valid_sets=[vad],
                                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                    pred = mdl.predict(Xva)
                    fold_losses.append(log_loss(yva, pred, labels=[0,1]))
                
                return np.mean(fold_losses)
            
            study = optuna.create_study(direction='minimize', 
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(optuna_objective, n_trials=15, timeout=200)
            
            best_params = study.best_params
            best_cv = study.best_value
            best_lgb_configs[target] = {**best_params, 'n_estimators': 500}
            pr(f"  {target}: optuna best CV={best_cv:.4f}, params={best_params}")
    else:
        for target in TARGET_COLS:
            best_lgb_configs[target] = {'objective':'binary','metric':'binary_logloss','verbose':-1,
                'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
                'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
                'min_child_samples':10,'n_jobs':1}
    
    # ── Model Comparison: LGBM vs XGB ──
    pr("\n── Model Comparison: LGBM vs XGBoost ──")
    
    feat_counts = [20, 30, 40, 50]
    lgb_results = {}
    xgb_results = {}
    
    for target in TARGET_COLS:
        ranked = final_ranked[target]
        y = _FEAT[target].values
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        # LGBM
        lgb_best_loss = float('inf')
        lgb_best_nf = None
        for nf in feat_counts:
            scols = [r[0] for r in ranked[:nf]]
            oof = cv_predict_lgb(scols, target, SEEDS, spw)
            cal = simple_cal(oof, y.mean())
            cal_loss = log_loss(y, cal, labels=[0,1])
            if cal_loss < lgb_best_loss:
                lgb_best_loss = cal_loss
                lgb_best_nf = nf
        lgb_results[target] = {'loss': lgb_best_loss, 'n_feats': lgb_best_nf}
        pr(f"  LGBM {target}: {lgb_best_loss:.6f} ({lgb_best_nf}f)")
        
        # XGB
        xgb_best_loss = float('inf')
        xgb_best_nf = None
        for nf in feat_counts:
            scols = [r[0] for r in ranked[:nf]]
            oof = cv_predict_xgb(scols, target, SEEDS, spw)
            cal = simple_cal(oof, y.mean())
            cal_loss = log_loss(y, cal, labels=[0,1])
            if cal_loss < xgb_best_loss:
                xgb_best_loss = cal_loss
                xgb_best_nf = nf
        xgb_results[target] = {'loss': xgb_best_loss, 'n_feats': xgb_best_nf}
        pr(f"  XGB  {target}: {xgb_best_loss:.6f} ({xgb_best_nf}f)")
    
    # ── Best per target ──
    pr(f"\n{'='*70}")
    pr("BEST MODEL PER TARGET")
    pr(f"{'='*70}")
    
    best_per_target = {}
    for target in TARGET_COLS:
        lgb_l = lgb_results[target]['loss']
        xgb_l = xgb_results[target]['loss']
        if lgb_l <= xgb_l:
            best_per_target[target] = {'model': 'LGBM', 'loss': lgb_l, 'n_feats': lgb_results[target]['n_feats']}
            pr(f"  {target}: LGBM={lgb_l:.6f}, XGB={xgb_l:.6f} → LGBM ✓")
        else:
            best_per_target[target] = {'model': 'XGB', 'loss': xgb_l, 'n_feats': xgb_results[target]['n_feats']}
            pr(f"  {target}: LGBM={lgb_l:.6f}, XGB={xgb_l:.6f} → XGB ✓")
    
    best_avg = np.mean([best_per_target[t]['loss'] for t in TARGET_COLS])
    pr(f"\nV22 best-per-target avg: {best_avg:.6f} (V10: 0.6038, Δ={best_avg-0.6038:+.6f})")
    
    # ── Train final models + submit ──
    pr(f"\n{'='*70}")
    pr("Training final models + submission")
    pr(f"{'='*70}")
    
    # Load test features (preprocessed)
    test_feat = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    pr(f"  Test features: {test_feat.shape}")
    
    # Add extended features to test
    test_ext_cols = add_extended_features(test_feat, get_feat_cols(test_feat)[:len(top30_all)], n_top=len(top30_all))
    
    # Add personalization
    test_feat_cols = get_feat_cols(test_feat)
    test_feat, _ = add_personalization(test_feat, test_feat_cols)
    
    predictions = test_feat[["subject_id", "sleep_date", "lifelog_date"]].copy()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    meta = {
        'version': 'v22',
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'base_features': len(base_feat_cols),
        'extended_features': len(ext_cols),
        'total_features': len(get_feat_cols(test_feat)),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'methods': ['extended_features', 'personalization', 'bayesian_optimization', 'xgboost_parallel'],
        'per_target': {},
    }
    
    for target in TARGET_COLS:
        best = best_per_target[target]
        ranked = final_ranked[target]
        scols = [r[0] for r in ranked[:best['n_feats']]]
        
        y = _FEAT[target].values
        train_rate = y.mean()
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        if best['model'] == 'LGBM':
            cfg = best_lgb_configs[target]
            cfg['n_estimators'] = 500
            sanitized = [sanitize(c) for c in scols]
            X_all = _FEAT[scols].fillna(0).values
            ds_all = lgb.Dataset(X_all, label=y, feature_name=sanitized)
            all_preds = np.zeros(len(test_feat))
            
            for seed in SEEDS:
                sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
                mdl = lgb.train(sc, ds_all, num_boost_round=cfg['n_estimators'])
                all_preds += mdl.predict(test_feat[scols].fillna(0).values)
            all_preds /= N_SEEDS
            
        else:
            xgb_cfg = {
                'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
                'max_depth':4,'learning_rate':0.03,'n_estimators':500,
                'subsample':0.7,'colsample_bytree':0.7,
                'reg_alpha':1.0,'reg_lambda':3.0,'min_child_weight':5,'n_jobs':1,
            }
            X_all = _FEAT[scols].fillna(0).values
            dtrain_all = xgb.DMatrix(X_all, label=y)
            all_preds = np.zeros(len(test_feat))
            
            for seed in SEEDS:
                sc = {**xgb_cfg, 'random_state': seed, 'scale_pos_weight': spw}
                mdl = xgb.train(sc, dtrain_all, num_boost_round=xgb_cfg['n_estimators'])
                all_preds += mdl.predict(xgb.DMatrix(test_feat[scols].fillna(0).values))
            all_preds /= N_SEEDS
        
        cal_preds = simple_cal(all_preds, train_rate)
        predictions[target] = cal_preds
        
        meta['per_target'][target] = {
            'model': best['model'], 'n_feature': best['n_feats'],
            'cal_loss_cv': best['loss'], 'train_rate': float(train_rate),
            'pred_mean': float(cal_preds.mean()), 'shift': float(cal_preds.mean()-train_rate),
        }
        pr(f"  {target}: {best['model']}+{best['n_feats']}f, mean={cal_preds.mean():.4f}, shift={cal_preds.mean()-train_rate:+.4f}")
    
    sub_path = SUBMIT_DIR / f"submission_v22_{timestamp}.csv"
    predictions.to_csv(sub_path, index=False)
    
    meta_path = SUBMIT_DIR / f"meta_v22_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    pr(f"\n✅ Submission saved: {sub_path}")
    pr(f"✅ Metadata saved: {meta_path}")
    
    pr(f"\n{'='*70}")
    pr("V22 FINAL SUMMARY")
    pr(f"{'='*70}")
    pr(f"{'Target':<6} {'Model':<8} {'Feats':<8} {'Train Rate':<12} {'Pred Mean':<12} {'Shift':<10}")
    for t in TARGET_COLS:
        m = meta['per_target'][t]
        pr(f"{t:<6} {m['model']:<8} {m['n_feature']:<8} {m['train_rate']:<12.3f} {m['pred_mean']:<12.4f} {m['shift']:+.4f}")
    
    pr(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")
    pr(f"\nV22 avg log_loss: {best_avg:.6f}")
    if best_avg < 0.6038:
        pr("🎯 BEATS V10!")
    else:
        pr("⚠️ Did NOT beat V10 — will proceed to Proposals 3,4,5")

if __name__ == "__main__":
    main()
