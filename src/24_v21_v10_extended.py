"""
V21: V10-based pipeline with Extended Features + XGBoost + Bayesian Opt

Based on V10 (07_v10_robust.py) with:
1. Extended features: lag, rolling, trend, interactions (top-100 features only)
2. XGBoost parallel + LightGBM
3. Per-target Bayesian hyperparameter optimization (optuna)
4. Same V10 approach: leakage fix, personalization, simple mean-match, 20-seed ensemble

Strategy: rank base features → take top-50 → add extended features on those → final ranking
"""
import sys, re, json, time, os, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB_INIT = True
except ImportError:
    HAS_XGB_INIT = False

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings('ignore')

def pr(msg):
    print(msg, flush=True)

sys.path.insert(0, 'src')
from config import TARGETS, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SEEDS = len(SEEDS)
N_SPLITS = 5

_SANITIZE_RE = re.compile(r'[^a-zA-Z0-9_]')
def sanitize(name):
    return _SANITIZE_RE.sub('_', name)

# ── Feature leakage fix ──
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
def add_extended_features(feat, feature_cols, n_top=50):
    """Add lag, rolling, trend, interactions for top N features only."""
    pr(f"  Selecting top {n_top} features for extended features...")
    
    df = feat.copy()
    df['_date_dt'] = pd.to_datetime(df['date'])
    
    added_cols = []
    selected = feature_cols[:n_top]
    
    for sid, grp in df.groupby('subject_id'):
        idx = grp.index
        df_sorted = df.loc[idx].sort_values('_date_dt')
        
        for col in selected:
            vals = df_sorted[col].fillna(0).values.astype(float)
            
            lag1 = np.roll(vals, 1)
            lag1[0] = vals[0]
            df.loc[idx, f'{col}_lag1'] = lag1
            added_cols.append(f'{col}_lag1')
            
            s = pd.Series(vals, index=df_sorted.index)
            roll3 = s.rolling(3, min_periods=1).mean()
            df.loc[idx, f'{col}_roll3'] = roll3.values
            added_cols.append(f'{col}_roll3')
            
            roll7 = s.rolling(7, min_periods=1).mean()
            df.loc[idx, f'{col}_roll7'] = roll7.values
            added_cols.append(f'{col}_roll7')
            
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
            
            dod = np.diff(vals, prepend=vals[0])
            df.loc[idx, f'{col}_dod'] = dod
            added_cols.append(f'{col}_dod')
    
    # Global interactions (always add)
    pr("  Computing global interactions...")
    if 'wPedo_pedo_step_mean' in df.columns and 'wLight_w_light_mean' in df.columns:
        df['step_x_light'] = df['wPedo_pedo_step_mean'] * df['wLight_w_light_mean']
        added_cols.append('step_x_light')
    if 'wHr_hr_std' in df.columns and 'mActivity_m_activity_mean' in df.columns:
        df['hr_std_x_activity'] = df['wHr_hr_std'] * df['mActivity_m_activity_mean']
        added_cols.append('hr_std_x_activity')
    
    screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_hour') and 'night' in c]
    if screen_cols:
        df['screen_night_ratio'] = df[screen_cols].sum(axis=1, min_count=1) / (len(screen_cols) + 1e-9)
        added_cols.append('screen_night_ratio')
    
    del df['_date_dt']
    
    pr(f"  Extended features: +{len(added_cols)} cols")
    return df, added_cols

# ── Personalization ──
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
def quick_rank(feat, feature_cols, target, n_trees=100):
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

# ── LGB base config ──
LGB_BASE = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_samples':10,'n_jobs':1,'verbose':-1,
}

# ── CV predict ──
def cv_predict_lgb(scols, target, seeds, spw, cfg):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    ne = cfg['n_estimators']
    sanitized = [sanitize(c) for c in scols]
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
        Xtr = feat.iloc[tr_idx][scols].fillna(0).values
        Xva = feat.iloc[va_idx][scols].fillna(0).values
        ytr, yva = y[tr_idx], y[va_idx]
        trd = lgb.Dataset(Xtr, label=ytr, feature_name=sanitized)
        seed_sum = np.zeros(len(va_idx))
        for seed in seeds:
            sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
            vad = lgb.Dataset(Xva, label=yva, feature_name=sanitized, reference=trd)
            mdl = lgb.train(sc, trd, num_boost_round=ne, valid_sets=[vad])
            seed_sum += mdl.predict(Xva)
        oof[va_idx] = seed_sum / N_SEEDS
    return oof

def cv_predict_xgb(scols, target, seeds, spw, cfg):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    ne = cfg['n_estimators']
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
        Xtr = feat.iloc[tr_idx][scols].fillna(0).values
        Xva = feat.iloc[va_idx][scols].fillna(0).values
        ytr, yva = y[tr_idx], y[va_idx]
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dval = xgb.DMatrix(Xva, label=yva)
        seed_sum = np.zeros(len(va_idx))
        for seed in seeds:
            sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
            mdl = xgb.train(sc, dtrain, num_boost_round=ne, evals=[(dval, 'val')])
            seed_sum += mdl.predict(dval)
        oof[va_idx] = seed_sum / N_SEEDS
    return oof

def main():
    global _FEAT
    t_total = time.time()
    pr("=" * 70)
    pr("V21: V10 Extended — Features + XGBoost + Bayesian Opt")
    pr("=" * 70)

    # Load features
    _FEAT = pd.read_parquet("data_processed/features.parquet")
    base_feat_cols = get_feat_cols(_FEAT)
    pr(f"Base features: {len(base_feat_cols)}")
    
    # Step 1: Rank base features per target → pick top-50
    pr("\n── [1/4] Rank base features → select top-50 per target ──")
    base_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(base_feat_cols, target)
        ranked = quick_rank(_FEAT, leak, target, n_trees=100)
        base_ranked[target] = ranked[:50]  # top 50
        top5 = [r[0] for r in ranked[:5]]
        pr(f"  {target}: top5={top5}, top50 count={len(ranked[:50])}")
    
    # Step 2: Add extended features on top-50
    pr("\n── [2/4] Extended features (top-50 only) ──")
    # Collect all top-50 columns across targets
    top50_all = set()
    for target in TARGET_COLS:
        for col, _ in base_ranked[target]:
            top50_all.add(col)
    top50_all = list(top50_all)
    pr(f"  Unique top-50 features across targets: {len(top50_all)}")
    
    _FEAT, ext_cols = add_extended_features(_FEAT, top50_all, n_top=len(top50_all))
    
    # Step 3: Personalization
    pr("\n── [3/4] Personalization (z-scores) ──")
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"Total features: {len(all_feat_cols)}")
    
    _FEAT, zscore_cols = add_personalization(_FEAT, all_feat_cols)
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"After personalization: {len(all_feat_cols)} features")
    
    # Step 4: Re-rank all features (base + extended + z-score) per target
    pr("\n── Final ranking (base+extended+z) per target ──")
    all_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(all_feat_cols, target)
        ranked = quick_rank(_FEAT, leak, target, n_trees=100)
        all_ranked[target] = ranked
        pr(f"  {target}: {len(leak)} features ranked")
    
    # ── Bayesian Hyperparameter Optimization ──
    pr("\n── [4/4] Bayesian Hyperparameter Optimization (Optuna) ──")
    
    try:
        import optuna
        HAS_OPTUNA = True
        pr("✅ Optuna available")
    except ImportError:
        HAS_OPTUNA = False
        pr("⚠️ Optuna not installed")
    
    best_lgb_configs = {}
    
    if HAS_OPTUNA:
        # Use top-30 ranked features for optuna search
        for target in TARGET_COLS:
            ranked = all_ranked[target]
            scols = [r[0] for r in ranked[:30]]
            y = _FEAT[target].values
            np_ = max((y==1).sum(), 1)
            nn = (y==0).sum()
            spw = nn / np_
            
            gkf = GroupKFold(n_splits=N_SPLITS)
            # Pre-compute fold splits
            fold_splits = list(gkf.split(_FEAT, y, _FEAT['subject_id']))
            
            # Pre-extract fold data
            X_fold = []
            y_fold = []
            va_indices = []
            for fold, (tr_idx, va_idx) in enumerate(fold_splits):
                X_fold.append(_FEAT.iloc[tr_idx][scols].fillna(0).values)
                y_fold.append(y[tr_idx])
                va_indices.append(va_idx)
            
            sanitized_cols = [sanitize(c) for c in scols]
            
            def optuna_objective(trial):
                nl = trial.suggest_int('num_leaves', 6, 40)
                md = trial.suggest_int('max_depth', 2, 8)
                lr = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
                ne = trial.suggest_int('n_estimators', 100, 800)
                ss = trial.suggest_float('subsample', 0.4, 0.9)
                cst = trial.suggest_float('colsample_bytree', 0.4, 0.9)
                ra = trial.suggest_float('reg_alpha', 0.0, 5.0)
                rl = trial.suggest_float('reg_lambda', 0.0, 10.0)
                mc = trial.suggest_int('min_child_samples', 3, 30)
                
                params = {
                    'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'num_leaves':nl,'max_depth':md,'learning_rate':lr,
                    'n_estimators':ne,'subsample':ss,'colsample_bytree':cst,
                    'reg_alpha':ra,'reg_lambda':rl,
                    'min_child_samples':mc,'n_jobs':1,
                    'scale_pos_weight':spw,
                }
                
                fold_losses = []
                for fold in range(N_SPLITS):
                    trd = lgb.Dataset(X_fold[fold], label=y_fold[fold], feature_name=sanitized_cols)
                    Xva = _FEAT.iloc[va_indices[fold]][scols].fillna(0).values
                    yva = y[va_indices[fold]]
                    vad = lgb.Dataset(Xva, label=yva, feature_name=sanitized_cols, reference=trd)
                    mdl = lgb.train(params, trd, num_boost_round=ne, valid_sets=[vad],
                                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                    pred = mdl.predict(Xva)
                    fold_losses.append(log_loss(yva, pred, labels=[0,1]))
                
                return np.mean(fold_losses)
            
            study = optuna.create_study(direction='minimize', 
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(optuna_objective, n_trials=20, timeout=300)
            
            best_params = study.best_params
            best_cv = study.best_value
            best_lgb_configs[target] = {**best_params, 'n_estimators': 500}
            pr(f"  {target}: optuna best CV={best_cv:.4f}")
    else:
        for target in TARGET_COLS:
            best_lgb_configs[target] = LGB_BASE.copy()
    
    # ── Model comparison: LGBM vs XGB ──
    pr("\n── Model Comparison: LGBM vs XGBoost ──")
    
    try:
        import xgboost as xgb
        HAS_XGB = True
        pr("✅ XGBoost available")
    except ImportError:
        HAS_XGB = False
        pr("⚠️ XGBoost not available")
    
    feat_counts = [20, 30, 40, 50]
    lgb_results = {}
    xgb_results = {}
    
    for target in TARGET_COLS:
        ranked = all_ranked[target]
        y = _FEAT[target].values
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        lgb_best_loss = float('inf')
        lgb_best_nf = None
        
        for nf in feat_counts:
            scols = [r[0] for r in ranked[:nf]]
            cfg = {**best_lgb_configs[target], 'n_estimators': 500}
            oof = cv_predict_lgb(scols, target, SEEDS, spw, cfg)
            
            shift = y.mean() - oof.mean()
            cal = np.clip(oof + shift, 0.0001, 0.9999)
            cal_loss = log_loss(y, cal, labels=[0,1])
            
            if cal_loss < lgb_best_loss:
                lgb_best_loss = cal_loss
                lgb_best_nf = nf
        
        lgb_results[target] = {'loss': lgb_best_loss, 'n_feats': lgb_best_nf}
        pr(f"  LGBM {target}: {lgb_best_loss:.6f} ({lgb_best_nf}f)")
    
    if HAS_XGB:
        xgb_base = {
            'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
            'max_depth':4,'learning_rate':0.03,'n_estimators':500,
            'subsample':0.7,'colsample_bytree':0.7,
            'reg_alpha':1.0,'reg_lambda':3.0,'min_child_weight':5,'n_jobs':1,
        }
        
        for target in TARGET_COLS:
            ranked = all_ranked[target]
            y = _FEAT[target].values
            np_ = max((y==1).sum(), 1)
            nn = (y==0).sum()
            spw = nn / np_
            
            xgb_best_loss = float('inf')
            xgb_best_nf = None
            
            for nf in feat_counts:
                scols = [r[0] for r in ranked[:nf]]
                oof = cv_predict_xgb(scols, target, SEEDS, spw, xgb_base)
                
                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
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
        xgb_l = xgb_results.get(target, {}).get('loss', float('inf'))
        
        if lgb_l <= xgb_l:
            best_per_target[target] = {'model': 'LGBM', 'loss': lgb_l, 'n_feats': lgb_results[target]['n_feats']}
            pr(f"  {target}: LGBM={lgb_l:.6f}, XGB={xgb_l:.6f} → LGBM ✓")
        else:
            best_per_target[target] = {'model': 'XGB', 'loss': xgb_l, 'n_feats': xgb_results[target]['n_feats']}
            pr(f"  {target}: LGBM={lgb_l:.6f}, XGB={xgb_l:.6f} → XGB ✓")
    
    best_avg = np.mean([best_per_target[t]['loss'] for t in TARGET_COLS])
    pr(f"\nV21 best-per-target avg: {best_avg:.6f} (V10: 0.6038, Δ={best_avg-0.6038:+.6f})")
    if best_avg < 0.6038:
        pr("🎯 BEATS V10!")
    
    # ── Train final models + submit ──
    pr(f"\n{'='*70}")
    pr("Training final models + submission")
    pr(f"{'='*70}")
    
    import importlib.util
    from config import DATA_DIR, PARQUET_FILES, SAMPLE_CSV
    
    spec = importlib.util.spec_from_file_location("01_load_data", Path('src/01_load_data.py'))
    load_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_data)
    
    spec2 = importlib.util.spec_from_file_location("02_feature_engineering", Path('src/02_feature_engineering.py'))
    feat_eng = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(feat_eng)
    
    parquet_dfs = {}
    for name in PARQUET_FILES:
        path = DATA_DIR / PARQUET_FILES[name]
        pr(f"  loading {name}...")
        df = pd.read_parquet(path)
        parquet_dfs[name] = load_data.build_merge_key(df)
    
    sample = pd.read_csv(SAMPLE_CSV)
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
    
    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    pr(f"  Test features: {test_features.shape}")
    
    # Apply same transformations
    test_feat_cols = get_feat_cols(test_features)
    test_features, _ = add_extended_features(test_features, test_feat_cols[:len(top50_all)], n_top=len(top50_all))
    test_feat_cols = get_feat_cols(test_features)
    test_features, _ = add_personalization(test_features, test_feat_cols)
    
    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    meta = {
        'version': 'v21',
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'base_features': len(base_feat_cols),
        'extended_features': len(ext_cols),
        'total_features': len(get_feat_cols(test_features)),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'methods': ['extended_features', 'personalization', 'bayesian_optimization', 'xgboost_parallel'],
        'per_target': {},
    }
    
    for target in TARGET_COLS:
        best = best_per_target[target]
        ranked = all_ranked[target]
        scols = [r[0] for r in ranked[:best['n_feats']]]
        
        y = _FEAT[target].values
        train_rate = y.mean()
        np_ = max((y==1).sum(), 1)
        nn = (y==0).sum()
        spw = nn / np_
        
        if best['model'] == 'LGBM':
            cfg = {**best_lgb_configs[target], 'n_estimators': 500}
            sanitized = [sanitize(c) for c in scols]
            X_all = _FEAT[scols].fillna(0).values
            ds_all = lgb.Dataset(X_all, label=y, feature_name=sanitized)
            all_preds = np.zeros(len(test_features))
            
            for seed in SEEDS:
                sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
                mdl = lgb.train(sc, ds_all, num_boost_round=cfg['n_estimators'])
                all_preds += mdl.predict(test_features[scols].fillna(0).values)
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
            all_preds = np.zeros(len(test_features))
            
            for seed in SEEDS:
                sc = {**xgb_cfg, 'random_state': seed, 'scale_pos_weight': spw}
                mdl = xgb.train(sc, dtrain_all, num_boost_round=xgb_cfg['n_estimators'])
                all_preds += mdl.predict(xgb.DMatrix(test_features[scols].fillna(0).values))
            all_preds /= N_SEEDS
        
        shift = train_rate - all_preds.mean()
        cal_preds = np.clip(all_preds + shift, 0.0001, 0.9999)
        predictions[target] = cal_preds
        
        meta['per_target'][target] = {
            'model': best['model'],
            'n_feature': best['n_feats'],
            'cal_loss_cv': best['loss'],
            'train_rate': float(train_rate),
            'pred_mean': float(cal_preds.mean()),
            'pred_min': float(cal_preds.min()),
            'pred_max': float(cal_preds.max()),
            'shift': float(shift),
        }
        
        pr(f"  {target}: {best['model']}+{best['n_feats']}f, mean={cal_preds.mean():.4f}, shift={shift:+.4f}")
    
    sub_path = SUBMIT_DIR / f"submission_v21_{timestamp}.csv"
    predictions.to_csv(sub_path, index=False)
    
    meta_path = SUBMIT_DIR / f"meta_v21_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    pr(f"\n✅ Submission saved: {sub_path}")
    pr(f"✅ Metadata saved: {meta_path}")
    
    pr(f"\n{'='*70}")
    pr("V21 FINAL SUMMARY")
    pr(f"{'='*70}")
    pr(f"{'Target':<6} {'Model':<8} {'Feats':<8} {'Train Rate':<12} {'Pred Mean':<12} {'Shift':<10}")
    for t in TARGET_COLS:
        m = meta['per_target'][t]
        pr(f"{t:<6} {m['model']:<8} {m['n_feature']:<8} {m['train_rate']:<12.3f} {m['pred_mean']:<12.4f} {m['shift']:+.4f}")
    
    pr(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")

if __name__ == "__main__":
    main()
