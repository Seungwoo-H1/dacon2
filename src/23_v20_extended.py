"""
V20: V8-based pipeline with Extended Features + XGBoost parallel

Improvements over V8:
1. Extended feature engineering: lag features, rolling stats, interactions, trends
2. XGBoost parallel training + LightGBM comparison
3. Same V8 approach: per-subject z-score personalization, simple mean-match calibration
4. Multi-config search with both LGBM and XGB

Author: 집가헤eng 🏠
"""
import sys, re, json, time, os, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

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

# ── Extended Feature Engineering ──
def add_extended_features(feat):
    """
    Add lag, rolling, trend, and interaction features.
    Operates on subject_id groups using date ordering.
    """
    pr("  Adding extended features (lag, rolling, trend, interactions)...")
    t0 = time.time()
    feature_cols_before = get_feat_cols(feat)
    
    df = feat.copy()
    # Convert date to datetime for ordering
    df['_date_dt'] = pd.to_datetime(df['date'])
    
    added_cols = []
    
    for sid, grp in df.groupby('subject_id'):
        idx = grp.index
        df_sorted = df.loc[idx].sort_values('_date_dt')
        
        for col in feature_cols_before:
            if col in META_COLS or col in TARGET_COLS:
                continue
            
            vals = df_sorted[col].fillna(0).values.astype(float)
            
            # Lag features (1-day lag)
            lag1 = np.roll(vals, 1)
            lag1[0] = vals[0]  # first day: repeat
            df.loc[idx, f'{col}_lag1'] = lag1
            added_cols.append(f'{col}_lag1')
            
            # 3-day rolling mean
            s = pd.Series(vals, index=df_sorted.index)
            roll3 = s.rolling(3, min_periods=1, center=False).mean()
            df.loc[idx, f'{col}_roll3'] = roll3.values
            added_cols.append(f'{col}_roll3')
            
            # 7-day rolling mean
            roll7 = s.rolling(7, min_periods=1, center=False).mean()
            df.loc[idx, f'{col}_roll7'] = roll7.values
            added_cols.append(f'{col}_roll7')
            
            # Trend (linear slope over available history, capped at 7 days)
            slopes = np.zeros(len(vals))
            for i in range(len(vals)):
                start = max(0, i - 6)
                window = vals[start:i+1]
                if len(window) >= 3:
                    x = np.arange(len(window))
                    m, b = np.polyfit(x, window, 1)
                    slopes[i] = m
            df.loc[idx, f'{col}_trend'] = slopes
            added_cols.append(f'{col}_trend')
            
            # Day-over-day change
            dod = np.diff(vals, prepend=vals[0])
            df.loc[idx, f'{col}_dod'] = dod
            added_cols.append(f'{col}_dod')
    
    # Interaction features (top feature pairs)
    pr("  Computing top interactions...")
    df['_date_dt'] = pd.to_datetime(df['date'])
    
    # Step-light interaction: physical activity × environment
    if 'wPedo_pedo_step_mean' in df.columns and 'wLight_w_light_mean' in df.columns:
        df['step_x_light'] = df['wPedo_pedo_step_mean'] * df['wLight_w_light_mean']
        added_cols.append('step_x_light')
    
    # HR variability × activity
    if 'wHr_hr_std' in df.columns and 'mActivity_m_activity_mean' in df.columns:
        df['hr_std_x_activity'] = df['wHr_hr_std'] * df['mActivity_m_activity_mean']
        added_cols.append('hr_std_x_activity')
    
    # Screen time ratio (hour-based)
    screen_cols = [c for c in df.columns if c.startswith('mScreenStatus_hour')]
    if screen_cols:
        df['screen_night_ratio'] = df[[c for c in screen_cols if 'night' in c]].sum(axis=1, min_count=1) / (len([c for c in screen_cols if 'night' in c]) + 1e-9)
        added_cols.append('screen_night_ratio')
    
    # Delete temp column
    del df['_date_dt']
    
    pr(f"  Extended features added: {len(added_cols)} new cols (total before: {len(feature_cols_before)})")
    pr(f"  New total features: {len(get_feat_cols(df))}")
    return df, added_cols

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
def quick_rank(feat, feature_cols, target, n_trees=100):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values.astype(np.float32)
    n_pos = max((y==1).sum(), 1)
    n_neg = (y==0).sum()
    spw = n_neg / n_pos
    
    params = {
        'objective':'binary','metric':'binary_logloss','verbose':-1,
        'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':n_trees,
        'subsample':0.7,'colsample_bytree':0.7,
        'reg_alpha':1.0,'reg_lambda':3.0,
        'scale_pos_weight':spw,'random_state':42,
        'min_child_samples':10,'n_jobs':1,
    }
    ds = lgb.Dataset(X, label=y)
    mdl = lgb.train(params, ds, num_boost_round=n_trees)
    imp = mdl.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    return ranked

# ── LightGBM Configs ──
LGB_V8 = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':10,'max_depth':3,'learning_rate':0.03,'n_estimators':300,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':0.5,'reg_lambda':1.0,
    'min_child_samples':10,'n_jobs':1,'verbose':-1,
}
LGB_V10_CFG = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_samples':10,'n_jobs':1,'verbose':-1,
}
LGB_REGULAR = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':400,
    'subsample':0.8,'colsample_bytree':0.8,
    'reg_alpha':0.3,'reg_lambda':0.5,
    'min_child_samples':8,'n_jobs':1,'verbose':-1,
}

# ── XGBoost Configs ──
XGB_V8 = {
    'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
    'max_depth':3,'learning_rate':0.03,'n_estimators':300,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':0.5,'reg_lambda':1.0,
    'min_child_weight':5,'random_state':42,'n_jobs':1,
}
XGB_V10_CFG = {
    'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
    'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_weight':5,'random_state':42,'n_jobs':1,
}
XGB_REGULAR = {
    'objective':'binary:logistic','eval_metric':'logloss','verbosity':0,
    'max_depth':5,'learning_rate':0.02,'n_estimators':400,
    'subsample':0.8,'colsample_bytree':0.8,
    'reg_alpha':0.3,'reg_lambda':0.5,
    'min_child_weight':3,'random_state':42,'n_jobs':1,
}

try:
    import xgboost as xgb
    HAS_XGB = True
    pr("✅ XGBoost available")
except ImportError:
    HAS_XGB = False
    pr("⚠️ XGBoost not available, skipping")

def cv_predict_lgb(scols, target, seeds, spw, cfg):
    feat = _FEAT
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(y))
    ne = cfg['n_estimators']
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
        Xtr = feat.iloc[tr_idx][scols].fillna(0).values
        Xva = feat.iloc[va_idx][scols].fillna(0).values
        ytr, yva = y[tr_idx], y[va_idx]
        trd = lgb.Dataset(Xtr, label=ytr)
        
        seed_sum = np.zeros(len(va_idx))
        for seed in seeds:
            sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
            vad = lgb.Dataset(Xva, label=yva, reference=trd)
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
            sc = {**cfg, 'random_state': seed}
            # XGBoost scale_pos_weight
            if 'scale_pos_weight' not in sc:
                sc['scale_pos_weight'] = spw
            mdl = xgb.train(sc, dtrain, num_boost_round=ne, evals=[(dval, 'val')])
            seed_sum += mdl.predict(dval)
        oof[va_idx] = seed_sum / N_SEEDS
    return oof

def main():
    global _FEAT
    t_total = time.time()
    pr("=" * 70)
    pr("V20: V8-based + Extended Features + XGBoost parallel")
    pr("=" * 70)

    # Load features
    _FEAT = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(_FEAT)
    pr(f"Base features: {len(feat_cols)}")
    
    # Extended features
    pr("\n── Extended Feature Engineering ──")
    _FEAT, ext_cols = add_extended_features(_FEAT)
    
    # Personalization
    pr("\n── Personalization (z-scores) ──")
    _FEAT, zscore_cols = add_personalization(_FEAT, get_feat_cols(_FEAT))
    all_feat_cols = get_feat_cols(_FEAT)
    pr(f"Total features after extensions: {len(all_feat_cols)}")
    
    # Feature ranking per target
    pr("\n── Feature ranking per target ──")
    all_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(all_feat_cols, target)
        ranked = quick_rank(_FEAT, leak, target, n_trees=100)
        all_ranked[target] = ranked
        top5 = [r[0] for r in ranked[:5]]
        pr(f"  {target}: top5={top5}")
    
    # ── Config search ──
    pr("\n── Config Grid Search ──")
    
    lgb_configs = [
        ('LGB-V8', LGB_V8),
        ('LGB-V10', LGB_V10_CFG),
        ('LGB-REG', LGB_REGULAR),
    ]
    xgb_configs = [
        ('XGB-V8', XGB_V8),
        ('XGB-V10', XGB_V10_CFG),
        ('XGB-REG', XGB_REGULAR),
    ] if HAS_XGB else []
    
    all_configs = lgb_configs + xgb_configs
    feat_counts = [10, 20, 30, 40, 50]
    
    all_results = []
    combo = 0
    total_combos = len(all_configs) * len(feat_counts) * len(TARGET_COLS)
    pr(f"Total combos: {total_combos}")
    
    cv_fn_lgb = cv_predict_lgb
    cv_fn_xgb = cv_predict_xgb
    
    for cname, cfg in all_configs:
        cv_fn = cv_fn_xgb if cname.startswith('XGB') else cv_fn_lgb
        pr(f"\n--- {cname} ---")
        
        for n_feats in feat_counts:
            for tidx, target in enumerate(TARGET_COLS):
                combo += 1
                ranked = all_ranked[target]
                scols = [r[0] for r in ranked[:n_feats]]
                y = _FEAT[target].values
                np_ = max((y==1).sum(), 1)
                nn = (y==0).sum()
                spw = nn / np_
                
                t0 = time.time()
                oof = cv_fn(scols, target, SEEDS, spw, cfg)
                elapsed = time.time() - t0
                
                # Calibrate
                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                cal_loss = log_loss(y, cal, labels=[0,1])
                
                all_results.append({
                    'target': target, 'config': cname,
                    'n_feats': n_feats, 'cal_loss': cal_loss,
                    'elapsed': elapsed, 'model_type': 'XGB' if cname.startswith('XGB') else 'LGB'
                })
                
                v10c = 0.6038  # V10 avg baseline for display
                if combo % 10 == 0:
                    pr(f"  [{combo}/{total_combos}] {target} {cname}+{n_feats}f cal={cal_loss:.4f} [{elapsed:.0f}s]")
    
    # ── Results summary ──
    pr(f"\n{'='*70}")
    pr("V20 RESULTS")
    pr(f"{'='*70}")
    
    for target in TARGET_COLS:
        tgt_res = [r for r in all_results if r['target'] == target]
        best_r = min(tgt_res, key=lambda x: x['cal_loss'])
        best_oof_for_target = best_r
        
        # Also show per-model-type best
        lgb_best = min([r for r in tgt_res if r['model_type'] == 'LGB'], key=lambda x: x['cal_loss'])
        xgb_best = min([r for r in tgt_res if r['model_type'] == 'XGB'], key=lambda x: x['cal_loss']) if HAS_XGB else None
        
        pr(f"{target}:")
        pr(f"  Overall best: {best_r['config']}+{best_r['n_feats']}f cal={best_r['cal_loss']:.6f}")
        pr(f"  LGB best: {lgb_best['config']}+{lgb_best['n_feats']}f cal={lgb_best['cal_loss']:.6f}")
        if xgb_best:
            pr(f"  XGB best: {xgb_best['config']}+{xgb_best['n_feats']}f cal={xgb_best['cal_loss']:.6f}")
            xgb_improve = lgb_best['cal_loss'] - xgb_best['cal_loss']
            pr(f"  XGB vs LGB: {xgb_improve:+.6f}")
    
    best_per_target = {}
    for target in TARGET_COLS:
        tgt_res = [r for r in all_results if r['target'] == target]
        best_per_target[target] = min(tgt_res, key=lambda x: x['cal_loss'])
    
    best_avg = np.mean([best_per_target[t]['cal_loss'] for t in TARGET_COLS])
    pr(f"\nV20 best-per-target avg: {best_avg:.6f} (V10: 0.6038, Δ={best_avg-0.6038:+.6f})")
    pr(f"{'🎯 BEATS V10!' if best_avg < 0.6038 else 'Not yet. Continuing...'}")
    
    # ── Train final models on full data + submit ──
    pr(f"\n{'='*70}")
    pr("Training final models for submission")
    pr(f"{'='*70}")
    
    # Load test data
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
        pr(f"  loading {name}: {path.name} ...")
        df = pd.read_parquet(path)
        parquet_dfs[name] = load_data.build_merge_key(df)
    
    sample = pd.read_csv(SAMPLE_CSV)
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"]).dt.date
    
    test_features = feat_eng.create_day_features(parquet_dfs, sample)
    pr(f"  Test features: {test_features.shape}")
    
    # Apply same extended features + personalization
    test_features, _ = add_extended_features(test_features)
    test_feat_cols = get_feat_cols(test_features)
    test_features, _ = add_personalization(test_features, test_feat_cols)
    
    predictions = test_features[["subject_id", "sleep_date", "lifelog_date"]].copy()
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Per-target: train best config on all data, predict test
    meta = {
        'version': 'v20',
        'timestamp': timestamp,
        'n_samples': len(predictions),
        'base_features': len(feat_cols),
        'extended_features': len(ext_cols),
        'total_features': len(get_feat_cols(test_features)),
        'n_seeds': N_SEEDS,
        'n_splits': N_SPLITS,
        'per_target': {},
        'best_per_target': {},
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
        
        # Load best config
        cfg_name = best['config']
        if cfg_name.startswith('LGB'):
            if cfg_name == 'LGB-V8': cfg = LGB_V8
            elif cfg_name == 'LGB-V10': cfg = LGB_V10_CFG
            else: cfg = LGB_REGULAR
            cv_fn = cv_predict_lgb
            
            # Train on all data
            X_all = _FEAT[scols].fillna(0).values
            ds_all = lgb.Dataset(X_all, label=y)
            best_model_seeds = []
            for seed in SEEDS:
                sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
                mdl = lgb.train(sc, ds_all, num_boost_round=cfg['n_estimators'])
                best_model_seeds.append(mdl)
            
            # Predict test
            test_X = test_features[scols].fillna(0).values
            all_preds = np.zeros(len(test_X))
            for mdl in best_model_seeds:
                all_preds += mdl.predict(test_X)
            all_preds /= N_SEEDS
            
            # Calibrate
            shift = train_rate - all_preds.mean()
            cal_preds = np.clip(all_preds + shift, 0.0001, 0.9999)
            
        elif cfg_name.startswith('XGB'):
            if cfg_name == 'XGB-V8': cfg = XGB_V8
            elif cfg_name == 'XGB-V10': cfg = XGB_V10_CFG
            else: cfg = XGB_REGULAR
            cv_fn = cv_predict_xgb
            
            X_all = _FEAT[scols].fillna(0).values
            dtrain_all = xgb.DMatrix(X_all, label=y)
            best_model_seeds = []
            for seed in SEEDS:
                sc = {**cfg, 'random_state': seed}
                sc['scale_pos_weight'] = spw
                mdl = xgb.train(sc, dtrain_all, num_boost_round=cfg['n_estimators'])
                best_model_seeds.append(mdl)
            
            test_X = test_features[scols].fillna(0).values
            dtest = xgb.DMatrix(test_X)
            all_preds = np.zeros(len(test_X))
            for mdl in best_model_seeds:
                all_preds += mdl.predict(dtest)
            all_preds /= N_SEEDS
            
            shift = train_rate - all_preds.mean()
            cal_preds = np.clip(all_preds + shift, 0.0001, 0.9999)
        else:
            continue
        
        predictions[target] = cal_preds
        
        meta['per_target'][target] = {
            'config': cfg_name,
            'n_features': best['n_feats'],
            'model_type': 'XGB' if cfg_name.startswith('XGB') else 'LGB',
            'train_rate': float(train_rate),
            'pred_mean': float(cal_preds.mean()),
            'pred_min': float(cal_preds.min()),
            'pred_max': float(cal_preds.max()),
            'shift': float(shift),
        }
        meta['best_per_target'][target] = {
            'config': cfg_name,
            'n_feats': best['n_feats'],
            'cal_loss': best['cal_loss'],
        }
        
        pr(f"  {target}: {cfg_name}+{best['n_feats']}f, mean={cal_preds.mean():.4f}, shift={shift:+.4f}")
    
    # Save
    sub_path = SUBMIT_DIR / f"submission_v20_{timestamp}.csv"
    predictions.to_csv(sub_path, index=False)
    
    meta_path = SUBMIT_DIR / f"meta_v20_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    pr(f"\n✅ Submission saved: {sub_path}")
    pr(f"✅ Metadata saved: {meta_path}")
    
    # Summary table
    pr(f"\n{'='*70}")
    pr("V20 FINAL SUMMARY")
    pr(f"{'='*70}")
    pr(f"{'Target':<6} {'Config':<12} {'Feats':<8} {'Train Rate':<12} {'Pred Mean':<12} {'Shift':<10}")
    for t in TARGET_COLS:
        m = meta['per_target'][t]
        pr(f"{t:<6} {m['config']:<12} {m['n_features']:<8} {m['train_rate']:<12.3f} {m['pred_mean']:<12.4f} {m['shift']:+.4f}")
    
    pr(f"\nTotal time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")

if __name__ == "__main__":
    main()
