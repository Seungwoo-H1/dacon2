"""
V13 - Better Calibration (Lightweight, with personalization)
Strategy: Test multiple calibration strategies:
1. Global mean-match (V10 baseline)
2. Logit-space mean-match
3. Per-subject mean-match

Uses personalization (z-score) but efficiently:
- Compute z-score stats ONCE, store as float32
- Use lgb.Dataset directly (memory efficient)

Memory-optimized: 10 seeds, 3 configs, 2 feature counts per target.
"""
import sys, re, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, "src")
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}
RANDOM_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SEEDS = len(RANDOM_SEEDS)
N_SPLITS = 5

LEAK_S = {"wLight_w_light_mean","wLight_w_light_std","wLight_w_light_min","wLight_w_light_max","wLight_w_light_count",
    "wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count",
    "wPedo_pedo_step_mean","wPedo_pedo_step_sum","wPedo_pedo_step_frequency_mean","wPedo_pedo_step_frequency_sum",
    "wPedo_pedo_running_step_mean","wPedo_pedo_running_step_sum","wPedo_pedo_walking_step_mean","wPedo_pedo_walking_step_sum",
    "wPedo_pedo_distance_mean","wPedo_pedo_distance_sum","wPedo_pedo_speed_mean","wPedo_pedo_speed_sum",
    "wPedo_pedo_burned_calories_mean","wPedo_pedo_burned_calories_sum"}
LEAK_Q = {"wHr_hr_mean","wHr_hr_std","wHr_hr_min","wHr_hr_max","wHr_hr_median","wHr_hr_count"}

def sanitize(n): return re.sub(r"[^a-zA-Z0-9_]", "_", n)
def get_feat_cols(f):
    return [c for c in f.columns if c not in META_COLS | set(TARGET_COLS)
            and f[c].dtype in [np.float64,np.int64,float,int,bool,np.bool_]]
def remove_leak(cols, t):
    if t.startswith("S"): return [c for c in cols if c not in LEAK_S]
    elif t.startswith("Q"): return [c for c in cols if c not in LEAK_Q]
    return cols

LGB_CFG = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,
    'force_row_wise':True,'n_jobs':-1,'verbose':-1,
}

CONFIGS = [
    {'name':'C1','nl':8,'md':3,'lr':0.02,'ne':200,'ss':0.6,'cst':0.6,'ra':2.0,'rl':5.0,'mc':15},
    {'name':'C3','nl':12,'md':4,'lr':0.03,'ne':200,'ss':0.7,'cst':0.7,'ra':1.0,'rl':3.0,'mc':10},
    {'name':'C5','nl':20,'md':5,'lr':0.02,'ne':300,'ss':0.7,'cst':0.7,'ra':0.5,'rl':2.0,'mc':8},
]

def add_personalization_fast(df, feat_cols):
    """Compute z-score personalization, return feature matrix (np array) + subject_id + datetime cols."""
    df = df.copy()
    X_base = df[feat_cols].fillna(0).values.astype(np.float32)
    
    # Per-subject z-score: compute mean/std per subject
    subj_means = df.groupby('subject_id')[feat_cols].mean().values.astype(np.float32)
    subj_stds = df.groupby('subject_id')[feat_cols].std().fillna(1).values.astype(np.float32)
    subj_stds[subj_stds < 1e-10] = 1.0
    
    # Map subject stats to rows
    subj_map = {sid: i for i, sid in enumerate(df['subject_id'].unique())}
    subj_indices = np.array([subj_map[sid] for sid in df['subject_id']])
    
    X_zscore = np.zeros_like(X_base)
    for j, col in enumerate(feat_cols):
        means = subj_means[:, j]
        stds = subj_stds[:, j]
        for i, (si, row) in enumerate(zip(subj_indices, X_base)):
            X_zscore[i, j] = (X_base[i, j] - means[si]) / stds[si]
    
    # Concatenate base + zscore
    X_all = np.hstack([X_base, X_zscore]).astype(np.float32)
    
    return X_all, df['subject_id'].values, df['lifelog_date'].values, df['sleep_date'].values

def rank_features(X, y, feat_cols, seed=42):
    n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
    spw = n_neg / n_pos
    cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
           'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
           'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
           'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10}
    ds = lgb.Dataset(X, label=y, feature_name=[f'f{i}' for i in range(X.shape[1])], params={'verbose':'-1'})
    model = lgb.train(cfg, ds, num_boost_round=100)
    imp = model.feature_importance(importance_type="gain")
    return sorted(zip(feat_cols, imp), key=lambda x: -x[1])

def lgb_cv_predict_fast(X, y, selected_indices, seeds, spw, subject_ids):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_full = np.zeros((len(y), len(seeds)))
    n_cols = len(selected_indices)
    feat_names = [f'f{i}' for i in selected_indices]
    
    for si, seed in enumerate(seeds):
        cfg = {**LGB_CFG, 'random_state': seed}
        for fold, (tri, vai) in enumerate(gkf.split(X, y, subject_ids)):
            Xtr = X[tri][:, selected_indices]
            Xva = X[vai][:, selected_indices]
            ytr, yva = y[tri], y[vai]
            tr_ds = lgb.Dataset(Xtr, label=ytr, feature_name=feat_names, params={'verbose':'-1'})
            va_ds = lgb.Dataset(Xva, label=yva, feature_name=feat_names, reference=tr_ds, params={'verbose':'-1'})
            params = {**cfg, 'scale_pos_weight': spw}
            mdl = lgb.train(params, tr_ds, num_boost_round=cfg['n_estimators'],
                          valid_sets=[va_ds],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof_full[vai, si] = mdl.predict(Xva)
    oof_avg = oof_full.mean(axis=1)
    return oof_avg, oof_full

def cal_mean_match(pred, target_rate):
    cal = pred + (target_rate - pred.mean())
    return np.clip(cal, 0.0001, 0.9999)

def cal_logit_shift(pred, target_rate):
    eps = 1e-10
    p = np.clip(pred, eps, 1-eps)
    logit_pred = np.log(p / (1-p))
    logit_target = np.log(target_rate / (1-target_rate))
    cal_logit = logit_pred + logit_target
    return np.clip(1 / (1 + np.exp(-cal_logit)), 0.0001, 0.9999)

def main():
    log.info("=" * 70)
    log.info("V13: Better Calibration (with personalization, memory-efficient)")
    log.info("=" * 70)
    
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Features: {feat.shape}")
    
    feat_cols = get_feat_cols(feat)
    log.info(f"Feature cols: {len(feat_cols)}")
    
    # Personalization (once)
    log.info("Computing z-score personalization...")
    start = time.time()
    X_all, subj_ids, lf_dates, sl_dates = add_personalization_fast(feat, feat_cols)
    log.info(f"Personalization done: X shape={X_all.shape}, time={time.time()-start:.1f}s")
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    
    results = {}
    
    for target in TARGET_COLS:
        log.info(f"\n{'='*40}")
        log.info(f"Training {target}...")
        
        y = feat[target].values
        
        # Build leak-free feature indices
        leak_cols_set = remove_leak(feat_cols, target)
        
        # Rank features using base + zscore cols
        ranked = rank_features(X_all, y, leak_cols_set, seed=42)
        
        best_loss = float('inf')
        best_oof = None
        best_selected_indices = None
        best_cfg = None
        best_cal_method = None
        
        for n_feat in [10, 20]:
            if n_feat > len(ranked):
                continue
            ranked_cols = [r[0] for r in ranked[:n_feat]]
            
            # Get indices in X_all
            col_to_idx = {c: i for i, c in enumerate(feat_cols)}
            selected_indices = [col_to_idx[c] for c in ranked_cols]
            
            for cfg in CONFIGS:
                test_cfg = {**LGB_CFG,
                           'num_leaves':cfg['nl'],'max_depth':cfg['md'],
                           'learning_rate':cfg['lr'],'n_estimators':cfg['ne'],
                           'subsample':cfg['ss'],'colsample_bytree':cfg['cst'],
                           'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
                           'min_child_samples':cfg['mc']}
                
                n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
                spw = n_neg / n_pos
                
                oof, oof_full = lgb_cv_predict_fast(X_all, y, selected_indices, RANDOM_SEEDS, spw, subj_ids)
                
                # 2 calibration methods
                cal_mm = cal_mean_match(oof, train_rate[target])
                cal_logit = cal_logit_shift(oof, train_rate[target])
                
                loss_mm = log_loss(y, cal_mm, labels=[0,1])
                loss_logit = log_loss(y, cal_logit, labels=[0,1])
                
                for cal_name, loss in [('mean_match', loss_mm), ('logit_shift', loss_logit)]:
                    if loss < best_loss:
                        best_loss = loss
                        best_oof = oof
                        best_selected_indices = selected_indices
                        best_cfg = {**cfg, '_n_feats': n_feat}
                        best_cal_method = cal_name
        
        # Log all cal methods for comparison
        for n_feat in [10, 20]:
            if n_feat > len(ranked):
                continue
            ranked_cols = [r[0] for r in ranked[:n_feat]]
            col_to_idx = {c: i for i, c in enumerate(feat_cols)}
            sel_idx = [col_to_idx[c] for c in ranked_cols]
            
            n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
            spw = n_neg / n_pos
            oof, _ = lgb_cv_predict_fast(X_all, y, sel_idx, RANDOM_SEEDS, spw, subj_ids)
            
            cal_mm = cal_mean_match(oof, train_rate[target])
            cal_logit = cal_logit_shift(oof, train_rate[target])
            
            log.info(f"  {target} {n_feat}feat: mean_match={log_loss(y, cal_mm, labels=[0,1]):.4f}, "
                    f"logit_shift={log_loss(y, cal_logit, labels=[0,1]):.4f}")
        
        # V10 baseline
        v10_meta = json.load(open('submissions/meta_v10_20260501_170715.json'))
        v10_mm_loss = v10_meta['per_target'][target]['cal_oof_loss']
        
        results[target] = {
            'best_config': best_cfg,
            'n_features': len(best_selected_indices),
            'cal_method': best_cal_method,
            'cal_oof_loss': float(best_loss),
            'v10_mm_loss': float(v10_mm_loss),
            'improvement': float(v10_mm_loss - best_loss),
        }
        
        log.info(f"  ✅ Best: {best_cal_method} ({best_loss:.4f}, {len(best_selected_indices)} feats)")
        log.info(f"  V10 MM: {v10_mm_loss:.4f}")
        log.info(f"  Improvement: {v10_mm_loss - best_loss:+.4f}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V13 SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Method':<15} {'N_Feat':<8} {'V10-MM':<10} {'V13-Cal':<10} {'Improvement':<12}")
    for t in TARGET_COLS:
        r = results[t]
        log.info(f"{t:<6} {r['cal_method']:<15} {r['n_features']:<8} {r['v10_mm_loss']:<10.4f} {r['cal_oof_loss']:<10.4f} {r['improvement']:+.4f}")
    
    avg_v13 = np.mean([r['cal_oof_loss'] for r in results.values()])
    avg_v10 = 0.603807
    log.info(f"\nV10 avg cal OOF: {avg_v10:.6f}")
    log.info(f"V13 avg cal OOF: {avg_v13:.6f}")
    log.info(f"Improvement: {avg_v10 - avg_v13:+.6f}")
    log.info(f"Improved: {avg_v13 < avg_v10}")
    
    meta = {
        'version': 'v13_calibration',
        'timestamp': pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        'avg_cal_oof': round(avg_v13, 6),
        'v10_avg_cal_oof': 0.603807,
        'improved': avg_v13 < avg_v10,
        'n_seeds_used': N_SEEDS,
        'results': {t: {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for t, r in results.items()},
    }
    meta_path = f'submissions/meta_v13_{meta["timestamp"]}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Meta saved: {meta_path}")

if __name__ == "__main__":
    main()
