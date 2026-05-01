"""
V14 - Feature Subset Optimization (with personalization)
Strategy: Compare feature selection methods:
1. LGBM importance ranking
2. Mutual information ranking  
3. LASSO selection

Uses personalization (z-score) efficiently via float32 numpy arrays.
Only 3 feature counts (10, 20, 30) × 3 configs × 1 method = 9 combos per target.
"""
import sys, re, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
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
    """Compute z-score personalization, return feature matrix as float32 numpy array."""
    df = df.copy()
    X_base = df[feat_cols].fillna(0).values.astype(np.float32)
    subj_means = df.groupby('subject_id')[feat_cols].mean().values.astype(np.float32)
    subj_stds = df.groupby('subject_id')[feat_cols].std().fillna(1).values.astype(np.float32)
    subj_stds[subj_stds < 1e-10] = 1.0
    subj_map = {sid: i for i, sid in enumerate(df['subject_id'].unique())}
    subj_indices = np.array([subj_map[sid] for sid in df['subject_id']])
    X_zscore = np.zeros_like(X_base)
    for j, col in enumerate(feat_cols):
        means = subj_means[:, j]
        stds = subj_stds[:, j]
        for i, si in enumerate(subj_indices):
            X_zscore[i, j] = (X_base[i, j] - means[si]) / stds[si]
    X_all = np.hstack([X_base, X_zscore]).astype(np.float32)
    return X_all, df['subject_id'].values, df['lifelog_date'].values, df['sleep_date'].values

def rank_features_lgb(X, y, feat_cols, seed=42):
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
    return oof_avg

def cal_mean_match(pred, target_rate):
    return np.clip(pred + (target_rate - pred.mean()), 0.0001, 0.9999)

def main():
    log.info("=" * 70)
    log.info("V14: Feature Subset Optimization (with personalization)")
    log.info("=" * 70)
    
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"Features: {feat.shape}")
    
    feat_cols = get_feat_cols(feat)
    log.info(f"Feature cols: {len(feat_cols)}")
    
    # Personalization
    log.info("Computing z-score personalization...")
    start = time.time()
    X_all, subj_ids, _, _ = add_personalization_fast(feat, feat_cols)
    log.info(f"Personalization done: X shape={X_all.shape}, time={time.time()-start:.1f}s")
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    
    results = {}
    
    for target in TARGET_COLS:
        log.info(f"\n{'='*40}")
        log.info(f"Feature selection for {target}...")
        
        y = feat[target].values
        leak_cols = remove_leak(feat_cols, target)
        
        col_to_idx = {c: i for i, c in enumerate(feat_cols)}
        
        # Method 1: LGBM importance (on base + zscore cols)
        log.info(f"  [1/3] LGBM importance...")
        ranked_lgb = rank_features_lgb(X_all, y, leak_cols, seed=42)
        
        best_lgbm_loss = float('inf')
        best_lgbm_cols = None
        best_lgbm_oof = None
        
        for n_feat in [10, 20, 30]:
            if n_feat > len(ranked_lgb):
                continue
            selected_cols = [r[0] for r in ranked_lgb[:n_feat]]
            selected_indices = [col_to_idx[c] for c in selected_cols]
            
            for cfg in CONFIGS:
                n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
                spw = n_neg / n_pos
                
                oof = lgb_cv_predict_fast(X_all, y, selected_indices, RANDOM_SEEDS, spw, subj_ids)
                cal = cal_mean_match(oof, train_rate[target])
                loss = log_loss(y, cal, labels=[0,1])
                
                if loss < best_lgbm_loss:
                    best_lgbm_loss = loss
                    best_lgbm_cols = selected_cols
                    best_lgbm_oof = oof
        
        # Method 2: MI ranking
        log.info(f"  [2/3] Mutual information...")
        X_base_only = X_all[:, :len(feat_cols)]
        mi_scores = mutual_info_classif(X_base_only[:, [feat_cols.index(c) for c in leak_cols]], y, random_state=42, n_jobs=-1)
        mi_ranked = sorted(zip(leak_cols, mi_scores), key=lambda x: -x[1])
        
        best_mi_loss = float('inf')
        best_mi_cols = None
        
        for n_feat in [10, 20, 30]:
            if n_feat > len(mi_ranked):
                continue
            selected_cols = [r[0] for r in mi_ranked[:n_feat]]
            selected_indices = [col_to_idx[c] for c in selected_cols]
            
            for cfg in CONFIGS:
                n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
                spw = n_neg / n_pos
                
                oof = lgb_cv_predict_fast(X_all, y, selected_indices, RANDOM_SEEDS, spw, subj_ids)
                cal = cal_mean_match(oof, train_rate[target])
                loss = log_loss(y, cal, labels=[0,1])
                
                if loss < best_mi_loss:
                    best_mi_loss = loss
                    best_mi_cols = selected_cols
        
        # Method 3: LASSO
        log.info(f"  [3/3] LASSO...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_base_only[:, [feat_cols.index(c) for c in leak_cols]])
        
        best_lasso_loss = float('inf')
        best_lasso_cols = None
        
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5]:
            lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
            lasso.fit(X_scaled, y)
            active_cols = [leak_cols[i] for i in range(len(leak_cols)) if lasso.coef_[i] != 0]
            
            if len(active_cols) == 0:
                continue
            
            for n_top in [min(10, len(active_cols)), min(20, len(active_cols))]:
                selected_cols = active_cols[:n_top]
                selected_indices = [col_to_idx[c] for c in selected_cols]
                n_pos = max((y==1).sum(),1); n_neg = (y==0).sum()
                spw = n_neg / n_pos
                
                oof = lgb_cv_predict_fast(X_all, y, selected_indices, RANDOM_SEEDS, spw, subj_ids)
                cal = cal_mean_match(oof, train_rate[target])
                loss = log_loss(y, cal, labels=[0,1])
                
                if loss < best_lasso_loss:
                    best_lasso_loss = loss
                    best_lasso_cols = selected_cols
        
        # Compare
        methods = {
            'LGBM_importance': (best_lgbm_loss, best_lgbm_cols),
            'MI_ranking': (best_mi_loss, best_mi_cols),
            'LASSO': (best_lasso_loss, best_lasso_cols),
        }
        best_method = min(methods, key=lambda k: methods[k][0])
        best_loss, best_cols = methods[best_method]
        
        v10_meta = json.load(open('submissions/meta_v10_20260501_170715.json'))
        v10_mm_loss = v10_meta['per_target'][target]['cal_oof_loss']
        v10_n_feats = v10_meta['per_target'][target]['n_features']
        
        results[target] = {
            'best_method': best_method,
            'n_features': len(best_cols) if best_cols else 0,
            'selected_features': best_cols[:10] if best_cols else [],
            'cal_oof_loss': float(best_loss),
            'v10_mm_loss': float(v10_mm_loss),
            'v10_n_feats': v10_n_feats,
            'improvement': float(v10_mm_loss - best_loss),
            'lgbm_loss': float(best_lgbm_loss),
            'mi_loss': float(best_mi_loss),
            'lasso_loss': float(best_lasso_loss),
        }
        
        log.info(f"  LGBM: {best_lgbm_loss:.4f} ({len(best_lgbm_cols) if best_lgbm_cols else 0} feats)")
        log.info(f"  MI: {best_mi_loss:.4f} ({len(best_mi_cols) if best_mi_cols else 0} feats)")
        log.info(f"  LASSO: {best_lasso_loss:.4f} ({len(best_lasso_cols) if best_lasso_cols else 0} feats)")
        log.info(f"  ✅ Best: {best_method} ({best_loss:.4f})")
        log.info(f"  V10 MM: {v10_mm_loss:.4f}")
        log.info(f"  Improvement: {v10_mm_loss - best_loss:+.4f}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V14 SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Method':<20} {'N_Feat':<8} {'V10-MM':<10} {'V14-Cal':<10} {'Improvement':<12}")
    for t in TARGET_COLS:
        r = results[t]
        log.info(f"{t:<6} {r['best_method']:<20} {r['n_features']:<8} {r['v10_mm_loss']:<10.4f} {r['cal_oof_loss']:<10.4f} {r['improvement']:+.4f}")
    
    avg_v14 = np.mean([r['cal_oof_loss'] for r in results.values()])
    avg_v10 = 0.603807
    log.info(f"\nV10 avg cal OOF: {avg_v10:.6f}")
    log.info(f"V14 avg cal OOF: {avg_v14:.6f}")
    log.info(f"Improvement: {avg_v10 - avg_v14:+.6f}")
    log.info(f"Improved: {avg_v14 < avg_v10}")
    
    meta = {
        'version': 'v14_feature_selection',
        'timestamp': pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        'avg_cal_oof': round(avg_v14, 6),
        'v10_avg_cal_oof': 0.603807,
        'improved': avg_v14 < avg_v10,
        'n_seeds_used': N_SEEDS,
        'results': {t: {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for t, r in results.items()},
    }
    meta_path = f'submissions/meta_v14_{meta["timestamp"]}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Meta saved: {meta_path}")

if __name__ == "__main__":
    main()
