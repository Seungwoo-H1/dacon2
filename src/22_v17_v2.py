"""
V17: Reproduce V10 exactly, then test LIGHT/TINY configs

Step 1: Run V10's ranking (same algorithm) and save rankings to meta
Step 2: Use V10's ranking to test V10, LIGHT, TINY configs with exactly the same feature sets
Step 3: If any config beats V10 with same feature set → real improvement
"""
import sys, re, json, time, os, warnings
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

def quick_rank(feat, feature_cols, target, n_trees=100):
    """Same ranking algorithm as V10."""
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
    return ranked  # return full ranked list

# V10's best config (C4)
LGB_V10 = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
    'subsample':0.7,'colsample_bytree':0.7,
    'reg_alpha':1.0,'reg_lambda':3.0,
    'min_child_samples':10,'n_jobs':1,'verbose':-1,
}
LGB_LIGHT = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':8,'max_depth':3,'learning_rate':0.02,'n_estimators':300,
    'subsample':0.6,'colsample_bytree':0.6,
    'reg_alpha':2.0,'reg_lambda':5.0,
    'min_child_samples':15,'n_jobs':1,'verbose':-1,
}
LGB_TINY = {
    'objective':'binary','metric':'binary_logloss',
    'num_leaves':6,'max_depth':2,'learning_rate':0.015,'n_estimators':200,
    'subsample':0.5,'colsample_bytree':0.5,
    'reg_alpha':3.0,'reg_lambda':8.0,
    'min_child_samples':20,'n_jobs':1,'verbose':-1,
}

def cv_predict(scols, target, seeds, spw, cfg):
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

def main():
    global _FEAT
    t_total = time.time()
    pr("=" * 70)
    pr("V17: Reproduce V10 + test LIGHT/TINY")
    pr("=" * 70)

    # Load V10 meta
    meta_files = sorted(Path("submissions").glob("meta_v10_*.json"))
    meta_v10 = json.load(open(meta_files[-1]))
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    pr(f"V10 avg cal OOF: {v10_avg:.6f}")

    # Load features + personalization
    _FEAT = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(_FEAT)
    pr(f"Base features: {len(feat_cols)}")
    
    pr("Adding personalization...")
    t0 = time.time()
    _FEAT, zscore_cols = add_personalization(_FEAT, feat_cols)
    pr(f"Personalization: {time.time()-t0:.1f}s → {len(get_feat_cols(_FEAT))} features")

    # Step 1: Rank features per target (save to meta for reproducibility)
    pr("\n=== Step 1: Ranking features ===")
    all_ranked = {}
    for target in TARGET_COLS:
        leak = remove_leak(feat_cols, target)
        ranked = quick_rank(_FEAT, leak, target, n_trees=100)
        all_ranked[target] = ranked
        top5 = [r[0] for r in ranked[:5]]
        pr(f"  {target}: top5={top5}")
    
    # Save rankings to file
    ranking_file = PROJECT_ROOT / "v17_rankings.json"
    ranking_data = {t: [r[0] for r in ranked] for t, ranked in all_ranked.items()}
    with open(ranking_file, 'w') as f:
        json.dump(ranking_data, f)
    pr(f"Rankings saved: {ranking_file}")

    # Step 2: Test V10, LIGHT, TINY configs with ALL feature counts
    pr("\n=== Step 2: Grid search ===")
    configs = [
        ('V10', LGB_V10),
        ('LIGHT', LGB_LIGHT),
        ('TINY', LGB_TINY),
    ]
    feat_counts = [10, 20, 30, 40, 50]
    
    all_results = []
    combo = 0
    total_combos = len(configs) * len(feat_counts) * len(TARGET_COLS)
    pr(f"Total combos: {total_combos}")
    
    for cname, cfg in configs:
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
                oof = cv_predict(scols, target, SEEDS, spw, cfg)
                elapsed = time.time() - t0
                
                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                cal_loss = log_loss(y, cal, labels=[0,1])
                
                all_results.append({
                    'target': target, 'config': cname,
                    'n_feats': n_feats, 'cal_loss': cal_loss,
                    'features': scols[:3] + ['...'] + scols[-2:]
                })
                
                v10c = v10_cal_oof[target]
                diff = cal_loss - v10c
                marker = "✓" if diff < -0.001 else ("~" if abs(diff) <= 0.001 else "✗")
                pr(f"[{combo}/{total_combos}] {target} {cname}+{n_feats}f cal={cal_loss:.4f} V10={v10c:.4f} Δ={diff:+.4f} [{elapsed:.0f}s] {marker}")
    
    # ── Results ──────────────────────────────────────────────
    pr(f"\n{'='*70}")
    pr("V17 RESULTS")
    pr(f"{'='*70}")
    
    for target in TARGET_COLS:
        tgt_res = [r for r in all_results if r['target'] == target]
        best_r = min(tgt_res, key=lambda x: x['cal_loss'])
        v10c = v10_cal_oof[target]
        diff = best_r['cal_loss'] - v10c
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        pr(f"{target}: V10={v10c:.4f} → V17={best_r['cal_loss']:.4f} Δ={diff:+.4f} [{best_r['config']}+{best_r['n_feats']}f] {marker}")
    
    best_avg = np.mean([min(r['cal_loss'] for r in all_results if r['target'] == t) for t in TARGET_COLS])
    pr(f"\nV17 best-per-target avg: {best_avg:.6f} (V10: {v10_avg:.6f}, Δ={best_avg-v10_avg:+.6f})")
    beat = best_avg < v10_avg
    pr(f"{'🎯 BEATS V10!' if beat else 'Not yet.'}")
    
    # Per-config best
    pr(f"\n{'='*70}")
    pr("Per-config best")
    for cname in ['V10', 'LIGHT', 'TINY']:
        cfg_res = [r for r in all_results if r['config'] == cname]
        best_c = min(cfg_res, key=lambda x: x['cal_loss'])
        pr(f"  {cname}: avg={np.mean([r['cal_loss'] for r in cfg_res]):.6f}, best={best_c['cal_loss']:.4f} ({best_c['n_feats']}f)")
    
    # Save
    result_file = SUBMIT_DIR / f"v17_results_{int(time.time())}.json"
    with open(result_file, 'w') as f:
        json.dump({"version":"v17","v10_avg":v10_avg,"best_per_target_avg":best_avg,"beat_v10":beat,"results":all_results}, f, indent=2)
    pr(f"Saved: {result_file}")
    
    pr(f"Total time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")

if __name__ == "__main__":
    main()
