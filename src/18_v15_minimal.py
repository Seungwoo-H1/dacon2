"""
V15: Minimal footprint — beat V10 (0.6038)

Strategy:
- NO feature ranking overhead — use all leak-free features directly
- NO personalization z-score (too many columns)
- 5 configs × 2 feature groups (all, variance-filtered) = 10 combos
- 10 seeds, 5-fold GroupKFold
- Each combo: 10 seeds × 5 folds = 50 trains on 450 samples
- Total: 10 combos × 7 targets × 50 trains = 3,500 trains
- But we skip early-stopping so actual iterations are much less
"""
import sys, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
import os
os.environ['PYTHONUNBUFFERED'] = '1'
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout, force=True)
log = logging.getLogger(__name__)

def pr(msg):
    """Print with immediate flush."""
    print(msg, flush=True)
    log.info(msg)

sys.path.insert(0, 'src')
from config import TARGETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
SUBMIT_DIR = PROJECT_ROOT / "submissions"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]
N_SEEDS = len(SEEDS)
N_SPLITS = 5

_SANITIZE_RE = __import__('re').compile(r'[^a-zA-Z0-9_]')
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

def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("V15: Minimal footprint grid search")
    log.info("=" * 70)

    # Load V10 meta
    meta_files = sorted(Path("submissions").glob("meta_v10_*.json"))
    meta_v10 = json.load(open(meta_files[-1]))
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    log.info(f"V10 avg cal OOF: {v10_avg:.6f}")

    # Load features
    feat = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(feat)
    log.info(f"Base features: {len(feat_cols)}")

    # Variance filter
    feat_filled = feat[feat_cols].fillna(0)
    variances = feat_filled.var()
    low_var = set(variances[variances < 1e-6].index.tolist())
    high_var_cols = [c for c in feat_cols if c not in low_var]
    log.info(f"Removed {len(low_var)} near-zero-variance features, {len(high_var_cols)} remaining")

    # ── Config grid (5 configs) — full LGBM param names ──────
    configs = [
        {'name':'C1_v10','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
         'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
         'min_child_samples':10,'force_row_wise':True,'n_jobs':-1},
        {'name':'C2_light','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':8,'max_depth':3,'learning_rate':0.02,'n_estimators':300,
         'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':2.0,'reg_lambda':5.0,
         'min_child_samples':15,'force_row_wise':True,'n_jobs':-1},
        {'name':'C3_tiny','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':6,'max_depth':2,'learning_rate':0.015,'n_estimators':200,
         'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':3.0,'reg_lambda':8.0,
         'min_child_samples':20,'force_row_wise':True,'n_jobs':-1},
        {'name':'C4_medium','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':12,'max_depth':4,'learning_rate':0.025,'n_estimators':400,
         'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
         'min_child_samples':10,'force_row_wise':True,'n_jobs':-1},
        {'name':'C5_deep','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':500,
         'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.5,'reg_lambda':2.0,
         'min_child_samples':8,'force_row_wise':True,'n_jobs':-1},
    ]

    # Feature groups
    feature_groups = [
        ('all', feat_cols),
        ('high_var', high_var_cols),
    ]

    # ── Run grid search ──────────────────────────────────────
    all_results = []

    for ci, cfg in enumerate(configs):
        for fg_name, fg_cols in feature_groups:
            # Remove leak features per target — but we do target loop below
            # First filter to leak-free
            combo_key = f"{cfg['name']}+{fg_name}"
            pr(f"\n[{ci+1}/{len(configs)*len(feature_groups)}] {combo_key} ({len(fg_cols)} feats)")
            
            for tidx, target in enumerate(TARGET_COLS):
                leak_cols = remove_leak(fg_cols, target)
                y = feat[target].values
                np_ = max((y==1).sum(), 1)
                nn = (y==0).sum()
                spw = nn / np_

                gkf = GroupKFold(n_splits=N_SPLITS)
                oof = np.zeros(len(y))
                
                for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
                    Xtr = feat.iloc[tr_idx][leak_cols].fillna(0).values.astype(np.float32)
                    Xva = feat.iloc[va_idx][leak_cols].fillna(0).values.astype(np.float32)
                    ytr, yva = y[tr_idx], y[va_idx]
                    
                    sn = [sanitize(c) for c in leak_cols]
                    trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn)
                    
                    fold_seed_sum = np.zeros(len(va_idx))
                    for seed in SEEDS:
                        sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
                        vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd)
                        mdl = lgb.train(sc, trd, num_boost_round=cfg['n_estimators'],
                            valid_sets=[vad],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                        fold_seed_sum += mdl.predict(Xva)
                    
                    oof[va_idx] = fold_seed_sum / N_SEEDS

                # Calibrate
                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                cal_loss = log_loss(y, cal, labels=[0,1])
                
                all_results.append({
                    'target': target, 'cfg': cfg['name'],
                    'fg': fg_name, 'n_feats': len(leak_cols),
                    'cal_loss': cal_loss
                })
                if (tidx + ci * len(TARGET_COLS)) % 3 == 0:
                    pr(f"  [{tidx+1}/7] {target}: cal={cal_loss:.4f}")

    # ── Results ──────────────────────────────────────────────
    pr(f"\n{'='*70}")
    pr("V15 RESULTS — Per-target best config")
    pr(f"{'='*70}")
    
    for target in TARGET_COLS:
        tgt_res = [r for r in all_results if r['target'] == target]
        best_r = min(tgt_res, key=lambda x: x['cal_loss'])
        v10c = v10_cal_oof[target]
        diff = best_r['cal_loss'] - v10c
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        pr(f"{target}: V10={v10c:.4f} → V15={best_r['cal_loss']:.4f} Δ={diff:+.4f} [{best_r['cfg']}+{best_r['fg']}] {marker}")

    # Best per-target average
    best_per_target_avg = np.mean([
        min(r['cal_loss'] for r in all_results if r['target'] == t)
        for t in TARGET_COLS
    ])
    
    # Best single config
    pr(f"\n{'='*70}")
    pr("Best single config (same for all targets)")
    for cfg in configs:
        cfg_name = cfg['name']
        cfg_losses = [r['cal_loss'] for r in all_results if r['cfg'] == cfg_name]
        cfg_avg = np.mean(cfg_losses) / len(feature_groups)
        pr(f"  {cfg_name}: avg={cfg_avg:.6f}")

    pr(f"\n{'='*70}")
    pr(f"V15 best-per-target avg: {best_per_target_avg:.6f} (V10: {v10_avg:.6f}, Δ={best_per_target_avg-v10_avg:+.6f})")
    beat = best_per_target_avg < v10_avg
    pr(f"{'🎯 BEATS V10!' if beat else 'Not yet — need more experiments.'}")
    
    # Save
    result_file = SUBMIT_DIR / f"v15_results_{int(time.time())}.json"
    result_file.parent.mkdir(exist_ok=True)
    meta_out = {"version":"v15","v10_avg":v10_avg,
        "best_per_target_avg":best_per_target_avg,
        "beat_v10":beat,
        "results":{r['target']: [] for r in all_results}}
    for r in all_results:
        meta_out["results"][r['target']].append({
            'cfg': r['cfg'], 'fg': r['fg'], 'n_feats': r['n_feats'],
            'cal_loss': round(r['cal_loss'], 6)
        })
    with open(result_file, 'w') as f:
        json.dump(meta_out, f, indent=2)
    log.info(f"Results saved: {result_file}")
    
    total_time = time.time() - t_total
    log.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

if __name__ == "__main__":
    main()
