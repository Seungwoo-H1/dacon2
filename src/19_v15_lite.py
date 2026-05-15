"""
V15-lite: Quick test — beat V10 (0.6038)

Same as V15 but n_estimators halved for speed.
5 configs × 2 feature groups × 7 targets × 10 seeds × 5 folds
= 7,000 model trains — should finish in ~30-60 min
"""
import sys, json, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')

def pr(msg):
    print(msg, flush=True)

sys.path.insert(0, 'src')
from config import TARGETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    pr("=" * 70)
    pr("V15-lite: Quick grid search")
    pr("=" * 70)

    meta_files = sorted(Path("submissions").glob("meta_v10_*.json"))
    meta_v10 = json.load(open(meta_files[-1]))
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    pr(f"V10 avg cal OOF: {v10_avg:.6f}")

    feat = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(feat)
    pr(f"Base features: {len(feat_cols)}")

    feat_filled = feat[feat_cols].fillna(0)
    variances = feat_filled.var()
    low_var = set(variances[variances < 1e-6].index.tolist())
    high_var_cols = [c for c in feat_cols if c not in low_var]
    pr(f"Removed {len(low_var)} near-zero-variance, {len(high_var_cols)} remaining")

    # Halved n_estimators for speed
    configs = [
        {'name':'C1_v10','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':250,
         'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
         'min_child_samples':10,'force_row_wise':True,'n_jobs':-1},
        {'name':'C2_light','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':8,'max_depth':3,'learning_rate':0.02,'n_estimators':150,
         'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':2.0,'reg_lambda':5.0,
         'min_child_samples':15,'force_row_wise':True,'n_jobs':-1},
        {'name':'C3_tiny','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':6,'max_depth':2,'learning_rate':0.015,'n_estimators':100,
         'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':3.0,'reg_lambda':8.0,
         'min_child_samples':20,'force_row_wise':True,'n_jobs':-1},
        {'name':'C4_medium','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':12,'max_depth':4,'learning_rate':0.025,'n_estimators':200,
         'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
         'min_child_samples':10,'force_row_wise':True,'n_jobs':-1},
        {'name':'C5_deep','objective':'binary','metric':'binary_logloss','verbose':-1,
         'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':250,
         'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':0.5,'reg_lambda':2.0,
         'min_child_samples':8,'force_row_wise':True,'n_jobs':-1},
    ]

    feature_groups = [
        ('all', feat_cols),
        ('high_var', high_var_cols),
    ]

    all_results = []
    total_combos = len(configs) * len(feature_groups)
    combo_count = 0

    for ci, cfg in enumerate(configs):
        for fg_name, fg_cols in feature_groups:
            combo_count += 1
            combo_key = f"{cfg['name']}+{fg_name}"
            pr(f"\n[{combo_count}/{total_combos}] {combo_key} ({len(fg_cols)} feats)")
            
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

                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                cal_loss = log_loss(y, cal, labels=[0,1])
                
                all_results.append({
                    'target': target, 'cfg': cfg['name'],
                    'fg': fg_name, 'n_feats': len(leak_cols),
                    'cal_loss': cal_loss
                })

                elapsed = time.time() - t_total
                pr(f"  [{tidx+1}/7] {target}: cal={cal_loss:.4f} (V10={v10_cal_oof[target]:.4f}) [{elapsed:.0f}s]")

    # ── Results ──────────────────────────────────────────────
    pr(f"\n{'='*70}")
    pr("V15-lite RESULTS — Per-target best config")
    pr(f"{'='*70}")
    
    for target in TARGET_COLS:
        tgt_res = [r for r in all_results if r['target'] == target]
        best_r = min(tgt_res, key=lambda x: x['cal_loss'])
        v10c = v10_cal_oof[target]
        diff = best_r['cal_loss'] - v10c
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        pr(f"{target}: V10={v10c:.4f} → V15={best_r['cal_loss']:.4f} Δ={diff:+.4f} [{best_r['cfg']}+{best_r['fg']}] {marker}")

    best_per_target_avg = np.mean([
        min(r['cal_loss'] for r in all_results if r['target'] == t)
        for t in TARGET_COLS
    ])

    pr(f"\n{'='*70}")
    pr("Best single config (same for all targets)")
    for cfg in configs:
        cfg_name = cfg['name']
        cfg_losses = [r['cal_loss'] for r in all_results if r['cfg'] == cfg_name]
        cfg_avg = np.mean(cfg_losses) / len(feature_groups)
        pr(f"  {cfg_name}: avg={cfg_avg:.6f}")

    pr(f"\n{'='*70}")
    pr(f"V15-lite best-per-target avg: {best_per_target_avg:.6f} (V10: {v10_avg:.6f}, Δ={best_per_target_avg-v10_avg:+.6f})")
    beat = best_per_target_avg < v10_avg
    pr(f"{'🎯 BEATS V10!' if beat else 'Not yet.'}")
    
    # Save
    result_file = SUBMIT_DIR / f"v15lite_results_{int(time.time())}.json"
    result_file.parent.mkdir(exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({"version":"v15-lite","v10_avg":v10_avg,
            "best_per_target_avg":best_per_target_avg,"beat_v10":beat,"results":all_results}, f, indent=2)
    pr(f"Saved: {result_file}")
    
    total_time = time.time() - t_total
    pr(f"Total: {total_time:.0f}s ({total_time/60:.1f}min)")

if __name__ == "__main__":
    main()
