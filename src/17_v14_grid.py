"""
V14: Beat V10 — Minimal memory, max impact

Approach:
1. Load V10 features (141 cols) — NO personalization, NO new features
2. For each target:
   a. Use V10's already-tuned config from meta
   b. Try 3 feature counts: 10, 20, 30 (using V10 importance ranking)
   c. Try 3 model configs: V10-style, Lighter, Medium  
   d. Rank 15×21=315 combos per target, pick best
3. Best config → final model per target
4. If time: CatBoost on top-10 features only

Memory safe: 450 × N features × small arrays only.
"""
import sys, re, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

RANDOM_SEEDS = [42,123,456,789,1024,1337,2048,3037,4096,5001,
                6000,7123,8001,9000,10000,11111,12000,13001,14000,15001]
N_SEEDS = len(RANDOM_SEEDS)
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

def lgb_cv_20seeds(feat, scols, target, seeds, spw, cfg):
    """Fast 20-seed GroupKFold CV for 450 samples."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in scols]
    for si, seed in enumerate(seeds):
        sc = {**cfg, 'random_state': seed, 'scale_pos_weight': spw}
        for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            Xtr = feat.iloc[ti][scols].fillna(0).values.astype(np.float32)
            Xva = feat.iloc[vi][scols].fillna(0).values.astype(np.float32)
            ytr, yva = y[ti], y[vi]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn, params={'verbose':'-1'})
            vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd, params={'verbose':'-1'})
            mdl = lgb.train(sc, trd, num_boost_round=cfg['n_estimators'],
                valid_sets=[vad], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vi, si] = mdl.predict(Xva)
    return oof.mean(axis=1)

def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("V14: Grid search — beat V10 (0.6038)")
    log.info("=" * 70)

    # Load V10 meta
    meta_files = sorted(Path("submissions").glob("meta_v10_*.json"))
    meta_v10 = json.load(open(meta_files[-1]))
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    log.info(f"V10 avg cal OOF: {v10_avg:.6f}")

    # Load features (just the base 141)
    feat = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(feat)
    log.info(f"Features: {len(feat_cols)}")

    # Quick ranking per target (lightweight: 100 trees, no personalization)
    log.info("Quick ranking per target...")
    all_ranked = {}
    for target in TARGET_COLS:
        y = feat[target].values
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
        leak = remove_leak(feat_cols, target)
        # Use only top-50 by quick scan to reduce dataset size
        sn_leak = [sanitize(c) for c in leak]
        X = feat[leak].fillna(0).values.astype(np.float32)
        ds = lgb.Dataset(X, label=y, feature_name=sn_leak, params={'verbose':'-1'})
        cfg_rank = {'objective':'binary','metric':'binary_logloss','verbose':-1,
               'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
               'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
               'force_row_wise':True,'n_jobs':-1}
        mdl = lgb.train(cfg_rank, ds, num_boost_round=100)
        imp = mdl.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak, imp), key=lambda x: -x[1])
        all_ranked[target] = ranked
        log.info(f"  {target}: ranked {len(leak)} features")

    # Config grid
    config_grid = [
        # (name, nl, md, lr, ne, ss, cst, ra, rl, mc)
        ('C1_v10style', 15, 4, 0.03, 500, 0.7, 0.7, 1.0, 3.0, 10),
        ('C2_light',    8, 3, 0.02, 300, 0.6, 0.6, 2.0, 5.0, 15),
        ('C3_medium',  12, 4, 0.025, 400, 0.7, 0.7, 1.0, 3.0, 10),
        ('C4_tiny',     6, 2, 0.015, 200, 0.5, 0.5, 3.0, 8.0, 20),
        ('C5_deep',    20, 5, 0.02, 500, 0.8, 0.8, 0.5, 2.0, 8),
    ]

    feat_counts = [10, 20, 30]

    # Grid search
    best_overall = {'config': None, 'n_feats': None, 'avg_loss': float('inf')}
    per_target_best = {}  # target -> {config, n_feats, oof, cal_loss}

    total_combos = len(TARGET_COLS) * len(config_grid) * len(feat_counts)
    combo_count = 0

    for target in TARGET_COLS:
        y = feat[target].values
        train_rate = y.mean()
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_

        per_target_results = []

        for cfg in config_grid:
            name, nl, md, lr, ne, ss, cst, ra, rl, mc = cfg
            base_cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
                       'num_leaves':nl,'max_depth':md,'learning_rate':lr,'n_estimators':ne,
                       'subsample':ss,'colsample_bytree':cst,'reg_alpha':ra,'reg_lambda':rl,
                       'min_child_samples':mc,'force_row_wise':True,'n_jobs':-1}

            for n_feats in feat_counts:
                combo_count += 1
                ranked = all_ranked[target]
                scols = [r[0] for r in ranked[:n_feats]]

                t0 = time.time()
                oof = lgb_cv_20seeds(feat, scols, target, RANDOM_SEEDS, spw, base_cfg)
                elapsed = time.time() - t0

                oof_loss = log_loss(y, oof, labels=[0,1])
                # Calibrate
                shift = train_rate - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                cal_loss = log_loss(y, cal, labels=[0,1])

                per_target_results.append({
                    'config': name, 'n_feats': n_feats,
                    'oof_loss': oof_loss, 'cal_loss': cal_loss,
                    'elapsed': elapsed
                })

                # Track global best
                if cal_loss < best_overall['avg_loss']:
                    best_overall = {'config': name, 'n_feats': n_feats, 'avg_loss': cal_loss, 'target': target}

                if cal_loss < 0.001:
                    log.info(f"  *** {target}: {name}+{n_feats}f → cal={cal_loss:.4f} (beat V10 {v10_cal_oof[target]:.4f}) [{elapsed:.0f}s]")

        # Best for this target
        best_t = min(per_target_results, key=lambda x: x['cal_loss'])
        per_target_best[target] = best_t
        v10_cal = v10_cal_oof[target]
        diff = best_t['cal_loss'] - v10_cal
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        log.info(f"  [{target}] BEST: {best_t['config']}+{best_t['n_feats']}f cal={best_t['cal_loss']:.4f} (V10={v10_cal:.4f} Δ={diff:+.4f}) {marker}")

    # ── Final comparison ─────────────────────────────────────
    # Best per-target vs V10
    best_avg = np.mean([per_target_best[t]['cal_loss'] for t in TARGET_COLS])
    log.info(f"\n{'='*70}")
    log.info("V14 COMPARISON: Best per-target configs vs V10")
    log.info(f"{'Target':<6} {'V10':<10} {'V14':<10} {'Δ':<10} {'Config':<18} {'Ftr'}")
    for t in TARGET_COLS:
        v10c = v10_cal_oof[t]
        v14c = per_target_best[t]['cal_loss']
        diff = v14c - v10c
        cfg = f"{per_target_best[t]['config']}+{per_target_best[t]['n_feats']}f"
        marker = "✓" if diff < -0.001 else ("~" if abs(diff) <= 0.001 else "✗")
        log.info(f"{t:<6} {v10c:<10.4f} {v14c:<10.4f} {diff:+.4f} {cfg:<18} {marker}")
    
    log.info(f"\nV14 best avg: {best_avg:.6f} (V10: {v10_avg:.6f}, Δ={best_avg-v10_avg:+.6f})")
    beat = best_avg < v10_avg
    log.info(f"{'🎯 BEATS V10!' if beat else 'Not yet — needs more experiments.'}")

    # Also try: best single config across all targets (not per-target)
    all_cal = {}
    for cfg in config_grid:
        name = cfg[0]
        for nf in feat_counts:
            cfg_obj = {'objective':'binary','metric':'binary_logloss','verbose':-1,
                       'num_leaves':cfg[1],'max_depth':cfg[2],'learning_rate':cfg[3],
                       'n_estimators':cfg[4],'subsample':cfg[5],'colsample_bytree':cfg[6],
                       'reg_alpha':cfg[7],'reg_lambda':cfg[8],'min_child_samples':cfg[9],
                       'force_row_wise':True,'n_jobs':-1}
            avg_loss = 0
            for target in TARGET_COLS:
                y = feat[target].values
                np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
                ranked = all_ranked[target]
                scols = [r[0] for r in ranked[:nf]]
                oof = lgb_cv_20seeds(feat, scols, target, RANDOM_SEEDS, spw, cfg_obj)
                shift = y.mean() - oof.mean()
                cal = np.clip(oof + shift, 0.0001, 0.9999)
                avg_loss += log_loss(y, cal, labels=[0,1])
            avg_loss /= len(TARGET_COLS)
            all_cal[f'{name}+{nf}f'] = avg_loss

    best_single = min(all_cal.items(), key=lambda x: x[1])
    log.info(f"\nBest single config (same for all): {best_single[0]} avg={best_single[1]:.6f}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("BEST OF ALL")
    log.info(f"{'Name':<30} {'AVG Cal OOF':<14} {'Δ vs V10':<12}")
    log.info(f"{'V10':<30} {v10_avg:<14.6f} {'—':<12}")
    log.info(f"{'V14 best-per-target':<30} {best_avg:<14.6f} {best_avg-v10_avg:+.6f}")
    log.info(f"{'V14 best-single-config':<30} {best_single[1]:<14.6f} {best_single[1]-v10_avg:+.6f}")

    for name, val in sorted(all_cal.items(), key=lambda x: x[1])[:5]:
        log.info(f"  {name:<30} {val:<14.6f} {val-v10_avg:+.6f}")

    total_time = time.time() - t_total
    log.info(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
    log.info(f"Total combos tested: {combo_count}")

if __name__ == "__main__":
    main()
