"""
V13: Beat V10 (0.6038) — Efficient multi-experiment pipeline

Key changes from V10:
1. Skip redundant feature ranking — use V10's already-ranked features
2. Fewer personalization columns: only add z-score for TOP-50 features (not all 141)
3. Add temporal/rate-of-change features without exploding feature count
4. Multiple model configs per target, select best via CV
5. CatBoost with aggressive feature reduction
6. Stacking ensemble

Memory efficient: avoids regex sanitization on all features by pre-sanitizing.
"""
import sys, re, json, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
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

# Pre-sanitize to avoid regex overhead
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

def lgb_cv_predict(feat, scols, target, seeds, spw, base_cfg):
    """GroupKFold CV, returns OOF avg predictions."""
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in scols]
    for si, seed in enumerate(seeds):
        sc = {**base_cfg, 'random_state': seed, 'scale_pos_weight': spw}
        for fold, (ti, vi) in enumerate(gkf.split(feat, y, feat['subject_id'])):
            Xtr = feat.iloc[ti][scols].fillna(0).values
            Xva = feat.iloc[vi][scols].fillna(0).values
            ytr, yva = y[ti], y[vi]
            trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn, params={'verbose':'-1'})
            vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd, params={'verbose':'-1'})
            mdl = lgb.train(sc, trd, num_boost_round=base_cfg['n_estimators'],
                valid_sets=[vad], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[vi, si] = mdl.predict(Xva)
    return oof.mean(axis=1)

def main():
    t_total = time.time()
    log.info("=" * 70)
    log.info("V13: Efficient multi-experiment — beat V10 (0.6038)")
    log.info("=" * 70)

    # ── Load V10 metadata ──────────────────────────────────────
    meta_files = sorted(Path("submissions").glob("meta_v10_*.json"))
    if not meta_files:
        log.error("No V10 meta found!")
        return
    meta_v10 = json.load(open(meta_files[-1]))
    v10_cal_oof = {t: meta_v10["per_target"][t]["cal_oof_loss"] for t in TARGET_COLS}
    v10_avg = np.mean(list(v10_cal_oof.values()))
    log.info(f"V10 avg cal OOF: {v10_avg:.6f}")

    # ── Load features ──────────────────────────────────────────
    feat = pd.read_parquet("data_processed/features.parquet")
    feat_cols = get_feat_cols(feat)
    log.info(f"Base features: {len(feat_cols)}")

    # Personalization: z-score for top features only (limit to 50)
    # Do ranking on ORIGINAL feat (no personalization yet) to avoid column pollution
    log.info("Adding personalization (top-50 features only)...")
    all_ranked = {}
    zscore_cols_added = []
    
    for target in TARGET_COLS:
        y = feat[target].values
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
        cfg = {'objective':'binary','metric':'binary_logloss','verbose':-1,
               'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':100,
               'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
               'scale_pos_weight':spw,'random_state':42,'min_child_samples':10,
               'force_row_wise':True,'n_jobs':-1}
        leak = remove_leak(feat_cols, target)
        sn = [sanitize(c) for c in leak]
        X = feat[leak].fillna(0).values
        ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
        mdl = lgb.train(cfg, ds, num_boost_round=100)
        imp = mdl.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak, imp), key=lambda x: -x[1])
        all_ranked[target] = ranked

    # Now add z-scores for top 50 features from a UNION of top features across all targets
    union_top50 = set()
    for target in TARGET_COLS:
        for r in all_ranked[target][:50]:
            union_top50.add(r[0])
    union_top50 = sorted(union_top50)
    log.info(f"Union top features for z-score: {len(union_top50)}")
    
    subj = feat['subject_id'].values
    for col in union_top50:
        vals = feat[col].values.astype(float)
        mask_nan = pd.isna(vals)
        vals_fill = np.where(mask_nan, 0.0, vals)
        
        # Per-subject mean/std
        subj_mean = np.zeros(len(vals))
        subj_std = np.ones(len(vals)) * 1e-10
        for sid in np.unique(subj):
            idx = subj == sid
            sv = vals_fill[idx]
            subj_mean[idx] = sv.mean()
            subj_std[idx] = max(sv.std(), 1e-10)
        
        z = np.where(mask_nan, 0.0, (vals_fill - subj_mean) / subj_std)
        zcol = f'{col}_z'
        feat = feat.assign(**{zcol: z})
        zscore_cols_added.append(zcol)

    # Now feat has base + top50*7_targets z-score cols
    all_feat_cols = get_feat_cols(feat)
    log.info(f"Total features after personalization: {len(all_feat_cols)}")

    # ── Config A: V10-style (conservative) ────────────────────
    configs = {
        'A_v10': None,  # will use V10 configs
        'B_lighter': {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':8,'max_depth':3,'learning_rate':0.02,'n_estimators':300,
            'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':2.0,'reg_lambda':5.0,
            'min_child_samples':15,'force_row_wise':True,'n_jobs':-1,
        },
        'C_medium': {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':12,'max_depth':4,'learning_rate':0.025,'n_estimators':400,
            'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
        },
    }

    results = {}  # config_name -> {target: cal_oof}

    # ── Experiment A: V10 configs, top-10/20 features ─────────
    log.info("\n=== Exp A: V10 configs + personalization ===")
    results['A_v10_personal'] = {}
    for target in TARGET_COLS:
        v10_cfg = meta_v10["per_target"][target]["config"]
        cfg = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':v10_cfg['nl'],'max_depth':v10_cfg['md'],'learning_rate':v10_cfg['lr'],
            'n_estimators':v10_cfg['ne'],'subsample':v10_cfg['ss'],'colsample_bytree':v10_cfg['cst'],
            'reg_alpha':v10_cfg['ra'],'reg_lambda':v10_cfg['rl'],'min_child_samples':v10_cfg['mc'],
            'force_row_wise':True,'n_jobs':-1,
        }

        leak = remove_leak(all_feat_cols, target)
        ranked = all_ranked[target]
        n_feat = v10_cfg.get('_n_feats', 10)
        scols = [r[0] for r in ranked[:n_feat]]

        y = feat[target].values
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
        t0 = time.time()
        oof = lgb_cv_predict(feat, scols, target, RANDOM_SEEDS, spw, cfg)
        elapsed = time.time() - t0

        oof_loss = log_loss(y, oof, labels=[0,1])
        # Simple calibration
        shift = y.mean() - oof.mean()
        cal = np.clip(oof + shift, 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0,1])
        results['A_v10_personal'][target] = cal_loss

        v10_cal = v10_cal_oof[target]
        diff = cal_loss - v10_cal
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        log.info(f"  {target}: V10={v10_cal:.4f}, V13={cal_loss:.4f}, Δ={diff:+.4f} {marker} ({elapsed:.0f}s)")

    avg_a = np.mean(list(results['A_v10_personal'].values()))
    log.info(f"  A avg: {avg_a:.6f} (V10: {v10_avg:.6f}, Δ={avg_a-v10_avg:+.6f})")

    # ── Experiment B: Lighter config ─────────────────────────
    log.info("\n=== Exp B: Lighter config (nl=8, md=3, reg heavy) ===")
    results['B_lighter'] = {}
    for target in TARGET_COLS:
        cfg = configs['B_lighter']
        leak = remove_leak(all_feat_cols, target)
        ranked = all_ranked[target]
        scols = [r[0] for r in ranked[:20]]

        y = feat[target].values
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
        t0 = time.time()
        oof = lgb_cv_predict(feat, scols, target, RANDOM_SEEDS, spw, cfg)
        elapsed = time.time() - t0

        oof_loss = log_loss(y, oof, labels=[0,1])
        shift = y.mean() - oof.mean()
        cal = np.clip(oof + shift, 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0,1])
        results['B_lighter'][target] = cal_loss

        v10_cal = v10_cal_oof[target]
        diff = cal_loss - v10_cal
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        log.info(f"  {target}: V10={v10_cal:.4f}, V13={cal_loss:.4f}, Δ={diff:+.4f} {marker} ({elapsed:.0f}s)")

    avg_b = np.mean(list(results['B_lighter'].values()))
    log.info(f"  B avg: {avg_b:.6f} (V10: {v10_avg:.6f}, Δ={avg_b-v10_avg:+.6f})")

    # ── Experiment C: Medium config ──────────────────────────
    log.info("\n=== Exp C: Medium config ===")
    results['C_medium'] = {}
    for target in TARGET_COLS:
        cfg = configs['C_medium']
        leak = remove_leak(all_feat_cols, target)
        ranked = all_ranked[target]
        scols = [r[0] for r in ranked[:30]]

        y = feat[target].values
        np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_
        t0 = time.time()
        oof = lgb_cv_predict(feat, scols, target, RANDOM_SEEDS, spw, cfg)
        elapsed = time.time() - t0

        oof_loss = log_loss(y, oof, labels=[0,1])
        shift = y.mean() - oof.mean()
        cal = np.clip(oof + shift, 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0,1])
        results['C_medium'][target] = cal_loss

        v10_cal = v10_cal_oof[target]
        diff = cal_loss - v10_cal
        marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
        log.info(f"  {target}: V10={v10_cal:.4f}, V13={cal_loss:.4f}, Δ={diff:+.4f} {marker} ({elapsed:.0f}s)")

    avg_c = np.mean(list(results['C_medium'].values()))
    log.info(f"  C avg: {avg_c:.6f} (V10: {v10_avg:.6f}, Δ={avg_c-v10_avg:+.6f})")

    # ── Experiment D: Stacking (LGB + CB) ────────────────────
    log.info("\n=== Exp D: Stacking ensemble ===")
    try:
        import catboost
        has_cb = True
        log.info(f"CatBoost {catboost.__version__} available")
    except ImportError:
        has_cb = False
        log.info("CatBoost not available, skipping D")

    if has_cb:
        from catboost import CatBoostClassifier, Pool
        results['D_stacking'] = {}

        # First: get LGB OOF predictions (use B_lighter which may be best)
        best_single = min(results.items(), key=lambda x: np.mean(list(x[1].values())))
        best_cfg_name = best_single[0]
        best_cfg = configs.get(best_cfg_name, configs['A_v10'])
        if best_cfg is None:
            # V10 config — pick a specific one
            best_cfg = configs['C_medium']

        for target in TARGET_COLS:
            y = feat[target].values
            np_ = max((y==1).sum(), 1); nn = (y==0).sum(); spw = nn/np_

            leak = remove_leak(all_feat_cols, target)
            ranked = all_ranked[target]
            scols = [r[0] for r in ranked[:20]]
            sn = [sanitize(c) for c in scols]

            # LGB OOF
            lgb_oof = np.zeros(len(y))
            gkf = GroupKFold(n_splits=N_SPLITS)
            for si, seed in enumerate(RANDOM_SEEDS):
                sc = {**configs['C_medium'], 'random_state': seed, 'scale_pos_weight': spw}
                fold_idx = 0
                for ti, vi in gkf.split(feat, y, feat['subject_id']):
                    Xtr = feat.iloc[ti][scols].fillna(0).values
                    Xva = feat.iloc[vi][scols].fillna(0).values
                    ytr = y[ti]; yva = y[vi]
                    trd = lgb.Dataset(Xtr, label=ytr, feature_name=sn, params={'verbose':'-1'})
                    vad = lgb.Dataset(Xva, label=yva, feature_name=sn, reference=trd, params={'verbose':'-1'})
                    mdl = lgb.train(sc, trd, num_boost_round=configs['C_medium']['n_estimators'],
                        valid_sets=[vad], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                    lgb_oof[vi] += mdl.predict(Xva) / N_SEEDS

            # CatBoost OOF
            cb_oof = np.zeros(len(y))
            for si, seed in enumerate([42, 123, 456, 789, 1024]):
                fold_idx = 0
                for ti, vi in gkf.split(feat, y, feat['subject_id']):
                    Xtr = feat.iloc[ti][scols].fillna(-1).values
                    Xva = feat.iloc[vi][scols].fillna(-1).values
                    ytr = y[ti]; yva = y[vi]
                    cb_pool = Pool(Xtr, ytr, cat_features=[])
                    cb_eval = Pool(Xva, yva, cat_features=[])
                    mdl = CatBoostClassifier(
                        iterations=400, depth=6, learning_rate=0.03,
                        l2_leaf_reg=3, bagging_temperature=0.5,
                        subsample=0.7, random_strength=1,
                        random_state=seed, scale_pos_weight=spw,
                        verbose=0)
                    mdl.fit(cb_pool, eval_set=cb_eval, early_stopping_rounds=50)
                    cb_oof[vi] += mdl.predict(Xva, prediction_type="Probability")[:, 1] / 5

            # Stacking: train meta-learner on OOF predictions
            train_X = np.column_stack([lgb_oof, cb_oof])
            meta = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            meta.fit(train_X, y)
            meta_pred = meta.predict_proba(train_X)[:, 1]
            cal_loss = log_loss(y, meta_pred, labels=[0,1])
            results['D_stacking'][target] = cal_loss

            v10_cal = v10_cal_oof[target]
            diff = cal_loss - v10_cal
            marker = "✓ BETTER" if diff < -0.001 else ("~ same" if abs(diff) <= 0.001 else "✗ worse")
            log.info(f"  {target}: V10={v10_cal:.4f}, V13={cal_loss:.4f}, Δ={diff:+.4f} {marker}")

        avg_d = np.mean(list(results['D_stacking'].values()))
        log.info(f"  D avg: {avg_d:.6f} (V10: {v10_avg:.6f}, Δ={avg_d-v10_avg:+.6f})")

    # ── Final comparison ─────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("V13 FINAL COMPARISON")
    log.info(f"{'='*70}")
    log.info(f"{'Config':<20} {'AVG Cal OOF':<14} {'Δ vs V10':<12} {'Winner'}")
    log.info(f"{'V10':<20} {v10_avg:<14.6f} {'—':<12} {'baseline'}")

    all_avgs = {'V10': v10_avg}
    for name, res in results.items():
        avg = np.mean(list(res.values()))
        all_avgs[name] = avg
        diff = avg - v10_avg
        is_best = "⭐" if diff < -0.0001 else ""
        log.info(f"{name:<20} {avg:<14.6f} {diff:+.6f} {is_best}")

    best_name = min(all_avgs, key=all_avgs.get)
    best_avg = all_avgs[best_name]
    beat = best_avg < v10_avg
    log.info(f"\n{'🎯 ' if beat else ''}BEST: {best_name} ({best_avg:.6f})")
    log.info(f"{'BEATS V10!' if beat else 'Does not beat V10 yet.'} (Δ={best_avg-v10_avg:+.6f})")

    total_time = time.time() - t_total
    log.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save results
    result_file = Path("submissions") / f"v13_results_{int(time.time())}.json"
    result_file.parent.mkdir(exist_ok=True)
    meta_out = {"version":"v13","timestamp":int(time.time()),
        "v10_avg":v10_avg,"best_config":best_name,"best_avg":best_avg,
        "beat_v10":beat,"results":{}}
    for name, res in results.items():
        meta_out["results"][name] = {t: float(v) for t, v in res.items()}
        meta_out["results"][name]["avg"] = float(np.mean(list(res.values())))
    with open(result_file, 'w') as f:
        json.dump(meta_out, f, indent=2)
    log.info(f"Results saved: {result_file}")

if __name__ == "__main__":
    main()
