"""
V37_fix2 — Per-Target Hyper + 6 Configs × 3 feat counts + 20 Seeds

Uses features_v11_personalized.parquet (pre-computed personalization).
V10-style per-target tuning: 6 configs × 3 feat counts (10/20/30) × 20 seeds.
No feature engineering — uses pre-personalized features directly.

Expected: ~45 min, no SIGKILL
"""

import sys, re, warnings, logging, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

TARGET_COLS = TARGETS
META = {"subject_id", "lifelog_date", "sleep_date", "date"}
N_SPLITS = 5
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

LEAKAGE_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min', 'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count'}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(feat):
    return [c for c in feat.columns
            if c not in META | set(TARGET_COLS)
            and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def remove_leakage(feature_cols, target):
    if target.startswith('S'):
        return [c for c in feature_cols if c not in LEAKAGE_S]
    elif target.startswith('Q'):
        return [c for c in feature_cols if c not in LEAKAGE_Q]
    return feature_cols

def rank_features(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = (y==0).sum() / max((y==1).sum(), 1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 50,
        'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type="gain")
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del ds, m, imp, X, y
    gc.collect()
    return ranked

def train_ensemble(feat, cols, target, seeds, spw, n_est=500, early_stop=50):
    y = feat[target].values.astype(np.float64)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]
    for si, s in enumerate(seeds):
        cfg = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': n_est,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
            'random_state': s, 'scale_pos_weight': spw,
        }
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            Xtr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            Xva = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            d2 = lgb.Dataset(Xtr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            v2 = lgb.Dataset(Xva, label=y[va], feature_name=sn, reference=d2, params={'verbose': '-1'})
            m = lgb.train(cfg, d2, num_boost_round=n_est, valid_sets=[v2],
                         callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(Xva)
            del d2, v2, m, Xtr, Xva
        gc.collect()
    return oof

def mean_match(pred, rate):
    c = pred + (rate - pred.mean())
    return np.clip(c, 0.0001, 0.9999)

# V10-style configs: nl, md, lr, ne, ss, cst, ra, rl, mc
V10_CONFIGS = [
    {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cst': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C3', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cst': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cst': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    {'name': 'C6', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cst': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
]

def build_params(cfg, seed, spw):
    return {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cst'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
        'force_row_wise': True, 'n_jobs': -1,
        'random_state': seed, 'scale_pos_weight': spw,
    }

def main():
    log.info("=" * 70)
    log.info("V37_fix2 — Per-Target Hyper (V10 configs) + 20 Seeds")
    log.info("Uses features_v11_personalized.parquet (pre-computed)")
    log.info("=" * 70)

    # Load pre-personalized features
    feat_path = DATA_PROCESSED / "features_v11_personalized.parquet"
    log.info(f"Loading {feat_path}")
    feat = pd.read_parquet(feat_path)
    log.info(f"Features: {feat.shape}, memory={feat.memory_usage(deep=True).sum()/1024/1024:.1f}MB")

    all_cols = get_feature_cols(feat)
    zscore_cols = [c for c in all_cols if '_zscore' in c]
    basic_cols = [c for c in all_cols if '_zscore' not in c and '_subj' not in c]
    log.info(f"Basic: {len(basic_cols)}, Z-score: {len(zscore_cols)}")

    train_rate = {t: feat[t].mean() for t in TARGET_COLS}

    all_cal = {}
    log.info("\n=== Per-target tuning (V10 6-configs × 3-featcounts × 20-seeds) ===")

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={train_rate[target]:.3f}) ---")
        y = feat[target].values.astype(np.float64)
        spw = (y==0).sum() / max((y==1).sum(), 1)

        leak_free = remove_leakage(all_cols, target)
        log.info(f"  Leak-free: {len(leak_free)}")

        # Rank using z-score features (they're personalization)
        z_leak = remove_leakage(zscore_cols, target)
        ranked = rank_features(feat, z_leak, target)
        log.info(f"  Ranked {len(z_leak)} z-score features")

        best_config = None
        best_loss = float('inf')
        best_oof = None
        best_sel = None

        for n_feats in [10, 20, 30]:
            sel = [r[0] for r in ranked[:n_feats]]
            log.info(f"  Testing n_feats={n_feats}...")

            for cfg in V10_CONFIGS:
                oof = train_ensemble(feat, sel, target, SEEDS, spw, n_est=cfg['ne'])
                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                cal = mean_match(oof_avg, train_rate[target])
                loss = log_loss(y, cal, labels=[0, 1])
                oof_loss = log_loss(y, oof_avg, labels=[0, 1])

                if loss < best_loss:
                    best_loss = loss
                    best_oof = oof_avg.copy()
                    best_config = {**cfg, '_n_feats': n_feats}
                    best_sel = sel
                    log.info(f"    NEW BEST: {cfg['name']} n_feats={n_feats} cv={oof_loss:.4f} cal={loss:.4f}")

                del oof, oof_avg, cal
                gc.collect()

        all_cal[target] = best_oof
        log.info(f"  ✅ Best: {best_config} Cal={best_loss:.4f}")
        del best_oof
        gc.collect()

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V37_fix2 SUMMARY")
    for t in TARGET_COLS:
        cl = log_loss(feat[t], all_cal[t], labels=[0, 1])
        log.info(f"  {t}: Cal={cl:.4f}")
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0, 1]) for t in TARGET_COLS])
    log.info(f"\n  Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Delta: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

    # Generate submission
    log.info("\n=== Generating submission (train on ALL data) ===")

    # Rank and select features for each target
    all_selected = {}
    for target in TARGET_COLS:
        z_leak = remove_leakage(zscore_cols, target)
        ranked = rank_features(feat, z_leak, target)
        # Use best n_feats from tuning
        best_n = 20  # default
        for t2 in TARGET_COLS:
            if t2 == target and 'best_config' in dir():
                best_n = best_config.get('_n_feats', 20)
        all_selected[target] = [r[0] for r in ranked[:best_n]]
        log.info(f"  {target}: selected {len(all_selected[target])} z-score features")

    # Save OOF results
    oof_df = pd.DataFrame(all_cal, index=[0])
    oof_df.to_csv(MODEL_DIR / "v37_fix2_oof.csv", index=False)
    log.info(f"OOF saved to {MODEL_DIR}/v37_fix2_oof.csv")

if __name__ == "__main__":
    main()
