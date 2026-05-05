"""
V50 — Isotonic Calibration + Feature Ensembling (Best from V48 + V47 insights)

Key findings from V47/V48/V49:
- V48: Isotonic calibration on binary OOF → -0.0153 (big improvement!)
- V47: Multi-config ensemble didn't help (overfits with 450 samples)
- V49: MI feature selection worse than LGBM importance

V50 combines:
1. Isotonic calibration (from V48, the best finding)
2. Multiple feature counts per target (5, 10, 15, 20) + isotonic cal
3. Also try isotonic on the per-config ensemble from V47
4. Deeper configs (V37-style per-target configs) + isotonic cal

This is the "isotonic-aware" version of the existing V10/V37 pipeline.
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"
DATA_RAW = ROOT / "data_raw"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


# ── Leakage columns ──
LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


# ── V37-style per-target configs ──
V37_CFGS = {
    "Q1": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cb": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q2": {"nl": 10, "md": 3, "lr": 0.05, "ne": 200, "ss": 0.8, "cb": 0.7, "ra": 0.5, "rl": 2.0, "mc": 10},
    "Q3": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cb": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
    "S1": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cb": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S2": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cb": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S3": {"nl": 6,  "md": 2, "lr": 0.02, "ne": 200, "ss": 0.5, "cb": 0.5, "ra": 10.0,"rl": 20.0,"mc": 25},
    "S4": {"nl": 8,  "md": 3, "lr": 0.03, "ne": 200, "ss": 0.6, "cb": 0.6, "ra": 2.0, "rl": 5.0, "mc": 15},
}

# Default config for V50
DEFAULT_CFG = {
    'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500,
    'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10,
}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def add_personalization(df, feature_cols):
    personal_cols = []
    df = df.copy()
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols


def rank_features_importance(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]


def train_cv_oof(feat, cols, target, seeds, cfg=None, n_folds=5):
    """Train LGBM with CV, return OOF predictions."""
    if cfg is None:
        cfg = DEFAULT_CFG
    y = feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]

    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }

    for si, seed in enumerate(seeds):
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
            gc.collect()
    return oof


def isotonic_calibrate(oof_preds, y_true):
    """Apply isotonic regression calibration to OOF predictions."""
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(oof_preds, y_true)
        return iso.predict(oof_preds), True
    except Exception:
        return oof_preds, False


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V50 — Isotonic Calibration + Feature Ensembling")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Loaded: {feat.shape}")

    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape}")

    all_cols = feat_cols_raw + zscore_cols
    train_rates = {t: feat[t].mean() for t in TARGET_COLS}

    # ── 2. Feature ranking ──
    log.info("\n--- 2. Feature ranking ---")
    ranked_map = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_map[target] = ranked

    # ── 3. Experiment: Isotonic calibration vs no cal ──
    log.info("\n--- 3. Isotonic calibration experiments ---")

    all_results = {}

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_map[target]

        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}) ===")

        # Evaluate multiple feature counts: 5, 10, 15, 20
        # With and without isotonic calibration
        best_overall_cal = float('inf')
        best_overall_oof = None
        best_config = None
        best_n = None
        results_detail = {}

        for n_feat in [5, 10, 15, 20]:
            sel_cols = ranked[:n_feat]
            oof = train_cv_oof(feat, sel_cols, target, SEEDS, cfg=V37_CFGS.get(target, DEFAULT_CFG))
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)

            # No calibration
            no_cal_loss = log_loss(y, oof_avg, labels=[0, 1])

            # Isotonic calibration
            iso_cal, ok = isotonic_calibrate(oof_avg, y)
            if ok:
                iso_cal = mean_match(iso_cal, train_rates[target])
                iso_cal_loss = log_loss(y, iso_cal, labels=[0, 1])
            else:
                iso_cal_loss = no_cal_loss
                iso_cal = oof_avg

            # Mean-match only (V10 style)
            mm_cal = mean_match(oof_avg, train_rates[target])
            mm_cal_loss = log_loss(y, mm_cal, labels=[0, 1])

            results_detail[n_feat] = {
                'no_cal': no_cal_loss,
                'iso_cal': iso_cal_loss,
                'mm_cal': mm_cal_loss,
            }

            log.info(f"    n={n_feat}: no_cal={no_cal_loss:.4f}, iso_cal={iso_cal_loss:.4f}, mm_cal={mm_cal_loss:.4f}")

            # Track best
            if iso_cal_loss < best_overall_cal:
                best_overall_cal = iso_cal_loss
                best_overall_oof = iso_cal
                best_config = f"iso_cal_n{n_feat}"
                best_n = n_feat

        log.info(f"    ✅ Best: {best_config} Cal={best_overall_cal:.4f}")

        all_results[target] = {
            'best_method': best_config,
            'cal_oof': best_overall_oof,
            'cal_loss': best_overall_cal,
            'n_feat': best_n,
            'results_detail': results_detail,
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V50 SUMMARY")
    log.info(f"{'='*70}")

    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['best_method']} Cal={r['cal_loss']:.4f} n={r['n_feat']}")

    avg_cal = np.mean([
        log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V50 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  V48 Avg Cal: 0.5885")
    log.info(f"  Δ vs V10: {avg_cal - 0.6038:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")
    log.info(f"  Δ vs V48: {avg_cal - 0.5885:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save OOF ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    oof_path = DATA_PROCESSED / "oof_v50.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 6. Save metadata ──
    meta = {
        'version': 'V50',
        'name': 'Isotonic Calibration + Feature Ensembling',
        'avg_cal_loss': avg_cal,
        'v10_cal_loss': 0.6038,
        'v48_cal_loss': 0.5885,
        'delta_v10': avg_cal - 0.6038,
        'delta_v48': avg_cal - 0.5885,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'best_method': r['best_method'],
            'cal_loss': r['cal_loss'],
            'n_feat': r['n_feat'],
            'results_detail': {
                str(k): {mk: float(mv) for mk, mv in v.items()}
                for k, v in r['results_detail'].items()
            },
        }
    meta_path = DATA_PROCESSED / "v50_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")

    return all_results


if __name__ == "__main__":
    main()
