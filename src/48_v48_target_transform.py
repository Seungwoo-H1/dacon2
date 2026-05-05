"""
V48 — Target Transformation (Power/Yeo-Johnson + Binarization)

Hypothesis: The targets (binary 0/1) may have distributions that make
classification harder than regression-like approaches. By applying Yeo-Johnson
transform and then using a different binarization threshold, we might find
a better decision boundary that separates classes more cleanly.

Additionally, we try:
1. Direct regression on targets (treat as continuous [0,1]) then clip
2. Regression on Yeo-Johnson transformed targets
3. Probability calibration comparison: isotonic vs platt vs none

Method:
1. For each target, try multiple approaches:
   a) Standard binary classification (baseline)
   b) Regression-based: train LGBM as regression → clip to [0,1] → threshold at target rate
   c) Yeo-Johnson transform on targets → binarize with optimal threshold → classify
   d) Two-stage: regression first (predict continuous), then classify the residual
2. Evaluate all on OOF CV
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


# ── Config: medium depth, moderate reg ──
CFG = {
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
    del model, ds, X, y
    gc.collect()
    return [r[0] for r in ranked]


def train_cv_model(feat, cols, y_train, seeds, n_folds=5, task='binary'):
    """Train model with CV, return OOF predictions (n_samples, n_seeds)."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y_train), len(seeds)))
    sn = [sanitize(c) for c in cols]

    obj = 'regression' if task == 'regression' else 'binary'
    metric = 'rmse' if task == 'regression' else 'binary_logloss'

    cfg_full = {
        'objective': obj, 'metric': metric,
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': CFG['nl'], 'max_depth': CFG['md'],
        'learning_rate': CFG['lr'], 'n_estimators': CFG['ne'],
        'subsample': CFG['ss'], 'colsample_bytree': CFG['cb'],
        'reg_alpha': CFG['ra'], 'reg_lambda': CFG['rl'],
        'min_child_samples': CFG['mc'],
    }

    for si, seed in enumerate(seeds):
        cfg_seed = {**cfg_full, 'random_state': seed}
        for tr, va in gkf.split(feat, y_train, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            y_tr = y_train[tr]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y_train[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=CFG['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
            gc.collect()
    return oof


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V48 — Target Transformation Experiments")
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
        log.info(f"  {target} top-10: {ranked[:10]}")

    # ── 3. Per-target experiments ──
    log.info("\n--- 3. Per-target experiments ---")

    all_results = {}

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_map[target]
        sel_cols = ranked[:20]

        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}) ===")

        # ── Method A: Binary classification (baseline) ──
        oof_a = train_cv_model(feat, sel_cols, y, SEEDS, n_folds=5, task='binary')
        oof_a_avg = np.clip(oof_a.mean(axis=1), 0.0001, 0.9999)
        cal_a = mean_match(oof_a_avg, train_rates[target])
        cal_loss_a = log_loss(y, cal_a, labels=[0, 1])
        log.info(f"    [A] Binary classification: OOF={log_loss(y, oof_a_avg, labels=[0,1]):.4f}, Cal={cal_loss_a:.4f}")

        # ── Method B: Regression (direct continuous prediction) ──
        oof_b = train_cv_model(feat, sel_cols, y, SEEDS, n_folds=5, task='regression')
        oof_b_avg = np.clip(oof_b.mean(axis=1), 0, 1)
        # Calibrate via Platt scaling (logistic on OOF)
        oof_b_clipped = np.clip(oof_b_avg, 0.001, 0.999)
        platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        platt.fit(oof_b_clipped.reshape(-1, 1), y)
        cal_b_raw = platt.predict_proba(oof_b_clipped.reshape(-1, 1))[:, 1]
        cal_b = mean_match(cal_b_raw, train_rates[target])
        cal_loss_b = log_loss(y, cal_b, labels=[0, 1])
        log.info(f"    [B] Regression + Platt cal: OOF={log_loss(y, np.clip(oof_b_avg,0.0001,0.9999), labels=[0,1]):.4f}, Cal={cal_loss_b:.4f}")

        # ── Method C: Yeo-Johnson transform on targets ──
        # Fit YJ on target distribution, transform, then classify
        try:
            yt_johnson = feat[target].values.copy()
            # Add small epsilon for YJ (needs strictly positive or use Yeo-Johnson which handles zeros)
            yt_yj = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(
                yt_johnson.reshape(-1, 1)
            ).ravel()

            # Now the transformed targets are approximately normal.
            # Use them as soft labels: train regression, then map back via inverse YJ
            oof_c_reg = train_cv_model(feat, sel_cols, yt_yj, SEEDS, n_folds=5, task='regression')
            oof_c_raw = oof_c_reg.mean(axis=1)

            # Inverse transform to get predictions in original space
            # (We need to fit the same transformer on train to get exact inverse)
            # Since we're using OOF predictions which are out-of-sample,
            # we approximate by fitting on full y and transforming predictions
            yj_fit = PowerTransformer(method='yeo-johnson', standardize=True)
            y_full_yj = yj_fit.fit_transform(y.reshape(-1, 1)).ravel()

            # Fit a simple mapping: predict_yj → predict_y via regression
            # Actually, better: fit mapping from reg_preds (in YJ space) to binary
            # Use isotonic: fit on full data, predict on OOF
            iso = IsotonicRegression(out_of_bounds='clip')
            # Train: use reg predictions as features, binary target as labels
            # But we need OOF here... let's use the reg OOF predictions
            oof_c_clipped = np.clip(oof_c_raw, -3, 3)  # YJ space is roughly [-3, 3]

            # Map back: use the YJ inverse
            try:
                oof_c_inv = yj_fit.inverse_transform(oof_c_clipped.reshape(-1, 1)).ravel()
                oof_c_inv = np.clip(oof_c_inv, 0, 1)
                cal_c = mean_match(oof_c_inv, train_rates[target])
            except Exception:
                oof_c_inv = oof_c_clipped
                cal_c = mean_match(oof_c_inv, train_rates[target])

            cal_loss_c = log_loss(y, cal_c, labels=[0, 1])
            log.info(f"    [C] Yeo-Johnson + regression + inverse: Cal={cal_loss_c:.4f}")
            method_c_valid = True
        except Exception as e:
            log.info(f"    [C] Yeo-Johnson FAILED: {e}")
            method_c_valid = False
            cal_loss_c = float('inf')

        # ── Method D: Isotonic calibration on binary OOF ──
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(oof_a_clipped := np.clip(oof_a_avg, 0.001, 0.999), y)
            cal_d = iso.predict(oof_a_avg)
            cal_d = mean_match(cal_d, train_rates[target])
        except Exception:
            cal_d = oof_a_avg
        cal_loss_d = log_loss(y, cal_d, labels=[0, 1])
        log.info(f"    [D] Binary + Isotonic cal: Cal={cal_loss_d:.4f}")

        # ── Method E: Ensemble of B+D (regression + binary, average) ──
        oof_ensemble = (cal_b + cal_d) / 2
        cal_e = mean_match(oof_ensemble, train_rates[target])
        cal_loss_e = log_loss(y, cal_e, labels=[0, 1])
        log.info(f"    [E] Ensemble (B+D): Cal={cal_loss_e:.4f}")

        # Find best
        losses = {
            'A': cal_loss_a,
            'B': cal_loss_b,
            'D': cal_loss_d,
            'E': cal_loss_e,
        }
        if method_c_valid:
            losses['C'] = cal_loss_c

        best_method = min(losses, key=losses.get)
        best_cal = losses[best_method]

        log.info(f"    ✅ Best for {target}: {best_method} (Cal={best_cal:.4f})")
        for m, l in sorted(losses.items()):
            marker = " ← BEST" if m == best_method else ""
            log.info(f"       {m}: {l:.4f}{marker}")

        # Get the best OOF for this target
        if best_method == 'A':
            best_oof = cal_a
        elif best_method == 'B':
            best_oof = cal_b
        elif best_method == 'C':
            best_oof = cal_c
        elif best_method == 'D':
            best_oof = cal_d
        elif best_method == 'E':
            best_oof = cal_e

        all_results[target] = {
            'best_method': best_method,
            'cal_oof': best_oof,
            'cal_loss': best_cal,
            'all_losses': losses,
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V48 SUMMARY")
    log.info(f"{'='*70}")

    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: best={r['best_method']} Cal={r['cal_loss']:.4f}")

    avg_cal = np.mean([
        log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V48 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save OOF ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    oof_path = DATA_PROCESSED / "oof_v48.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 6. Save metadata ──
    meta = {
        'version': 'V48',
        'name': 'Target Transformation',
        'avg_cal_loss': avg_cal,
        'v10_cal_loss': 0.6038,
        'delta': avg_cal - 0.6038,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'best_method': r['best_method'],
            'cal_loss': r['cal_loss'],
            'all_losses': {k: float(v) for k, v in r['all_losses'].items()},
        }
    meta_path = DATA_PROCESSED / "v48_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")

    return all_results


if __name__ == "__main__":
    main()
