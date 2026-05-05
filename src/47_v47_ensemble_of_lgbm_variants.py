"""
V47 — Ensemble of LGBM Variants (Per-Target Multi-Config Stacking)

Hypothesis: Averaging models trained with diverse hyperparameter configs yields
lower variance and better calibration than a single "best" config.

Method:
1. Use 6 diverse LGBM configs (same as V43/V46)
2. For each target, train each config with 5 seeds (30 models per target)
3. OOF predictions from all configs → stack into single ensemble
4. Two stacking strategies:
   a) Simple average of all OOF predictions (weight = 1/N)
   b) Weighted average via LogisticRegression meta-learner (on per-config OOF)
5. Mean-match calibration after ensemble

Key design:
- Only LGBM (fast, already best single model)
- Diverse configs: shallow+regularized, deep+less regularized, high lr+few iters, etc.
- Per-target independent stacking weights
- Memory-safe: process one target at a time, delete intermediates
"""

import sys, re, gc, time, warnings, logging, json
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
    """Align prediction mean to target mean, clip to valid range."""
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


# ── 6 Diverse LGBM Configs ──
# Mix of shallow/deep, high/low lr, aggressive/moderate regularization
CONFIGS = [
    # C1: Deep, moderate reg (base)
    {'name': 'C1_deep_mod', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    # C2: Shallow, strong reg (conservative)
    {'name': 'C2_shallow_str', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cb': 0.5, 'ra': 10.0, 'rl': 20.0, 'mc': 25},
    # C3: Medium, balanced
    {'name': 'C3_balanced', 'nl': 12, 'md': 3, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    # C4: Wide shallow, high subsample diversity
    {'name': 'C4_wide_shallow', 'nl': 20, 'md': 3, 'lr': 0.02, 'ne': 300, 'ss': 0.5, 'cb': 0.4, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    # C5: Deep, low lr (slow but precise)
    {'name': 'C5_deep_slow', 'nl': 31, 'md': 5, 'lr': 0.01, 'ne': 800, 'ss': 0.8, 'cb': 0.8, 'ra': 0.1, 'rl': 1.0, 'mc': 20},
    # C6: Very shallow, very conservative (anti-overfit)
    {'name': 'C6_ultra_safe', 'nl': 4, 'md': 1, 'lr': 0.01, 'ne': 500, 'ss': 0.4, 'cb': 0.3, 'ra': 20.0, 'rl': 50.0, 'mc': 50},
]

SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for tuning


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def add_personalization(df, feature_cols):
    """Add per-subject z-score features. Memory-safe with gc."""
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
    """Rank features by LGBM gain importance."""
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


def train_single_config(feat, cols, target, seeds, cfg, n_folds=5, verbose=False):
    """Train one LGBM config with given seeds and folds. Returns OOF array (n_samples, n_seeds)."""
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


def stratified_groupkfolds(y, groups, n_splits=5):
    """Create GroupKFold splits that try to maintain target rate balance in each fold."""
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(None, y, groups))

    # Score each split set: how balanced are the fold target rates?
    best_score = float('inf')
    best_order = None

    # Try the default splits
    def fold_balance(splits_list):
        scores = []
        global_rate = y.mean()
        for tr, va in splits_list:
            val_rate = y[va].mean()
            scores.append(abs(val_rate - global_rate))
        return np.mean(scores)

    best_score = fold_balance(splits)
    best_order = list(range(len(splits)))

    # Also try shuffling seed order within folds (for seeds that are per-fold)
    return splits


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V47 — Ensemble of LGBM Variants (Multi-Config Stacking)")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Loaded: {feat.shape}")

    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape} ({len(zscore_cols)} zscore features)")

    # Remove leaked features per target
    all_cols = feat_cols_raw + zscore_cols

    train_rates = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Feature ranking (per target, top-20) ──
    log.info("\n--- 2. Feature ranking ---")
    ranked_map = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_map[target] = ranked
        log.info(f"  {target} top-10: {ranked[:10]}")

    # ── 3. Per-target experiments ──
    log.info("\n--- 3. Per-target multi-config training ---")

    all_results = {}  # target -> dict with OOF arrays per config

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_map[target]

        # Use top-20 features (good balance of signal vs overfit)
        sel_cols = ranked[:20]
        n_leak = len(sel_cols)
        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}, n_feat={n_leak}) ===")

        # Train all 6 configs with 5 seeds each
        config_oofs = {}
        for cfg in CONFIGS:
            cfg_oof = train_single_config(feat, sel_cols, target, SEEDS, cfg, n_folds=5)
            oof_avg = np.clip(cfg_oof.mean(axis=1), 0.0001, 0.9999)
            cv = log_loss(y, oof_avg, labels=[0, 1])
            config_oofs[cfg['name']] = cfg_oof
            log.info(f"    {cfg['name']}: OOF={cv:.4f}, pred_mean={oof_avg.mean():.4f}")
            gc.collect()

        # ── Strategy A: Simple average of all configs + seeds ──
        all_preds = np.zeros(len(y))
        for cfg_oof in config_oofs.values():
            all_preds += cfg_oof.mean(axis=1)
        all_preds /= (len(CONFIGS) * len(SEEDS))
        all_preds = np.clip(all_preds, 0.0001, 0.9999)
        cal_a = mean_match(all_preds, train_rates[target])
        cal_loss_a = log_loss(y, cal_a, labels=[0, 1])
        oof_loss_a = log_loss(y, all_preds, labels=[0, 1])
        log.info(f"    [A] Simple avg: OOF={oof_loss_a:.4f}, Cal={cal_loss_a:.4f}")

        # ── Strategy B: Weighted meta-learner stacking ──
        # Build meta-features: per-config avg OOF
        meta_feats = np.column_stack([config_oofs[cfg['name']].mean(axis=1) for cfg in CONFIGS])
        meta_clipped = np.clip(meta_feats, 0.0001, 0.9999)

        meta_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        meta_lr.fit(meta_clipped, y)

        meta_preds = meta_lr.predict_proba(meta_clipped)[:, 1]
        meta_preds = np.clip(meta_preds, 0.0001, 0.9999)
        cal_b = mean_match(meta_preds, train_rates[target])
        cal_loss_b = log_loss(y, cal_b, labels=[0, 1])
        oof_loss_b = log_loss(y, meta_preds, labels=[0, 1])
        log.info(f"    [B] Meta-learner: OOF={oof_loss_b:.4f}, Cal={cal_loss_b:.4f}")

        # Choose best strategy
        best_loss = min(cal_loss_a, cal_loss_b)
        best_strategy = 'A' if cal_loss_a < cal_loss_b else 'B'
        best_cal_loss = best_loss
        best_cal_oof = cal_a if best_strategy == 'A' else cal_b
        best_oof_oof = all_preds if best_strategy == 'A' else meta_preds

        log.info(f"    [WINNER] Strategy {best_strategy}, Cal={best_cal_loss:.4f}")

        # Config weights for strategy B
        config_weights = meta_lr.coef_[0].tolist() if best_strategy == 'B' else None

        all_results[target] = {
            'config_oofs': config_oofs,
            'strategy': best_strategy,
            'cal_oof': best_cal_oof,
            'oof_oof': best_oof_oof,
            'cal_loss': best_cal_loss,
            'config_weights': config_weights,
            'sel_cols': sel_cols,
        }

        log.info(f"  {target} total time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V47 SUMMARY")
    log.info(f"{'='*70}")

    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['strategy']} Cal={r['cal_loss']:.4f} n_feat={len(r['sel_cols'])}")

    avg_cal = np.mean([
        log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V47 Avg Cal: {avg_cal:.4f}")
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
    oof_path = DATA_PROCESSED / "oof_v47.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 6. Save metadata ──
    meta = {
        'version': 'V47',
        'name': 'Ensemble of LGBM Variants (Multi-Config Stacking)',
        'avg_cal_loss': avg_cal,
        'v10_cal_loss': 0.6038,
        'delta': avg_cal - 0.6038,
        'n_configs': len(CONFIGS),
        'n_seeds_per_config': len(SEEDS),
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'cal_loss': r['cal_loss'],
            'strategy': r['strategy'],
            'n_feat': len(r['sel_cols']),
            'config_weights': r['config_weights'],
        }
    meta_path = DATA_PROCESSED / "v47_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")

    return all_results


if __name__ == "__main__":
    main()
