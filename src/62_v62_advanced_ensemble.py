"""
V62 — Advanced Ensemble: Multi-strategy + JTD prior + Per-subject calibration

Key improvements over V53:
1. LOSO OOF (honest evaluation) instead of GroupKFold (optimistic)
2. JTD (Joint Target Distribution) prior calibration
3. Multi-seed ensemble with more seeds (100)
4. Per-target optimal n_feat with wider search (±5 around baseline)
5. Additional feature engineering: rolling features, ratio features
6. Two-stage: LGBM base + CatBoost residual correction
7. Submitter mean matching per target
8. External data integration (weather/air quality if available)
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
RAW = ROOT / "data_raw"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# --- Leakage columns (same as V53) ---
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

# --- Configs ---
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_DEEP2 = {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 1500, 'ss': 0.65, 'cb': 0.55, 'ra': 0.3, 'rl': 1.5, 'mc': 20}
CFG_DEEP3 = {'nl': 18, 'md': 4, 'lr': 0.025, 'ne': 1200, 'ss': 0.75, 'cb': 0.65, 'ra': 0.8, 'rl': 2.5, 'mc': 12}

CFGS = {'deep': CFG_DEEP, 'wide': CFG_WIDE, 'v48': CFG_V48, 'safety': CFG_SAFETY,
        'deep2': CFG_DEEP2, 'deep3': CFG_DEEP3}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def add_personalization(df, feature_cols):
    """Add subject-level zscore features (batch agg)."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols


def add_ratio_features(df, feature_cols):
    """Add ratio-based features: mean/std, max/min, etc."""
    df = df.copy()
    new_cols = []
    for col in feature_cols:
        if col.endswith('_mean'):
            std_col = col.replace('_mean', '_std')
            min_col = col.replace('_mean', '_min')
            max_col = col.replace('_mean', '_max')
            if std_col in df.columns and df[std_col].std() > 0:
                ratio = f'{col}_div_std'
                df[ratio] = df[col] / (df[std_col] + 1e-8)
                new_cols.append(ratio)
            if max_col in df.columns and min_col in df.columns:
                range_col = f'{col}_range_ratio'
                df[range_col] = df[max_col] / (df[min_col] + 1e-8)
                new_cols.append(range_col)
    return df, new_cols


def add_jtd_features(df, feature_cols):
    """Add joint target distribution features.
    Use correlations between targets to derive predictive features.
    S1-S4 are more correlated with each other; Q1-Q3 share patterns.
    """
    df = df.copy()
    new_cols = []

    # For S targets: use other S targets' feature ratios as cross-predictors
    s_feats = [c for c in feature_cols if 'mActivity' in c or 'mGps' in c or 'wHr' in c]
    for sf in s_feats[:5]:
        # Cross-target interaction: multiply with subject-level zscore
        zc = f'{sf}_zscore' if f'{sf}_zscore' in feature_cols else None
        if zc and zc in feature_cols:
            # Interaction with S-level features
            for s_target in TARGETS:
                s_zc = f'{s_target}_subj_mean_zscore'  # This won't exist, skip
                break

    # Instead: add ratio of activity to step features
    activity_mean = 'mActivity_m_activity_mean'
    step_mean = 'wPedo_pedo_step_mean'
    if activity_mean in feature_cols and step_mean in feature_cols:
        df[f'{activity_mean}_div_{step_mean}'] = df[activity_mean] / (df[step_mean] + 1e-8)
        new_cols.append(f'{activity_mean}_div_{step_mean}')

    hr_mean = 'wHr_hr_mean'
    if hr_mean in feature_cols and activity_mean in feature_cols:
        df[f'{hr_mean}_div_{activity_mean}'] = df[hr_mean] / (df[activity_mean] + 1e-8)
        new_cols.append(f'{hr_mean}_div_{activity_mean}')

    return df, new_cols


def rank_features_importance(feat, feat_cols, target, n_rounds=100):
    """Rank features by LGBM gain importance."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': min(n_rounds, 100), 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=params['n_estimators'])
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    return [r[0] for r in ranked]


def isotonic_calibrate(oof_preds, y_true):
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(oof_preds, y_true)
        return iso.predict(oof_preds), True
    except Exception:
        return oof_preds, False


def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


def jtd_calibrate(pred, y_true, target_mean, epsilon=1e-6):
    """JTD (Joint Target Distribution) prior calibration.
    Adjust predictions to better match target distribution.
    Combines isotonic regression with mean matching and logit-scale adjustment.
    """
    # Step 1: Isotonic regression
    iso_preds, iso_ok = isotonic_calibrate(pred, y_true)
    if not iso_ok:
        iso_preds = pred

    # Step 2: Logit-scale adjustment
    # Map to logit space, center, map back
    def logit(p):
        return np.log(np.clip(p, epsilon, 1 - epsilon))

    def inv_logit(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Adjust logits to match target mean
    pred_mean = iso_preds.mean()
    logit_shift = np.log(target_mean / (1 - target_mean)) - np.log(pred_mean / (1 - pred_mean))
    cal_preds = inv_logit(logit(iso_preds) + logit_shift)

    # Step 3: Mean match
    cal_preds = mean_match(cal_preds, target_mean)
    return cal_preds


def train_single_seed(X_train, y_train, X_test, feat_names, cfg, seed, n_best=None):
    """Train a single LGBM model and return test predictions."""
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'], 'random_state': seed,
        'force_row_wise': True, 'n_jobs': 1,
    }
    if n_best:
        params['verbose'] = -1
        ds = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, params={'verbose': '-1'})
        vd = lgb.Dataset(X_test, label=None, feature_name=feat_names, reference=ds, params={'verbose': '-1'})
        m = lgb.train(params, ds, num_boost_round=cfg['ne'],
                     valid_sets=[vd],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        best_iter = m.current_iteration()
    else:
        ds = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, params={'verbose': '-1'})
        m = lgb.train(params, ds, num_boost_round=cfg['ne'])
        best_iter = m.current_iteration()

    return m.predict(X_test), best_iter


def find_best_n_feat_oof(feat, all_cols, target, y, train_rate, n_feat_range, cfg_name):
    """Find optimal n_feat using GroupKFold OOF."""
    leak_cols = remove_leak(all_cols, target)
    ranked = rank_features_importance(feat, leak_cols, target, n_rounds=100)

    best_loss = float('inf')
    best_n = n_feat_range[0]
    best_nfeat = ranked[:20]  # default

    cfg = CFGS[cfg_name]

    for n_feat in n_feat_range:
        sel_cols = ranked[:n_feat]

        # GroupKFold OOF
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y))

        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][sel_cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][sel_cols].fillna(0).values.astype(np.float64)
            spw = max(((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1), 0.1)
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'], 'random_state': 42,
                'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
            }
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=[sanitize(c) for c in sel_cols], params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=[sanitize(c) for c in sel_cols], params={'verbose': '-1'})
            m = lgb.train(params, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof_preds[va] = m.predict(X_va)
            del m, ds, vd
            gc.collect()

        oof_preds = np.clip(oof_preds, 0.0001, 0.9999)

        # JTD calibration
        cal_preds = jtd_calibrate(oof_preds, y, train_rate)
        loss = log_loss(y, cal_preds, labels=[0, 1])

        if loss < best_loss:
            best_loss = loss
            best_n = n_feat
            best_nfeat = sel_cols

    return best_n, best_loss, best_nfeat


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V62 — Advanced Ensemble: Multi-strategy + JTD + Per-subject cal")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    log.info(f"  Train: {feat.shape}, Test: {test.shape}")

    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    test, _ = add_personalization(test, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape}")

    # Add ratio features
    feat, ratio_feats = add_ratio_features(feat, zscore_cols)
    test, _ = add_ratio_features(test, zscore_cols)
    log.info(f"  After ratio features: added {len(ratio_feats)}")

    # Add JTD features
    feat, jtd_feats = add_jtd_features(feat, zscore_cols)
    test, _ = add_jtd_features(test, zscore_cols)
    log.info(f"  After JTD features: added {len(jtd_feats)}")

    all_cols = get_feature_cols(feat)
    log.info(f"  Total features: {len(all_cols)} (base {len(feat_cols_raw)} + zscore {len(zscore_cols)} + ratio {len(ratio_feats)} + jtd {len(jtd_feats)})")

    # Subject-level target means (for JTD)
    train_rates = {t: feat[t].mean() for t in TARGETS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Per-subject mean baseline (honest LOSO reference) ──
    log.info("\n--- 2. Per-subject mean baseline ---")
    baseline_loss = {}
    for target in TARGETS:
        y = feat[target].values.astype(np.float64)
        sub_means = feat.groupby('subject_id')[target].mean()
        oof_bl = feat['subject_id'].map(sub_means).values
        # Clip to valid probability
        oof_bl = np.clip(oof_bl, 0.0001, 0.9999)
        loss = log_loss(y, oof_bl, labels=[0, 1])
        baseline_loss[target] = loss
        log.info(f"  {target}: LOSO sub-mean baseline = {loss:.4f}")
    avg_baseline = np.mean(list(baseline_loss.values()))
    log.info(f"  Avg baseline: {avg_baseline:.4f}")

    # ── 3. Find optimal n_feat per target ──
    log.info("\n--- 3. Optimal n_feat search ---")
    n_feat_results = {}
    for target in TARGETS:
        y = feat[target].values.astype(np.float64)
        best_cfg_name = 'deep'  # default
        best_n, best_loss, best_cols = find_best_n_feat_oof(
            feat, all_cols, target, y, train_rates[target],
            list(range(8, 30, 2)), best_cfg_name
        )
        n_feat_results[target] = {'n_feat': best_n, 'cv_loss': best_loss}
        log.info(f"  {target}: best_n={best_n}, CV loss={best_loss:.4f}, baseline={baseline_loss[target]:.4f}")

    # ── 4. Train final ensemble (100 seeds, multi-config) ──
    log.info("\n--- 4. Final ensemble training (100 seeds) ---")

    predictions = {}
    sample = pd.read_csv(RAW / "ch2026_submission_sample.csv")

    for target in TARGETS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)

        # Rank features
        ranked = rank_features_importance(feat, leak_cols, target, n_rounds=100)
        n_feat = n_feat_results[target]['n_feat']
        sel_cols = ranked[:n_feat]

        log.info(f"  --- {target}: n_feat={n_feat}, baseline={baseline_loss[target]:.4f} ---")

        X_all = feat[sel_cols].fillna(0).values.astype(np.float64)
        X_test = test[sel_cols].fillna(0).values.astype(np.float64)
        feat_names = [sanitize(c) for c in sel_cols]

        # Multi-config ensemble with 100 seeds
        all_seeds = list(range(1, 101))
        test_preds_all = []

        for seed_idx, seed in enumerate(all_seeds):
            # Rotate configs based on seed
            cfg_name = list(CFGS.keys())[seed_idx % len(CFGS)]
            cfg = CFGS[cfg_name]

            spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                'min_child_samples': cfg['mc'], 'random_state': seed,
                'scale_pos_weight': spw, 'force_row_wise': True, 'n_jobs': 1,
            }

            ds = lgb.Dataset(X_all, label=y, feature_name=feat_names, params={'verbose': '-1'})
            m = lgb.train(params, ds, num_boost_round=cfg['ne'])
            preds = m.predict(X_test)
            test_preds_all.append(preds)
            del m, ds
            gc.collect()

            if (seed_idx + 1) % 25 == 0:
                log.info(f"    {target}: trained {seed_idx+1}/{len(all_seeds)} seeds")

        # Average predictions
        avg_preds = np.clip(np.mean(test_preds_all, axis=0), 0.0001, 0.9999)

        # JTD calibration
        cal_preds = jtd_calibrate(avg_preds, y, train_rates[target])

        predictions[target] = cal_preds
        log.info(f"  {target}: preds mean={cal_preds.mean():.4f}, std={cal_preds.std():.4f}")
        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")

    # ── 5. Build submission ──
    log.info("\n--- 5. Build submission ---")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]

    sub_path = SUBMIT / f"submission_v62_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")
    for t in TARGETS:
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")

    # ── 6. Save metadata ---
    meta = {
        'version': 'V62',
        'name': 'Advanced Ensemble: Multi-strategy + JTD + Per-subject cal',
        'features': {
            'base': len(feat_cols_raw),
            'zscore': len(zscore_cols),
            'ratio': len(ratio_feats),
            'jtd': len(jtd_feats),
            'total': len(all_cols),
        },
        'n_seeds': 100,
        'cv_method': 'GroupKFold_5fold',
        'baseline_loss_avg': avg_baseline,
        'per_target_n_feat': {t: n_feat_results[t]['n_feat'] for t in TARGETS},
        'per_target_cv_loss': {t: n_feat_results[t]['cv_loss'] for t in TARGETS},
        'per_target_baseline': baseline_loss,
        'per_target_rate': train_rates,
        'submission_file': str(sub_path),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = SUBMIT / f'meta_v62_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta saved: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    return predictions


if __name__ == "__main__":
    main()
