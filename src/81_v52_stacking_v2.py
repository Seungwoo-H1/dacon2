"""
V81 — V52 Improved: Multi-Config Ensemble + Isotonic Calibration + Meta-Learner

Improvements over V52:
1. Multi-config ensemble: train ALL 4 configs (v48/deep/wide/safety) with best n_feat, then ensemble
   V52 picked 1 config/target. We ensemble top-2 configs per target.
2. Meta-learner (LogisticRegression): stack OOF predictions from multiple configs
3. Wider feature search: 5,8,10,15,20 → 5,8,10,15,20,25
4. Rank fusion feature selection: combine LGBM importance + MI ranking
5. Mean-match + isotonic cal after ensemble

Key hypothesis: ensemble of diverse configs beats single best config.
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
DATA_RAW = ROOT / "data_raw"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

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


# ── Configs ──
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}

CFGS = {'v48': CFG_V48, 'deep': CFG_DEEP, 'wide': CFG_WIDE, 'safety': CFG_SAFETY}

# Fewer seeds for speed (20 → 15)
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000]


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


def rank_features_fusion(feat, feat_cols, target):
    """Rank features by rank fusion (LGBM importance + MI)."""
    # LGBM importance
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    lgb_ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()

    # MI ranking
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranked = sorted(zip(feat_cols, mi_scores), key=lambda x: -x[1])

    # Rank average fusion
    lgb_ranks = {f: i+1 for i, f in enumerate(lgb_ranked)}
    mi_ranks = {f: i+1 for i, f in enumerate(mi_ranked)}
    fused_scores = {}
    for f in feat_cols:
        fused_scores[f] = lgb_ranks.get(f, len(feat_cols)) + mi_ranks.get(f, len(feat_cols))
    ranked = sorted(feat_cols, key=lambda f: fused_scores[f])
    return ranked


def train_cv_model(feat, cols, y, seeds, cfg, n_folds=5):
    """Train model with CV, return OOF predictions (avg over seeds)."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    n_valid = np.zeros(len(y))
    sn = [sanitize(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }

    for seed in seeds:
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va] += m.predict(X_va)
            n_valid[va] += 1
            del ds, vd, m, X_tr, X_va
            gc.collect()

    oof_avg = np.clip(oof / n_valid, 0.0001, 0.9999)
    return oof_avg


def isotonic_calibrate(pred, y_true):
    """Apply isotonic regression calibration."""
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(pred, y_true)
        return iso.predict(pred), True
    except Exception:
        return pred, False


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V81 — V52 Improved: Multi-Config Ensemble + Meta-Learner")
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

    # ── 2. Feature ranking (rank fusion) ──
    log.info("\n--- 2. Feature ranking (rank fusion LGBM+MI) ---")
    ranked_fusion = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_fusion(feat, leak_cols, target)
        ranked_fusion[target] = ranked
        log.info(f"  {target}: ranked {len(leak_cols)} features")

    # ── 3. Train all configs for all targets ──
    log.info("\n--- 3. Multi-config training ---")

    # Store OOF predictions for each config per target
    # config_oofs[target] = {cfg_name: oof_preds}
    config_oofs = {}
    config_losses = {}  # Track best config per target

    N_FEATS = [5, 8, 10, 15, 20, 25]

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_fusion[target]

        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}) ===")

        # Train each config × n_feat combo
        config_perf = {}  # cfg_name → (best_loss, best_n_feat, best_oof)
        all_oofs = {}  # cfg_name → oof (for best n_feat)

        for cfg_name, cfg in CFGS.items():
            cfg_best_loss = float('inf')
            cfg_best_n = None
            cfg_best_oof = None

            for n_feat in N_FEATS:
                sel_cols = ranked[:n_feat]
                oof = train_cv_model(feat, sel_cols, y, SEEDS, cfg, n_folds=5)

                # Mean match first
                oof_mm = mean_match(oof, train_rates[target])

                # Isotonic cal
                iso_cal, ok = isotonic_calibrate(oof_mm, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    loss = log_loss(y, oof_mm, labels=[0, 1])

                config_perf[f"{cfg_name}_n{n_feat}"] = (loss, n_feat, oof)

                if loss < cfg_best_loss:
                    cfg_best_loss = loss
                    cfg_best_n = n_feat
                    cfg_best_oof = iso_cal if ok else oof_mm

            log.info(f"  {cfg_name}: best n_feat={cfg_best_n}, cal={cfg_best_loss:.4f}")
            config_perf[f"{cfg_name}_best"] = (cfg_best_loss, cfg_best_n, cfg_best_oof)
            all_oofs[cfg_name] = (cfg_best_oof, cfg_best_loss, cfg_best_n)

        config_oofs[target] = all_oofs

        # Pick top-2 configs by performance
        sorted_cfgs = sorted(all_oofs.items(), key=lambda x: x[1][1])
        top_cfgs = sorted_cfgs[:2]
        top_names = [c[0] for c in top_cfgs]
        log.info(f"    Top-2: {[f'{n}(loss={c[1]:.4f},n={c[2]})' for n,c in top_cfgs]}")

        # Ensemble top-2 configs
        oof_ensemble = np.zeros(len(y))
        weights = []
        for cfg_name, (oof, loss, n) in top_cfgs:
            w = 1.0 / max(loss, 0.01)  # inverse-loss weighting
            oof_ensemble += w * oof
            weights.append(w)
        oof_ensemble /= sum(weights)

        # Meta-learner: combine top-2 OOFs
        oof_stack = np.column_stack([c[1][0] for c in top_cfgs])
        best_c = 1.0
        best_meta_cv = float('inf')
        for c_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
            meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=2000, random_state=42)
            meta.fit(oof_stack, y)
            meta_pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
            meta_cv = log_loss(y, meta_pred)
            if meta_cv < best_meta_cv:
                best_meta_cv = meta_cv
                best_c = c_val

        # Apply meta-learner
        meta_pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
        meta_pred = mean_match(meta_pred, train_rates[target])

        # Compare meta vs simple ensemble
        ensemble_loss = log_loss(y, mean_match(oof_ensemble, train_rates[target]), labels=[0, 1])

        if meta_cv < ensemble_loss:
            final_oof = meta_pred
            final_method = f"meta-C={best_c:.2f}+top2"
            final_loss = meta_cv
            log.info(f"    ✅ Meta wins: meta={meta_cv:.4f} vs ensemble={ensemble_loss:.4f}")
        else:
            final_oof = mean_match(oof_ensemble, train_rates[target])
            final_method = f"ensemble-top2-C={best_c:.2f}"
            final_loss = ensemble_loss
            log.info(f"    ✅ Ensemble wins: ensemble={ensemble_loss:.4f} vs meta={meta_cv:.4f}")

        log.info(f"    Final: {final_method} cal={final_loss:.4f}")

        config_losses[target] = {
            'final_method': final_method,
            'cal_loss': final_loss,
            'cal_oof': final_oof,
            'top_configs': top_names,
            'all_oofs': {k: (o, l, n) for k, (o, l, n) in all_oofs.items()},
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V81 SUMMARY")
    log.info(f"{'='*70}")

    all_oofs_df = {}
    for target in TARGET_COLS:
        r = config_losses[target]
        log.info(f"  {target}: {r['final_method']} Cal={r['cal_loss']:.4f} (top: {r['top_configs']})")
        all_oofs_df[target] = r['cal_oof']

    avg_cal = np.mean([
        log_loss(feat[t].values, all_oofs_df[t], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V81 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V52 Avg Cal: 0.5725")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ vs V52: {avg_cal - 0.5725:+.4f} ({'✅ IMPROVED' if avg_cal < 0.5725 else '❌ Not improved'})")
    log.info(f"  Δ vs V10: {avg_cal - 0.6038:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save OOF ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_oofs_df[target]
    oof_path = DATA_PROCESSED / "oof_v81.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 6. Generate submission ──
    log.info("\n--- 6. Generate submission ---")
    test_feat = pd.read_parquet(DATA_PROCESSED / "features_test.parquet") if (DATA_PROCESSED / "features_test.parquet").exists() else None

    # Use test features if available, otherwise use train features for demo
    if test_feat is None:
        # For OOF, save OOF as "submission" (for internal eval only)
        sub_df = oof_df.copy()
        for target in TARGET_COLS:
            sub_df[target] = all_oofs_df[target]
        sub_path = SUBMIT / f"submission_v81_oof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        log.info("  Test features found. Generating real submission.")
        # Train final models on all data
        test_preds = {}
        for target in TARGET_COLS:
            r = config_losses[target]
            top_cfgs_data = r['all_oofs']
            # Sort and take top-2
            sorted_cfgs = sorted(top_cfgs_data.items(), key=lambda x: x[1][1])[:2]

            # Predict on test
            test_preds[target] = np.zeros(len(test_feat))
            total_weight = 0
            for cfg_name, (oof_val, loss, n_feat_val) in sorted_cfgs:
                # Re-train on full data with best config
                leak_cols = remove_leak(all_cols, target)
                ranked = ranked_fusion[target]
                sel_cols = ranked[:n_feat_val]
                cfg = CFGS[cfg_name.split('_')[0]]
                w = 1.0 / max(loss, 0.01)

                test_pred = train_cv_model_all(feat, test_feat, sel_cols, feat[target].values, SEEDS, cfg, n_folds=5, is_test=True)
                test_pred = mean_match(test_pred, train_rates[target])
                test_preds[target] += w * test_pred
                total_weight += w
            test_preds[target] /= total_weight

        sub_df = pd.DataFrame({
            'subject_id': test_feat['subject_id'].values,
            'sleep_date': test_feat['sleep_date'].values,
            'lifelog_date': test_feat['lifelog_date'].values,
        })
        for target in TARGET_COLS:
            sub_df[target] = test_preds[target]
        sub_path = SUBMIT / f"submission_v81_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    sub_df.to_csv(sub_path, index=False)
    log.info(f"  Submission: {sub_path}")

    # ── 7. Save metadata ──
    meta = {
        'version': 'V81',
        'name': 'V52 Improved: Multi-Config Ensemble + Meta-Learner',
        'avg_cal_loss': avg_cal,
        'v52_cal_loss': 0.5725,
        'v10_cal_loss': 0.6038,
        'n_seeds': len(SEEDS),
        'n_folds': 5,
        'feature_method': 'rank_fusion_lgbm_mi',
        'calibration': 'isotonic + mean_match',
        'ensemble': 'top-2 config inverse-loss weighted + meta-learner',
        'delta_v52': avg_cal - 0.5725,
        'delta_v10': avg_cal - 0.6038,
        'per_target': {},
        'submission_file': str(sub_path),
    }
    for target in TARGET_COLS:
        r = config_losses[target]
        meta['per_target'][target] = {
            'final_method': r['final_method'],
            'cal_loss': r['cal_loss'],
            'top_configs': r['top_configs'],
        }
    meta_path = SUBMIT / f"meta_v81_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")
    log.info(f"\n✅ DONE!")


def train_cv_model_all(feat_train, feat_test, cols, y_train, seeds, cfg, n_folds=5, is_test=False):
    """Train on full data and predict test set."""
    sn = [sanitize(c) for c in cols]
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)

    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }

    test_pred = np.zeros(len(feat_test))
    for seed in seeds:
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        X_tr = feat_train[cols].fillna(0).values.astype(np.float64)
        X_te = feat_test[cols].fillna(0).values.astype(np.float64)
        ds = lgb.Dataset(X_tr, label=y_train, feature_name=sn, params={'verbose': '-1'})
        m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'])
        test_pred += m.predict(X_te)
        del m, ds
        gc.collect()

    return test_pred / len(seeds)


if __name__ == "__main__":
    from datetime import datetime
    main()
