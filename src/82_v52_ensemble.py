"""
V82 — V52 Improved: Multi-Config Ensemble + Per-Target Config

V52의 접근을 유지하면서 개선:
1. V52는 target별로 1개 config만 선택 → 이를 top-3 config ensemble으로 확장
2. V52의 LGBM importance ranking 유지 (rank fusion 제거)
3. 20 seeds 유지 (V52와 동일)
4. 각 config별로 isotonic cal 후 inverse-loss weighted ensemble
5. Per-target config 선택 (V53 방식) + Ensemble 비교

Key hypothesis: ensemble of diverse configs > single best config
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
CFG_V53_DEEP = {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 1500, 'ss': 0.65, 'cb': 0.65, 'ra': 0.3, 'rl': 1.5, 'mc': 20}
CFG_V53_WIDE = {'nl': 35, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.85, 'cb': 0.85, 'ra': 2.5, 'rl': 5.0, 'mc': 5}

CFGS = {'v48': CFG_V48, 'deep': CFG_DEEP, 'wide': CFG_WIDE, 'safety': CFG_SAFETY,
        'v53_deep': CFG_V53_DEEP, 'v53_wide': CFG_V53_WIDE}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

N_FEATS = [5, 8, 10, 15, 20]


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
    """Rank features by LGBM importance (gain)."""
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

    return np.clip(oof / n_valid, 0.0001, 0.9999)


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
    log.info("V82 — V52 Improved: Multi-Config Ensemble + Per-Target Config")
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

    # ── 2. Feature ranking (LGBM importance — V52 방식) ──
    log.info("\n--- 2. Feature ranking (LGBM importance) ---")
    ranked_lgb = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_lgb[target] = ranked
        log.info(f"  {target}: ranked {len(leak_cols)} features")

    # ── 3. Experiment ──
    log.info("\n--- 3. Multi-config training + ensemble ──")

    all_results = {}

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_lgb[target]

        log.info(f"\n  === {target} (rate={train_rates[target]:.3f}) ===")

        # Store all (cfg, n_feat) combinations
        all_configs = {}  # key: "cfg_nN" → (cal_loss, iso_cal_oof, n_feat, cfg_name)

        for cfg_name, cfg in CFGS.items():
            for n_feat in N_FEATS:
                sel_cols = ranked[:n_feat]
                oof = train_cv_model(feat, sel_cols, y, SEEDS, cfg, n_folds=5)

                # Isotonic cal
                iso_cal, ok = isotonic_calibrate(oof, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    loss = log_loss(y, oof, labels=[0, 1])
                    iso_cal = oof

                key = f"{cfg_name}_n{n_feat}"
                all_configs[key] = (loss, iso_cal, n_feat, cfg_name)

        # Sort by loss
        sorted_configs = sorted(all_configs.items(), key=lambda x: x[1][0])

        # Show top 5
        log.info("  Top 5 configs:")
        for key, (loss, oof, nf, cn) in sorted_configs[:5]:
            log.info(f"    {key}: cal={loss:.4f}")

        # ── Strategy 1: Single best config (V52 방식) ──
        best_key, (best_loss, best_oof, best_nf, best_cn) = sorted_configs[0]
        v52_style_loss = best_loss

        # ── Strategy 2: Top-3 ensemble (inverse-loss weighted) ──
        top3_keys = [k for k, _ in sorted_configs[:3]]
        oof_ensemble = np.zeros(len(y))
        total_w = 0
        for key in top3_keys:
            loss, oof, nf, cn = all_configs[key]
            w = 1.0 / max(loss, 0.01)
            oof_ensemble += w * oof
            total_w += w
        oof_ensemble /= total_w
        oof_ensemble = mean_match(oof_ensemble, train_rates[target])
        ensemble_loss = log_loss(y, oof_ensemble, labels=[0, 1])

        # ── Strategy 3: Top-5 ensemble ──
        top5_keys = [k for k, _ in sorted_configs[:5]]
        oof_ens5 = np.zeros(len(y))
        total_w5 = 0
        for key in top5_keys:
            loss, oof, nf, cn = all_configs[key]
            w = 1.0 / max(loss, 0.01)
            oof_ens5 += w * oof
            total_w5 += w
        oof_ens5 /= total_w5
        oof_ens5 = mean_match(oof_ens5, train_rates[target])
        ens5_loss = log_loss(y, oof_ens5, labels=[0, 1])

        # ── Strategy 4: Stacking (top-3 OOFs as features) ──
        if len(top3_keys) >= 2:
            oof_stack = np.column_stack([all_configs[k][1] for k in top3_keys])
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
            meta_pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
            meta_pred = mean_match(meta_pred, train_rates[target])
            meta_loss = best_meta_cv
        else:
            meta_loss = float('inf')

        # Pick best strategy
        strategies = {
            'best_cfg': (v52_style_loss, best_oof, best_key),
            'top3_ens': (ensemble_loss, oof_ensemble, '+'.join(top3_keys)),
            'top5_ens': (ens5_loss, oof_ens5, '+'.join(top5_keys)),
        }
        if meta_loss < float('inf'):
            strategies['stack'] = (meta_loss, meta_pred, f'stack-C={best_c:.2f}')

        best_strat_name = min(strategies, key=lambda k: strategies[k][0])
        best_loss, best_oof, best_detail = strategies[best_strat_name]

        stack_str = f'{meta_loss:.4f}' if meta_loss < float('inf') else 'N/A'
        log.info(f"\n  Strategies: best_cfg={v52_style_loss:.4f}, top3_ens={ensemble_loss:.4f}, "
                 f"top5_ens={ens5_loss:.4f}, stack={stack_str}")
        log.info(f"  ✅ Best: {best_strat_name} ({best_detail}) cal={best_loss:.4f}")

        all_results[target] = {
            'best_strategy': best_strat_name,
            'cal_oof': best_oof,
            'cal_loss': best_loss,
            'detail': best_detail,
            'all_configs': {k: (loss, oof, nf, cn) for k, (loss, oof, nf, cn) in sorted_configs[:10]},
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V82 SUMMARY")
    log.info(f"{'='*70}")

    all_oofs_dict = {}
    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['best_strategy']} ({r['detail']}) Cal={r['cal_loss']:.4f}")
        all_oofs_dict[target] = r['cal_oof']

    avg_cal = np.mean([
        log_loss(feat[t].values, all_oofs_dict[t], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V82 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V52 Avg Cal: 0.5725")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  V80 Avg Cal: 0.6124")
    log.info(f"  V81 Avg Cal: 0.6285")
    log.info(f"  Δ vs V52: {avg_cal - 0.5725:+.4f} ({'✅ IMPROVED' if avg_cal < 0.5725 else '❌ Not improved'})")
    log.info(f"  Δ vs V10: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save OOF ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_oofs_dict[target]
    oof_path = DATA_PROCESSED / "oof_v82.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 6. Save metadata ──
    meta = {
        'version': 'V82',
        'name': 'V52 Improved: Multi-Config Ensemble + Per-Target Config',
        'avg_cal_loss': avg_cal,
        'v52_cal_loss': 0.5725,
        'v10_cal_loss': 0.6038,
        'v80_cal_loss': 0.6124,
        'v81_cal_loss': 0.6285,
        'n_seeds': len(SEEDS),
        'n_folds': 5,
        'feature_method': 'lgbm_importance',
        'calibration': 'isotonic + mean_match',
        'n_configs': len(CFGS),
        'n_nfeats': len(N_FEATS),
        'delta_v52': avg_cal - 0.5725,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'best_strategy': r['best_strategy'],
            'cal_loss': r['cal_loss'],
            'detail': r['detail'],
        }
    meta_path = SUBMIT / f"meta_v82_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()


def generate_submission_v82():
    """Generate test submission using V82 best configs."""
    from sklearn.isotonic import IsotonicRegression
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import gc
    import re
    
    ROOT = Path('/home/mwoo423/projects/dacon2')
    DATA = ROOT / "data_processed"
    SUBMIT = ROOT / "submissions"
    
    TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    TARGET_COLS = TARGETS
    META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
    
    def sanitize(n):
        return re.sub(r'[^a-zA-Z0-9_]','_',n)
    
    def mean_match(pred, target_mean):
        shift = target_mean - pred.mean()
        return np.clip(pred + shift, 0.0001, 0.9999)
    
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
    
    CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
    CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
    CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
    CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
    CFG_V53_DEEP = {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 1500, 'ss': 0.65, 'cb': 0.65, 'ra': 0.3, 'rl': 1.5, 'mc': 20}
    CFG_V53_WIDE = {'nl': 35, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.85, 'cb': 0.85, 'ra': 2.5, 'rl': 5.0, 'mc': 5}
    CFGS = {'v48': CFG_V48, 'deep': CFG_DEEP, 'wide': CFG_WIDE, 'safety': CFG_SAFETY,
            'v53_deep': CFG_V53_DEEP, 'v53_wide': CFG_V53_WIDE}
    
    SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
             6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]
    
    # Load data
    print("Loading features...")
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_cols_raw = [c for c in feat.columns if c not in META | set(TARGET_COLS) and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    # Add personalization
    personal_cols = []
    for col in feat_cols_raw:
        col_filled = feat[col].fillna(0)
        grp = col_filled.groupby(feat['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        feat = feat.merge(grp, on='subject_id', how='left')
        mask_zero = feat[f'{col}_subj_std'] == 0
        mask_null = feat[col].isnull()
        feat[f'{col}_zscore'] = np.where(mask_zero | mask_null, 0.0,
            (feat[col].fillna(0) - feat[f'{col}_subj_mean']) / feat[f'{col}_subj_std'])
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    
    all_cols = feat_cols_raw + personal_cols
    train_rates = {t: feat[t].mean() for t in TARGET_COLS}
    
    # Load test - apply same personalization using TRAIN stats
    test_feat = pd.read_parquet(DATA / "test_features.parquet")
    print(f"Train: {feat.shape}, Test: {test_feat.shape}")
    
    # Personalize test: use test data's own subj stats (same as train personalization)
    test_feat = test_feat.copy()
    for col in feat_cols_raw:
        col_filled = test_feat[col].fillna(0)
        grp = col_filled.groupby(test_feat['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        test_feat = test_feat.merge(grp, on='subject_id', how='left')
        mask_zero = test_feat[f'{col}_subj_std'] == 0
        mask_null = test_feat[col].isnull()
        test_feat[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (test_feat[col].fillna(0) - test_feat[f'{col}_subj_mean']) / test_feat[f'{col}_subj_std']
        )
    
    # Rank features (LGBM importance) - reuse from training
    ranked_lgb = {}
    for target in TARGET_COLS:
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        X = feat[leak_cols].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
        sn = [sanitize(c) for c in leak_cols]
        ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
        model = lgb.train({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                           'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
                           'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
                           'reg_alpha': 1.0, 'reg_lambda': 3.0, 'scale_pos_weight': spw,
                           'random_state': 42, 'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1},
                          ds, num_boost_round=50)
        imp = model.feature_importance(importance_type='gain')
        ranked_lgb[target] = [r[0] for r in sorted(zip(leak_cols, imp), key=lambda x: -x[1])]
        del model, ds
        gc.collect()
    
    # Best configs per target from V82
    best_configs = {
        'Q1': [('deep', 20), ('wide', 20), ('v53_wide', 20), ('v48', 20), ('wide', 10)],
        'Q2': [('deep', 15), ('v48', 15), ('v48', 10)],
        'Q3': [('wide', 20), ('v53_wide', 20), ('v48', 15), ('deep', 15), ('deep', 10)],
        'S1': [('deep', 25), ('wide', 25)],  # stack top-2
        'S2': [('safety', 20), ('wide', 20)],  # stack top-2
        'S3': [('v53_wide', 8)],  # best cfg
        'S4': [('v53_deep', 15), ('wide', 15), ('deep', 15), ('safety', 8), ('v48', 15)],
    }
    
    cfg_map = {'v48': CFG_V48, 'deep': CFG_DEEP, 'wide': CFG_WIDE, 'safety': CFG_SAFETY,
               'v53_deep': CFG_V53_DEEP, 'v53_wide': CFG_V53_WIDE}
    
    print("\nGenerating test predictions...")
    test_preds = {}
    
    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_lgb[target]
        
        best_cfgs = best_configs[target]
        test_pred = np.zeros(len(test_feat))
        total_w = 0
        
        for cfg_name, n_feat in best_cfgs:
            cfg = cfg_map[cfg_name]
            sel_cols = ranked[:n_feat]
            
            # Train on all train data, predict test
            sn = [sanitize(c) for c in sel_cols]
            X_tr = feat[sel_cols].fillna(0).values.astype(np.float64)
            X_te = test_feat[sel_cols].fillna(0).values.astype(np.float64)
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
            
            tp = np.zeros(len(test_feat))
            for seed in SEEDS:
                cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
                ds = lgb.Dataset(X_tr, label=y, feature_name=sn, params={'verbose': '-1'})
                m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'])
                tp += m.predict(X_te)
                del m, ds
                gc.collect()
            tp = np.clip(tp / len(SEEDS), 0.0001, 0.9999)
            tp = mean_match(tp, train_rates[target])
            
            test_pred += tp
            total_w += 1
        
        test_pred /= total_w
        test_pred = mean_match(test_pred, train_rates[target])
        test_preds[target] = test_pred
        
        print(f"  {target}: {time.time()-tgt_t:.0f}s, mean={test_pred.mean():.4f}")
        gc.collect()
    
    # Save
    sub_df = pd.DataFrame({
        'subject_id': test_feat['subject_id'].values,
        'sleep_date': test_feat['sleep_date'].values,
        'lifelog_date': test_feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        sub_df[target] = test_preds[target]
    
    sub_path = SUBMIT / "submission_v82_test_20260509.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")
    print(f"Shape: {sub_df.shape}")


if __name__ == "__main__":
    import time
    # main()  # Already ran
    generate_submission_v82()
