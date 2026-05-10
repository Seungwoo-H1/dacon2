"""
V83 — V52 Improvement: More Diversity + Finer Sweep + Better Ensemble

From V52 (0.5725) → V82 (0.5654):
- Added v53 configs + multi-config ensemble + stacking

V83 improvements over V82:
1. 8 configs (6 + aggr_deep, ultra_wide) for more diversity
2. 7 n_feats (5,8,10,15,20 + 3,12,25) for finer sweep
3. Better strategy search: single, top-3, top-5, top-7, top-10, stack_LR, stack_Ridge, stack_KRR
4. Wider LR C search (15 values)
5. Same calibration: isotonic + mean_match

Strategy: same 20 seeds, same GroupKFold 5-fold, same leakage handling
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
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


# ── 8 configs ──
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_V53_DEEP = {'nl': 25, 'md': 6, 'lr': 0.015, 'ne': 1500, 'ss': 0.65, 'cb': 0.65, 'ra': 0.3, 'rl': 1.5, 'mc': 20}
CFG_V53_WIDE = {'nl': 35, 'md': 3, 'lr': 0.04, 'ne': 400, 'ss': 0.85, 'cb': 0.85, 'ra': 2.5, 'rl': 5.0, 'mc': 5}
CFG_AGGR_DEEP = {'nl': 30, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.6, 'cb': 0.55, 'ra': 0.2, 'rl': 1.0, 'mc': 25}
CFG_ULTRA_WIDE = {'nl': 40, 'md': 3, 'lr': 0.06, 'ne': 200, 'ss': 0.9, 'cb': 0.9, 'ra': 3.0, 'rl': 8.0, 'mc': 3}

CFGS = {
    'v48': CFG_V48, 'deep': CFG_DEEP, 'wide': CFG_WIDE, 'safety': CFG_SAFETY,
    'v53_deep': CFG_V53_DEEP, 'v53_wide': CFG_V53_WIDE,
    'aggr_deep': CFG_AGGR_DEEP, 'ultra_wide': CFG_ULTRA_WIDE,
}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

N_FEATS = [5, 8, 10, 12, 15, 20, 25]


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


def train_cv_model(feat, cols, y, seeds, cfg, n_folds=5):
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
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(pred, y_true)
        return iso.predict(pred), True
    except Exception:
        return pred, False


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V83 — V52 Improvement: 8 configs × 7 n_feats × 20 seeds")
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
    ranked_lgb = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        ranked = rank_features_importance(feat, leak_cols, target)
        ranked_lgb[target] = ranked
        log.info(f"  {target}: {len(leak_cols)} features ranked")

    # ── 3. Training ──
    log.info(f"\n--- 3. Training: {len(CFGS)} configs × {len(N_FEATS)} n_feats × {len(SEEDS)} seeds = {len(CFGS)*len(N_FEATS)*len(SEEDS)} models/target ---")

    all_results = {}

    for ti, target in enumerate(TARGET_COLS):
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_lgb[target]

        log.info(f"\n  === {target} ({ti+1}/7, rate={train_rates[target]:.3f}) ===")

        all_configs = {}

        for cfg_name, cfg in CFGS.items():
            for n_feat in N_FEATS:
                sel_cols = ranked[:n_feat]
                oof = train_cv_model(feat, sel_cols, y, SEEDS, cfg, n_folds=5)

                iso_cal, ok = isotonic_calibrate(oof, y)
                if ok:
                    iso_cal = mean_match(iso_cal, train_rates[target])
                    loss = log_loss(y, iso_cal, labels=[0, 1])
                else:
                    loss = log_loss(y, oof, labels=[0, 1])
                    iso_cal = oof

                key = f"{cfg_name}_n{n_feat}"
                all_configs[key] = (loss, iso_cal, n_feat, cfg_name)

        sorted_cfgs = sorted(all_configs.items(), key=lambda x: x[1][0])
        log.info(f"  Top 5 configs:")
        for k, (l, o, nf, cn) in sorted_cfgs[:5]:
            log.info(f"    {k}: cal={l:.4f}")

        # ── Strategies ──
        strategies = {}

        # Single best (V52 style)
        s1_loss, s1_oof, s1_nf, s1_cn = sorted_cfgs[0][1]
        strategies['single'] = (s1_loss, s1_oof, sorted_cfgs[0][0])

        # Top-N ensembles
        for top_n in [3, 5, 7, 10]:
            top_k = [k for k, _ in sorted_cfgs[:min(top_n, len(sorted_cfgs))]]
            oof_ens = np.zeros(len(y))
            tw = 0
            for k in top_k:
                loss, oof, nf, cn = all_configs[k]
                w = 1.0 / max(loss, 0.01)
                oof_ens += w * oof
                tw += w
            oof_ens /= tw
            oof_ens = mean_match(oof_ens, train_rates[target])
            loss = log_loss(y, oof_ens, labels=[0, 1])
            strategies[f'top{top_n}_ens'] = (loss, oof_ens, '+'.join(top_k))

        # Stacking: LR with wide C search
        top5 = [k for k, _ in sorted_cfgs[:5]]
        oof_stack = np.column_stack([all_configs[k][1] for k in top5])
        for c_val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=5000, random_state=42)
            meta.fit(oof_stack, y)
            pred = np.clip(meta.predict_proba(oof_stack)[:, 1], 0.0001, 0.9999)
            pred = mean_match(pred, train_rates[target])
            ml = log_loss(y, pred)
            strategies[f'stack_LR_C{c_val}'] = (ml, pred, f'LR-C={c_val}')

        # Stacking: LR with top-7
        top7 = [k for k, _ in sorted_cfgs[:7]]
        oof_stack7 = np.column_stack([all_configs[k][1] for k in top7])
        for c_val in [0.01, 0.1, 1.0, 10.0, 50.0]:
            meta = LogisticRegression(C=c_val, solver='lbfgs', max_iter=5000, random_state=42)
            meta.fit(oof_stack7, y)
            pred = np.clip(meta.predict_proba(oof_stack7)[:, 1], 0.0001, 0.9999)
            pred = mean_match(pred, train_rates[target])
            ml = log_loss(y, pred)
            strategies[f'stack_LR7_C{c_val}'] = (ml, pred, f'LR7-C={c_val}')

        # Stacking: KRR with top-5
        for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            meta = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
            meta.fit(oof_stack, y)
            pred = np.clip(meta.predict(oof_stack), 0.0001, 0.9999)
            pred = mean_match(pred, train_rates[target])
            ml = log_loss(y, pred)
            strategies[f'stack_KRR_g{gamma}'] = (ml, pred, f'KRR-gamma={gamma}')

        # Pick best
        best_name = min(strategies, key=lambda k: strategies[k][0])
        best_loss, best_oof, best_detail = strategies[best_name]

        # Report
        sorted_strats = sorted(strategies.items(), key=lambda x: x[1][0])
        log.info(f"  Best: {best_name} ({best_detail}) cal={best_loss:.4f}")
        log.info(f"  Top strategies:")
        for sn, (sl, so, sd) in sorted_strats[:5]:
            log.info(f"    {sn}: cal={sl:.4f}")

        all_results[target] = {
            'best_strategy': best_name,
            'cal_oof': best_oof,
            'cal_loss': best_loss,
            'detail': best_detail,
            'sorted_cfgs': sorted_cfgs,
            'all_configs': all_configs,
        }

        log.info(f"  {target} time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Summary ──
    log.info(f"\n{'='*70}")
    log.info("V83 SUMMARY")
    log.info(f"{'='*70}")

    all_oofs_dict = {}
    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['best_strategy']} cal={r['cal_loss']:.4f} ({r['detail']})")
        all_oofs_dict[target] = r['cal_oof']

    avg_cal = np.mean([
        log_loss(feat[t].values, all_oofs_dict[t], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V83 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V52: 0.5725 | V82: 0.5654 | V10: 0.6038")
    log.info(f"  Δ vs V52: {avg_cal - 0.5725:+.4f} ({'✅ IMPROVED' if avg_cal < 0.5725 else '❌'})")
    log.info(f"  Δ vs V82: {avg_cal - 0.5654:+.4f} ({'✅ IMPROVED' if avg_cal < 0.5654 else '❌'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 5. Save ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_oofs_dict[target]
    oof_path = DATA_PROCESSED / "oof_v83.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    meta = {
        'version': 'V83',
        'name': 'V52 Improvement: 8 configs × 7 n_feats × 20 seeds, multi-strategy ensemble',
        'avg_cal_loss': avg_cal,
        'v52_cal_loss': 0.5725,
        'v82_cal_loss': 0.5654,
        'v10_cal_loss': 0.6038,
        'n_seeds': len(SEEDS),
        'n_configs': len(CFGS),
        'n_nfeats': len(N_FEATS),
        'configs': list(CFGS.keys()),
        'nfeats': N_FEATS,
        'delta_v52': avg_cal - 0.5725,
        'delta_v82': avg_cal - 0.5654,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'best_strategy': r['best_strategy'],
            'cal_loss': r['cal_loss'],
            'detail': r['detail'],
        }
    meta_path = SUBMIT / f"meta_v83_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata: {meta_path}")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
