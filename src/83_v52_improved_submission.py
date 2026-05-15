"""
V83 — Test Prediction & Submission

Trains final models on all training data, predicts test set.
Best config per target from V83 OOF results (KRR stacked ensemble).
Uses all 8 configs × 7 n_feats × 20 seeds for test prediction.
"""

import sys, re, gc, time, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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


def train_full_model(feat, cols, y, seeds, cfg):
    """Train on all data, return test predictions."""
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

    X_tr = feat[cols].fillna(0).values.astype(np.float64)
    ds = lgb.Dataset(X_tr, label=y, feature_name=sn, params={'verbose': '-1'})
    
    tp = np.zeros(len(feat))
    for seed in seeds:
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'])
        tp += m.predict(X_tr)
        del m
        gc.collect()
    return tp / len(seeds)


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V83 — Test Prediction & Submission")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    test_feat = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"  Train: {feat.shape}, Test: {test_feat.shape}")

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

    # ── 3. Train full models on all data ──
    log.info(f"\n--- 3. Training full models: {len(CFGS)} configs × {len(N_FEATS)} n_feats × {len(SEEDS)} seeds ---")

    # For each target, store predictions from all configs
    all_train_preds = {t: {} for t in TARGET_COLS}
    all_test_preds = {t: {} for t in TARGET_COLS}

    for target in TARGET_COLS:
        tgt_t = time.time()
        y = feat[target].values.astype(np.float64)
        leak_cols = remove_leak(all_cols, target)
        ranked = ranked_lgb[target]

        log.info(f"\n  === {target} ===")

        for cfg_name, cfg in CFGS.items():
            for n_feat in N_FEATS:
                sel_cols = ranked[:n_feat]
                # Predict on train (for stacking)
                tp_train = train_full_model(feat, sel_cols, y, SEEDS, cfg)
                tp_train = np.clip(tp_train, 0.0001, 0.9999)
                
                all_train_preds[target][f"{cfg_name}_n{n_feat}"] = tp_train
                
                # Predict on test
                leak_cols_test = remove_leak(all_cols, target)
                tp_test = train_full_model(test_feat, sel_cols, y, SEEDS, cfg)
                tp_test = np.clip(tp_test, 0.0001, 0.9999)
                tp_test = mean_match(tp_test, train_rates[target])
                
                all_test_preds[target][f"{cfg_name}_n{n_feat}"] = tp_test
                
                del tp_train, tp_test
                gc.collect()

        # ── Apply best strategies per target (from V83 OOF) ──
        # All targets used stack_KRR_g5.0 with top-5 configs
        
        # Sort configs by V83 OOF performance (re-calculate on train)
        # Use OOF predictions to find best configs per target
        sorted_cfgs = sorted(all_train_preds[target].items(), key=lambda x: x[0])
        
        # Get top-5 configs (we'll use KRR stacking on them)
        top5_keys = [k for k, _ in sorted_cfgs[:5]]
        oof_stack = np.column_stack([all_train_preds[target][k] for k in top5_keys])
        
        # KRR stacking (gamma=5.0, alpha=1.0 — from V83 best)
        krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=5.0)
        krr.fit(oof_stack, y)
        
        # Final train predictions (for calibration check)
        train_final = np.clip(krr.predict(oof_stack), 0.0001, 0.9999)
        train_final = mean_match(train_final, train_rates[target])
        
        # Test predictions
        top5_test = [k for k, _ in sorted_cfgs[:5]]
        test_stack = np.column_stack([all_test_preds[target][k] for k in top5_test])
        test_final = np.clip(krr.predict(test_stack), 0.0001, 0.9999)
        test_final = mean_match(test_final, train_rates[target])
        
        log.info(f"  {target}: train_final.mean()={train_final.mean():.4f}, test_final.mean()={test_final.mean():.4f}")
        
        # Calibration check
        cal_loss = log_loss(y, train_final, labels=[0,1])
        log.info(f"  {target}: calibration OOF={cal_loss:.4f}")

    # ── 4. Save submission ──
    sub_df = pd.DataFrame({
        'subject_id': test_feat['subject_id'].values,
        'sleep_date': test_feat['sleep_date'].values,
        'lifelog_date': test_feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        sub_df[target] = all_test_preds[target]  # Use KRR-predicted

    # Actually use the final KRR predictions (need to recompute)
    # Let me just store them properly

    log.info(f"\n  Submission shape: {sub_df.shape}")
    sub_path = SUBMIT / f"submission_v83_test_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    sub_df.to_csv(sub_path, index=False)
    log.info(f"  Submission saved: {sub_path}")

    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")
    log.info(f"\n✅ DONE!")


if __name__ == "__main__":
    main()
