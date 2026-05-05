"""
V36 — Stacking with LGBM (4 variants, 5 seeds)

Uses precomputed features.parquet.
No XGB — only LGBM to save memory.

Reference: V10(0.6038)
"""

import sys, re, json, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
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

SEEDS = [42, 123, 456, 789, 1024]
N_SPLITS = 5

LEAKAGE_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    return np.clip(pred + (target_mean - pred.mean()), 0.0001, 0.9999)

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        subj_stats = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        subj_stats.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        subj_stats = subj_stats.reset_index()
        df = df.merge(subj_stats, on='subject_id', how='left')
        mask_std_zero = (df[f'{col}_subj_std'] == 0)
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(mask_std_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std'])
        personal_cols.append(f'{col}_zscore')
    return df, personal_cols

def rank_features(feat, feature_cols, target, seed=42):
    y = feat[target].values
    X = feat[feature_cols].fillna(0).values
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':50,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
              'scale_pos_weight':spw,'random_state':seed,'min_child_samples':10,
              'force_row_wise':True,'n_jobs':-1}
    sn = [sanitize(c) for c in feature_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose':'-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type="gain")
    return sorted(zip(feature_cols, imp), key=lambda x: -x[1])

def run_lgb_cv(feat, cols, target, seeds, cfg):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]
    for si, s in enumerate(seeds):
        params = {**cfg, 'random_state': s, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'], valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
    return oof

def main():
    log.info("=" * 70)
    log.info("V36 — LGBM Stacking (4 variants × 5 seeds)")
    log.info("=" * 70)

    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Features: {feat.shape}")

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    feat, personal_cols = add_personalization(feat, raw_base)

    total_cols = get_feature_cols(feat)
    total_cols = [c for c in total_cols if c not in META_COLS and c not in set(TARGET_COLS)]
    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in total_cols if c not in leak_all]

    log.info(f"  Features: base={len(raw_base)} + personalization={len(personal_cols)}, available={len(available)}")

    LGB_CFGS = [
        {'name':'C1','num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1},
        {'name':'C2','num_leaves':10,'max_depth':3,'learning_rate':0.03,'n_estimators':300,'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,'verbose':-1},
        {'name':'C3','num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':300,'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_samples':8,'force_row_wise':True,'n_jobs':-1,'verbose':-1},
        {'name':'C4','num_leaves':6,'max_depth':2,'learning_rate':0.02,'n_estimators':200,'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':5.0,'reg_lambda':10.0,'min_child_samples':25,'force_row_wise':True,'n_jobs':-1,'verbose':-1},
    ]

    all_cal = {}
    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()

        ranked = rank_features(feat, available, target, seed=42)
        sel_cols = [c for c, _ in ranked[:20]]

        # Train 4 LGBM variants
        oofs = {}
        for cfg in LGB_CFGS:
            name = cfg['name']
            log.info(f"  Training {name}...")
            oofs[name] = run_lgb_cv(feat, sel_cols, target, SEEDS, cfg)
            oofs[name] = np.clip(oofs[name].mean(axis=1), 0.0001, 0.9999)
            loss = log_loss(feat[target], oofs[name], labels=[0,1])
            log.info(f"    {name}: {loss:.4f}")

        # Build meta features
        meta_X = np.column_stack([oofs[c] for c in LGB_CFGS])
        y = feat[target].values

        # Meta-learner via LOSO
        gkf = GroupKFold(n_splits=N_SPLITS)
        meta_oof = np.zeros(len(y))
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            meta_model = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
            meta_model.fit(meta_X[tr], y[tr])
            meta_oof[va] = meta_model.predict_proba(meta_X[va])[:, 1]

        cal = mean_match(meta_oof, train_rate)
        cal_loss = log_loss(y, cal, labels=[0,1])
        all_cal[target] = cal

        # Compare with best single
        best_single = min(log_loss(y, oofs[c], labels=[0,1]) for c in LGB_CFGS)
        log.info(f"  Stacking: {cal_loss:.4f}  Best single: {best_single:.4f}")

    log.info(f"\n{'='*70}")
    log.info("V36 SUMMARY")
    for t in TARGET_COLS:
        cal_l = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"  {t}: Cal={cal_l:.4f}")
    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Delta: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

if __name__ == "__main__":
    main()
