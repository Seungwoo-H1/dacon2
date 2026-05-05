"""
V35 — XGB(3 seeds) + LGBM(5 seeds) simple blend

Uses precomputed features.parquet to skip slow 02_feature_engineering.
XGB only 3 seeds to save memory. LGBM 5 seeds.

Reference: V10(0.6038 avg cal)
"""

import sys, re, json, warnings, logging, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import TARGETS, DATA_PROCESSED, MODEL_DIR, SUBMIT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
TARGET_COLS = TARGETS
META_COLS = {"subject_id", "lifelog_date", "sleep_date", "date"}

LGB_SEEDS = [42, 123, 456, 789, 1024]
XGB_SEEDS = [42, 123, 456]
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
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

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

def run_lgb_cv(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]
    for si, s in enumerate(seeds):
        cfg = {
            'objective':'binary','metric':'binary_logloss','verbose':-1,
            'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
            'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_samples':10,'force_row_wise':True,'n_jobs':-1,
            'random_state': s, 'scale_pos_weight': spw,
        }
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose':'-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose':'-1'})
            m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
        gc_collect()
    return oof

def run_xgb_cv(feat, cols, target, seeds):
    y = feat[target].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    spw = ((y == 0).sum()) / max((y == 1).sum(), 1)
    oof = np.zeros((len(y), len(seeds)))
    for si, s in enumerate(seeds):
        params = {
            'objective':'binary:logistic','tree_method':'hist','n_estimators':500,
            'max_depth':4,'learning_rate':0.03,
            'subsample':0.7,'colsample_bytree':0.7,
            'reg_alpha':1.0,'reg_lambda':3.0,
            'min_child_weight':10,'scale_pos_weight':spw,
            'early_stopping_rounds':50,
            'verbosity':0,'n_jobs':-1,'random_state':s,
        }
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values
            X_va = feat.iloc[va][cols].fillna(0).values
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
            oof[va, si] = model.predict_proba(X_va)[:, 1]
            del model, X_tr, X_va
        gc_collect()
    return oof

def gc_collect():
    import gc; gc.collect()

def main():
    log.info("=" * 70)
    log.info("V35 — XGB(3 seeds) + LGBM(5 seeds) simple blend")
    log.info("=" * 70)

    # Load precomputed features
    feat_path = DATA_PROCESSED / "features.parquet"
    if feat_path.exists():
        log.info(f"Loading precomputed features: {feat_path}")
        feat = pd.read_parquet(feat_path)
    else:
        log.error(f"Precomputed features not found: {feat_path}")
        sys.exit(1)

    log.info(f"  Features loaded: {feat.shape}")

    all_feat_cols = get_feature_cols(feat)
    raw_base = [c for c in all_feat_cols
                if '_subj_mean' not in c and '_subj_std' not in c and '_zscore' not in c]
    log.info(f"  Base features: {len(raw_base)}")

    feat, personal_cols = add_personalization(feat, raw_base)
    log.info(f"  Personalization added: {len(personal_cols)}")

    total_cols = get_feature_cols(feat)
    total_cols = [c for c in total_cols if c not in META_COLS and c not in set(TARGET_COLS)]
    leak_all = LEAKAGE_S | LEAKAGE_Q
    available = [c for c in total_cols if c not in leak_all]
    log.info(f"  Available features: {len(available)}")

    log.info(f"\n=== Per-target tuning ===")
    all_cal = {}
    all_oof = {}

    for target in TARGET_COLS:
        log.info(f"\n--- {target} (rate={feat[target].mean():.3f}) ---")
        train_rate = feat[target].mean()

        ranked = rank_features(feat, available, target, seed=42)
        sel_cols = [c for c, _ in ranked[:20]]
        log.info(f"  Selected top-20 features")

        # LGBM
        log.info("  Training LGBM (5 seeds)...")
        oof_lgb = run_lgb_cv(feat, sel_cols, target, LGB_SEEDS)
        oof_lgb_avg = np.clip(oof_lgb.mean(axis=1), 0.0001, 0.9999)
        del oof_lgb; gc_collect()

        # XGB
        log.info("  Training XGB (3 seeds)...")
        oof_xgb = run_xgb_cv(feat, sel_cols, target, XGB_SEEDS)
        oof_xgb_avg = np.clip(oof_xgb.mean(axis=1), 0.0001, 0.9999)
        del oof_xgb; gc_collect()

        # Find best blend weight
        best_w = 0.5; best_loss = float('inf')
        for w in np.arange(0.0, 1.05, 0.05):
            blend = w * oof_xgb_avg + (1-w) * oof_lgb_avg
            loss = log_loss(feat[target], blend, labels=[0,1])
            if loss < best_loss:
                best_loss = loss; best_w = w

        cal_blend = mean_match(best_w * oof_xgb_avg + (1-best_w) * oof_lgb_avg, train_rate)
        cal_loss = log_loss(feat[target], cal_blend, labels=[0,1])
        oof_blend = best_w * oof_xgb_avg + (1-best_w) * oof_lgb_avg

        all_oof[target] = oof_blend
        all_cal[target] = cal_blend

        oof_lgb_loss = log_loss(feat[target], oof_lgb_avg, labels=[0,1])
        oof_xgb_loss = log_loss(feat[target], oof_xgb_avg, labels=[0,1])
        log.info(f"  LGBM: {oof_lgb_loss:.4f}  XGB: {oof_xgb_loss:.4f}  Blend(w={best_w:.2f}): {best_loss:.4f}  Cal: {cal_loss:.4f}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V35 SUMMARY")
    for t in TARGET_COLS:
        oof_l = log_loss(feat[t], all_oof[t], labels=[0,1])
        cal_l = log_loss(feat[t], all_cal[t], labels=[0,1])
        log.info(f"  {t}: OOF={oof_l:.4f} Cal={cal_l:.4f}")

    avg_cal = np.mean([log_loss(feat[t], all_cal[t], labels=[0,1]) for t in TARGET_COLS])
    avg_oof = np.mean([log_loss(feat[t], all_oof[t], labels=[0,1]) for t in TARGET_COLS])
    log.info(f"\n  Avg OOF:  {avg_oof:.4f}")
    log.info(f"  Avg Cal:  {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Delta: {0.6038 - avg_cal:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")

if __name__ == "__main__":
    main()
