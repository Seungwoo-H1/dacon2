"""
V98: Root Cause Analysis + Alternative Approach

Temperature scaling (V97) failed because:
1. High temperature (T=3.246 for Q1, T=2.260 for S2) crushed prediction variance
2. OOF variance ≠ test variance → temperature fit on OOF doesn't transfer

Key insight: Temperature scaling compresses the prediction distribution.
If the test set has MORE separation than OOF, compression hurts.

New approach: V53 Swept with careful calibration that PRESERVES variance.
Try: mean-shift calibration with variance preservation.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
SUBMIT.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety','n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide':   {'nl':30, 'md':3, 'lr':0.05, 'ne':300, 'ss':0.8, 'cb':0.8, 'ra':2.0, 'rl':5.0, 'mc':5},
    'deep':   {'nl':20, 'md':5, 'lr':0.02, 'ne':1000,'ss':0.7, 'cb':0.6, 'ra':0.5, 'rl':2.0, 'mc':15},
    'v48':    {'nl':15, 'md':4, 'lr':0.03, 'ne':500, 'ss':0.7, 'cb':0.7, 'ra':1.0, 'rl':3.0, 'mc':10},
    'safety': {'nl':10, 'md':3, 'lr':0.02, 'ne':1000,'ss':0.6, 'cb':0.6, 'ra':3.0, 'rl':10.0,'mc':20},
}

def sanitize(n): return re.sub(r'[^a-zA-Z0-9_]','_',n)
def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols
def get_feature_cols(df):
    return [c for c in df.columns if c not in META | set(TARGETS) and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
def add_personalization(df, feature_cols):
    df = df.copy(); zscore_cols = []; batch_size = 50
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]; agg_parts = []
        for col in batch:
            col_filled = df[col].fillna(0)
            grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
            grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']; grp = grp.reset_index(); agg_parts.append(grp)
        agg_df = agg_parts[0]
        for part in agg_parts[1:]: agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}; zscore_cols = []
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]
        for col in batch:
            zc = f'{col}_zscore'; mean_c = f'{col}_subj_mean'; std_c = f'{col}_subj_std'
            zcols_dict[zc] = np.where((df[std_c]==0)|df[col].isnull(), 0.0, (df[col].fillna(0)-df[mean_c])/df[std_c])
            zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index); df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols

def main():
    t_start = time.time()
    log.info("="*80)
    log.info("V98: Root Cause Analysis + V53 Baseline Reproduction")
    log.info("="*80)

    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    train_cols_order = list(train.columns)
    test = test[train_cols_order]

    feat_cols = get_feature_cols(train)
    base_cols = [c for c in feat_cols if not c.endswith('_zscore') and '_x_' not in c]
    train_p, zscore_cols = add_personalization(train, base_cols)
    test_p, _ = add_personalization(test, base_cols)
    all_cols = base_cols + zscore_cols

    n_seeds = 50
    gkf = GroupKFold(n_splits=5)

    # V98-1: Reproduce V53 Swept baseline (linear cal) to verify
    log.info("\n=== V98-1: V53 Swept Reproduction (linear cal) ===")
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]['cfg']
        n_feat = V53_SWEEP[target]['n_feat']
        cfg = CFGS[cfg_name]
        t0 = time.time()

        leak_cols = remove_leak(all_cols, target)
        y = train_p[target].values.astype(np.float64)
        train_rate = float(y.mean())
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)

        X_all = train_p[leak_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in leak_cols]

        rank_params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
            'n_estimators':50,'scale_pos_weight':spw,'random_state':42,
            'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
            'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'min_child_samples':cfg['mc']}
        ds = lgb.Dataset(X_all, label=y, feature_name=sn)
        m_rank = lgb.train(rank_params, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked[:n_feat]]
        sn_sel = [sanitize(c) for c in sel_cols]

        X = train_p[sel_cols].fillna(0).values.astype(np.float64)
        Xts = test_p[sel_cols].fillna(0).values.astype(np.float64)

        # OOF
        oof_preds = np.zeros(len(y))
        seed_preds = []
        for seed in range(42, 42+n_seeds):
            oof_fold = np.zeros(len(y))
            for fold_i, (tr_i, va_i) in enumerate(gkf.split(X, y, train_p['subject_id'])):
                p_tr = {k:v for k,v in rank_params.items() if k not in ('n_estimators','random_state')}
                p_tr.update({'n_estimators':cfg['ne'],'random_state':seed})
                ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel)
                m = lgb.train(p_tr, ds_tr, num_boost_round=cfg['ne'])
                oof_fold[va_i] += m.predict(X[va_i])
            oof_preds += oof_fold

            ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
            p_all = {k:v for k,v in rank_params.items() if k not in ('n_estimators','random_state')}
            p_all.update({'n_estimators':cfg['ne'],'random_state':seed})
            m = lgb.train(p_all, ds_all, num_boost_round=cfg['ne'])
            seed_preds.append(m.predict(Xts))

        oof_preds /= n_seeds
        test_preds = np.mean(seed_preds, axis=0)

        # Linear calibration: shift mean to train_rate
        shift = train_rate - oof_preds.mean()
        cal_oof = np.clip(oof_preds + shift, 0.0001, 0.9999)
        cal_test = np.clip(test_preds + shift, 0.0001, 0.9999)

        cv_loss = log_loss(y, oof_preds, labels=[0,1])
        cal_oof_loss = log_loss(y, cal_oof, labels=[0,1])

        log.info(f"  {target}: CV={cv_loss:.4f} CalOOF={cal_oof_loss:.4f} "
                 f"oof_std={oof_preds.std():.4f} test_std={test_preds.std():.4f} "
                 f"oof_mean={oof_preds.mean():.4f} test_mean={test_preds.mean():.4f}")

        del X, Xts, oof_preds, seed_preds, m_rank, ds
        gc.collect()

    # V98-2: Try variance-preserving calibration
    # Key idea: instead of compressing via temperature, 
    # fit per-sample correction using OOF quantile mapping
    log.info("\n=== V98-2: Quantile-based calibration ===")
    for target in TARGETS:
        cfg_name = V53_SWEEP[target]['cfg']
        n_feat = V53_SWEEP[target]['n_feat']
        cfg = CFGS[cfg_name]
        t0 = time.time()

        leak_cols = remove_leak(all_cols, target)
        y = train_p[target].values.astype(np.float64)
        train_rate = float(y.mean())

        X_all = train_p[leak_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in leak_cols]

        rank_params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
            'n_estimators':50,'scale_pos_weight':train_rate/(1-train_rate+1e-8),'random_state':42,
            'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
            'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'min_child_samples':cfg['mc']}
        ds = lgb.Dataset(X_all, label=y, feature_name=sn)
        m_rank = lgb.train(rank_params, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked[:n_feat]]
        sn_sel = [sanitize(c) for c in sel_cols]

        X = train_p[sel_cols].fillna(0).values.astype(np.float64)
        Xts = test_p[sel_cols].fillna(0).values.astype(np.float64)

        # OOF
        oof_preds = np.zeros(len(y))
        seed_preds = []
        for seed in range(42, 42+n_seeds):
            oof_fold = np.zeros(len(y))
            for fold_i, (tr_i, va_i) in enumerate(gkf.split(X, y, train_p['subject_id'])):
                p_tr = {k:v for k,v in rank_params.items() if k not in ('n_estimators','random_state')}
                p_tr.update({'n_estimators':cfg['ne'],'random_state':seed})
                ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel)
                m = lgb.train(p_tr, ds_tr, num_boost_round=cfg['ne'])
                oof_fold[va_i] += m.predict(X[va_i])
            oof_preds += oof_fold

            ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
            p_all = {k:v for k,v in rank_params.items() if k not in ('n_estimators','random_state')}
            p_all.update({'n_estimators':cfg['ne'],'random_state':seed})
            m = lgb.train(p_all, ds_all, num_boost_round=cfg['ne'])
            seed_preds.append(m.predict(Xts))

        oof_preds /= n_seeds
        test_preds = np.mean(seed_preds, axis=0)

        # Quantile mapping: preserve ranks, adjust distribution
        # Map OOF predictions to match train distribution quantiles
        from scipy.stats import rankdata
        
        # OOF rank-based: sort OOF, map to train_rate-based expected values
        n = len(y)
        # Expected value of k-th order statistic from Beta(train_rate*(n), (1-train_rate)*(n))
        # Simpler: use rank to map OOF -> train expected values
        
        oof_ranks = rankdata(oof_preds) / n  # [0, 1]
        
        # Use a simple approach: map OOF rank to linear interpolation of train probs
        # For each OOF prediction, its rank determines its "strength" relative to train
        # Map: rank[i] -> expected value given that rank
        # Simple: use the OOF CDF values as the calibration
        cal_oof = oof_ranks * 0.99 + 0.005  # preserve rank order, map to [0.005, 0.995]
        
        # Alternative: keep original OOF ranks but scale to match train rate mean
        cal_oof2 = np.interp(oof_preds, np.sort(oof_preds), np.sort(np.full(n, train_rate)))
        
        # Best approach: keep original OOF, just clip and adjust extreme values
        # This is V53's approach but with better extreme value handling
        shift = train_rate - oof_preds.mean()
        cal_oof3 = np.clip(oof_preds + shift, 0.001, 0.999)
        
        cv3 = log_loss(y, cal_oof3, labels=[0,1])
        log.info(f"  {target}: CV={cv3:.4f} test_mean={cal_oof3.mean():.4f} oof_std={oof_preds.std():.4f} test_std={test_preds.std():.4f}")

        del X, Xts, oof_preds, seed_preds, m_rank, ds
        gc.collect()

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
