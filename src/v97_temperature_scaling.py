"""
V97: V53 Swept + Temperature Scaling Calibration
Instead of isotonic (overfits), use temperature scaling:
1. Train isotonic on OOF to get a mapping
2. Fit a temperature T to minimize log_loss(OOF/T) vs labels
3. Apply same T to test predictions
This is more robust than full isotonic regression.

Also try: V53 original + alternative seed ranges
Also try: stacking V53 with different configs
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
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

def temperature_scale(probs, T):
    """Apply temperature scaling to probabilities."""
    probs = np.clip(probs, 1e-7, 1-1e-7)
    logit = np.log(probs / (1 - probs))
    scaled_logit = logit / T
    return 1 / (1 + np.exp(-scaled_logit))

def fit_temperature(oof_preds, y_true):
    """Find optimal temperature T that minimizes log_loss."""
    def loss(T):
        scaled = temperature_scale(oof_preds, T)
        return log_loss(y_true, scaled, labels=[0,1])
    result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
    return result.x, result.fun

def main():
    t_start = time.time()
    log.info("="*80)
    log.info("V97: V53 Swept + Temperature Scaling")
    log.info("="*80)

    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    train_cols_order = list(train.columns)
    test = test[train_cols_order]
    log.info(f"  Train: {train.shape}, Test: {test.shape}")

    feat_cols = get_feature_cols(train)
    base_cols = [c for c in feat_cols if not c.endswith('_zscore') and '_x_' not in c]
    train_p, zscore_cols = add_personalization(train, base_cols)
    test_p, _ = add_personalization(test, base_cols)
    all_cols = base_cols + zscore_cols
    log.info(f"  Features: {len(base_cols)} base + {len(zscore_cols)} zscore = {len(all_cols)}")

    n_seeds = 50
    gkf = GroupKFold(n_splits=5)

    # Experiments:
    # 1. V53 Swept + temperature scaling (fit T on OOF, apply to test)
    # 2. V53 Swept + mean shift (baseline)
    # 3. V53 Swept + both (blend)

    cal_results = {}

    for cal_method in ['linear', 'temperature']:
        log.info(f"\n{'#'*60}")
        log.info(f"Calibration: {cal_method}")
        log.info(f"{'#'*60}")

        target_results = {}
        for target in TARGETS:
            t0 = time.time()
            cfg_name = V53_SWEEP[target]['cfg']
            n_feat = V53_SWEEP[target]['n_feat']
            cfg = CFGS[cfg_name]

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

            # OOF predictions
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

            if cal_method == 'linear':
                shift = train_rate - oof_preds.mean()
                cal_oof = np.clip(oof_preds + shift, 0.0001, 0.9999)
                cal_test = np.clip(test_preds + shift, 0.0001, 0.9999)
                T = 1.0
            else:
                # Temperature scaling
                T, cal_loss = fit_temperature(oof_preds, y)
                cal_oof = np.clip(temperature_scale(oof_preds, T), 0.0001, 0.9999)
                cal_test = np.clip(temperature_scale(test_preds, T), 0.0001, 0.9999)

            cv_loss = log_loss(y, oof_preds, labels=[0,1])
            cal_oof_loss = log_loss(y, cal_oof, labels=[0,1])

            target_results[target] = {
                'cal_oof': round(cal_oof_loss, 6),
                'cv': round(cv_loss, 6),
                'train_rate': round(train_rate, 6),
                'oof_mean': round(oof_preds.mean(), 6),
                'test_mean': round(cal_test.mean(), 6),
                'shift': round(cal_test.mean()-train_rate, 6),
                'T': round(T, 4),
                'n_feat': n_feat, 'cfg': cfg_name, 'time_s': round(time.time()-t0, 0),
            }
            log.info(f"  {target}: CV={cv_loss:.4f} CalOOF={cal_oof_loss:.4f} T={T:.3f} "
                     f"oof_mean={oof_preds.mean():.4f} test_mean={cal_test.mean():.4f} "
                     f"shift={cal_test.mean()-train_rate:+.4f}")

            del X, Xts, oof_preds, seed_preds, m_rank, ds
            gc.collect()

        avg_cal = np.mean([v['cal_oof'] for v in target_results.values()])
        cal_results[cal_method] = target_results
        log.info(f"\n  AVG CalOOF ({cal_method}): {avg_cal:.4f}")

    # Summary
    log.info("\n" + "="*80)
    log.info("V97 SUMMARY: Calibration Method Comparison")
    log.info("="*80)
    for method, results in cal_results.items():
        avg = np.mean([v['cal_oof'] for v in results.values()])
        log.info(f"  {method}: AVG CalOOF={avg:.4f}")

    log.info(f"Total time: {time.time()-t_start:.0f}s")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(SUBMIT / f'v97_results_{ts}.json', 'w') as f:
        json.dump(cal_results, f, indent=2, default=str)
    log.info(f"  Results saved: v97_results_{ts}.json")


if __name__ == "__main__":
    main()
