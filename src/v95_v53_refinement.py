"""
V95: V53 Swept Replicate + Calibration Refinement
Goal: Reproduce V53 Swept's LB success (0.65358) and try to improve.

Key hypotheses:
1. V53 Swept's calibration style + n_feat swept combo works well on LB
2. V10 OOF 0.6038 vs V53 CV 0.6813 → V53 Swept has worse OOF but better LB
3. This suggests V53 Swept's calibration is better aligned with test distribution

Approach:
- Reproduce V53 Swept exactly (same seed range, same cfg, same n_feat)
- Try: isotonic calibration instead of linear shift
- Try: more seeds (100 instead of 50)
- Try: different random seed ranges
- Try: stacking multiple V53 Swept with different seed sets
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

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

# V53 Swept optimal config
V53_SWEEP = {
    'Q1':  {'cfg': 'deep', 'nf': 19},
    'Q2':  {'cfg': 'deep', 'nf': 14},
    'Q3':  {'cfg': 'v48', 'nf': 11},
    'S1':  {'cfg': 'wide', 'nf': 21},
    'S2':  {'cfg': 'deep', 'nf': 19},
    'S3':  {'cfg': 'safety','nf': 23},
    'S4':  {'cfg': 'wide', 'nf': 20},
}

CFGS = {
    'wide':   {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep':   {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

def build_lgb_params(cfg, **extras):
    params = {
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }
    params.update(extras)
    return params

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    df = df.copy(); zscore_cols = []; batch_size = 50
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]; agg_parts = []
        for col in batch:
            col_filled = df[col].fillna(0)
            grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
            grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
            grp = grp.reset_index(); agg_parts.append(grp)
        agg_df = agg_parts[0]
        for part in agg_parts[1:]: agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}; zscore_cols = []
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]
        for col in batch:
            zc = f'{col}_zscore'; mean_c = f'{col}_subj_mean'; std_c = f'{col}_subj_std'
            zcols_dict[zc] = np.where((df[std_c]==0)|df[col].isnull(), 0.0,
                (df[col].fillna(0)-df[mean_c])/df[std_c])
            zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index); df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols


def train_and_predict(target, n_seeds, seed_start=42, cal_method='linear'):
    """Train V95 model for one target.
    
    cal_method: 'linear' (default V53 style) or 'isotonic'
    """
    log.info(f"\n{'='*60}")
    log.info(f"V95 Training: {target} (seeds={n_seeds}, cal={cal_method})")
    log.info(f"{'='*60}")
    t_start = time.time()

    cfg_name = V53_SWEEP[target]['cfg']
    n_feat = V53_SWEEP[target]['nf']
    cfg = CFGS[cfg_name]

    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    train_cols_order = list(train.columns)
    test = test[train_cols_order]

    feat_cols = get_feature_cols(train)
    base_cols = [c for c in feat_cols if not c.endswith('_zscore') and not c.endswith('_rm*') and not c.endswith('_rs*') and '_x_' not in c]

    train, zscore_cols = add_personalization(train, base_cols)
    test, _ = add_personalization(test, base_cols)
    all_feat_cols = base_cols + zscore_cols
    gc.collect()

    leak_cols = remove_leak(all_feat_cols, target)
    y = train[target].values.astype(np.float64)
    train_rate = float(y.mean())
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    all_available = leak_cols
    train_r = train.fillna(0)
    test_r = test.fillna(0)

    # Feature ranking
    X_all = train_r[all_available].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in all_available]

    p_rank = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
              n_estimators=50, scale_pos_weight=spw, random_state=42,
              force_row_wise=True, n_jobs=1)
    ds = lgb.Dataset(X_all, label=y, feature_name=sn)
    m_rank = lgb.train(p_rank, ds, num_boost_round=50)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_available, imp), key=lambda x: -x[1])

    sel_cols = [r[0] for r in ranked[:n_feat]]
    sn_sel = [sanitize(c) for c in sel_cols]

    X = train_r[sel_cols].fillna(0).values.astype(np.float64)
    Xts = test_r[sel_cols].fillna(0).values.astype(np.float64)

    # OOF for calibration data
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    for s in range(seed_start, seed_start + n_seeds):
        for fold_i, (tr_i, va_i) in enumerate(gkf.split(X, y, train_r['subject_id'])):
            ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel)
            cfg_seed = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                    random_state=s, scale_pos_weight=spw,
                    force_row_wise=True, n_jobs=1)
            m = lgb.train(cfg_seed, ds_tr, num_boost_round=cfg['ne'])
            oof_preds[va_i] += m.predict(X[va_i])
    oof_preds /= n_seeds

    # Test prediction
    test_preds = np.zeros(len(Xts))
    for s in range(seed_start, seed_start + n_seeds):
        ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
        cfg_seed = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                random_state=s, scale_pos_weight=spw,
                force_row_wise=True, n_jobs=1)
        m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg['ne'])
        test_preds += m.predict(Xts)
    test_preds /= n_seeds

    # Calibration
    if cal_method == 'linear':
        # V53 style: shift by train_rate - mean
        cal_oof = np.clip(oof_preds + (train_rate - oof_preds.mean()), 0.0001, 0.9999)
        cal_test = np.clip(test_preds + (train_rate - test_preds.mean()), 0.0001, 0.9999)
    elif cal_method == 'isotonic':
        # Isotonic regression on OOF
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_preds, y)
        cal_oof = iso.predict(oof_preds)
        cal_test = iso.predict(test_preds)
        cal_oof = np.clip(cal_oof, 0.0001, 0.9999)
        cal_test = np.clip(cal_test, 0.0001, 0.9999)

    cv_loss = log_loss(y, oof_preds, labels=[0,1])
    cal_oof_loss = log_loss(y, cal_oof, labels=[0,1])

    time_s = time.time() - t_start
    log.info(f"  CV={cv_loss:.4f}, CalOOF={cal_oof_loss:.4f}, "
             f"train_rate={train_rate:.4f}, test_mean={cal_test.mean():.4f}, "
             f"shift={cal_test.mean()-train_rate:+.4f}, time={time_s:.0f}s")

    del train_r, test_r, X_all, X, Xts, ds, ds_tr, ds_all, m, m_rank
    gc.collect()

    return sel_cols, cal_test, train_rate, time_s, cfg_name, cal_oof_loss, cv_loss


def main():
    t_global = time.time()
    log.info("=" * 80)
    log.info("V95: V53 Swept Replicate + Calibration Refinement")
    log.info("=" * 80)

    # Experiments to try:
    # 1. V53 Swept replicate: 50 seeds, linear cal (exact V53)
    # 2. V53 Swept + 100 seeds, linear cal
    # 3. V53 Swept + 50 seeds, isotonic cal
    # 4. V53 Swept + 100 seeds, isotonic cal
    # 5. Different seed start (0 instead of 42)
    # 6. Ensemble: 50 seeds (start=42) + 50 seeds (start=1000)

    experiments = [
        ('v53_50s_linear', 50, 42, 'linear'),
        ('v53_100s_linear', 100, 42, 'linear'),
        ('v53_50s_iso', 50, 42, 'isotonic'),
        ('v53_100s_iso', 100, 42, 'isotonic'),
        ('v53_50s_start0', 50, 0, 'linear'),
        ('v53_100s_start0', 100, 0, 'linear'),
    ]

    all_results = {}

    for exp_name, n_seeds, seed_start, cal_method in experiments:
        log.info(f"\n{'#'*60}")
        log.info(f"Experiment: {exp_name} (n_seeds={n_seeds}, seed_start={seed_start}, cal={cal_method})")
        log.info(f"{'#'*60}")

        target_results = {}
        for target in TARGETS:
            sel_cols, cal_test, train_rate, time_s, cfg_name, cal_oof_loss, cv_loss = \
                train_and_predict(target, n_seeds, seed_start, cal_method)
            target_results[target] = {
                'exp': exp_name, 'n_seeds': n_seeds, 'seed_start': seed_start,
                'cal_method': cal_method, 'n_feat': V53_SWEEP[target]['nf'],
                'cfg': cfg_name, 'sel_cols': sel_cols, 'cal_test': cal_test,
                'train_rate': train_rate, 'time_s': time_s,
                'cal_oof_loss': cal_oof_loss, 'cv_loss': cv_loss,
            }
            log.info(f"  {target}: cal_oof={cal_oof_loss:.4f}")

        # Average
        avg_cal = np.mean([v['cal_oof_loss'] for v in target_results.values()])
        all_results[exp_name] = target_results
        log.info(f"\n  AVG CalOOF: {avg_cal:.4f}")

    # Summary
    log.info("\n" + "=" * 80)
    log.info("V95 SUMMARY")
    log.info("=" * 80)
    for exp_name, target_results in all_results.items():
        avg_cal = np.mean([v['cal_oof_loss'] for v in target_results.values()])
        log.info(f"  {exp_name}: AVG CalOOF={avg_cal:.4f}")

    log.info(f"\nTotal time: {time.time() - t_global:.0f}s")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(SUBMIT / f'v95_results_{ts}.json', 'w') as f:
        json.dump({k: {t: {'cal_oof_loss': v['cal_oof_loss'], 'cv_loss': v['cv_loss'],
                           'n_feat': v['n_feat'], 'cfg': v['cfg']}
                       for t, v in res.items()}
                   for k, res in all_results.items()}, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    main()
