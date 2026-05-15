"""
V93: Leaderboard Submission — Rolling-enhanced V53
Per-target optimal configs from V92:
  Q1:  deep, nf23, rolling
  Q2:  deep, nf23, rolling
  Q3:  v48,  nf20, rolling
  S1:  wide, nf17, rolling
  S2:  deep, nf23, rolling
  S3:  safety, nf17, rolling
  S4:  wide, nf17, rolling

Uses 30 seeds for stability, 5-fold GroupKFold OOF.
Trains on full data, predicts test set.
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

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

# V92 optimal per-target config
TARGET_CFG_V92 = {
    'Q1':  {'cfg': 'deep',  'nf': 23},
    'Q2':  {'cfg': 'deep',  'nf': 23},
    'Q3':  {'cfg': 'v48',   'nf': 20},
    'S1':  {'cfg': 'wide',  'nf': 17},
    'S2':  {'cfg': 'deep',  'nf': 23},
    'S3':  {'cfg': 'safety','nf': 17},
    'S4':  {'cfg': 'wide',  'nf': 17},
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
    """Add subject-level zscore features (memory-safe)."""
    df = df.copy()
    zscore_cols = []
    batch_size = 50
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]
        agg_parts = []
        for col in batch:
            col_filled = df[col].fillna(0)
            grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
            grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
            grp = grp.reset_index()
            agg_parts.append(grp)
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for start in range(0, len(feature_cols), batch_size):
        batch = feature_cols[start:start+batch_size]
        for col in batch:
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

def add_rolling(df, cols):
    """Add rolling mean/std features."""
    df = df.copy().sort_values(['subject_id', 'date'])
    new_cols = []
    for c in cols:
        g = df.groupby('subject_id')[c]
        for w in [3, 7]:
            rm = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            rs = g.rolling(w, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            df[f'{c}_rm{w}'] = rm.values
            df[f'{c}_rs{w}'] = rs.values
            new_cols.extend([f'{c}_rm{w}', f'{c}_rs{w}'])
    return df, new_cols


def train_and_predict(target):
    """Train V93 model for one target and return test predictions."""
    log.info(f"\n{'='*60}")
    log.info(f"V93 Training: {target}")
    log.info(f"{'='*60}")
    t_start = time.time()

    cfg_name = TARGET_CFG_V92[target]['cfg']
    n_feat = TARGET_CFG_V92[target]['nf']
    cfg = CFGS[cfg_name]
    n_seeds = 30
    use_rolling = True

    # Load data
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    train_cols_order = list(train.columns)
    test = test[train_cols_order]

    feat_cols = get_feature_cols(train)
    base_cols = [c for c in feat_cols if not c.endswith('_zscore') and not c.endswith('_rm*') and not c.endswith('_rs*') and '_x_' not in c]

    # Personalization
    train, zscore_cols = add_personalization(train, base_cols)
    test, _ = add_personalization(test, base_cols)
    all_feat_cols = base_cols + zscore_cols
    gc.collect()

    leak_cols = remove_leak(all_feat_cols, target)
    y = train[target].values.astype(np.float64)
    train_rate = float(y.mean())
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    all_available = leak_cols

    if use_rolling:
        rolling_base = [c for c in all_available if '_rm3' not in c and '_rs3' not in c and '_rm7' not in c and '_rs7' not in c]
        train_r, added_rolling = add_rolling(train, rolling_base)
        test_r, _ = add_rolling(test, rolling_base)
        all_available = all_available + added_rolling
    else:
        train_r = train
        test_r = test

    train_r = train_r.fillna(0)
    test_r = test_r.fillna(0)

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

    # Train on full data with multiple seeds
    test_preds = np.zeros(len(Xts))
    for s in range(42, 42 + n_seeds):
        ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
        cfg_seed = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                random_state=s, scale_pos_weight=spw,
                force_row_wise=True, n_jobs=1)
        m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg['ne'])
        test_preds += m.predict(Xts)
    test_preds /= n_seeds

    # Calibration
    cal_test = np.clip(test_preds + (train_rate - test_preds.mean()), 0.0001, 0.9999)

    time_s = time.time() - t_start
    log.info(f"  {target}: CalOOF train_rate={train_rate:.4f}, "
             f"test_mean={cal_test.mean():.4f}, shift={cal_test.mean()-train_rate:+.4f}, "
             f"time={time_s:.0f}s")

    del train_r, test_r, X_all, X, Xts, ds_all, m, m_rank
    gc.collect()

    return sel_cols, cal_test, train_rate, time_s, cfg_name


def main():
    t_global = time.time()
    log.info("=" * 80)
    log.info("V93: Rolling-enhanced V53 Submission")
    log.info("=" * 80)
    log.info(f"Per-target configs from V92:")
    for t, c in TARGET_CFG_V92.items():
        log.info(f"  {t}: {c['cfg']}, nf={c['nf']}")

    all_preds = {}
    all_info = {}

    for target in TARGETS:
        sel_cols, test_preds, train_rate, time_s, cfg_name = train_and_predict(target)
        all_preds[target] = test_preds
        all_info[target] = {'sel_cols': sel_cols, 'train_rate': train_rate, 'time_s': time_s, 'cfg': cfg_name}

    # Build submission
    test = pd.read_parquet(DATA / "test_features.parquet")
    submit = pd.DataFrame({
        'subject_id': test['subject_id'],
        'lifelog_date': test['lifelog_date'],
    })
    for target in TARGETS:
        submit[target] = all_preds[target]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT / f'v93_submission_{ts}.csv'
    submit.to_csv(sub_path, index=False)
    log.info(f"\nSubmission saved: {sub_path}")
    log.info(f"Total time: {time.time() - t_global:.0f}s")

    # Print summary
    log.info("\n" + "=" * 80)
    log.info("V93 SUBMISSION SUMMARY")
    log.info("=" * 80)
    for t in TARGETS:
        info = all_info[t]
        log.info(f"  {t}: cfg={info['cfg']}, nf={len(info['sel_cols'])}, "
                 f"mean={all_preds[t].mean():.4f}, train_rate={info['train_rate']:.4f}, "
                 f"time={info['time_s']:.0f}s")

    return submit, all_preds


if __name__ == "__main__":
    main()
