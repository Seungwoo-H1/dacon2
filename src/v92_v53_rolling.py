"""
V92: Rolling Feature Optimized V53
- Focus: Rolling features + minimal cfg sweep + 20 seeds (fixed)
- Target: 1 plan per target → very fast
- Tests: rolling vs no-rolling with 20 seeds × 5 folds

Expected runtime: ~3-4 min total (7 targets × 2 configs × 5 folds)
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
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

# CFGs per target (V53 baseline assignments)
TARGET_CFG = {
    'Q1': 'deep', 'Q2': 'deep', 'S2': 'deep',
    'Q3': 'v48',
    'S1': 'wide', 'S4': 'wide',
    'S3': 'safety',
}

CFGS = {
    'wide':   {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep':   {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

def build_lgb_params(cfg, **extras):
    """Build LightGBM params from cfg dict with explicit key mapping."""
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


def run_target(target):
    """Run V92 experiment for one target."""
    log.info(f"\n{'='*60}")
    log.info(f"TARGET: {target}")
    log.info(f"{'='*60}")
    t_start = time.time()

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

    cfg_name = TARGET_CFG[target]
    cfg = CFGS[cfg_name]
    n_seeds = 20
    gkf = GroupKFold(n_splits=5)

    results = {}

    # Experiment: no-rolling vs rolling
    for use_rolling in [False, True]:
        exp_name = f"{cfg_name}_roll{int(use_rolling)}"
        log.info(f"\n  Exp: {exp_name}")
        t0 = time.time()

        # Feature engineering
        all_available = leak_cols  # base + zscore

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

        X_all = train_r[all_available].fillna(0).values.astype(np.float64)
        Xt = test_r[all_available].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in all_available]

        # Feature ranking (1 seed, 50 trees)
        p_rank = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                  n_estimators=50, scale_pos_weight=spw, random_state=42,
                  force_row_wise=True, n_jobs=1)
        ds = lgb.Dataset(X_all, label=y, feature_name=sn)
        m_rank = lgb.train(p_rank, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(all_available, imp), key=lambda x: -x[1])

        # Test n_feat: 17, 20, 23
        for n_feat in [17, 20, 23]:
            sel_cols = [r[0] for r in ranked[:n_feat]]
            sn_sel = [sanitize(c) for c in sel_cols]

            X = train_r[sel_cols].fillna(0).values.astype(np.float64)
            Xts = test_r[sel_cols].fillna(0).values.astype(np.float64)

            # OOF
            oof_preds = np.zeros(len(y))
            n_total = n_seeds * 5

            for s in range(42, 42 + n_seeds):
                for fold_i, (tr_i, va_i) in enumerate(gkf.split(X, y, train_r['subject_id'])):
                    ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel)
                    cfg_seed = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                            random_state=s, scale_pos_weight=spw,
                            force_row_wise=True, n_jobs=1)
                    m = lgb.train(cfg_seed, ds_tr, num_boost_round=cfg['ne'])
                    oof_preds[va_i] += m.predict(X[va_i])

            oof_preds /= n_total
            cv_loss = log_loss(y, oof_preds, labels=[0, 1])
            cal_oof = np.clip(oof_preds + (train_rate - oof_preds.mean()), 0.0001, 0.9999)
            cal_oof_loss = log_loss(y, cal_oof, labels=[0, 1])

            # Test prediction
            test_preds = np.zeros(len(Xts))
            for s in range(42, 42 + n_seeds):
                ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
                cfg_seed = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                        random_state=s, scale_pos_weight=spw,
                        force_row_wise=True, n_jobs=1)
                m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg['ne'])
                test_preds += m.predict(Xts)
            test_preds /= n_seeds
            cal_test = np.clip(test_preds + (train_rate - test_preds.mean()), 0.0001, 0.9999)

            key = f"{exp_name}_nf{n_feat}"
            results[key] = {
                'target': target,
                'n_feat': n_feat,
                'cv_loss': round(float(cv_loss), 6),
                'cal_oof_loss': round(float(cal_oof_loss), 6),
                'train_rate': round(train_rate, 4),
                'test_mean': round(float(cal_test.mean()), 6),
                'test_shift': round(float(cal_test.mean() - train_rate), 6),
                'n_seeds': n_seeds,
                'cfg_name': cfg_name,
                'rolling': use_rolling,
                'time_s': round(time.time() - t0, 1),
            }
            log.info(f"    nf{n_feat}: CV={cv_loss:.4f}, CalOOF={cal_oof_loss:.4f}, "
                     f"test_mean={cal_test.mean():.4f}, shift={cal_test.mean()-train_rate:+.4f}")

            del X, Xts, oof_preds, test_preds, ds_tr, ds_all, m
            gc.collect()

        total_time = time.time() - t_start
        log.info(f"  {exp_name} total time: {total_time:.0f}s")

    return target, results


def main():
    t_global = time.time()
    log.info("=" * 80)
    log.info("V92: Rolling Feature Optimized V53")
    log.info("=" * 80)

    all_results = {}
    all_best = {}

    for target in TARGETS:
        log.info(f"\n{'#'*60}")
        log.info(f"Processing target: {target}")
        log.info(f"{'#'*60}")

        target_results = run_target(target)
        t, r = target_results
        all_results.update(r)

        # Find best per target
        target_res = {k: v for k, v in r.items() if v['target'] == t}
        best = min(target_res.items(), key=lambda x: x[1]['cal_oof_loss'])
        all_best[t] = {'key': best[0], **best[1]}
        log.info(f"\n  Best for {t}: {best[0]} → CalOOF={best[1]['cal_oof_loss']:.4f}")

    # Summary
    log.info("\n" + "=" * 80)
    log.info("BEST PER TARGET")
    log.info("=" * 80)
    avg_cal = 0
    best_overall = None
    best_cal = float('inf')
    for t in TARGETS:
        v = all_best[t]
        log.info(f"  {t}: {v['key']} → CalOOF={v['cal_oof_loss']:.4f} "
                 f"(cfg={v['cfg_name']}, rolling={v['rolling']}, nf={v['n_feat']}, seeds={v['n_seeds']})")
        avg_cal += v['cal_oof_loss']
        if v['cal_oof_loss'] < best_cal:
            best_cal = v['cal_oof_loss']
            best_overall = v

    avg_cal /= len(TARGETS)
    log.info(f"\nAVG Cal OOF: {avg_cal:.4f}")
    log.info(f"Best overall: {best_overall['key']} → {best_overall['cal_oof_loss']:.4f}")
    log.info(f"Total time: {time.time() - t_global:.0f}s")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(SUBMIT / f'v92_results_{ts}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(SUBMIT / f'v92_best_{ts}.json', 'w') as f:
        json.dump({'best_per_target': all_best, 'avg_cal_oof': round(avg_cal, 6),
                    'best_overall': best_overall['key']}, f, indent=2, default=str)
    log.info(f"Saved: submissions/v92_{ts}_*.json")

    # Print rolling improvement summary
    log.info("\n" + "=" * 80)
    log.info("ROLLING IMPROVEMENT SUMMARY")
    log.info("=" * 80)
    for t in TARGETS:
        no_roll = {k: v for k, v in all_results.items() if v['target'] == t and not v['rolling']}
        roll = {k: v for k, v in all_results.items() if v['target'] == t and v['rolling']}
        if no_roll and roll:
            best_no = min(no_roll.values(), key=lambda x: x['cal_oof_loss'])
            best_roll = min(roll.values(), key=lambda x: x['cal_oof_loss'])
            imp = best_no['cal_oof_loss'] - best_roll['cal_oof_loss']
            log.info(f"  {t}: no-rolling={best_no['cal_oof_loss']:.4f} → rolling={best_roll['cal_oof_loss']:.4f} (Δ={imp:+.4f})")

    return all_results, all_best


if __name__ == "__main__":
    main()
