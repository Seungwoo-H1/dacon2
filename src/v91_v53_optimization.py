"""
V91: Lightweight V53 Optimization - Target-wise parallel experiments
Tests key combinations efficiently:
1. Seed count: 30 vs 50
2. n_feat: baseline vs swept vs +-3
3. Rolling features: with vs without
4. Per-target optimal config search

Uses GroupKFold 5-fold OOF. Only 24 cores, n_jobs=1 for LGBM (avoids thrashing).
"""

import sys, gc, logging, json, re, time, warnings, multiprocessing
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

CONSTANT_COLS = [
    'mACStatus_m_charging_min','mACStatus_m_charging_max','mLight_m_light_min',
    'mScreenStatus_m_screen_use_min','mScreenStatus_m_screen_use_max',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean','mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max','mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min','mUsageStats_usage_game_ratio_min',
]
COLLINEAR_DROP = [
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'mBle_ble_device_count_mean','mBle_ble_device_count_std',
    'mBle_ble_device_count_max',
    'mWifi_wifi_bssid_count_mean','mWifi_wifi_bssid_count_std',
    'mWifi_wifi_bssid_count_max',
]

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':  {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
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
    # Process in batches to avoid memory spike
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
    # Compute zscores in batches
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


def run_target_experiment(args):
    """Run all experiments for a single target (for multiprocessing)."""
    target, experiment_plan = args
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

    # Add personalization (done once per target)
    train, zscore_cols = add_personalization(train, base_cols)
    test, _ = add_personalization(test, base_cols)
    all_feat_cols = base_cols + zscore_cols
    gc.collect()

    leak_cols = remove_leak(all_feat_cols, target)
    y = train[target].values.astype(np.float64)
    train_rate = float(y.mean())
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

    # Run experiment plan for this target
    results = {}
    gkf = GroupKFold(n_splits=5)

    for exp_key in experiment_plan:
        exp = experiment_plan[exp_key]
        feat_type = exp['feat_type']
        cfg_name = exp['cfg_name']
        cfg = CFGS[cfg_name]
        n_seeds = exp['n_seeds']
        n_feat_base = exp['n_feat_base']  # base n_feat for ranking
        use_rolling = exp.get('rolling', False)

        exp_name = f"{cfg_name}_s{n_seeds}_nf{n_feat_base}_roll{int(use_rolling)}"
        log.info(f"\n  Exp: {exp_name}")
        t0 = time.time()

        # Feature engineering
        if feat_type == 'zscore':
            all_available = leak_cols  # base + zscore
        elif feat_type == 'base':
            all_available = [c for c in leak_cols if not c.endswith('_zscore')]
        else:
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

        # Test multiple n_feat values around the base
        for delta in [0, -3, +3, -5, +5]:
            n_feat = max(5, n_feat_base + delta)
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

            # Calibrate
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

            key = f"nf{n_feat}"
            results[key] = {
                'experiment': exp_name,
                'target': target,
                'n_feat': n_feat,
                'cv_loss': round(float(cv_loss), 6),
                'cal_oof_loss': round(float(cal_oof_loss), 6),
                'train_rate': round(train_rate, 4),
                'test_mean': round(float(cal_test.mean()), 6),
                'test_shift': round(float(cal_test.mean() - train_rate), 6),
                'n_seeds': n_seeds,
                'cfg_name': cfg_name,
                'feat_type': feat_type,
                'rolling': use_rolling,
                'time_s': round(time.time() - t0, 1),
            }
            log.info(f"    nf{n_feat}: CV={cv_loss:.4f}, CalOOF={cal_oof_loss:.4f}, "
                     f"test_mean={cal_test.mean():.4f}, shift={cal_test.mean()-train_rate:+.4f}")

            # Cleanup
            del X, Xts, oof_preds, test_preds
            gc.collect()

        total_time = time.time() - t_start
        log.info(f"  {exp_name} total time: {total_time:.0f}s")

    return target, results


def main():
    t_global = time.time()
    log.info("=" * 80)
    log.info("V91: Lightweight V53 Optimization")
    log.info("=" * 80)

    # Experiment plans per target
    # Each plan: dict of {config_name: {feat_type, cfg_name, n_seeds, n_feat_base, rolling}}
    experiment_plans = {}

    for target in TARGETS:
        experiment_plans[target] = {}

        # Use appropriate cfg per target (V53 baseline)
        if target in ['Q1', 'Q2', 'S2']:
            base_cfg = 'deep'
        elif target == 'Q3':
            base_cfg = 'v48'
        else:
            base_cfg = 'wide' if target in ['S1', 'S4'] else 'safety'

        # Plan 1: n_seeds=30, baseline n_feat
        experiment_plans[target]['plan1'] = {
            'feat_type': 'zscore', 'cfg_name': base_cfg,
            'n_seeds': 30, 'n_feat_base': 20, 'rolling': False
        }

        # Plan 2: n_seeds=50, baseline n_feat
        experiment_plans[target]['plan2'] = {
            'feat_type': 'zscore', 'cfg_name': base_cfg,
            'n_seeds': 50, 'n_feat_base': 20, 'rolling': False
        }

        # Plan 3: n_seeds=30, with rolling
        experiment_plans[target]['plan3'] = {
            'feat_type': 'zscore', 'cfg_name': base_cfg,
            'n_seeds': 30, 'n_feat_base': 20, 'rolling': True
        }

        # Plan 4: n_seeds=50, V53 swept n_feat
        swept_nfeat = {'Q1': 19, 'Q2': 14, 'Q3': 5, 'S1': 21, 'S2': 19, 'S3': 21, 'S4': 20}
        experiment_plans[target]['plan4'] = {
            'feat_type': 'zscore', 'cfg_name': base_cfg,
            'n_seeds': 50, 'n_feat_base': swept_nfeat.get(target, 20), 'rolling': False
        }

        # Plan 5: n_seeds=100, baseline n_feat
        experiment_plans[target]['plan5'] = {
            'feat_type': 'zscore', 'cfg_name': base_cfg,
            'n_seeds': 100, 'n_feat_base': 20, 'rolling': False
        }

        # Plan 6: CFG sweep - wide variant
        experiment_plans[target]['plan6'] = {
            'feat_type': 'zscore', 'cfg_name': 'wide',
            'n_seeds': 30, 'n_feat_base': 20, 'rolling': False
        }

        # Plan 7: CFG sweep - safety variant
        experiment_plans[target]['plan7'] = {
            'feat_type': 'zscore', 'cfg_name': 'safety',
            'n_seeds': 30, 'n_feat_base': 20, 'rolling': False
        }

    log.info(f"\nExperiment plans per target:")
    for target, plans in experiment_plans.items():
        log.info(f"  {target}: {len(plans)} plans")
        for pname, p in plans.items():
            log.info(f"    {pname}: {p['cfg_name']}_s{p['n_seeds']}_nf{p['n_feat_base']}_r{int(p['rolling'])}")

    all_results = {}
    all_best = {}

    # Run sequentially to avoid memory issues (24 cores but LGBM uses n_jobs=1)
    for target in TARGETS:
        target_start = time.time()
        log.info(f"\n{'#'*60}")
        log.info(f"Processing target: {target}")
        log.info(f"{'#'*60}")

        target_results = {}
        plans = experiment_plans[target]
        gkf = GroupKFold(n_splits=5)

        for plan_key in sorted(plans.keys()):
            exp = plans[plan_key]
            exp_name = f"{exp['cfg_name']}_s{exp['n_seeds']}_nf{exp['n_feat_base']}_r{int(exp['rolling'])}"

            log.info(f"\n[{plan_key}] {exp_name}")
            t0 = time.time()

            # Run this single experiment
            # Feature engineering
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

            if exp['feat_type'] == 'zscore':
                all_available = leak_cols
            else:
                all_available = [c for c in leak_cols if not c.endswith('_zscore')]

            if exp['rolling']:
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

            cfg = CFGS[exp['cfg_name']]
            n_seeds = exp['n_seeds']
            n_feat_base = exp['n_feat_base']

            # Feature ranking
            p_rank = build_lgb_params(cfg, objective='binary', metric='binary_logloss', verbose=-1,
                  n_estimators=50, scale_pos_weight=spw, random_state=42,
                  force_row_wise=True, n_jobs=1)
            ds = lgb.Dataset(X_all, label=y, feature_name=sn)
            m_rank = lgb.train(p_rank, ds, num_boost_round=50)
            imp = m_rank.feature_importance(importance_type='gain')
            ranked = sorted(zip(all_available, imp), key=lambda x: -x[1])

            # Sweep n_feat: base + delta
            for delta in [0, -3, +3]:
                n_feat = max(5, n_feat_base + delta)
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

                # Test
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
                all_results[key] = {
                    'target': target,
                    'n_feat': n_feat,
                    'cv_loss': round(float(cv_loss), 6),
                    'cal_oof_loss': round(float(cal_oof_loss), 6),
                    'train_rate': round(train_rate, 4),
                    'test_mean': round(float(cal_test.mean()), 6),
                    'test_shift': round(float(cal_test.mean() - train_rate), 6),
                    'n_seeds': n_seeds,
                    'cfg_name': exp['cfg_name'],
                    'feat_type': exp['feat_type'],
                    'rolling': exp['rolling'],
                    'time_s': round(time.time() - t0, 1),
                }
                log.info(f"    nf{n_feat}: CV={cv_loss:.4f}, CalOOF={cal_oof_loss:.4f}, "
                         f"test_mean={cal_test.mean():.4f}, shift={cal_test.mean()-train_rate:+.4f}")

                del X, Xts, oof_preds, test_preds, ds_tr, ds_all, m
                gc.collect()

        target_total = time.time() - target_start
        log.info(f"\n{target} total time: {target_total:.0f}s")

    # Find best per target
    log.info("\n" + "=" * 80)
    log.info("BEST RESULTS PER TARGET")
    log.info("=" * 80)

    best_overall_cal = float('inf')
    best_config = None

    # Group by target, find best cal_oof_loss
    for target in TARGETS:
        target_results = {k: v for k, v in all_results.items() if v['target'] == target}
        best = min(target_results.items(), key=lambda x: x[1]['cal_oof_loss'])
        all_best[target] = {'key': best[0], **best[1]}
        log.info(f"  {target}: {best[0]} → CalOOF={best[1]['cal_oof_loss']:.4f}")

        if best[1]['cal_oof_loss'] < best_overall_cal:
            best_overall_cal = best[1]['cal_oof_loss']
            best_config = best

    avg_cal_oof = np.mean([v['cal_oof_loss'] for v in all_best.values()])
    log.info(f"\nAVG Cal OOF (best per target): {avg_cal_oof:.4f}")
    log.info(f"Best overall: {best_config[0]} → {best_config[1]['cal_oof_loss']:.4f}")
    log.info(f"Total time: {time.time() - t_global:.0f}s")

    # Save results
    res_path = SUBMIT / f'v91_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(res_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Results saved: {res_path}")

    # Save best config
    best_path = SUBMIT / f'v91_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(best_path, 'w') as f:
        json.dump({'best_per_target': all_best, 'avg_cal_oof': round(avg_cal_oof, 6),
                    'best_overall': best_config[0]}, f, indent=2, default=str)
    log.info(f"Best config saved: {best_path}")

    return all_results, all_best


if __name__ == "__main__":
    main()
