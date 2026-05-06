"""
V53 feature count sweep: test n_feat ±3 around baseline and find per-target optimum.
Baseline: V53_CONFIGS (Q1:20, Q2:15, Q3:8, S1:20, S2:20, S3:20, S4:20)
Explores range(max(2, n-3), n+4) for each target independently.
Reports CV-like OOF log-loss on train data using 50-seed ensemble.
"""

import sys, gc, logging, json, re, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

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
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

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
    """Add subject-level zscore features (batch agg, no fragmentation)."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
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

def train_and_predict_noseed(train_feat, cols, y_train, target, cfgs, v53_cfgs, n_seeds=50):
    """Train on all train data, return OOF predictions (train data) using target-specific CFG.
    Uses GroupKFold-style: split into 3 groups by subject_id, predict held-out groups.
    For simplicity: use random KFold with group awareness.
    """
    X = train_feat[cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in cols]
    spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)

    v53_cfg = v53_cfgs.get(target, {'cfg': 'deep', 'n_feat': 20})
    cfg_name = v53_cfg['cfg']
    base_cfg = cfgs.get(cfg_name, cfgs['deep'])
    n_trees = base_cfg['ne']

    # GroupKFold: group by subject_id, predict held-out subjects
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=3)
    subjects = train_feat['subject_id'].values

    oof_preds = np.zeros(len(X))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_train, subjects)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        results_fold = []
        for seed in range(1, n_seeds + 1):
            cfg_seed = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'force_row_wise': True, 'n_jobs': 1,
                'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
                'learning_rate': base_cfg['lr'], 'n_estimators': n_trees,
                'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
                'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
                'min_child_samples': base_cfg['mc'], 'random_state': seed, 'scale_pos_weight': spw,
            }
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=n_trees)
            pred = m.predict(X_va)
            results_fold.append(pred)
            del ds, m
            gc.collect()

        oof_preds[va_idx] = np.clip(np.mean(results_fold, axis=0), 0.0001, 0.9999)
        del results_fold, X_tr, X_va, y_tr, y_va
        gc.collect()

    return oof_preds

def log_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("V53 Feature Count Sweep")
    log.info("=" * 60)

    # Load data
    train = pd.read_parquet(DATA / "features.parquet")
    log.info(f"  Train: {train.shape}")

    # V53 baseline configs
    V53_CONFIGS = {
        'Q1': {'cfg': 'deep', 'n_feat': 20},
        'Q2': {'cfg': 'deep', 'n_feat': 15},
        'Q3': {'cfg': 'v48', 'n_feat': 8},
        'S1': {'cfg': 'wide', 'n_feat': 20},
        'S2': {'cfg': 'deep', 'n_feat': 20},
        'S3': {'cfg': 'safety', 'n_feat': 20},
        'S4': {'cfg': 'wide', 'n_feat': 20},
    }

    CFGS = {
        'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
        'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
        'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
        'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
    }

    # Get base features and add personalization
    feat_cols = get_feature_cols(train)
    train, zscore_cols = add_personalization(train, feat_cols)
    all_cols = feat_cols + zscore_cols
    log.info(f"  Features: {len(all_cols)} total (base {len(feat_cols)} + zscore {len(zscore_cols)})")

    n_seeds = 10  # Reduced for sweep speed (baseline was 50, use 10 for exploration)
    log.info(f"  Seeds per config: {n_seeds}")

    sweep_results = {}
    best_per_target = {}

    for target in TARGETS:
        cfg_name = V53_CONFIGS[target]['cfg']
        base_n = V53_CONFIGS[target]['n_feat']
        lo = max(2, base_n - 3)
        hi = base_n + 4

        log.info(f"\n{'='*40}")
        log.info(f"  Target: {target} (cfg={cfg_name}, baseline n_feat={base_n}, range={lo}~{hi-1})")
        log.info(f"{'='*40}")

        # Get non-leak features
        non_leak = remove_leak(all_cols, target)

        # First, rank features with baseline config
        y = train[target].values.astype(np.float64)
        X = train[non_leak].fillna(0).values.astype(np.float64)
        spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)

        base_cfg = CFGS.get(cfg_name, CFGS['deep'])
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'], 'learning_rate': base_cfg['lr'],
            'n_estimators': min(base_cfg['ne'], 100), 'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
            'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': base_cfg['mc'], 'force_row_wise': True, 'n_jobs': 1,
        }
        sn = [sanitize(c) for c in non_leak]
        ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds, num_boost_round=params_rank['n_estimators'])
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(non_leak, imp), key=lambda x: -x[1])
        del model_rank, ds, X
        gc.collect()

        target_results = {}
        for n_feat in range(lo, hi):
            sel_cols = [ranked[i][0] for i in range(min(n_feat, len(ranked)))]
            t0 = time.time()

            oof = train_and_predict_noseed(train, sel_cols, y, target, CFGS, V53_CONFIGS, n_seeds)
            cv_loss = log_loss(y, oof)
            elapsed = time.time() - t0

            log.info(f"    n_feat={n_feat:3d}: CV log_loss={cv_loss:.4f} ({elapsed:.0f}s) | features={len(sel_cols)}")
            target_results[n_feat] = {'cv_loss': cv_loss, 'elapsed': elapsed, 'cols': sel_cols}

        # Find best
        best_n = min(target_results, key=lambda k: target_results[k]['cv_loss'])
        best_cv = target_results[best_n]['cv_loss']
        baseline_cv = target_results.get(base_n, {}).get('cv_loss', float('inf'))
        delta = baseline_cv - best_cv  # positive = improvement

        best_per_target[target] = {
            'best_n_feat': best_n,
            'best_cv': best_cv,
            'baseline_cv': baseline_cv,
            'delta': delta,
            'all_results': target_results,
            'cols': target_results[best_n]['cols'],
            'cfg': cfg_name,
        }

        sweep_results[target] = {
            r: {'cv_loss': target_results[r]['cv_loss'], 'elapsed': target_results[r]['elapsed']}
            for r in target_results
        }

        log.info(f"  ✅ {target} best n_feat={best_n} (baseline={base_n}, delta={delta:+.4f})")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SWEEP SUMMARY")
    log.info(f"{'='*60}")

    avg_baseline = 0
    avg_best = 0
    for target in TARGETS:
        b = best_per_target[target]
        avg_baseline += b['baseline_cv']
        avg_best += b['best_cv']
        sign = "✅" if b['delta'] > 0 else "❌"
        log.info(f"  {sign} {target}: baseline={b['baseline_cv']:.4f} → best_n={b['best_n_feat']} cv={b['best_cv']:.4f} delta={b['delta']:+.4f}")

    avg_baseline /= len(TARGETS)
    avg_best /= len(TARGETS)
    avg_delta = avg_baseline - avg_best
    log.info(f"  AVG: baseline={avg_baseline:.4f} → best={avg_best:.4f} delta={avg_delta:+.4f}")

    # Save sweep results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_seeds': n_seeds,
        'sweep_results': sweep_results,
        'best_per_target': {
            t: {
                'best_n_feat': b['best_n_feat'],
                'best_cv': b['best_cv'],
                'baseline_cv': b['baseline_cv'],
                'delta': b['delta'],
                'cols': b['cols'],
                'cfg': b['cfg'],
            } for t, b in best_per_target.items()
        },
        'avg_baseline': avg_baseline,
        'avg_best': avg_best,
        'avg_delta': avg_delta,
    }
    out_path = SUBMIT / f'v53_sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"  Saved: {out_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")
    log.info(f"{'='*60}")

    return output

if __name__ == "__main__":
    main()
