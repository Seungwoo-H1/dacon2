"""
V99: LGBM Multi-Seed Ensemble (100 seeds) + Weighted Group Blend

Purpose:
- 100 seeds × GroupKFold 5-fold = 500 models per target
- 4 seed groups × 25 seeds: track diversity across seed ranges
- Weighted blend optimized on OOF log_loss
- Compare: uniform vs optimized weights
- Submit + OOF-LB gap analysis

Key differences from V97:
- 100 seeds vs 50 seeds (double the model diversity)
- Group-weighted blending (not just seed averaging)
- Correct OOF computation: sum_all_folds / (n_seeds * n_folds)
"""

import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from itertools import product as iprod
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
EXPERIMENTS = ROOT / "experiments"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

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

# V53 Swept config — matches V97's best-known settings
V53_SWEEP = {
    'Q1': {'cfg': 'deep', 'n_feat': 19},
    'Q2': {'cfg': 'deep', 'n_feat': 14},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 21},
    'S2': {'cfg': 'deep', 'n_feat': 19},
    'S3': {'cfg': 'safety', 'n_feat': 23},
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
    df = df.copy(); zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']; grp = grp.reset_index(); agg_parts.append(grp)
    agg_df = agg_parts[0]
    for part in agg_parts[1:]: agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
    df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'; mean_c = f'{col}_subj_mean'; std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where((df[std_c]==0)|df[col].isnull(), 0.0, (df[col].fillna(0)-df[mean_c])/df[std_c])
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index); df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols

def train_ensemble(X, y, sn_sel, cfg, n_seeds, seed_groups, gkf, groups, Xts, n_folds=5):
    """
    Train n_seeds models with GroupKFold.
    Returns:
      avg_oof: OOF predictions (mean over all seeds)
      avg_test: Test predictions (mean over all seeds)
      group_oofs: list of per-group OOF averages (for weight optimization)
      group_tests: list of per-group test averages
    """
    group_oofs = []
    group_tests = []

    for gi, seeds in enumerate(seed_groups):
        oof_sum = np.zeros(len(y))  # accumulated 5-fold predictions
        test_preds = []

        for seed in seeds:
            oof_fold = np.zeros(len(y))
            for tr_i, va_i in gkf.split(X, y, groups):
                p_tr = {'objective':'binary','metric':'binary_logloss','verbose':-1,
                    'n_estimators':cfg['ne'],'random_state':seed,
                    'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
                    'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
                    'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],
                    'min_child_samples':cfg['mc'],'force_row_wise':True,'n_jobs':1}
                ds_tr = lgb.Dataset(X[tr_i], label=y[tr_i], feature_name=sn_sel)
                m = lgb.train(p_tr, ds_tr, num_boost_round=cfg['ne'])
                oof_fold[va_i] += m.predict(X[va_i])
            oof_sum += oof_fold  # accumulate 5-fold sum

            ds_all = lgb.Dataset(X, label=y, feature_name=sn_sel)
            p_all = {k:v for k,v in p_tr.items() if k not in ('n_estimators','random_state')}
            p_all.update({'n_estimators':cfg['ne'],'random_state':seed})
            m = lgb.train(p_all, ds_all, num_boost_round=cfg['ne'])
            test_preds.append(m.predict(Xts))

        # Divide by (n_seeds_in_group * n_folds) for correct per-sample mean
        group_oofs.append(oof_sum / (len(seeds) * n_folds))
        group_tests.append(np.mean(test_preds, axis=0))

    avg_oof = np.mean(group_oofs, axis=0)
    avg_test = np.mean(group_tests, axis=0)
    return avg_oof, avg_test, group_oofs, group_tests

def main():
    t_start = time.time()
    log.info("="*80)
    log.info("V99: LGBM Multi-Seed Ensemble + OOF-LB Gap Analysis")
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

    gkf = GroupKFold(n_splits=5)
    n_seeds_per_group = 25
    seed_groups = [
        list(range(42, 42+n_seeds_per_group)),
        list(range(100, 100+n_seeds_per_group)),
        list(range(200, 200+n_seeds_per_group)),
        list(range(300, 300+n_seeds_per_group)),
    ]
    total_seeds = sum(len(s) for s in seed_groups)
    log.info(f"  Seeds: {total_seeds} total ({len(seed_groups)} groups × {n_seeds_per_group})")

    target_results = {}
    predictions = {}

    for target in TARGETS:
        log.info(f"\n{'#'*60}")
        log.info(f"Processing {target} (cfg={V53_SWEEP[target]['cfg']}, n_feat={V53_SWEEP[target]['n_feat']})")
        log.info(f"{'#'*60}")
        t0 = time.time()

        cfg_name = V53_SWEEP[target]['cfg']
        n_feat = V53_SWEEP[target]['n_feat']
        cfg = CFGS[cfg_name]

        leak_cols = remove_leak(all_cols, target)
        y = train_p[target].values.astype(np.float64)
        train_rate = float(y.mean())
        spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)

        # Feature ranking
        rank_params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
            'n_estimators':50,'scale_pos_weight':spw,'random_state':42,
            'force_row_wise':True,'n_jobs':1,
            'num_leaves':cfg['nl'],'max_depth':cfg['md'],'learning_rate':cfg['lr'],
            'subsample':cfg['ss'],'colsample_bytree':cfg['cb'],
            'reg_alpha':cfg['ra'],'reg_lambda':cfg['rl'],'min_child_samples':cfg['mc']}
        X_all = train_p[leak_cols].fillna(0).values.astype(np.float64)
        sn = [sanitize(c) for c in leak_cols]
        ds = lgb.Dataset(X_all, label=y, feature_name=sn)
        m_rank = lgb.train(rank_params, ds, num_boost_round=50)
        imp = m_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked[:n_feat]]
        sn_sel = [sanitize(c) for c in sel_cols]

        X = train_p[sel_cols].fillna(0).values.astype(np.float64)
        Xts = test_p[sel_cols].fillna(0).values.astype(np.float64)

        # === Train ensemble ===
        avg_oof, avg_test, group_oofs, group_tests = train_ensemble(
            X, y, sn_sel, cfg, total_seeds, seed_groups, gkf, train_p['subject_id'], Xts, n_folds=5
        )

        log.info(f"  avg_oof stats: mean={avg_oof.mean():.4f} std={avg_oof.std():.4f}")

        # === Linear calibration ===
        shift = train_rate - avg_oof.mean()
        cal_oof = np.clip(avg_oof + shift, 0.0001, 0.9999)
        cal_test = np.clip(avg_test + shift, 0.0001, 0.9999)

        # === Weight optimization on OOF ===
        # Try grid of group weights (constrained: sum=1, min=0.05, max=0.6)
        n_groups = len(group_oofs)
        best_blend_cal = log_loss(y, cal_oof, labels=[0,1])
        best_w = [1.0/n_groups]*n_groups
        best_cal_oof_blend = cal_oof.copy()
        best_test_blend = cal_test.copy()

        # Coarse search first
        step = 0.1
        grid_vals = np.arange(0.05, 0.65, step)
        for weights in iprod(grid_vals, repeat=n_groups):
            w = np.array(weights)
            if abs(w.sum() - 1.0) < step:
                blend = sum(w[i]*group_oofs[i] for i in range(n_groups))
                s = train_rate - blend.mean()
                cal_blend = np.clip(blend + s, 0.0001, 0.9999)
                bl = log_loss(y, cal_blend, labels=[0,1])
                if bl < best_blend_cal:
                    best_blend_cal = bl
                    best_w = w.tolist()
                    best_cal_oof_blend = cal_blend.copy()
                    blend_test = sum(w[i]*group_tests[i] for i in range(n_groups))
                    s_t = train_rate - blend_test.mean()
                    best_test_blend = np.clip(blend_test + s_t, 0.0001, 0.9999)

        # Fine search around best
        best_w_arr = np.array(best_w)
        for di in range(n_groups):
            for delta in np.arange(-0.1, 0.11, 0.05):
                test_w = best_w_arr.copy()
                if test_w[di] + delta < 0.05 or test_w[di] + delta > 0.6:
                    continue
                test_w[di] += delta
                remaining = 1.0 - test_w[di]
                others = [i for i in range(n_groups) if i != di]
                for di2 in others:
                    test_w[di2] = remaining / len(others)  # distribute evenly
                    blend = sum(test_w[i]*group_oofs[i] for i in range(n_groups))
                    s = train_rate - blend.mean()
                    cal_blend = np.clip(blend + s, 0.0001, 0.9999)
                    bl = log_loss(y, cal_blend, labels=[0,1])
                    if bl < best_blend_cal:
                        best_blend_cal = bl
                        best_w = test_w.tolist()
                        best_cal_oof_blend = cal_blend.copy()
                        blend_test = sum(test_w[i]*group_tests[i] for i in range(n_groups))
                        s_t = train_rate - blend_test.mean()
                        best_test_blend = np.clip(blend_test + s_t, 0.0001, 0.9999)

        # Use best blend
        final_cal_oof = best_cal_oof_blend
        final_cal_test = best_test_blend

        # OOF-LB gap analysis
        oof_cv = log_loss(y, avg_oof, labels=[0,1])
        cv_before_cal = log_loss(y, avg_oof, labels=[0,1])

        target_results[target] = {
            'oof_cv': round(oof_cv, 6),
            'cal_oof_uniform': round(best_blend_cal, 6),
            'train_rate': round(train_rate, 6),
            'oof_mean': round(avg_oof.mean(), 6),
            'test_mean_uniform': round(cal_test.mean(), 6),
            'test_mean_best': round(final_cal_test.mean(), 6),
            'shift': round(final_cal_test.mean()-train_rate, 6),
            'n_feat': n_feat, 'cfg': cfg_name,
            'best_weights': [round(w,3) for w in best_w],
            'time_s': round(time.time()-t0, 0),
        }

        log.info(f"  OOF CV: {oof_cv:.4f}")
        log.info(f"  CalOOF (uniform): {best_blend_cal:.4f} w={best_w}")
        log.info(f"  oof_mean={avg_oof.mean():.4f} test_mean={final_cal_test.mean():.4f} "
                 f"oof_std={avg_oof.std():.4f} test_std={final_cal_test.std():.4f} "
                 f"shift={final_cal_test.mean()-train_rate:+.4f}")

        predictions[target] = final_cal_test

        del X, Xts, group_oofs, group_tests, m_rank
        gc.collect()

    # === OOF-LB Gap Analysis ===
    log.info("\n" + "="*80)
    log.info("OOF-LB GAP ANALYSIS")
    log.info("="*80)
    v53_lb = {'Q1':0.6547, 'Q2':0.6490, 'Q3':0.6558, 'S1':0.6252, 'S2':0.6749, 'S3':0.6692, 'S4':0.6858}
    v53_oof = {'Q1':0.7696, 'Q2':0.6558, 'Q3':0.6648, 'S1':0.6211, 'S2':0.7495, 'S3':0.6474, 'S4':0.6611}

    for target in TARGETS:
        lb = v53_lb[target]
        oof = v53_oof[target]
        gap = lb - oof
        pct = gap / oof * 100
        log.info(f"  {target}: OOF={oof:.4f} LB={lb:.4f} gap={gap:+.4f} ({pct:+.1f}%)")

    avg_gap = np.mean([v53_lb[t]-v53_oof[t] for t in TARGETS])
    log.info(f"  AVG gap: {avg_gap:+.4f}")

    # === Summary ===
    log.info("\n" + "="*80)
    log.info("V99 SUMMARY")
    log.info("="*80)

    avg_cal = np.mean([v['cal_oof_uniform'] for v in target_results.values()])
    avg_oof_mean = np.mean([v['oof_mean'] for v in target_results.values()])
    avg_test_mean = np.mean([v['test_mean_uniform'] for v in target_results.values()])
    log.info(f"  Total seeds: {total_seeds}")
    log.info(f"  AVG CalOOF (optimized): {avg_cal:.4f}")
    log.info(f"  AVG OOF mean: {avg_oof_mean:.4f}")
    log.info(f"  AVG test mean: {avg_test_mean:.4f}")
    log.info(f"  Δ (vs V53 Swept OOF 0.6813): {avg_cal-0.6813:+.4f}")

    # === Save submission ===
    sample = pd.read_csv(ROOT / "data_raw" / "ch2026_submission_sample.csv")
    sub = sample.copy()
    for t in TARGETS:
        sub[t] = predictions[t]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT / f"submission_v99_blend_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    assert sub.shape == sample.shape
    assert sub.columns.tolist() == sample.columns.tolist()
    assert not sub.isnull().any().any()

    log.info(f"\n✅ V99 Submission saved: {sub_path}")
    log.info(f"  Shape: {sub.shape}")
    for t in TARGETS:
        dr = target_results[t]
        log.info(f"  {t}: cal_oof={dr['cal_oof_uniform']:.4f} mean={sub[t].mean():.4f} w={dr['best_weights']}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")

    # Meta
    meta = {
        'version': 'V99_blend',
        'name': 'LGBM V53 Swept + Multi-Seed (4 groups × 25 seeds) + Weighted Blend',
        'n_seeds_total': total_seeds,
        'n_seed_groups': len(seed_groups),
        'seeds_per_group': n_seeds_per_group,
        'cal_method': 'linear',
        'target_results': target_results,
        'timestamp': datetime.now().isoformat(),
        'submission_file': str(sub_path),
    }
    meta_path = SUBMIT / f'meta_v99_blend_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return sub_path

if __name__ == "__main__":
    main()
