"""
V255: Confirmatory Run for V254 Top Approach (E) with Multiple Seeds

V254 showed Approach E (Cross-target raw features) as best: AVG OOF 0.60177.
Now confirm with multiple seeds per target to match V127 methodology.
Also add a variant with fewer shared features.
"""

import os, sys, gc, re, json, time, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}
N_FOLDS = 5
N_SEEDS = 5  # seeds per target per fold

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def rank_features_once(train_feat, all_feat_cols, target, seed=42):
    y = train_feat[target].values.astype(np.float64)
    X = train_feat[all_feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]

    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in all_feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(all_feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def approach_e_seed_variants(feat):
    """
    Approach E variants with multi-seed ensemble:
    - E1: Cross-target raw features, 1 seed (baseline from V254)
    - E2: Cross-target raw features, 5 seeds (mean ensemble)
    - E3: Cross-target top-100 features only (reduce noise), 5 seeds
    - E4: Cross-target top-50 features only, 5 seeds
    """
    all_feat_cols = get_feature_cols(feat)
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    results = {}

    for variant_name, n_feat_limit, n_seeds in [
        ('E1_CrossTarget_1seed', None, 1),
        ('E2_CrossTarget_5seed', None, 5),
        ('E3_CrossTop100_5seed', 100, 5),
        ('E4_CrossTop50_5seed', 50, 5),
    ]:
        print(f"\n  [{variant_name}] n_feat_limit={n_feat_limit}, n_seeds={n_seeds}")

        oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

        for t in TARGETS:
            y = feat[t].values.astype(np.float64)
            other_targets = [ot for ot in TARGETS if ot != t]
            extended_cols = all_feat_cols + other_targets

            ranked = rank_features_once(feat, extended_cols, t)
            if n_feat_limit:
                sel_cols = ranked[:n_feat_limit]
            else:
                sel_cols = ranked[:V53_SWEEP[t]['n_feat']]

            # For single-seed, use fold-based seeds; for multi-seed, use multiple seeds per fold
            all_fold_seeds = [SEED + i for i in range(n_seeds)]

            for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values
                X_val = feat[sel_cols].iloc[val_idx].fillna(0).values
                y_tr, y_val = y[tr_idx], y[val_idx]

                if n_seeds == 1:
                    spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                    cfg_name = V53_SWEEP[t]['cfg']
                    base_cfg = CFGS[cfg_name]
                    params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
                    oof_preds[t][val_idx] = m.predict(X_val)
                else:
                    # Multiple seeds: average predictions
                    fold_seed_preds = []
                    for seed_idx, seed in enumerate(all_fold_seeds):
                        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                        cfg_name = V53_SWEEP[t]['cfg']
                        base_cfg = CFGS[cfg_name]
                        params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': seed,
                                  'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                        sn = [sanitize_col(c) for c in sel_cols]
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                        m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
                        fold_seed_preds.append(m.predict(X_val))
                    oof_preds[t][val_idx] = np.mean(fold_seed_preds, axis=0)

        # Compute OOF
        per_target_oof = {}
        for t in TARGETS:
            ll = log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999))
            per_target_oof[t] = ll

        avg_oof = np.mean(list(per_target_oof.values()))
        print(f"    AVG OOF: {avg_oof:.5f}")
        for t in TARGETS:
            print(f"      {t}: {per_target_oof[t]:.5f}")

        results[variant_name] = {'avg_oof': avg_oof, 'per_target_oof': per_target_oof}

        # Clean up
        for t in TARGETS:
            del oof_preds[t]
        gc.collect()

    return results


def approach_g_adaptive_cross_target(feat):
    """
    Approach G: Adaptive cross-target features.
    Only add other targets that are strongly correlated with the current target.
    This avoids adding noisy "target noise" features.
    """
    print(f"\n  [G_AdaptiveCrossTarget]")

    # First, compute inter-target correlation on training data
    base_feat_cols = get_feature_cols(feat)
    target_matrix = feat[TARGETS].values

    corr_matrix = np.corrcoef(target_matrix.T)
    print("    Inter-target correlation:")
    for i, t1 in enumerate(TARGETS):
        for j, t2 in enumerate(TARGETS):
            if i < j:
                print(f"      {t1}-{t2}: {corr_matrix[i, j]:.3f}")

    # For each target, add only correlated other targets as features
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    threshold = 0.3  # only add if correlation > threshold

    results = {}
    oof_preds = {t: np.zeros(len(feat)) for t in TARGETS}

    for t in TARGETS:
        t_idx = TARGETS.index(t)
        # Find correlated targets
        corr_scores = [corr_matrix[t_idx, j] for j in range(len(TARGETS)) if j != t_idx]
        # For simplicity, use top-3 correlated targets
        correlated = [TARGETS[j] for j in range(len(TARGETS)) if j != TARGETS.index(t) and corr_matrix[t_idx, j] > threshold]
        if len(correlated) > 3:
            correlated = [correlated[k] for k in sorted(range(len(correlated)), key=lambda k: -corr_matrix[t_idx, TARGETS.index(correlated[k])])[:3]]

        extended_cols = base_feat_cols + correlated

        y = feat[t].values.astype(np.float64)
        ranked = rank_features_once(feat, extended_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        sel_cols = ranked[:n_feat]

        fold_lls = []
        fold_pred = np.zeros(len(feat))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(feat, y, group)):
            X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values
            X_val = feat[sel_cols].iloc[val_idx].fillna(0).values
            y_tr, y_val = y[tr_idx], y[val_idx]

            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            base_cfg = CFGS[cfg_name]
            params = {**base_cfg, 'scale_pos_weight': spw, 'random_state': SEED + fold,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            sn = [sanitize_col(c) for c in sel_cols]
            ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
            m = lgb.train(params, ds, num_boost_round=base_cfg['n_estimators'])
            fold_pred[val_idx] = m.predict(X_val)
            fold_lls.append(log_loss(y_val, np.clip(fold_pred[val_idx], 0.001, 0.999)))

        oof_preds[t] = fold_pred
        print(f"    {t}: OOF={np.mean(fold_lls):.5f} (corr targets={correlated}, feat={len(sel_cols)})")

    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    print(f"    AVG OOF: {avg_oof:.5f}")
    results['G_AdaptiveCrossTarget'] = {'avg_oof': avg_oof, 'per_target_oof': {t: log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS}}

    for t in TARGETS:
        del oof_preds[t]
    gc.collect()
    return results


SEED = 42


def main():
    t_start = time.time()
    print("=" * 70)
    print("V255: Confirmatory Multi-Seed Cross-Target Experiments")
    print("=" * 70)

    feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
    print(f"\nData: {feat.shape}, Features: {len(get_feature_cols(feat))}")

    results = {}

    # Run variants
    print("\n" + "=" * 70)
    print("Part 1: E1-E4 (Cross-target with varying seeds/feature counts)")
    print("=" * 70)
    e_results = approach_e_seed_variants(feat)
    results.update(e_results)

    print("\n" + "=" * 70)
    print("Part 2: G (Adaptive cross-target)")
    print("=" * 70)
    g_results = approach_g_adaptive_cross_target(feat)
    results.update(g_results)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'Approach':<40} {'AVG OOF':>10} {'Δ vs E1':>10}")
    e1_oof = results.get('E1_CrossTarget_1seed', {}).get('avg_oof', None)

    for name, data in sorted(results.items(), key=lambda x: x[1]['avg_oof']):
        avg = data['avg_oof']
        delta = avg - e1_oof if e1_oof else 0
        print(f"{name:<40} {avg:>10.5f} {delta:>+10.5f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = EXPERIMENTS / f'v255_cross_target_confirmatory_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {result_path}")

    return results


if __name__ == '__main__':
    main()
