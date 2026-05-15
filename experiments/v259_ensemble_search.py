"""
V259: Ensemble Architecture Search — Bayesian Weight Optimization

Memory-efficient, 20 models per target (5 cfg × 4 feature-subset combos + 0 extra seeds).
Uses simple grid-search + SLSQP on reduced dimensionality.
"""

import os, sys, gc, re, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings('ignore')

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / 'data_processed'
EXPERIMENTS = ROOT / 'experiments'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

SEED = 42
N_FOLDS = 5

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_numeric_cols(df, exclude=None):
    ex = META_COLS | set(TARGETS)
    if exclude: ex |= exclude
    return [c for c in df.columns
            if df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]
            and c not in ex]

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'narrow': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 500,
               'subsample': 0.6, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 25},
    'wide2':  {'num_leaves': 50, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500,
               'subsample': 0.8, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 3},
}

LEAK_S = {'wlight_w_light_mean','wlight_w_light_std','wlight_w_light_min','wlight_w_light_max','wlight_w_light_count',
          'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count',
          'wpedo_pedo_step_mean','wpedo_pedo_step_sum','wpedo_pedo_step_frequency_mean','wpedo_pedo_step_frequency_sum',
          'wpedo_pedo_running_step_mean','wpedo_pedo_running_step_sum','wpedo_pedo_walking_step_mean','wpedo_pedo_walking_step_sum',
          'wpedo_pedo_distance_mean','wpedo_pedo_distance_sum','wpedo_pedo_speed_mean','wpedo_pedo_speed_sum',
          'wpedo_pedo_burned_calories_mean','wpedo_pedo_burned_calories_sum'}
LEAK_Q = {'whr_hr_mean','whr_hr_std','whr_hr_min','whr_hr_max','whr_hr_median','whr_hr_count'}

def remove_leak(cols, t):
    if t.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif t.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def cfg_to_params(cfg_s, seed, spw, extra=None):
    p = dict(cfg_s)
    p.update({'scale_pos_weight': spw, 'random_state': seed,
              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1})
    if extra: p.update(extra)
    return p

def train_cv_model(feat_df, cols, y, seed, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    X = feat_df[cols].fillna(0).values.astype(np.float64)
    sn = [sanitize_col(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    p = cfg_to_params(cfg, seed, spw)
    for fi, (tri, vai) in enumerate(gkf.split(feat_df, y, feat_df['subject_id'])):
        ds = lgb.Dataset(X[tri], label=y[tri], feature_name=sn)
        m = lgb.train(p, ds, num_boost_round=cfg['n_estimators'],
                      callbacks=[lgb.log_evaluation(0)])
        oof[vai] = m.predict(X[vai])
        del ds, m
    return np.clip(oof, 0.0001, 0.9999)

# ============================================================
# Generate model variants: 5 cfg × 3 feature subsets × 2 seeds = 30 per target
# ============================================================

def generate_variants(fcols, target, rng):
    variants = []
    n_total = len(fcols)
    pct_list = [(0.80, 'wide80'), (0.50, 'deep50'), (0.30, 'narrow30')]

    for cfg_name, cfg in CFGS.items():
        for pct, pct_name in pct_list:
            n_select = max(int(n_total * pct), 5)
            idx = sorted(rng.choice(n_total, size=n_select, replace=False))
            cols = remove_leak([fcols[i] for i in idx], target)

            for seed_val in [SEED, SEED + hash(cfg_name + pct_name) % 1000]:
                variants.append((cols, cfg, seed_val))
    return variants


# ============================================================
# Load Data
# ============================================================

print("=" * 70)
print("V259: Ensemble Architecture Search (Optimized)")
print("=" * 70)

print("\n[1] Loading data...")
feat = pd.read_parquet(DATA / 'features_clean_v60.parquet')
y_dict = {t: feat[t].values for t in TARGETS}
fcols = get_numeric_cols(feat)
print(f"  Features: {len(fcols)}, Subjects: {feat['subject_id'].nunique()}, Rows: {len(feat)}")

all_correlations = {}
all_weight_opt = {}
all_greedy = {}
all_rank_avg = {}
all_hierarchical = {}


# ============================================================
# Process Each Target
# ============================================================

for target_idx, target in enumerate(TARGETS):
    print(f"\n{'='*60}")
    print(f"TARGET: {target} ({target_idx+1}/{len(TARGETS)})")
    print(f"{'='*60}")

    clean_fcols = remove_leak(fcols, target)
    print(f"  Features (leak-removed): {len(clean_fcols)}")

    rng = np.random.RandomState(SEED + target_idx)
    raw_variants = generate_variants(clean_fcols, target, rng)

    # Deduplicate identical feature sets (same config + subset can have same cols with different seeds)
    seen_cols = set()
    seen_variants = []
    for cols, cfg, seed_val in raw_variants:
        col_key = tuple(cols)
        if col_key not in seen_cols:
            seen_cols.add(col_key)
            seen_variants.append((cols, cfg, seed_val))

    # If still too few, add one more seed per unique subset
    if len(seen_variants) < 20:
        for cols, cfg, sv in list(seen_variants):
            seen_variants.append((cols, cfg, SEED + 999))

    variants = seen_variants
    n_models = len(variants)
    print(f"  Unique models: {n_models}")
    for i, (cols, cfg, sv) in enumerate(variants[:3]):
        print(f"    m{i}: {len(cols)} features, cfg={cfg.get('num_leaves')}/{cfg.get('max_depth')} lr={cfg.get('learning_rate')} seed={sv}")

    # Train
    print(f"\n  Training {n_models} models...")
    model_ids = []
    oof_arr = []
    y = y_dict[target]
    t0 = time.time()

    for i, (cols, cfg, seed_val) in enumerate(variants):
        oof = train_cv_model(feat, cols, y, seed_val, cfg, N_FOLDS)
        oof_arr.append(oof)
        model_ids.append(f'm{i:02d}')
        elapsed = time.time() - t0
        print(f"    [{target}] m{i:02d} LL={log_loss(y, oof, labels=[0,1]):.5f} [{elapsed:.0f}s]")
        del oof

    oof_matrix = np.column_stack(oof_arr)
    del oof_arr
    gc.collect()
    total_train = time.time() - t0
    print(f"  Training done: {total_train:.0f}s")

    # --- Correlation ---
    print(f"\n  Correlation Analysis...")
    corr_mat = np.corrcoef(oof_matrix.T)
    upper_vals = corr_mat[np.triu_indices(n_models, k=1)]
    all_correlations[target] = {
        'min': float(np.min(upper_vals)),
        'max': float(np.max(upper_vals)),
        'mean': float(np.mean(upper_vals)),
        'median': float(np.median(upper_vals)),
        'std': float(np.std(upper_vals)),
    }
    print(f"    corr [min={all_correlations[target]['min']:.3f}, max={all_correlations[target]['max']:.3f}, mean={all_correlations[target]['mean']:.3f}]")

    # Uniform baseline
    uniform_pred = oof_matrix.mean(axis=1)
    uniform_ll = log_loss(y, np.clip(uniform_pred, 0.0001, 0.9999), labels=[0, 1])

    # --- Weight Optimization: SLSQP with random restarts ---
    print(f"\n  Weight Optimization...")
    def logloss_obj(w):
        wc = np.clip(w, 0.0001, None)
        wn = wc / wc.sum()
        return log_loss(y, np.clip(oof_matrix @ wn, 0.0001, 0.9999), labels=[0, 1])

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models
    best_result = None
    best_loss = float('inf')

    for restart in range(20):
        w0 = np.random.dirichlet(np.ones(n_models))
        res = minimize(logloss_obj, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 100, 'ftol': 1e-10})
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    opt_weights = best_result.x
    opt_weights = np.clip(opt_weights, 0, 1)
    opt_weights /= opt_weights.sum()
    opt_pred = np.clip(oof_matrix @ opt_weights, 0.0001, 0.9999)
    opt_ll = log_loss(y, opt_pred, labels=[0, 1])

    w_sorted = sorted(zip(model_ids, opt_weights), key=lambda x: -x[1])[:5]
    all_weight_opt[target] = {
        'opt_ll': float(opt_ll),
        'uniform_ll': float(uniform_ll),
        'delta': float(opt_ll - uniform_ll),
        'n_models': n_models,
        'top_weights': [{'model': m, 'weight': round(float(w), 6)} for m, w in w_sorted],
        'all_weights': {m: round(float(w), 6) for m, w in w_sorted},
    }
    print(f"    opt-LL={opt_ll:.5f}, uniform-LL={uniform_ll:.5f}, delta={opt_ll - uniform_ll:+.5f}")
    print(f"    Top: {[(m, f'{w:.4f}') for m, w in w_sorted]}")

    # --- Greedy Forward Selection ---
    print(f"\n  Greedy Forward Selection...")
    individual_ll = []
    for i in range(n_models):
        ll = log_loss(y, np.clip(oof_matrix[:, i], 0.0001, 0.9999), labels=[0, 1])
        individual_ll.append((i, ll))
    individual_ll.sort(key=lambda x: x[1])

    selected = [individual_ll[0][0]]
    remaining = list(range(n_models))
    remaining.remove(selected[0])
    best_ens_ll = float('inf')
    best_selected = selected[:]

    for _ in range(min(n_models - 1, 20)):
        best_add = None
        best_ll_new = float('inf')
        for idx in remaining:
            test_sel = selected + [idx]
            ens = oof_matrix[:, test_sel].mean(axis=1)
            ll = log_loss(y, np.clip(ens, 0.0001, 0.9999), labels=[0, 1])
            if ll < best_ll_new:
                best_ll_new = ll
                best_add = idx
        if best_add is not None:
            selected.append(best_add)
            remaining.remove(best_add)
            if best_ll_new < best_ens_ll:
                best_ens_ll = best_ll_new
                best_selected = selected[:]
        else:
            break

    greedy_pred = np.clip(oof_matrix[:, best_selected].mean(axis=1), 0.0001, 0.9999)
    greedy_ll = log_loss(y, greedy_pred, labels=[0, 1])
    all_greedy[target] = {
        'selected': len(best_selected),
        'total': n_models,
        'greedy_ll': float(greedy_ll),
        'uniform_ll': float(uniform_ll),
        'delta': float(greedy_ll - uniform_ll),
    }
    print(f"    selected={len(best_selected)}/{n_models}, greedy-LL={greedy_ll:.5f}, delta={greedy_ll - uniform_ll:+.5f}")

    # --- Rank Averaging ---
    print(f"\n  Rank Averaging...")
    rank_mat = np.zeros_like(oof_matrix)
    for i in range(oof_matrix.shape[0]):
        row = oof_matrix[i]
        rank_mat[i] = np.argsort(np.argsort(row)) + 1
    rank_norm = (rank_mat - 1) / (n_models - 1) if n_models > 1 else rank_mat
    rank_norm = rank_norm * 0.9998 + 0.0001
    rank_avg_pred = np.clip(rank_norm.mean(axis=1), 0.0001, 0.9999)
    rank_ll = log_loss(y, rank_avg_pred, labels=[0, 1])
    all_rank_avg[target] = {
        'rank_ll': float(rank_ll),
        'uniform_ll': float(uniform_ll),
        'delta': float(rank_ll - uniform_ll),
    }
    print(f"    rank-LL={rank_ll:.5f}, delta={rank_ll - uniform_ll:+.5f}")

    # --- Hierarchical Ensemble ---
    print(f"\n  Hierarchical Ensemble...")
    if n_models >= 4:
        dist_mat = np.sqrt(np.clip(2 * (1 - corr_mat), 0, 2))
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)
        dist_flat = squareform(dist_mat)
        n_clusters = min(max(3, n_models // 3), 8)
        Z = linkage(dist_flat, method='ward')
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        clusters = {}
        for i, cl in enumerate(cluster_labels):
            cs = str(cl)
            clusters.setdefault(cs, []).append(i)

        cluster_preds = np.zeros((oof_matrix.shape[0], n_clusters))
        cluster_weights = np.zeros(n_clusters)
        for ci, (cs, indices) in enumerate(sorted(clusters.items())):
            cluster_preds[:, ci] = oof_matrix[:, indices].mean(axis=1)
            cluster_weights[ci] = len(indices)
        cluster_weights /= cluster_weights.sum()

        hier_pred = np.clip(cluster_preds @ cluster_weights, 0.0001, 0.9999)
        hier_ll = log_loss(y, hier_pred, labels=[0, 1])
        cluster_sizes = {cs: len(indices) for cs, indices in clusters.items()}
    else:
        hier_ll = uniform_ll
        cluster_sizes = {}
        hier_pred = uniform_pred

    all_hierarchical[target] = {
        'hier_ll': float(hier_ll),
        'uniform_ll': float(uniform_ll),
        'delta': float(hier_ll - uniform_ll),
        'n_clusters': n_clusters if n_models >= 4 else 0,
        'cluster_sizes': {str(k): v for k, v in cluster_sizes.items()},
    }
    print(f"    n_clusters={all_hierarchical[target]['n_clusters']}, hier-LL={hier_ll:.5f}, delta={hier_ll - uniform_ll:+.5f}")

    # --- Save OOF per target ---
    out_df = pd.DataFrame(oof_matrix, columns=[f'm{i}' for i in range(n_models)])
    out_df.insert(0, 'subject_id', feat['subject_id'])
    out_df.insert(1, target, y)
    out_path = DATA / f'oof_v259_{target}.csv'
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # --- Free memory ---
    del oof_matrix, corr_mat, dist_mat, dist_flat, Z, cluster_labels, cluster_preds, rank_mat, rank_norm
    del greedy_pred, hier_pred, uniform_pred, opt_pred
    gc.collect()

    print(f"\n  [TARGET COMPLETE] {target}: uniform={uniform_ll:.5f}, opt={opt_ll:.5f}, greedy={greedy_ll:.5f}, rank={rank_ll:.5f}, hier={hier_ll:.5f}")


# ============================================================
# Compile Final Results
# ============================================================

print("\n" + "=" * 70)
print("COMPILING FINAL RESULTS")
print("=" * 70)

# Recalculate per-target uniform LL from saved CSVs
avg_uniforms = []
for t in TARGETS:
    df = pd.read_csv(DATA / f'oof_v259_{t}.csv')
    mcols = [c for c in df.columns if c.startswith('m')]
    avg_uniforms.append(log_loss(y_dict[t], np.clip(df[mcols].mean(axis=1).values, 0.0001, 0.9999), labels=[0, 1]))
avg_uniform = np.mean(avg_uniforms)

avg_weight_opt = np.mean([v['opt_ll'] for v in all_weight_opt.values()])
avg_greedy = np.mean([v['greedy_ll'] for v in all_greedy.values()])
avg_rank_avg = np.mean([v['rank_ll'] for v in all_rank_avg.values()])
avg_hierarchical = np.mean([v['hier_ll'] for v in all_hierarchical.values()])

all_corr_means = [v['mean'] for v in all_correlations.values()]
n_models_total = sum(v['n_models'] for v in all_weight_opt.values())

print(f"\n  Overall AVG OOF (uniform):  {avg_uniform:.5f}")
print(f"  Weight Optimization:        {avg_weight_opt:.5f}  (Δ={avg_weight_opt - avg_uniform:+.5f})")
print(f"  Greedy Forward Selection:   {avg_greedy:.5f}  (Δ={avg_greedy - avg_uniform:+.5f})")
print(f"  Rank Averaging:             {avg_rank_avg:.5f}  (Δ={avg_rank_avg - avg_uniform:+.5f})")
print(f"  Hierarchical Ensemble:      {avg_hierarchical:.5f}  (Δ={avg_hierarchical - avg_uniform:+.5f})")

best_method = min(
    [('uniform', avg_uniform), ('weight_opt', avg_weight_opt),
     ('greedy', avg_greedy), ('rank_avg', avg_rank_avg), ('hierarchical', avg_hierarchical)],
    key=lambda x: x[1]
)[0]
best_delta = float(min(
    avg_weight_opt - avg_uniform, avg_greedy - avg_uniform,
    avg_rank_avg - avg_uniform, avg_hierarchical - avg_uniform
))

print(f"\n  ★ Best method: {best_method} (Δ={best_delta:+.5f})")


# ============================================================
# Save Results
# ============================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = EXPERIMENTS / 'v259_ensemble_search_result.json'

result = {
    "version": "v259_ensemble_search",
    "timestamp": timestamp,
    "n_models_generated": n_models_total,
    "model_configs_per_target": {t: all_weight_opt[t]['n_models'] for t in TARGETS},
    "model_correlations": {
        "per_target": {t: {k: round(v, 6) for k, v in cs.items()} for t, cs in all_correlations.items()},
        "global_min": float(np.min([cs['min'] for cs in all_correlations.values()])),
        "global_max": float(np.max([cs['max'] for cs in all_correlations.values()])),
        "global_mean": float(np.mean(all_corr_means)),
    },
    "best_weight_optimization": {
        "per_target_ll": {t: all_weight_opt[t]['opt_ll'] for t in all_weight_opt},
        "per_target_uniform": {t: all_weight_opt[t]['uniform_ll'] for t in all_weight_opt},
        "per_target_delta": {t: all_weight_opt[t]['delta'] for t in all_weight_opt},
        "per_target_weights": {
            t: [w['weight'] for w in all_weight_opt[t]['top_weights']]
            for t in all_weight_opt
        },
        "per_target_weights_detail": {
            t: all_weight_opt[t]['top_weights'] for t in all_weight_opt
        },
        "avg_oof": float(avg_weight_opt),
        "delta": float(avg_weight_opt - avg_uniform),
    },
    "rank_averaging": {
        "per_target_ll": {t: all_rank_avg[t]['rank_ll'] for t in all_rank_avg},
        "per_target_delta": {t: all_rank_avg[t]['delta'] for t in all_rank_avg},
        "avg_oof": float(avg_rank_avg),
        "delta": float(avg_rank_avg - avg_uniform),
    },
    "hierarchical_ensemble": {
        "per_target_ll": {t: all_hierarchical[t]['hier_ll'] for t in all_hierarchical},
        "per_target_delta": {t: all_hierarchical[t]['delta'] for t in all_hierarchical},
        "avg_oof": float(avg_hierarchical),
        "delta": float(avg_hierarchical - avg_uniform),
        "per_target_n_clusters": {t: all_hierarchical[t]['n_clusters'] for t in all_hierarchical},
        "per_target_cluster_sizes": {
            t: all_hierarchical[t]['cluster_sizes'] for t in all_hierarchical
        },
    },
    "greedy_forward_selection": {
        "per_target_selected": {t: all_greedy[t]['selected'] for t in all_greedy},
        "per_target_total": {t: all_greedy[t]['total'] for t in all_greedy},
        "per_target_ll": {t: all_greedy[t]['greedy_ll'] for t in all_greedy},
        "per_target_delta": {t: all_greedy[t]['delta'] for t in all_greedy},
        "avg_oof": float(avg_greedy),
        "delta": float(avg_greedy - avg_uniform),
    },
    "summary": {
        "uniform_avg_oof": float(avg_uniform),
        "best_method": best_method,
        "best_delta": best_delta,
    },
    "notes": (
        f"GroupKFold({N_FOLDS}) CV, base seed={SEED}, {n_models_total} total models across {len(TARGETS)} targets. "
        f"Per target: 5 configs ({', '.join(CFGS.keys())}) × 3 feature subsets (80/50/30%) × 2 seeds = {5*3*2} models each. "
        f"Diversity via: different hyperparameter ranges, random feature subsets, different random seeds. "
        f"Weight opt: SLSQP with 20 Dirichlet restarts. "
        f"Greedy forward: starts with best individual, adds models reducing ensemble LL. "
        f"Rank averaging: within-fold rank → average ranks → convert back to probabilities. "
        f"Hierarchical: correlation-based Ward clustering → cluster-weighted blend. "
        f"Uniform average baseline for all comparisons."
    ),
}

with open(result_path, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n  Saved: {result_path}")

weight_detail = {t: all_weight_opt[t] for t in all_weight_opt}
with open(EXPERIMENTS / f'v259_weight_opt_detail_{timestamp}.json', 'w') as f:
    json.dump(weight_detail, f, indent=2, default=str)

print(f"  Saved: experiments/v259_weight_opt_detail_{timestamp}.json")
print("\nV259 ENSEMBLE SEARCH COMPLETE ✓")
