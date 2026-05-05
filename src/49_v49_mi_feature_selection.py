"""
V49 — MI-Based Feature Selection

Hypothesis: Current feature selection uses LightGBM importance (gain-based),
which is biased toward high-cardinality features and features with more split
opportunities. Mutual Information (MI) captures non-linear relationships
independently of the model, providing a complementary view.

Method:
1. Compute MI between each feature and each target using sklearn's
   mutual_info_classif (for binary targets)
2. Compare MI ranking vs LGBM importance ranking
3. Select top-N features using:
   a) MI only
   b) LGBM importance only
   c) Combined: average rank of MI + LGBM (rank fusion)
   d) Combined: sum of normalized scores
4. Evaluate all 4 approaches with same LGBM config
5. Also test feature pruning: remove highly correlated features first

Method details:
- MI uses sklearn's mutual_info_classif with discrete=False (continuous features)
- For speed: use 10 nearest neighbors (n_neighbors=10), which is reasonable for 450 samples
- Correlation pruning: remove one of any pair with |corr| > 0.95
"""

import sys, re, gc, time, warnings, logging, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"
DATA_RAW = ROOT / "data_raw"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)


# ── Leakage columns ──
LEAK_S = {
    'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',
}
LEAK_Q = {
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
}


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


# ── Config ──
CFG = {
    'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500,
    'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10,
}

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def add_personalization(df, feature_cols):
    personal_cols = []
    df = df.copy()
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols


def rank_lgbm_importance(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]


def rank_mi(feat, feat_cols, target, n_neighbors=10):
    """Rank features by mutual information with target."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)

    # MI works better with standardized features for continuous data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False,
                                     n_neighbors=n_neighbors, random_state=42,
                                     copy=True)

    ranked = sorted(zip(feat_cols, mi_scores), key=lambda x: -x[1])
    return [r[0] for r in ranked]


def rankmi_combined(feat, feat_cols, target, n_neighbors=10):
    """Combine MI and LGBM importance via rank fusion."""
    mi_ranked = rank_mi(feat, feat_cols, target, n_neighbors)
    lgb_ranked = rank_lgbm_importance(feat, feat_cols, target)

    # Convert to ranks (1 = best)
    mi_ranks = {f: i + 1 for i, f in enumerate(mi_ranked)}
    lgb_ranks = {f: i + 1 for i, f in enumerate(lgb_ranked)}

    # Combined score: average rank (lower is better)
    combined = {}
    for f in feat_cols:
        mr = mi_ranks.get(f, len(feat_cols))
        lr = lgb_ranks.get(f, len(feat_cols))
        combined[f] = (mr + lr) / 2.0

    ranked = sorted(combined.keys(), key=lambda f: combined[f])
    return ranked


def rank_score_combined(feat, feat_cols, target, n_neighbors=10):
    """Combine MI and LGBM importance via normalized score sum."""
    mi_scores = mutual_info_classif(
        StandardScaler().fit_transform(
            feat[feat_cols].fillna(0).values.astype(np.float64)
        ),
        feat[target].values.astype(np.float64),
        discrete_features=False, n_neighbors=n_neighbors,
        random_state=42, copy=True
    )
    mi_ranked = sorted(zip(feat_cols, mi_scores), key=lambda x: -x[1])

    # LGBM importance
    lgb_ranked = rank_lgbm_importance(feat, feat_cols, target)
    lgb_ranks = {f: i + 1 for i, f in enumerate(lgb_ranked)}

    # Normalize MI scores to [0,1] range
    mi_vals = np.array([s for _, s in mi_ranked])
    if mi_vals.max() > mi_vals.min():
        mi_norm = (mi_vals - mi_vals.min()) / (mi_vals.max() - mi_vals.min())
    else:
        mi_norm = np.zeros_like(mi_vals)

    mi_dict = {f: float(s) for f, s in zip([f for f, _ in mi_ranked], mi_norm)}

    # Combined: normalized MI + normalized LGBM inverse rank
    combined = {}
    for f in feat_cols:
        mi_s = mi_dict.get(f, 0.0)
        lgb_s = 1.0 / lgb_ranks.get(f, len(feat_cols))
        combined[f] = mi_s * 0.5 + lgb_s * 0.5

    ranked = sorted(combined.keys(), key=lambda f: -combined[f])
    return ranked


def remove_correlated_features(cols, X_df, threshold=0.95):
    """Remove one of each highly correlated feature pair (|corr| > threshold)."""
    X = X_df[cols].fillna(0).values
    corr = np.corrcoef(X.T)
    kept = list(cols)
    for i in range(len(cols)):
        if cols[i] not in kept:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] not in kept:
                continue
            if abs(corr[i, j]) > threshold:
                # Remove the one with lower MI (we'll recompute per target later)
                kept.remove(cols[j])
    return kept


def train_cv_oof(feat, cols, target, seeds, n_folds=5):
    """Train LGBM with CV, return OOF average predictions."""
    y = feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]

    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': CFG['nl'], 'max_depth': CFG['md'],
        'learning_rate': CFG['lr'], 'n_estimators': CFG['ne'],
        'subsample': CFG['ss'], 'colsample_bytree': CFG['cb'],
        'reg_alpha': CFG['ra'], 'reg_lambda': CFG['rl'],
        'min_child_samples': CFG['mc'],
    }

    for si, seed in enumerate(seeds):
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=CFG['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
            gc.collect()
    return np.clip(oof.mean(axis=1), 0.0001, 0.9999)


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V49 — MI-Based Feature Selection")
    log.info("=" * 70)

    # ── 1. Load features ──
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Loaded: {feat.shape}")

    feat_cols_raw = get_feature_cols(feat)
    feat, zscore_cols = add_personalization(feat, feat_cols_raw)
    log.info(f"  After personalization: {feat.shape}")

    all_cols = feat_cols_raw + zscore_cols
    train_rates = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Target rates: {train_rates}")

    # ── 2. Correlation pruning (pre-filter) ──
    log.info("\n--- 2. Correlation pruning ---")
    # We do correlation pruning per target since leakage differs
    # First compute correlation on all available (leak-free) features
    X_for_corr = feat[all_cols].fillna(0)
    corr_cols = {}
    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        pruned = remove_correlated_features(leak_cols, feat, threshold=0.95)
        corr_cols[target] = set(pruned)
        log.info(f"  {target}: {len(leak_cols)} → {len(pruned)} after corr pruning")

    # ── 3. Feature ranking for each target/method ──
    log.info("\n--- 3. Feature ranking ---")

    # We'll focus on the first 3 targets (Q1, Q2, Q3) for speed,
    # and use the best findings to infer all 7

    ranking_results = {}

    for target in TARGET_COLS[:3]:  # Q1, Q2, Q3
        tgt_t = time.time()
        leak_cols = remove_leak(all_cols, target)
        leak_cols_pruned = [c for c in leak_cols if c in corr_cols[target]]
        log.info(f"\n  === {target}: {len(leak_cols_pruned)} leak-free, corr-pruned features ===")

        # Method 1: LGBM importance only
        log.info("    Computing LGBM ranking...")
        lgb_ranked = rank_lgbm_importance(feat, leak_cols_pruned, target)

        # Method 2: MI only
        log.info("    Computing MI ranking...")
        mi_ranked = rank_mi(feat, leak_cols_pruned, target, n_neighbors=10)

        # Method 3: Rank fusion (average rank)
        log.info("    Computing rank fusion...")
        fused_ranked = rankmi_combined(feat, leak_cols_pruned, target, n_neighbors=10)

        # Method 4: Score fusion (normalized scores)
        log.info("    Computing score fusion...")
        score_ranked = rank_score_combined(feat, leak_cols_pruned, target, n_neighbors=10)

        # ── Evaluate each ranking method with different n_feat ──
        n_feats = [5, 10, 15, 20]
        method_names = ['lgb', 'mi', 'rank_fusion', 'score_fusion']
        method_ranks = {
            'lgb': lgb_ranked,
            'mi': mi_ranked,
            'rank_fusion': fused_ranked,
            'score_fusion': score_ranked,
        }

        best_method_score = float('inf')
        best_method = None
        best_n = None
        all_method_scores = {}

        for method in method_names:
            all_method_scores[method] = {}
            for n in n_feats:
                cols = method_ranks[method][:n]
                oof = train_cv_oof(feat, cols, target, SEEDS)
                cal = mean_match(oof, train_rates[target])
                cal_loss = log_loss(feat[target].values, cal, labels=[0, 1])
                all_method_scores[method][n] = cal_loss

                if cal_loss < best_method_score:
                    best_method_score = cal_loss
                    best_method = method
                    best_n = n
                    log.info(f"      {method} n={n}: Cal={cal_loss:.4f} ← BEST")
                else:
                    log.info(f"      {method} n={n}: Cal={cal_loss:.4f}")

        ranking_results[target] = {
            'all_scores': all_method_scores,
            'best_method': best_method,
            'best_n': best_n,
            'best_cal': best_method_score,
            'lgb_ranked': lgb_ranked,
            'mi_ranked': mi_ranked,
            'fused_ranked': fused_ranked,
            'score_ranked': score_ranked,
        }

        log.info(f"  {target} ranking+eval time: {time.time()-tgt_t:.0f}s")
        gc.collect()

    # ── 4. Cross-method analysis ──
    log.info(f"\n{'='*70}")
    log.info("V49 SUMMARY (Q1, Q2, Q3)")
    log.info(f"{'='*70}")

    for target in TARGET_COLS[:3]:
        r = ranking_results[target]
        log.info(f"\n  {target}:")
        log.info(f"    Best: {r['best_method']} n={r['best_n']} Cal={r['best_cal']:.4f}")
        for method in ['lgb', 'mi', 'rank_fusion', 'score_fusion']:
            best_for_method = min(r['all_scores'][method].values())
            best_n_for_method = min(r['all_scores'][method], key=r['all_scores'][method].get)
            log.info(f"    {method}: best Cal={best_for_method:.4f} (n={best_n_for_method})")

        # Feature overlap analysis
        lgb_top20 = set(r['lgb_ranked'][:20])
        mi_top20 = set(r['mi_ranked'][:20])
        fused_top20 = set(r['fused_ranked'][:20])
        log.info(f"    LGBM∩MI overlap (top20): {len(lgb_top20 & mi_top20)}")
        log.info(f"    LGBM∩Fused overlap (top20): {len(lgb_top20 & fused_top20)}")
        log.info(f"    MI∩Fused overlap (top20): {len(mi_top20 & fused_top20)}")

    # ── 5. Extend to all targets with best method ──
    # Use the most common best method across Q1-Q3 to predict for S1-S4
    method_counts = {}
    for target in TARGET_COLS[:3]:
        m = ranking_results[target]['best_method']
        method_counts[m] = method_counts.get(m, 0) + 1
    winning_method = max(method_counts, key=method_counts.get)
    winning_n = min(
        [ranking_results[t]['best_n'] for t in TARGET_COLS[:3]
         if ranking_results[t]['best_method'] == winning_method],
        default=20
    )
    log.info(f"\n  Winning method across Q1-Q3: {winning_method} (n_feat={winning_n})")

    # Apply to all targets
    log.info("\n--- 5. Full evaluation on all targets ---")
    all_results = {}

    for target in TARGET_COLS:
        leak_cols = remove_leak(all_cols, target)
        leak_cols_pruned = [c for c in leak_cols if c in corr_cols[target]]

        # Use the winning method's ranking
        if target in ranking_results:
            ranked = ranking_results[target][winning_method + '_ranked']
        else:
            ranked = rankmi_combined(feat, leak_cols_pruned, target, n_neighbors=10)

        sel_cols = ranked[:winning_n]
        oof = train_cv_oof(feat, sel_cols, target, SEEDS)
        cal = mean_match(oof, train_rates[target])
        cal_loss = log_loss(feat[target].values, cal, labels=[0, 1])

        all_results[target] = {
            'method': winning_method,
            'n_feat': winning_n,
            'cal_oof': cal,
            'cal_loss': cal_loss,
            'sel_cols': sel_cols,
        }
        log.info(f"  {target}: {winning_method} n={winning_n} Cal={cal_loss:.4f}")
        gc.collect()

    avg_cal = np.mean([
        log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
        for t in TARGET_COLS
    ])
    log.info(f"\n  V49 Avg Cal (all targets): {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f} ({'✅ IMPROVED' if avg_cal < 0.6038 else '❌ Not improved'})")
    log.info(f"  Total: {time.time()-t_start:.0f}s ({time.time()-t_start:.1f}min)")

    # ── 6. Save OOF ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    oof_path = DATA_PROCESSED / "oof_v49.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")

    # ── 7. Save metadata ──
    meta = {
        'version': 'V49',
        'name': 'MI-Based Feature Selection',
        'avg_cal_loss': avg_cal,
        'v10_cal_loss': 0.6038,
        'delta': avg_cal - 0.6038,
        'best_method': winning_method,
        'best_n_feat': winning_n,
        'per_target': {},
    }
    for target in TARGET_COLS:
        r = all_results[target]
        meta['per_target'][target] = {
            'cal_loss': r['cal_loss'],
            'method': r['method'],
            'n_feat': r['n_feat'],
        }
    if ranking_results:
        # Include comparison for Q1-Q3
        meta['comparison'] = {}
        for target in TARGET_COLS[:3]:
            meta['comparison'][target] = {
                m: {str(k): float(v) for k, v in ranking_results[target]['all_scores'][m].items()}
                for m in ['lgb', 'mi', 'rank_fusion', 'score_fusion']
            }
    meta_path = DATA_PROCESSED / "v49_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"  Metadata saved: {meta_path}")

    return all_results


if __name__ == "__main__":
    main()
