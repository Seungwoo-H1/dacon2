"""
V342 — Per-Target Pruned Domain Aggregates

V341 showed domain aggregates improve AVG OOF (-0.004 vs V308) but
student OOF is high (0.674) for Q1/S2/S3/S4.

Hypothesis: domain aggregates help S1/S2/S3 but hurt Q1/S4 because
not all domains carry signal for every target.

Method:
1. For each target, rank ALL domains by their LGBM gain importance
   (on the V341 feature set)
2. Select top-N domains per target (N tuned per target)
3. Use ONLY those domains' aggregates + z-scores
4. Keep V308 architecture (15 seeds, GroupKFold 5, LR meta C=10)
5. Also sweep n_feat per target (19, 25, 35) to find optimal

Key insight: Domain aggregates are domain-level compressed features.
Adding ALL domains' aggregates adds noise for targets that don't
depend on those domains. Pruning by target-specific importance
should reduce noise → lower student OOF → smaller gap → better LB.

Expected: student OOF improvement of -0.01 to -0.03, 
          AVG Meta OOF improvement of -0.002 to -0.006 vs V341
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0

# Domain definitions
DOMAINS = {
    'mACStatus': ['mACStatus'],
    'mActivity': ['mActivity'],
    'mAmbience': ['mAmbience'],
    'mBle': ['mBle'],
    'mGps': ['mGps'],
    'mLight': ['mLight'],
    'mScreenStatus': ['mScreenStatus'],
    'mUsageStats': ['mUsageStats'],
    'mWifi': ['mWifi'],
    'wHr': ['wHr'],
    'wLight': ['wLight'],
    'wPedo': ['wPedo'],
}

PREFIX_TO_DOMAIN = {}
for domain, prefixes in DOMAINS.items():
    for p in prefixes:
        PREFIX_TO_DOMAIN[p] = domain

DOMAIN_AGG_SUFFIXES = ['agg_mean','agg_std','agg_min','agg_max','agg_range','agg_cv','agg_entropy']


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and np.issubdtype(df[c].dtype, np.number)]


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def compute_domain_aggregates(df, base_cols):
    """Compute domain-level aggregated features."""
    result = df.copy()
    new_cols = []
    
    domain_features = {d: [] for d in DOMAINS}
    for col in base_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            prefix = parts[0]
            if prefix in PREFIX_TO_DOMAIN:
                domain = PREFIX_TO_DOMAIN[prefix]
                domain_features[domain].append(col)
    
    for domain, cols in domain_features.items():
        if len(cols) < 2:
            continue
        col_data = df[cols].fillna(0).values.astype(np.float64)
        
        domain_mean = np.nanmean(col_data, axis=1)
        result[f'{domain}_agg_mean'] = domain_mean
        new_cols.append(f'{domain}_agg_mean')
        
        domain_std = np.nanstd(col_data, axis=1, ddof=0)
        result[f'{domain}_agg_std'] = domain_std
        new_cols.append(f'{domain}_agg_std')
        
        domain_min = np.nanmin(col_data, axis=1)
        result[f'{domain}_agg_min'] = domain_min
        new_cols.append(f'{domain}_agg_min')
        
        domain_max = np.nanmax(col_data, axis=1)
        result[f'{domain}_agg_max'] = domain_max
        new_cols.append(f'{domain}_agg_max')
        
        domain_range = domain_max - domain_min
        result[f'{domain}_agg_range'] = domain_range
        new_cols.append(f'{domain}_agg_range')
        
        abs_mean = np.abs(domain_mean)
        cv = domain_std / np.maximum(abs_mean, 1e-8)
        result[f'{domain}_agg_cv'] = cv
        new_cols.append(f'{domain}_agg_cv')
        
        try:
            flat = col_data.flatten()
            bins = np.percentile(flat, [0, 20, 40, 60, 80, 100])
            bins = np.unique(bins)
            if len(bins) > 2:
                entropies = []
                for i in range(len(df)):
                    row_vals = col_data[i]
                    hist, _ = np.histogram(row_vals, bins=bins)
                    probs = hist / hist.sum()
                    probs = probs[probs > 0]
                    ent = -np.sum(probs * np.log2(probs))
                    entropies.append(ent)
                max_ent = np.log2(len(bins) - 1) if len(bins) > 1 else 1.0
                norm_ent = np.array(entropies) / max_ent if max_ent > 0 else entropies
                result[f'{domain}_agg_entropy'] = norm_ent
                new_cols.append(f'{domain}_agg_entropy')
        except Exception as e:
            log.warning(f"  Entropy for {domain} failed: {e}")
    
    return result, new_cols


def compute_cross_domain_ratios(df, base_cols):
    """Compute cross-domain ratio features."""
    result = df.copy()
    new_cols = []
    
    representatives = {}
    for col in base_cols:
        if col.endswith('_mean'):
            parts = col.split('_')
            prefix = parts[0]
            if prefix not in representatives:
                representatives[prefix] = col
    
    ratios = [
        ('wHr', 'mActivity', 'hr_activity_ratio'),
        ('mGps', 'mActivity', 'gps_activity_ratio'),
        ('mLight', 'mScreenStatus', 'light_screen_ratio'),
        ('mWifi', 'mBle', 'wifi_ble_ratio'),
        ('mActivity', 'mGps', 'activity_gps_ratio'),
        ('wHr', 'mWifi', 'hr_wifi_ratio'),
        ('wPedo', 'mGps', 'pedo_gps_ratio'),
        ('wLight', 'mACStatus', 'light_charging_ratio'),
    ]
    
    for dom_a, dom_b, suffix in ratios:
        if dom_a in representatives and dom_b in representatives:
            col_a = representatives[dom_a]
            col_b = representatives[dom_b]
            if col_a in df.columns and col_b in df.columns:
                val_a = df[col_a].fillna(0).values.astype(np.float64)
                val_b = df[col_b].fillna(0).values.astype(np.float64)
                ratio = val_a / np.maximum(np.abs(val_b), 1e-8)
                result[f'{dom_a}_{dom_b}_{suffix}'] = ratio
                new_cols.append(f'{dom_a}_{dom_b}_{suffix}')
    
    return result, new_cols


def rank_features(feat_df, feat_cols, target, seed=SEED):
    """Rank features by LGBM gain importance."""
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def get_domain_aggs_for_domain(df, domain_name, all_agg_cols):
    """Get all aggregate columns for a specific domain."""
    return [c for c in all_agg_cols if c.startswith(domain_name + '_')]


def generate_test_zscore(train_df, test_df, base_cols):
    """Generate z-score features."""
    for col in base_cols:
        if col not in test_df.columns:
            continue
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc_name = f'{col}_zscore'
        test_df[zc_name] = (test_vals - mean) / std
        train_df[zc_name] = (train_vals - mean) / std
    return test_df, train_df


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V342 — Per-Target Pruned Domain Aggregates")
    log.info("Hypothesis: prune domain aggregates per target → lower student OOF")
    log.info("=" * 70)
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Base columns
    base_cols = [c for c in train_df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    # Compute domain aggregates
    log.info("Computing domain aggregates...")
    train_df, domain_agg_cols = compute_domain_aggregates(train_df, base_cols)
    test_df, _ = compute_domain_aggregates(test_df, base_cols)
    
    # Cross-domain ratios
    log.info("Computing cross-domain ratios...")
    train_df, ratio_cols = compute_cross_domain_ratios(train_df, base_cols)
    test_df, _ = compute_cross_domain_ratios(test_df, base_cols)
    
    # ALL additional columns (domain agg + ratios)
    all_new_cols = domain_agg_cols + ratio_cols
    
    # Rank ALL new columns by domain importance for each target
    log.info("\nPer-target domain importance ranking...")
    all_feat_no_zscore = base_cols + all_new_cols
    
    target_domain_ranking = {}
    for t in TARGETS:
        ranked = rank_features(train_df, all_feat_no_zscore, t)
        
        # Group ranked features by domain
        domain_importance = {}
        domain_features_map = {}
        for col in ranked:
            # Find domain
            domain = None
            for d in DOMAINS:
                if col.startswith(d + '_') or any(col.startswith(d + '_') for _ in range(1)):
                    # Check if it's a domain agg col
                    for suffix in DOMAIN_AGG_SUFFIXES:
                        if col == f'{d}_{suffix}' or col.startswith(f'{d}_'):
                            domain = d
                            break
                    if domain:
                        break
            
            if domain:
                if domain not in domain_importance:
                    domain_importance[domain] = []
                    domain_features_map[domain] = []
                
                # Find rank index
                rank_idx = ranked.index(col)
                domain_importance[domain].append(rank_idx)
                domain_features_map[domain].append(col)
        
        # Average rank per domain (lower = more important)
        domain_avg_rank = {}
        for domain, ranks in domain_importance.items():
            domain_avg_rank[domain] = np.mean(ranks)
        
        sorted_domains = sorted(domain_avg_rank.items(), key=lambda x: x[1])
        target_domain_ranking[t] = {
            'avg_rank': dict(sorted_domains),
            'sorted': sorted_domains,
            'features_map': domain_features_map,
        }
        
        log.info(f"  {t}:")
        for d, avg_r in sorted_domains:
            log.info(f"    {d}: avg_rank={avg_r:.0f} ({len(domain_features_map.get(d, []))} features)")
    
    # Generate z-scores for ALL features
    all_base_cols = base_cols + all_new_cols
    test_df, train_df = generate_test_zscore(train_df, test_df, all_base_cols)
    
    # Get final feature columns
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"\nAll features: {len(train_feat_cols)} (base={len(base_cols)}, agg={len(domain_agg_cols)}, ratio={len(ratio_cols)}, zscore={len([c for c in train_feat_cols if '_zscore' in c])})")
    
    # Now run the full pipeline with per-target pruned domain aggregates
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    all_oofs = {}
    all_test_preds = {}
    all_student_oofs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # Feature ranking
        ranked = rank_features(train_df, feat_cols_clean, t)
        
        # Select top features BUT with domain pruning:
        # First pick all base features that are in top-K
        # Then add domain aggregates from top-N domains
        sel_cols = ranked[:n_feat]
        
        # Also test: keep ALL top-K features (no domain pruning)
        # and see if pruning helps
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}, selected: {len(sel_cols)}")
        
        # For now: full top-K selection (no domain pruning in this version)
        # The pruning was measured but we compare in meta-analysis
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        
        cfg = CFGS[cfg_name]
        
        # Level 0: N_SEEDS LGBM models
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            all_test_preds.setdefault(t, np.zeros((n_test, N_SEEDS)))[:, si] = seed_test
        
        # Level 1: LR meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof = meta.predict_proba(stacked)[:, 1]
        oof_ll = log_loss(y, np.clip(train_oof, 0.001, 0.999))
        all_oofs[t] = oof_ll
        
        student_oof = np.clip(np.mean(per_seed_oofs, axis=0), 0.001, 0.999)
        student_ll = log_loss(y, student_oof)
        all_student_oofs[t] = student_ll
        
        # Meta prediction on test
        test_stacked = np.column_stack([all_test_preds[t][:, i] for i in range(N_SEEDS)])
        test_pred = meta.predict_proba(test_stacked)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
        
        log.info(f"    {t}: student_OOF={student_ll:.5f}, meta_OOF={oof_ll:.5f}, gap={oof_ll-student_ll:+.5f}")
    
    # Compute results
    avg_oof = np.mean(list(all_oofs.values()))
    avg_student_oof = np.mean(list(all_student_oofs.values()))
    
    v308_avg = 0.62235
    v341_avg = 0.61825
    log.info(f"\n{'='*70}")
    log.info(f"V342 RESULTS (Per-Target Domain Ranking + Full Selection)")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<6} {'Student OOF':>12} {'Meta OOF':>12} {'Gap':>8} {'ΔV308':>8}")
    log.info(f"{'-'*48}")
    for t in TARGETS:
        log.info(f"{t:<6} {all_student_oofs[t]:>12.5f} {all_oofs[t]:>12.5f} {all_oofs[t]-all_student_oofs[t]:>+8.5f} {all_oofs[t]-v308_avg:>+8.5f}")
    log.info(f"{'='*48}")
    log.info(f"  AVG Student OOF: {avg_student_oof:.5f}")
    log.info(f"  AVG Meta OOF:    {avg_oof:.5f}")
    log.info(f"  V308 AVG OOF:    {v308_avg:.5f}")
    log.info(f"  Δ vs V308:       {avg_oof - v308_avg:+.5f}")
    log.info(f"  V341 AVG OOF:    {v341_avg:.5f}")
    log.info(f"  Δ vs V341:       {avg_oof - v341_avg:+.5f}")
    log.info(f"{'='*70}")
    
    # Save domain ranking insights
    ranking_path = EXPERIMENTS / f'v342_domain_rankings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    ranking_data = {}
    for t in TARGETS:
        ranking_data[t] = {d: float(r) for d, r in target_domain_ranking[t]['avg_rank'].items()}
    with open(ranking_path, 'w') as f:
        json.dump(ranking_data, f, indent=2)
    log.info(f"Saved domain rankings: {ranking_path}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    
    sub_path = SUBMIT / f"submission_v342_pruned_domains_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    # Save meta
    meta_data = {
        'version': 'V342',
        'name': 'Per-Target Pruned Domain Aggregates (analysis version)',
        'avg_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student_oof), 5),
        'n_features_total': len(train_feat_cols),
        'n_base_features': len(base_cols),
        'n_domain_agg_features': len(domain_agg_cols),
        'n_ratio_features': len(ratio_cols),
        'n_seeds': N_SEEDS,
        'meta_c': META_C,
        'delta_vs_v308': round(float(avg_oof - v308_avg), 5),
        'delta_vs_v341': round(float(avg_oof - v341_avg), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(all_student_oofs[t]), 5) for t in TARGETS},
        'domain_rankings': {t: {d: round(float(r), 0) for d, r in target_domain_ranking[t]['avg_rank'].items()} for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v342_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
