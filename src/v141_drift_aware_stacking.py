"""
V141 — Drift-Aware Proper Stacking

Hypothesis: V140 succeeds because OOF≈LB (stable generalization).
V141 improves by:
  1. Removing HIGH-PSI drift features that cause train/test distribution mismatch
  2. Weighting features by fold-stability (low CV importance)
  3. Using 5 seeds for more diversity + LR C=3.0 (slightly less regularized)
  4. Adversarial sample weighting to make validation more test-like

Architecture:
  Level 0: 5 LGBM models per target (5 seeds, GroupKFold 5-fold)
           with drift-weighted features
  Level 1: LR meta-learner (C=3.0)
  Feature selection: stability-filtered top-K

Known from analysis:
  - Adv AUC = 0.656 (moderate drift)
  - 15+ HIGH PSI features (GPS, HR, wifi, screen)
  - Stable features: mACStatus_m_charging_max/min, mLight_m_light_min, etc.
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
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
N_SEEDS = 5
META_C = 3.0  # slightly less regularized than V140 (0.1)


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


# ================================================================
# Adversarial Validation (for feature importance)
# ================================================================

def compute_adversarial_importance(train_df, test_df, feat_cols):
    """Train domain classifier → get drift importance for each feature."""
    X_all = pd.concat([train_df[feat_cols], test_df[feat_cols]], axis=0).fillna(0).values.astype(np.float64)
    y_adv = np.array([0]*len(train_df) + [1]*len(test_df))
    
    nzv = np.ptp(X_all, axis=0)
    feat_mask = nzv > 1e-6
    san_names = [sanitize_col(feat_cols[i]) for i in np.where(feat_mask)[0]]
    X_adv = X_all[:, feat_mask]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    adv_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv, y_adv)):
        pos_w = (y_adv[tr_idx] == 0).sum() / max((y_adv[tr_idx] == 1).sum(), 1)
        params = {'num_leaves': 31, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w,
                  'random_state': 42, 'verbose': -1, 'n_jobs': 1}
        ds = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx], feature_name=san_names)
        m = lgb.train(params, ds, num_boost_round=200)
        pred = m.predict(X_adv[va_idx])
        adv_aucs.append(roc_auc_score(y_adv[va_idx], pred))
    
    adv_auc = np.mean(adv_aucs)
    
    # Full model for feature importance
    pos_w = (y_adv == 0).sum() / max((y_adv == 1).sum(), 1)
    params = {'num_leaves': 31, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w,
              'random_state': 42, 'verbose': -1, 'n_jobs': 1}
    ds_full = lgb.Dataset(X_adv, label=y_adv, feature_name=san_names)
    m_full = lgb.train(params, ds_full, num_boost_round=200)
    imp = m_full.feature_importance(importance_type='gain')
    
    # Build mapping: san_name → orig_idx
    feat_idx_map = {san_names[j]: np.where(feat_mask)[0][j] for j in range(len(san_names))}
    
    # Compute PSI for each feature
    n_train = len(train_df)
    psi_scores = {}
    for i, san_name in enumerate(san_names):
        orig_idx = feat_idx_map[san_name]
        tr_dist = X_all[:n_train, orig_idx]
        te_dist = X_all[n_train:, orig_idx]
        bins = np.percentile(tr_dist, np.arange(0, 101, 5))
        bins = np.unique(bins)
        if len(bins) < 3:
            psi_scores[feat_cols[orig_idx]] = 999.0
            continue
        tr_hist, _ = np.histogram(tr_dist, bins=bins)
        te_hist, _ = np.histogram(te_dist, bins=bins)
        tr_pct = (tr_hist + 0.5) / n_train
        te_pct = (te_hist + 0.5) / n_train
        psi = np.sum((te_pct - tr_pct) * np.log(te_pct / tr_pct))
        psi_scores[feat_cols[orig_idx]] = float(psi)
    
    return adv_auc, psi_scores


# ================================================================
# Feature stability computation
# ================================================================

def compute_feature_stability(train_df, feat_cols):
    """Compute per-target feature importance CV across GroupKFold folds."""
    group = train_df['subject_id'].values
    stability = {}
    
    for t in TARGETS:
        y = train_df[t].values.astype(np.float64)
        fc_leaked = remove_leak(feat_cols, t)
        gkf = GroupKFold(n_splits=N_FOLDS)
        
        imps = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
            X_tr = train_df[fc_leaked].iloc[tr_idx].fillna(0).values.astype(np.float64)
            y_tr = y[tr_idx]
            spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            cfg_name = V53_SWEEP[t]['cfg']
            cfg = CFGS[cfg_name]
            params = {**cfg, 'scale_pos_weight': spw, 'random_state': SEED,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            ds = lgb.Dataset(X_tr, label=y_tr)
            m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
            imp = m.feature_importance(importance_type='gain')
            imps.append(imp)
        
        imps = np.array(imps)
        mean_imp = imps.mean(axis=0)
        std_imp = imps.std(axis=0)
        cv_imp = std_imp / (mean_imp + 1e-10)
        stability[t] = {
            'feat_cols': fc_leaked,
            'mean_imp': mean_imp,
            'cv_imp': cv_imp,
        }
    
    return stability


# ================================================================
# Stability-weighted feature selection
# ================================================================

def select_features_stability_weighted(train_df, feat_cols, psi_scores, stability, target, n_feat):
    """
    Select top-K features, but:
    1. Penalize HIGH-PSI drift features (multiply importance by 1/(1+PSI))
    2. Prefer stable features (low CV importance)
    """
    y = train_df[target].values.astype(np.float64)
    fc_leaked = remove_leak(feat_cols, target)
    
    # Get stability info for this target
    stab = stability[target]
    fc_idx = {c: i for i, c in enumerate(fc_leaked)}
    
    # Combine: base importance × drift penalty × stability bonus
    adjusted_imp = np.zeros(len(fc_leaked))
    for i, col in enumerate(fc_leaked):
        base_imp = stab['mean_imp'][i]
        psi = psi_scores.get(col, 0.0)
        
        # Drift penalty: high PSI → lower weight
        drift_penalty = 1.0 / (1.0 + psi)
        
        # Stability bonus: low CV → higher weight
        stability_bonus = 1.0 / (1.0 + stab['cv_imp'][i])
        
        adjusted_imp[i] = base_imp * drift_penalty * stability_bonus
    
    # Rank by adjusted importance
    ranked = sorted(zip(fc_leaked, adjusted_imp), key=lambda x: -x[1])
    
    # Select top-K
    selected = [r[0] for r in ranked[:n_feat]]
    
    # Report
    top_drift_in_selection = 0
    for col in selected:
        psi = psi_scores.get(col, 0.0)
        if psi > 0.1:
            top_drift_in_selection += 1
    
    return selected, top_drift_in_selection


# ================================================================
# Drift-aware sample weighting
# ================================================================

def compute_sample_weights(train_df, test_df, feat_cols):
    """
    Compute sample weights based on how similar each train sample is to the test distribution.
    Uses adversarial classifier probabilities as proximity scores.
    """
    X_all = pd.concat([train_df[feat_cols], test_df[feat_cols]], axis=0).fillna(0).values.astype(np.float64)
    y_adv = np.array([0]*len(train_df) + [1]*len(test_df))
    
    nzv = np.ptp(X_all, axis=0)
    feat_mask = nzv > 1e-6
    san_names = [sanitize_col(feat_cols[i]) for i in np.where(feat_mask)[0]]
    X_adv = X_all[:, feat_mask]
    
    # Train on 80% to get calibration scores on 20%
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs = np.zeros(len(X_adv))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_adv, y_adv)):
        pos_w = (y_adv[tr_idx] == 0).sum() / max((y_adv[tr_idx] == 1).sum(), 1)
        params = {'num_leaves': 31, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 200,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w,
                  'random_state': 42, 'verbose': -1, 'n_jobs': 1}
        ds = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx], feature_name=san_names)
        m = lgb.train(params, ds, num_boost_round=200)
        train_probs[va_idx] = m.predict(X_adv[va_idx])
    
    # Samples with higher prob of being "test-like" get higher weight
    # But cap to avoid extreme weighting
    sample_weights = train_probs[:len(train_df)]  # only train portion
    sample_weights = (sample_weights + 0.01)  # small baseline + signal
    sample_weights = sample_weights / sample_weights.mean()  # normalize
    sample_weights = np.clip(sample_weights, 0.5, 2.0)  # cap range
    
    return sample_weights


# ================================================================
# Main: Proper CV Stacking with Drift Awareness
# ================================================================

def drift_aware_stacking(train_df, test_df, feat_cols):
    """
    1. Compute adversarial importance → PSI scores
    2. Compute feature stability per target
    3. Select features with drift penalty + stability bonus
    4. Compute drift-aware sample weights
    5. Train stacking with weighted features
    """
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V141 — Drift-Aware Proper Stacking")
    log.info(f"  Seeds: {N_SEEDS}, Meta C: {META_C}, Folds: {N_FOLDS}")
    log.info("=" * 70)
    
    # Step 1: Adversarial validation
    log.info("\n[1/5] Adversarial validation...")
    adv_auc, psi_scores = compute_adversarial_importance(train_df, test_df, feat_cols)
    log.info(f"  Adv AUC: {adv_auc:.4f}")
    high_drift = sum(1 for v in psi_scores.values() if v > 0.1)
    mod_drift = sum(1 for v in psi_scores.values() if 0.05 < v <= 0.1)
    log.info(f"  HIGH PSI features (>0.1): {high_drift}, MODERATE (0.05-0.1): {mod_drift}")
    
    # Step 2: Feature stability
    log.info("\n[2/5] Feature stability analysis...")
    stability = compute_feature_stability(train_df, feat_cols)
    
    # Step 3: Drift-aware sample weights
    log.info("\n[3/5] Computing drift-aware sample weights...")
    sample_weights = compute_sample_weights(train_df, test_df, feat_cols)
    log.info(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}], mean={sample_weights.mean():.3f}")
    
    # Step 4: Stacking
    log.info("\n[4/5] Training stacking...")
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n  --- {t} ---")
        y = train_df[t].values.astype(np.float64)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        
        # Stability-weighted feature selection
        sel_cols, n_drift = select_features_stability_weighted(
            train_df, feat_cols, psi_scores, stability, t, n_feat
        )
        cfg = CFGS[cfg_name]
        
        log.info(f"  Selected {len(sel_cols)} features, {n_drift} HIGH-PSI drift features in selection")
        
        # Level 0: N_SEEDS models
        per_seed_oofs = []
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                w_tr = sample_weights[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            log.info(f"    Seed {si} train OOF: {log_loss(y, seed_oof):.5f}")
        
        # Level 1: Stack → LR meta-learner
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"    Stacking OOF: {ll:.5f}")
        
        test_stacked = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    
    # Step 5: Results
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999)) 
                       for t in TARGETS])
    log.info(f"\n[5/5] Results")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  Adv AUC: {adv_auc:.4f}")
    
    # Build submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v141_drift_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    # Meta
    meta_data = {
        'version': 'V141',
        'name': 'Drift-Aware Proper Stacking (5 seeds + stability-weighted feat sel + adversarial weights)',
        'avg_oof': round(float(avg_oof), 5),
        'adv_auc': round(float(adv_auc), 4),
        'n_high_psi_features': high_drift,
        'n_mod_psi_features': mod_drift,
        'meta_C': META_C,
        'n_seeds': N_SEEDS,
        'per_target_oof': {t: round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5) 
                          for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v141_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    # Save adversarial + stability analysis
    adv_path = EXPERIMENTS / 'adversarial_validation_v1.json'
    adv_save = {
        'adversarial_auc': round(float(adv_auc), 4),
        'psi_scores': {k: round(v, 4) for k, v in psi_scores.items()},
        'timestamp': ts,
    }
    with open(adv_path, 'w') as f:
        json.dump(adv_save, f, indent=2)
    
    stab_path = EXPERIMENTS / 'feature_stability_v1.json'
    stab_save = {}
    for t in TARGETS:
        stab = stability[t]
        items = []
        for i, name in enumerate(stab['feat_cols']):
            items.append({
                'feature': name,
                'mean_imp': round(float(stab['mean_imp'][i]), 2),
                'cv': round(float(stab['cv_imp'][i]), 3),
            })
        stab_save[t] = sorted(items, key=lambda x: -x['cv'])
    with open(stab_path, 'w') as f:
        json.dump(stab_save, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    return avg_oof, meta_data


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    t_start = time.time()
    
    # Load data
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    # Normalize dates
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    feat_cols = get_feature_cols(train_df)
    log.info(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {len(feat_cols)}")
    log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")
    
    # Run
    avg_oof, meta = drift_aware_stacking(train_df, test_df, feat_cols)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
