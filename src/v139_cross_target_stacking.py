"""
V139 — Cross-Target Raw + Stacking with Personalization (z-score)

V138 Approach 3에서 Cross-Target Raw Features + Stacking이 avg OOF 0.575로 가장 좋음.
S2가 0.488로 0.5 돌파!

But: V138 had NO personalization (z-score per subject).
V136 (with personalization) had equal weight avg OOF 0.543.

So: Cross-Target Raw + Stacking + Personalization = should beat 0.543

Also: Approach 2 (Proper Stacking, no cross-target) was 0.643
With personalization should be ~0.55 range.

Architecture:
┌─────────────────────────────────────────────────────┐
│ Features:                                             │
│   - 141 base features                                 │
│   - 141 z-score per-subject                           │
│   - 6 raw target columns (cross-target)               │
│   Total: 288 base cols + 6 targets = 294              │
├─────────────────────────────────────────────────────┘
│
│ Per-target:
│   - Remove leakage columns
│   - Rank by LGBM importance
│   - Select top-K features (+ cross-target cols always included)
│
│ Level 0: 3 LGBM models (different seeds, GroupKFold 5-fold)
│ Level 1: LogisticRegression on out-of-fold LGBM predictions
│
│ Compare against: V136 equal weight, V136 Bayesian optimized
"""
import sys, gc, logging, json, re, time, warnings
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

def add_personalization(df, feature_cols):
    df = df.copy()
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        subj_mean = grp[f'{col}_subj_mean']
        subj_std = grp[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        zc = f'{col}_zscore'
        df[zc] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(zc)
        gc.collect()
    return df, personal_cols

def rank_features(feat, feat_cols, target, seed=SEED):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**base, 'n_estimators': 50, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]

def train_with_cross_target(feat, feat_test, all_feat_cols, targets, n_seeds=3, n_feat=25):
    """
    Approach: Cross-Target Raw + Stacking with Personalization
    
    For each target:
    - Features = base + z-score + 6 raw target columns
    - Remove leakage
    - Rank by importance
    - Select top-K
    - Level 0: 3 seeds × GroupKFold 5-fold
    - Level 1: LogisticRegression meta-learner on OOF preds
    """
    log.info(f"  Training with cross-target features (personalized)")
    log.info(f"  {len(all_feat_cols)} base features + 141 z-score + 6 raw targets = {len(all_feat_cols) + 141 + 6} total")
    
    group = feat['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    oof_preds = {t: np.zeros(len(feat)) for t in targets}
    test_preds = {t: np.zeros(len(feat_test)) for t in targets}
    per_target_info = {}
    
    for t in targets:
        t_t = time.time()
        y = feat[t].values.astype(np.float64)
        
        # Cross-target: other 6 targets as raw features
        other_targets = [ot for ot in targets if ot != t]
        
        # Feature set
        base_cols = remove_leak(all_feat_cols, t)
        cross_cols = other_targets
        extended_cols = base_cols + cross_cols
        
        # Rank
        ranked = rank_features(feat, extended_cols, t)
        sel_cols = ranked[:n_feat]
        
        # Ensure cross-target cols are always included
        for cc in cross_cols:
            if cc in sel_cols:
                # Move to front if selected
                sel_cols.remove(cc)
                sel_cols.insert(0, cc)
        
        log.info(f"    {t}: base={len(base_cols)} + cross={len(cross_cols)} → selected={len(sel_cols)} (top={n_feat})")
        log.info(f"    {t}: top-5 = {sel_cols[:5]}")
        
        cfg = CFGS[V53_SWEEP[t]['cfg']]
        
        # Level 0: 3 seeds × 5 folds
        # For test: remove cross-target columns (they don't exist in test set)
        test_sel_cols = [c for c in sel_cols if c not in other_targets]
        if len(test_sel_cols) != len(sel_cols):
            log.info(f"    {t}: removed cross-target cols for test: {set(sel_cols) - set(test_sel_cols)}")
        
        seed_oofs = []
        seed_test_preds = []
        
        for si, seed in enumerate(range(SEED, SEED + n_seeds)):
            seed_oof = np.zeros(len(feat))
            seed_test = np.zeros(len(feat_test))
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, group)):
                X_tr = feat[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = feat[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                X_te = feat_test[test_sel_cols].fillna(0).values.astype(np.float64)
                y_tr, y_va = y[tr_idx], y[va_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                # Test may have fewer features (cross-target cols removed)
                # Match training feature order: zero-fill missing cols
                feat_names_train = list(m.feature_name())
                test_col_set = set(test_sel_cols)
                X_te = np.zeros((len(feat_test), len(feat_names_train)), dtype=np.float64)
                for fi, fname in enumerate(feat_names_train):
                    if fname in test_col_set:
                        X_te[:, fi] = feat_test[fname].fillna(0).values
                seed_test += m.predict(X_te)
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            seed_test = np.clip(seed_test, 0.001, 0.999)
            
            seed_oofs.append(seed_oof)
            seed_test_preds.append(seed_test)
            
            del seed_oof, seed_test
            gc.collect()
        
        # Level 1: Stack 3 seeds
        stacked_oof = np.column_stack(seed_oofs)  # (n_train, 3)
        stacked_test_raw = np.column_stack(seed_test_preds)  # (n_test, 3)
        
        # Meta-learner: train on 3 seed OOF preds
        meta = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        meta.fit(stacked_oof, y)
        
        oof_preds[t] = meta.predict_proba(stacked_oof)[:, 1]
        # Test: feed each seed's prediction through meta-learner, then average
        test_preds[t] = meta.predict_proba(stacked_test_raw)[:, 1]
        
        ll = log_loss(y, np.clip(oof_preds[t], 0.001, 0.999))
        meta_weights = meta.coef_[0].round(3)
        
        per_target_info[t] = {
            'll': round(ll, 5),
            'n_feat': len(sel_cols),
            'meta_weights': meta_weights.tolist(),
            'time': round(time.time() - t_t, 0),
        }
        
        log.info(f"    {t}: OOF={ll:.5f} (time: {time.time()-t_t:.0f}s) weights={meta_weights}")
        
        del stacked_oof, stacked_test_raw, seed_oofs, seed_test_preds
        gc.collect()
    
    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in targets])
    log.info(f"\n  AVG OOF: {avg_oof:.5f}")
    return oof_preds, test_preds, per_target_info


# ================================================================
# MAIN
# ================================================================

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V139 — Cross-Target Raw + Stacking with Personalization")
    log.info("=" * 70)
    
    # Load data
    feat = pd.read_parquet(DATA / "features.parquet")
    feat_test = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [feat, feat_test]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    all_feat_cols = get_feature_cols(feat)
    log.info(f"Base features: {len(all_feat_cols)}")
    
    # Add personalization
    feat, zscore_cols = add_personalization(feat, all_feat_cols)
    feat_test_z, _ = add_personalization(feat_test, all_feat_cols)
    log.info(f"Personalized: train={feat.shape}, test={feat_test_z.shape}")
    log.info(f"z-score columns: {len(zscore_cols)}")
    
    # Target means for submission
    train_rates = {t: feat[t].mean() for t in TARGETS}
    
    # === MAIN TRAINING ===
    log.info("\n--- Training Cross-Target + Stacking ---")
    oof_preds, test_preds, per_target_info = train_with_cross_target(
        feat, feat_test_z, all_feat_cols, TARGETS, n_seeds=3, n_feat=25
    )
    
    # === Build submission ===
    log.info("\n--- Building submission ---")
    sub = pd.DataFrame()
    sub['subject_id'] = feat_test_z['subject_id'].values
    sub['sleep_date'] = feat_test_z['sleep_date'].values
    sub['lifelog_date'] = feat_test_z['lifelog_date'].values
    
    for t in TARGETS:
        sub[t] = test_preds[t]
        log.info(f"    {t}: min={sub[t].min():.4f} max={sub[t].max():.4f} mean={sub[t].mean():.4f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMIT / f"submission_v139_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"\n  Saved: {sub_path}")
    
    # === Save metadata ===
    avg_oof = np.mean([log_loss(feat[t].values, np.clip(oof_preds[t], 0.001, 0.999)) for t in TARGETS])
    
    meta = {
        'version': 'V139',
        'name': 'Cross-Target Raw + Stacking with Personalization',
        'features': '141 base + 141 zscore + 6 cross-target raw',
        'level0': '3 LGBM seeds × GroupKFold 5-fold',
        'level1': 'LogisticRegression meta-learner (C=0.1)',
        'n_feat': 25,
        'avg_oof': round(avg_oof, 5),
        'per_target_oof': {t: per_target_info[t]['ll'] for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    meta_path = SUBMIT / f'meta_v139_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s ({time.time()-t_start/60:.1f}min)")
    
    # Compare with V136
    log.info("\n" + "=" * 70)
    log.info("COMPARISON")
    log.info("=" * 70)
    log.info(f"  V136 equal weight:       OOF avg ≈ 0.543")
    log.info(f"  V139 cross-target+stack: OOF avg = {avg_oof:.5f}")
    log.info(f"  Δ = {avg_oof - 0.543:+.5f}")
    
    return sub


if __name__ == '__main__':
    main()
