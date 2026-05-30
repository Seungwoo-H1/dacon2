"""
V157 — Feature-Rich Stacking (V146 + Wider Feature Selection)

Hypothesis: V146's top-K (11-23) is too conservative. V156 showed that 
564 group features are pure noise. Instead, we keep V146's 141 base features 
but rank more of them and select wider top-K sets.

Changes from V146:
1. Feature selection: top-K×2 instead of top-K (e.g., 19→38, 14→28, 11→22, 21→42, 23→46, 20→40)
2. No group features (V156 proved these are harmful)
3. No cross-domain interactions (V156's 55 interactions were noise)
4. No multi-meta (single LR with optimized C)
5. Keep V146's exact config→target mapping and leak removal

Risk: Low — same 141 base features, only selection width changes
Expected: OOF improvement 0.001-0.003

Why this time: V156 proved group features hurt. V146's features are proven.
We just need MORE of V146's proven features, not new noisy features.
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
N_SEEDS = 5
META_C = 10.0
FEAT_MULT = 2  # top-K × 2


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


def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    cfg_name = V53_SWEEP[target]['cfg']
    base = CFGS[cfg_name]
    params = {**{k: base[k] for k in ['num_leaves', 'max_depth', 'n_estimators']},
              'learning_rate': 0.05, 'scale_pos_weight': spw,
              'random_state': seed, 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def proper_stacking_v157(train_df, test_df, feat_cols):
    """
    V157: Wider feature selection from V146's base features.
    Same 5 seeds × GroupKFold 5-fold → LR meta (C=10).
    """
    global t_start
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    train_oof = {t: np.zeros(len(train_df)) for t in TARGETS}
    test_preds = {t: np.zeros((len(test_df), N_SEEDS)) for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n--- {t} ---")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat'] * FEAT_MULT  # ×2 wider
        cfg_name = V53_SWEEP[t]['cfg']
        
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:min(n_feat, len(ranked))]
        cfg = CFGS[cfg_name]
        
        log.info(f"    Features: {len(feat_cols_clean)} available, top-{n_feat} selected")
        
        per_seed_oofs = []
        for si, seed in enumerate(range(SEED, SEED + N_SEEDS * 7, 7)):
            seed_oof = np.zeros(len(train_df))
            seed_test = np.zeros(len(test_df))
            
            cols_for_train = [c for c in sel_cols if c in feat_cols_clean]
            cols_for_test = [c for c in sel_cols if c in feat_cols]
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[cols_for_train].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[cols_for_train].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in cols_for_train]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[cols_for_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            test_preds[t][:, si] = seed_test
            
            log.info(f"    Seed {si} (s{seed}): OOF={log_loss(y, seed_oof):.5f}")
        
        # Level 1: LR meta-learner (C=10, same as V146)
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        
        train_oof[t] = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_oof[t], 0.001, 0.999))
        log.info(f"    Stacking OOF (C={META_C}): {ll:.5f}")
        
        test_stacked = np.column_stack([test_preds[t][:, i] for i in range(N_SEEDS)])
        test_preds[t] = meta.predict_proba(test_stacked)[:, 1]
    
    # Compute average OOF
    avg_oof = np.mean([log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))
                       for t in TARGETS])
    log.info(f"\n{'='*70}")
    log.info(f"V157 AVG OOF: {avg_oof:.5f}")
    log.info(f"V146 AVG OOF: 0.63169")
    log.info(f"Δ vs V146: {avg_oof - 0.63169:+.5f}")
    log.info(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = test_preds[t]
    
    sub_path = SUBMIT / f"submission_v157_wider_feat_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved: {sub_path}")
    
    per_target_oof = {}
    for t in TARGETS:
        per_target_oof[t] = round(float(log_loss(train_df[t].values, np.clip(train_oof[t], 0.001, 0.999))), 5)
    
    meta_data = {
        'version': 'V157',
        'name': 'Feature-Rich Stacking (top-K × 2)',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'meta_C': META_C,
        'feat_mult': FEAT_MULT,
        'per_target_oof': per_target_oof,
        'v146_avg_oof': 0.63169,
        'delta_vs_v146': round(float(avg_oof - 0.63169), 5),
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = SUBMIT / f'meta_v157_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {meta_path}")
    
    return avg_oof, meta_data


# ================================================================
# MAIN
# ================================================================

t_start = time.time()
log.info("=" * 70)
log.info("V157 — Feature-Rich Stacking (V146 + Wider Feature Selection)")
log.info(f"Change: top-K → top-K×{FEAT_MULT}, same 141 base features")
log.info("=" * 70)

train_df = pd.read_parquet(DATA / "features.parquet")
test_df = pd.read_parquet(DATA / "test_features.parquet")

for df in [train_df, test_df]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols = get_feature_cols(train_df)
log.info(f"Base features: {len(feat_cols)}")
log.info(f"Target means: {[f'{t}: {train_df[t].mean():.3f}' for t in TARGETS]}")

avg_oof, meta = proper_stacking_v157(train_df, test_df, feat_cols)

log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
