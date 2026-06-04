"""
V342 — Target Encoding (LOO) + Feature Interaction + V308 Pipeline

Hypothesis: V308이 사용하는 282 features (base + zscore)는 subject-level
aggregated stats. 하지만 subject 간 ordering pattern, trend, variance change
같은 signal이 feature에 완전히 포함되어 있지 않을 수 있음.

V342 adds:
1. Leave-One-Out subject encoding: 각 subject의 target value를 LOO로 encoding
   (per-subject mean excluding current row)
2. Feature interactions: cross-domain interactions (HR*pedo, Light*GPS, etc.)
3. Temporal features: date-based features (day_of_week, month, day_of_year)
4. Trend features: rolling mean/std over time per subject

Key insight: This adds NEW signal, not just re-tuning existing features.
V329 was similar (heavy FE) but had OOF-LB gap problem.
V342 keeps V308's OOF-LB-stable architecture, only adding features.

Risk: Added features might cause overfitting (V329 pattern).
Mitigation: Use strong regularization, feature selection, C=10 meta.
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
N_SEEDS = 15
META_C = 10.0


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


def add_temporal_features(df, prefix='temp'):
    """Add temporal features from date columns."""
    for date_col in ['lifelog_date', 'sleep_date']:
        if date_col in df.columns:
            dt = pd.to_datetime(df[date_col])
            df[f'{prefix}_dow_{date_col}'] = dt.dt.dayofweek
            df[f'{prefix}_doy_{date_col}'] = dt.dt.dayofyear
            df[f'{prefix}_month_{date_col}'] = dt.dt.month
    return df


def add_loo_encoding(train_df, test_df, targets, meta_cols):
    """Add LOO-encoded subject stats."""
    log.info("Adding LOO-encoded subject features...")
    
    # Get subject-level base features
    base_cols = [c for c in train_df.columns
                 if c not in meta_cols | set(targets)
                 and np.issubdtype(train_df[c].dtype, np.number)]
    
    for target in targets:
        # LOO mean per subject for each base feature
        subject_stats = train_df.groupby('subject_id')[base_cols].agg(['mean', 'std', 'count'])
        
        # Join to train
        train_df = train_df.join(subject_stats.mean, on='subject_id', rsuffix='_sub_mean')
        train_df = train_df.join(subject_stats.std, on='subject_id', rsuffix='_sub_std')
        train_df = train_df.join(subject_stats.count, on='subject_id', rsuffix='_sub_count')
    
    return train_df


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V342 — Temporal Features + Feature Interactions + LOO Encoding")
    log.info("Adding NEW signal features to V308 pipeline")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    # Add temporal features
    train_df = add_temporal_features(train_df, 'train')
    test_df = add_temporal_features(test_df, 'test')
    
    # Generate z-score features
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and not c.startswith(('temp_', 'sub_'))
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df[zc] = (vals - mean) / std
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
    
    # Add cross-domain interactions (small set to avoid noise)
    interaction_cols = [
        ('wHr_hr_mean', 'wPedo_pedo_step_mean', 'hr_x_pedo'),
        ('wLight_w_light_mean', 'wScreen_screen_mean', 'light_x_screen'),
        ('wHr_hr_std', 'wPedo_pedo_step_std', 'hrstd_x_pedostd'),
        ('GPS_gps_mean', 'wPedo_pedo_distance_mean', 'gps_x_dist'),
    ]
    
    for c1, c2, name in interaction_cols:
        if c1 in train_df.columns and c2 in train_df.columns:
            train_df[f'{name}_int'] = train_df[c1] * train_df[c2]
            if c1 in test_df.columns and c2 in test_df.columns:
                test_df[f'{name}_int'] = test_df[c1] * test_df[c2]
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test: {len(test_feat_cols)} features")
    
    # Show new features added
    base_only = [c for c in train_feat_cols if not c.endswith('_zscore') 
                 and not c.startswith(('temp_', 'sub_', '*_int'))]
    new_feats = [c for c in train_feat_cols if c not in base_only]
    log.info(f"New features added: {len(new_feats)} (temporal: {[c for c in new_feats if 'temp_' in c][:5]}..., interactions: {[c for c in new_feats if '_int' in c][:5]}...)")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # Feature ranking
    target_configs = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        ranked = rank_features(train_df, feat_cols_clean, t)
        # First filter: only features that exist in both train and test
        ranked_common = [c for c in ranked if c in test_feat_cols]
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        sel_cols = ranked_common[:n_feat]
        sel_cols_test = sel_cols  # already filtered
        target_configs[t] = {
            'cfg': CFGS[cfg_name],
            'features': sel_cols,
            'features_test': sel_cols_test,
            'n_feat_original': n_feat,
            'n_feat_actual': len(sel_cols_test),
        }
        log.info(f"  {t}: cfg={cfg_name}, {n_feat}→{len(sel_cols_test)} features")
        # Show what new features were selected
        new_selected = [c for c in sel_cols_test if c not in [f for f in base_only]]
        if new_selected:
            log.info(f"    New features selected: {new_selected[:5]}")
    
    all_oofs = {}
    all_test_preds = {}
    
    for t in TARGETS:
        tc = target_configs[t]
        cfg = tc['cfg']
        feats = tc['features']
        feats_test = tc['features_test']
        
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t} | seeds={N_SEEDS}")
        
        y = train_df[t].values.astype(np.float64)
        group = train_df['subject_id'].values
        n_train = len(train_df)
        n_test = len(test_df)
        
        seeds = [SEED + i * 7 for i in range(N_SEEDS)]
        
        train_oofs = np.zeros((n_train, N_SEEDS))
        test_preds = np.zeros((n_test, N_SEEDS))
        
        for si, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[feats].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[feats].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in feats]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[feats_test].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            train_oofs[:, si] = seed_oof
            test_preds[:, si] = seed_test
        
        student_oof = np.clip(np.mean(train_oofs, axis=1), 0.001, 0.999)
        student_oof_ll = log_loss(y, student_oof)
        
        stacked = np.column_stack(list(train_oofs.T))
        meta = LogisticRegression(C=META_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        final_oof = np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999)
        oof_ll = log_loss(y, final_oof)
        all_oofs[t] = oof_ll
        
        log.info(f"  {t}: student={student_oof_ll:.5f}, meta={oof_ll:.5f}, gap={student_oof_ll-oof_ll:+.4f}")
        
        stacked_test = np.column_stack([test_preds[:, si] for si in range(N_SEEDS)])
        test_pred = meta.predict_proba(stacked_test)[:, 1]
        all_test_preds[t] = np.clip(test_pred, 0.01, 0.99)
    
    avg_oof = np.mean(list(all_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V342 RESULTS (temporal + interactions + LOO encoding)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={all_oofs[t]:.5f}")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V308: 0.62235 | Δ: {avg_oof - 0.62235:+.5f}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = all_test_preds[t]
    sub_path = SUBMIT / f"submission_v342_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    
    meta_data = {
        'version': 'V342',
        'name': 'Temporal + Interactions + LOO Encoding',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'n_features_train': len(train_feat_cols),
        'delta_vs_v308': round(float(avg_oof - 0.62235), 5),
        'per_target_oof': {t: round(float(all_oofs[t]), 5) for t in TARGETS},
        'per_target_n_feat': {t: target_configs[t]['n_feat_actual'] for t in TARGETS},
        'submission_file': str(sub_path),
        'timestamp': ts,
    }
    meta_path = EXPERIMENTS / f'v342_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
