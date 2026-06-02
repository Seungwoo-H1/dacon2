"""
V337 — V329 + V308 Cross-Validated Ensemble

Hypothesis: V329 (heavy features) and V308 (simple z-score) capture different signal.
Blending them should give complementary predictions.

Method:
- For each fold, train V329 and V308 students on training fold, predict validation fold
- Find optimal blend weight per target using OOF predictions
- Final test = weighted blend of full-model predictions
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15


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

def get_cfgs():
    return {
        'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
        'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
        'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
                   'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
        'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
                   'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
    }

def get_sweep():
    return {
        'Q1':  {'cfg': 'deep',   'n_feat': 19},
        'Q2':  {'cfg': 'deep',   'n_feat': 14},
        'Q3':  {'cfg': 'v48',    'n_feat': 11},
        'S1':  {'cfg': 'wide',   'n_feat': 21},
        'S2':  {'cfg': 'deep',   'n_feat': 19},
        'S3':  {'cfg': 'safety', 'n_feat': 23},
        'S4':  {'cfg': 'wide',   'n_feat': 20},
    }


def build_features(df, prefix, date_col='sleep_date'):
    """Build features with given prefix for column names.
    Handles copy internally to avoid SettingWithCopyWarning."""
    df = df.copy()
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    
    # Global z-scores
    base_cols = [c for c in df.columns
                 if c not in META_COLS | set(TARGETS)
                 and not c.endswith('_zscore')
                 and np.issubdtype(df[c].dtype, np.number)]
    
    for col in base_cols:
        vals = df[col].fillna(0).values.astype(np.float64)
        mean, std = np.mean(vals), np.std(vals, ddof=0)
        if std < 1e-8: std = 1e-8
        zc = f'{prefix}_zscore_{col}'
        df[zc] = (vals - mean) / std
    
    # Per-subject features
    clean_base = [c for c in base_cols if not c.endswith('_zscore')]
    
    for col in clean_base:
        grp = df.groupby('subject_id')[col]
        for w in [3, 5]:
            rm = grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(df.index)
            df[f'{prefix}_rmean{w}_{col}'] = rm.values
        for w in [3, 5]:
            rs = grp.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).reindex(df.index)
            df[f'{prefix}_rstd{w}_{col}'] = rs.fillna(0).values
        for sn, sf in [('min', 'min'), ('max', 'max'), ('median', 'median')]:
            df[f'{prefix}_{sn}_{col}'] = grp.transform(sf).values
        for q, qn in [(0.25, 'q25'), (0.75, 'q75')]:
            df[f'{prefix}_{qn}_{col}'] = grp.quantile(q).reindex(df['subject_id']).values
        smean = grp.transform('mean')
        df[f'{prefix}_ratio_{col}'] = df[col] / (smean + 1e-8)
        df[f'{prefix}_dev_{col}'] = df[col] - df[col].mean()
        d1 = df[col].diff().fillna(0)
        d2 = d1.diff().fillna(0)
        df[f'{prefix}_accel_{col}'] = d2.values
    
    # Cross-subject z-scores (first 50 only)
    for col in clean_base[:50]:
        grp = df.groupby('subject_id')[col]
        subj_mean = grp.transform('mean')
        g_mean, g_std = df[col].mean(), df[col].std()
        if g_std < 1e-8: g_std = 1e-8
        df[f'{prefix}_cross_z_{col}'] = (subj_mean - g_mean) / g_std
    
    # Day-of-week
    if date_col in df.columns:
        df['dow'] = df[date_col].dt.dayofweek
        df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
        df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)
    
    return df


def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V337 — V329 + V308 Cross-Validated Ensemble")
    log.info("=" * 70)
    
    train_raw = pd.read_parquet(DATA / "features.parquet")
    test_raw = pd.read_parquet(DATA / "test_features.parquet")
    
    # Build V329 and V308 feature sets
    log.info("Building V329 features...")
    v329_train = build_features(train_raw, 'v329')
    v329_test = build_features(test_raw, 'v329')
    log.info("Building V308 features...")
    v308_train = build_features(train_raw, 'v308')
    v308_test = build_features(test_raw, 'v308')
    
    CFGS = get_cfgs()
    SWEEP = get_sweep()
    group = train_raw['subject_id'].values
    
    n_train = len(v329_train)
    n_test = len(v329_test)
    
    # Get feature columns for each pipeline
    v329_feat_cols = get_feature_cols(v329_train)
    v308_feat_cols = get_feature_cols(v308_train)
    
    log.info(f"V329 features: {len(v329_feat_cols)}, V308 features: {len(v308_feat_cols)}")
    
    # Cross-validated predictions
    v329_fold_preds = {t: np.zeros((n_train, N_SEEDS)) for t in TARGETS}
    v308_fold_preds = {t: np.zeros((n_train, N_SEEDS)) for t in TARGETS}
    v329_test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    v308_test_preds = {t: np.zeros((n_test, N_SEEDS)) for t in TARGETS}
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    for t in TARGETS:
        log.info(f"\n--- Target: {t} ---")
        y = train_raw[t].values.astype(np.float64)
        n_feat = SWEEP[t]['n_feat']
        cfg_name = SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        v329_feat_clean = remove_leak(v329_feat_cols, t)
        v308_feat_clean = remove_leak(v308_feat_cols, t)
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(v329_train, y, group)):
            X_tr_v329 = v329_train.iloc[tr_idx]
            X_va_v329 = v329_train.iloc[va_idx]
            X_tr_v308 = v308_train.iloc[tr_idx]
            X_va_v308 = v308_train.iloc[va_idx]
            y_tr = y[tr_idx]
            
            for feat_name, X_tr_df, X_va_df in [
                ('v329', X_tr_v329, X_va_v329),
                ('v308', X_tr_v308, X_va_v308),
            ]:
                ranked = v329_feat_clean if feat_name == 'v329' else v308_feat_clean
                test_feat_cols = v329_feat_cols if feat_name == 'v329' else v308_feat_cols
                test_df = v329_test if feat_name == 'v329' else v308_test
                
                # Feature bagging
                rng = np.random.RandomState(SEED)
                n_bag = max(int(len(ranked) * 0.75), n_feat)
                bag = rng.choice(ranked, size=n_bag, replace=False)
                bag_set = set(bag)
                bag_feats = [f for f in ranked if f in bag_set][:n_feat]
                if len(bag_feats) < n_feat:
                    remaining = [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                    bag_feats.extend(remaining)
                sel_cols = [c for c in bag_feats if c in test_df.columns]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                
                for si in range(N_SEEDS):
                    seed = SEED + si * 7
                    X_tr = X_tr_df[sel_cols].fillna(0).values.astype(np.float64)
                    X_va = X_va_df[sel_cols].fillna(0).values.astype(np.float64)
                    
                    params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                              'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                    sn = [sanitize_col(c) for c in sel_cols]
                    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                    m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                    
                    if feat_name == 'v329':
                        v329_fold_preds[t][va_idx, si] = m.predict(X_va)
                        v329_test_preds[t][:, si] += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
                    else:
                        v308_fold_preds[t][va_idx, si] = m.predict(X_va)
                        v308_test_preds[t][:, si] += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            log.info(f"  Fold {fold+1}/{N_FOLDS} done")
        
        v329_test_preds[t] /= N_FOLDS
        v308_test_preds[t] /= N_FOLDS
    
    # Blend and evaluate
    target_oofs = {}
    blend_weights = {}
    v329_avg_oofs = {}
    v308_avg_oofs = {}
    
    for t in TARGETS:
        y = train_raw[t].values.astype(np.float64)
        v329_avg = np.mean(v329_fold_preds[t], axis=1)
        v308_avg = np.mean(v308_fold_preds[t], axis=1)
        
        v329_avg_oofs[t] = log_loss(y, v329_avg)
        v308_avg_oofs[t] = log_loss(y, v308_avg)
        
        # Find optimal weight
        best_w, best_loss = 0.5, float('inf')
        for w in np.arange(0, 1.05, 0.05):
            blended = np.clip(w * v329_avg + (1-w) * v308_avg, 0.001, 0.999)
            loss = log_loss(y, blended)
            if loss < best_loss:
                best_loss = loss
                best_w = w
        
        blend_weights[t] = best_w
        blended = np.clip(best_w * v329_avg + (1-best_w) * v308_avg, 0.001, 0.999)
        target_oofs[t] = log_loss(y, blended)
        
        log.info(f"{t}: V329={v329_avg_oofs[t]:.5f} V308={v308_avg_oofs[t]:.5f} w={best_w:.2f} blend={target_oofs[t]:.5f}")
    
    avg_oof = np.mean(list(target_oofs.values()))
    
    log.info(f"\n{'='*70}")
    log.info(f"V337 RESULTS")
    log.info(f"{'='*70}")
    for t in TARGETS:
        log.info(f"  {t}: OOF={target_oofs[t]:.5f} (w={blend_weights[t]:.2f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  V329: 0.54365 | Δ: {avg_oof - 0.54365:+.5f}")
    
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")
    
    # Save submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_raw['subject_id'].values
    sub['sleep_date'] = test_raw['sleep_date'].dt.strftime('%Y-%m-%d')
    sub['lifelog_date'] = test_raw['lifelog_date'].dt.strftime('%Y-%m-%d')
    
    for t in TARGETS:
        w = blend_weights[t]
        sub[t] = w * np.mean(v329_test_preds[t], axis=1) + (1-w) * np.mean(v308_test_preds[t], axis=1)
    
    sub_path = SUBMIT / f"submission_v337_ensemble_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    
    meta_data = {
        'version': 'V337',
        'name': 'V329 + V308 Cross-Validated Ensemble',
        'avg_oof': round(float(avg_oof), 5),
        'n_seeds': N_SEEDS,
        'v329_avg_oof': 0.54365,
        'delta_vs_v329': round(float(avg_oof - 0.54365), 5),
        'per_target_oof': {t: round(float(target_oofs[t]), 5) for t in TARGETS},
        'blend_weights': {t: round(float(blend_weights[t]), 3) for t in TARGETS},
        'v329_target_oof': {t: round(float(v329_avg_oofs[t]), 5) for t in TARGETS},
        'v308_target_oof': {t: round(float(v308_avg_oofs[t]), 5) for t in TARGETS},
        'predicted_lb': round(float(pred_lb), 5),
        'v308_actual_lb': 0.63893,
        'submission_file': str(sub_path),
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    meta_path = EXPERIMENTS / f'v337_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved: {sub_path}, {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
