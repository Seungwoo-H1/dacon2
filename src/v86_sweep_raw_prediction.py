"""
V86 — Full Pipeline with V53_Sweep n_feat + Raw Prediction

Key changes from V85:
- Use V53 sweep best n_feat instead of V53 original
- Raw prediction (no mean_match on test)
- Submission files created

Feature set:
  V53 original: Q1:20, Q2:15, Q3:8, S1:20, S2:20, S3:20, S4:20 → AVG OOF 0.6674
  V53 sweep:    Q1:17, Q2:17, Q3:11, S1:17, S2:20, S3:23, S4:23 → AVG OOF 0.6500 (-0.0174)
"""

import sys, re, gc, logging, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min',
    'wLight_w_light_max','wLight_w_light_count',
    'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count',
    'wPedo_pedo_step_mean','wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean','wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

# V53 sweep best n_feat (confirmed: AVG 0.6500)
V86_CONFIGS = {
    'Q1': {'cfg': 'deep', 'n_feat': 17},
    'Q2': {'cfg': 'deep', 'n_feat': 17},
    'Q3': {'cfg': 'v48', 'n_feat': 11},
    'S1': {'cfg': 'wide', 'n_feat': 17},
    'S2': {'cfg': 'deep', 'n_feat': 20},
    'S3': {'cfg': 'safety', 'n_feat': 23},
    'S4': {'cfg': 'wide', 'n_feat': 23},
}

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_', n)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def add_personalization(df, feature_cols):
    """Add subject-level zscore features (batch agg, no fragmentation)."""
    df = df.copy()
    zscore_cols = []
    agg_parts = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        agg_parts.append(grp)
    if agg_parts:
        agg_df = agg_parts[0]
        for part in agg_parts[1:]:
            agg_df = pd.merge(agg_df, part, on='subject_id', how='left')
        df = pd.merge(df, agg_df, on='subject_id', how='left')
    zcols_dict = {}
    for col in feature_cols:
        zc = f'{col}_zscore'
        mean_c = f'{col}_subj_mean'
        std_c = f'{col}_subj_std'
        zcols_dict[zc] = np.where(
            (df[std_c] == 0) | df[col].isnull(), 0.0,
            (df[col].fillna(0) - df[mean_c]) / df[std_c]
        )
        zscore_cols.append(zc)
    if zcols_dict:
        zdf = pd.DataFrame(zcols_dict, index=df.index)
        df = pd.concat([df, zdf], axis=1)
    drop_cols = [f'{c}_subj_mean' for c in feature_cols] + [f'{c}_subj_std' for c in feature_cols]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df, zscore_cols


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def mean_match(pred, rate):
    """Apply mean matching: shift prediction to match target rate mean.
    Used for OOF calibration only — NOT applied to test predictions.
    """
    return np.clip(pred + (rate - pred.mean()), 0.0001, 0.9999)


def rank_features_importance(train_p, leak_cols, y, cfg_name, target):
    """Rank features on FULL data, return top N selected columns."""
    base_cfg = CFGS[cfg_name]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params_rank = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
        'learning_rate': base_cfg['lr'], 'n_estimators': min(base_cfg['ne'], 100),
        'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
        'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
        'scale_pos_weight': spw, 'random_state': 42,
        'min_child_samples': base_cfg['mc'], 'force_row_wise': True, 'n_jobs': 1,
    }
    
    X_full = train_p[leak_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in leak_cols]
    ds = lgb.Dataset(X_full, label=y, feature_name=sn, params={'verbose': '-1'})
    model_rank = lgb.train(params_rank, ds, num_boost_round=params_rank['n_estimators'])
    imp = model_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
    
    n_feat = V86_CONFIGS[target]['n_feat']
    sel_cols = [r[0] for r in ranked[:n_feat]]
    
    del model_rank, ds, X_full
    gc.collect()
    
    return sel_cols, ranked


def train_full_and_predict(train_p, test_p, sel_cols, y, cfg_name, target):
    """
    Train models on FULL train data (1 fold = 1 model, random_state=42),
    then predict on test data.
    Returns (oof_raw, test_raw).
    """
    base_cfg = CFGS[cfg_name]
    
    X = train_p[sel_cols].fillna(0).values.astype(np.float64)
    X_test = test_p[sel_cols].fillna(0).values.astype(np.float64)
    
    # ===== OOF =====
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    
    for train_idx, val_idx in gkf.split(train_p, y, train_p['subject_id'].values):
        spw_fold = max(((y[train_idx] == 0).sum()) / max((y[train_idx] == 1).sum(), 1), 0.1)
        params_tr = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
            'learning_rate': base_cfg['lr'], 'n_estimators': base_cfg['ne'],
            'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
            'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
            'min_child_samples': base_cfg['mc'], 'random_state': 42, 'scale_pos_weight': spw_fold,
            'force_row_wise': True, 'n_jobs': 1,
        }
        ds_tr = lgb.Dataset(X[train_idx], label=y[train_idx], 
                          feature_name=[sanitize(c) for c in sel_cols], params={'verbose': '-1'})
        m = lgb.train(params_tr, ds_tr, num_boost_round=base_cfg['ne'])
        oof[val_idx] = m.predict(X[val_idx])
        del m, ds_tr
        gc.collect()
    
    # ===== Full train + Test predict =====
    spw_full = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params_full = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
        'learning_rate': base_cfg['lr'], 'n_estimators': base_cfg['ne'],
        'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
        'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
        'min_child_samples': base_cfg['mc'], 'random_state': 42, 'scale_pos_weight': spw_full,
        'force_row_wise': True, 'n_jobs': 1,
    }
    ds_full = lgb.Dataset(X, label=y, feature_name=[sanitize(c) for c in sel_cols], params={'verbose': '-1'})
    m_full = lgb.train(params_full, ds_full, num_boost_round=base_cfg['ne'])
    test_raw = m_full.predict(X_test)
    
    del m_full, ds_full
    gc.collect()
    
    return oof, test_raw


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V86 — Full Pipeline with V53_Sweep n_feat + Raw Prediction")
    log.info("=" * 70)
    
    # Load data
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    log.info(f"  Train: {train.shape}, Test: {test.shape}")
    
    # Personalization
    feat_cols = get_feature_cols(train)
    train_p, zscore_cols = add_personalization(train, feat_cols)
    test_p, _ = add_personalization(test, feat_cols)
    all_cols = feat_cols + zscore_cols
    log.info(f"  Features: {len(all_cols)} total ({len(feat_cols)} base + {len(zscore_cols)} zscore)")
    
    # Train rate for OOF calibration
    train_rate = {t: train_p[t].mean() for t in TARGETS}
    log.info(f"  Train rates: { {t: f'{r:.3f}' for t, r in train_rate.items()} }")
    
    # V86: per-target config + GroupKFold + test prediction
    target_results = {}
    
    for target in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"  Processing {target} ...")
        
        config = V86_CONFIGS[target]
        cfg_name = config['cfg']
        n_feat = config['n_feat']
        base_cfg = CFGS[cfg_name]
        
        leak_cols = remove_leak(all_cols, target)
        y = train_p[target].values.astype(np.float64)
        
        # Feature ranking
        sel_cols, ranked = rank_features_importance(train_p, leak_cols, y, cfg_name, target)
        
        log.info(f"  Config: {cfg_name}, n_feat={n_feat}")
        log.info(f"  Top 5 features: {[r[0] for r in ranked[:5]]}")
        
        # OOF + Test prediction
        oof_raw, test_raw = train_full_and_predict(train_p, test_p, sel_cols, y, cfg_name, target)
        
        # OOF calibration
        oof_cal = mean_match(oof_raw, train_rate[target])
        
        # Compute losses
        oof_loss_raw = log_loss(y, oof_raw, labels=[0, 1])
        oof_loss_cal = log_loss(y, oof_cal, labels=[0, 1])
        
        # Test loss (test targets are all 0 — logloss will be high)
        y_test = test_p[target].values.astype(np.float64)
        test_loss_raw = log_loss(y_test, test_raw, labels=[0, 1]) if y_test.var() > 0 else float('inf')
        
        target_results[target] = {
            'config': config,
            'n_feat': n_feat,
            'oof_raw': oof_raw,
            'oof_cal': oof_cal,
            'test_raw': test_raw,
            'oof_loss_raw': oof_loss_raw,
            'oof_loss_cal': oof_loss_cal,
            'test_loss_raw': test_loss_raw,
        }
        
        log.info(f"  OOF Raw logloss: {oof_loss_raw:.4f}")
        log.info(f"  OOF Cal logloss: {oof_loss_cal:.4f}")
        log.info(f"  Raw mean: train={oof_raw.mean():.4f} test={test_raw.mean():.4f} target={train_rate[target]:.4f}")
        
        del oof_raw, oof_cal, test_raw
        gc.collect()
    
    # Summary
    avg_oof_raw = np.mean([r['oof_loss_raw'] for r in target_results.values()])
    avg_oof_cal = np.mean([r['oof_loss_cal'] for r in target_results.values()])
    
    log.info(f"\n{'='*70}")
    log.info("V86 RESULTS")
    log.info(f"{'='*70}")
    log.info(f"{'Target':<8} {'OOF Raw':<12} {'OOF Cal':<12} {'Δ(OOF)'}")
    log.info(f"{'-'*36}")
    for t in TARGETS:
        r = target_results[t]
        delta = r['oof_loss_raw'] - r['oof_loss_cal']
        log.info(f"{t:<8} {r['oof_loss_raw']:<12.4f} {r['oof_loss_cal']:<12.4f} {delta:+.4f}")
    
    log.info(f"{'='*70}")
    log.info(f"  AVG OOF Raw:  {avg_oof_raw:.4f}")
    log.info(f"  AVG OOF Cal:  {avg_oof_cal:.4f}")
    log.info(f"  V53 sweep:    0.6500 (Sweep n_feat)")
    log.info(f"  V53 orig:     0.6674 (Orig n_feat)")
    log.info(f"  V10 baseline: 0.6038")
    log.info(f"  Δ vs V10 (OOF cal):  {avg_oof_cal - 0.6038:+.4f}")
    log.info(f"  Time: {time.time() - t_start:.0f}s")
    
    # ===== Create submission (raw test predictions) =====
    log.info(f"\n{'='*70}")
    log.info("Creating submission files...")
    SUBMIT_DIR = ROOT / 'submissions'
    SUBMIT_DIR.mkdir(exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Raw prediction submission (no mean_match on test)
    submission_raw = test[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for t in TARGETS:
        submission_raw[t] = target_results[t]['test_raw']
    
    sp_raw = SUBMIT_DIR / f'submission_v86_raw_{ts}.csv'
    submission_raw.to_csv(sp_raw, index=False)
    log.info(f"  Raw submission saved: {sp_raw}")
    for t in TARGETS:
        log.info(f"    {t}: mean={submission_raw[t].mean():.4f} (train={train_rate[t]:.4f})")
    
    # Calibrated OOF-only submission (for comparison)
    submission_cal = test[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    for t in TARGETS:
        submission_cal[t] = mean_match(target_results[t]['test_raw'], train_rate[t])
    
    sp_cal = SUBMIT_DIR / f'submission_v86_cal_{ts}.csv'
    submission_cal.to_csv(sp_cal, index=False)
    log.info(f"  Cal submission saved: {sp_cal}")
    
    # Save meta
    meta = {
        'version': 'V86',
        'method': 'V53_Sweep_n_feat + GroupKFold_5fold + Raw_Prediction',
        'mean_match': 'OOF_only (NOT on test)',
        'per_target': {},
        'avg_oof_raw': float(avg_oof_raw),
        'avg_oof_cal': float(avg_oof_cal),
        'submission_raw': str(sp_raw),
        'submission_cal': str(sp_cal),
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    for t in TARGETS:
        r = target_results[t]
        meta['per_target'][t] = {
            'config': r['config'],
            'n_feat': r['n_feat'],
            'oof_raw': float(r['oof_loss_raw']),
            'oof_cal': float(r['oof_loss_cal']),
            'train_rate': float(train_rate[t]),
            'test_mean_pred': float(submission_raw[t].mean()),
        }
    
    meta_path = ROOT / 'experiments' / 'v86_sweep_raw_prediction.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved: {meta_path}")
    
    return meta


if __name__ == "__main__":
    main()
