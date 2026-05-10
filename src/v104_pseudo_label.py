"""
V104: Pseudo-Labeling Experiment

Strategy:
- Train LGBM on train data (5-fold CV)
- Predict on test data
- Select high-confidence test predictions (confidence > 0.8) as pseudo-labels
- Retrain on train + pseudo-labels
- Compare train LL and predicted LB with V53

Note: Since Q1-Q3 and S1-S4 are different label types (Q: relative to individual average,
S: absolute compliance), we train separate models per target.
"""

import sys, gc, logging, json, time, warnings, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

np.random.seed(42)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

# V53 optimal per-target config from V93
TARGET_CFG = {
    'Q1': {'cfg': 'deep', 'nf': 19},
    'Q2': {'cfg': 'deep', 'nf': 14},
    'Q3': {'cfg': 'v48', 'nf': 11},
    'S1': {'cfg': 'wide', 'nf': 21},
    'S2': {'cfg': 'deep', 'nf': 19},
    'S3': {'cfg': 'safety', 'nf': 23},
    'S4': {'cfg': 'wide', 'nf': 20},
}

CFGS = {
    'deep':   {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48':    {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'wide':   {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}

LEAK_S = {'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min',
    'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',}

LEAK_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count'}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META_COLS | set(TARGETS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]


def compute_lb(preds_df, oof_df):
    """LB prediction formula."""
    entropies = []
    shifts = []
    skews = []
    from scipy.stats import skew as skew_func
    from scipy.stats import entropy as entropy_func
    for t in TARGETS:
        p = np.clip(preds_df[t].values, 0.005, 0.995)
        entropies.append(entropy_func([p.mean(), 1 - p.mean()], base=2))
        shifts.append(preds_df[t].mean() - oof_df[t].mean())
        skews.append(skew_func(p))
    S3_shift = shifts[5]
    S4_shift = shifts[6]
    avg_entropy = np.mean(entropies)
    avg_shift = np.mean(shifts)
    avg_skew = np.mean(skews)
    lb = (0.0896 * avg_entropy 
          - 0.4205 * avg_shift 
          + 0.1877 * avg_skew 
          + 0.4262 * S3_shift 
          + 0.2194 * S4_shift 
          + 0.7740)
    return lb


def train_model(X, y, params, use_weights=False):
    """Train a single LGBM model."""
    if use_weights:
        weight_col = lgb.Dataset(X, label=y, weight=np.ones(len(y)))
    else:
        weight_col = lgb.Dataset(X, label=y)
    model = lgb.train(params, weight_col, num_boost_round=params['n_estimators'])
    return model


def pseudo_labeling_experiment(target, train_df, test_df, pseudo_threshold=0.8, 
                               pseudo_weight=1.0, n_pseudo_boost=0):
    """
    Run pseudo-labeling experiment for one target.
    
    Returns:
        (oof_preds, test_preds, model, n_pseudo_included)
    """
    cfg_name = TARGET_CFG[target]['cfg']
    n_feat = TARGET_CFG[target]['nf']
    cfg = CFGS[cfg_name]
    
    # Get features
    feat_cols = get_feature_cols(train_df)
    base_cols = [c for c in feat_cols if '_zscore' not in c and '_rm' not in c and '_rs' not in c and '_x_' not in c]
    
    # Simple personalization (just add mean/std per subject)
    train_p = train_df.copy()
    test_p = test_df.copy()
    
    # Calculate subject means for base features
    for col in base_cols:
        subj_mean = train_p.groupby('subject_id')[col].transform('mean')
        subj_std = train_p.groupby('subject_id')[col].transform('std').fillna(1)
        train_p[f'{col}_z'] = (train_p[col] - subj_mean) / subj_std.clip(lower=1)
        test_p[f'{col}_z'] = (test_p[col] - subj_mean.mean()) / 1.0
    
    all_feat_cols = base_cols + [f'{c}_z' for c in base_cols]
    
    # Remove leaked features
    leak_cols = remove_leak(all_feat_cols, target)
    
    y = train_p[target].values.astype(np.float64)
    train_rate = float(y.mean())
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    train_p = train_p.fillna(0)
    test_p = test_p.fillna(0)
    
    # Feature ranking on base features (without zscore)
    X_base = train_p[leak_cols].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in leak_cols]
    
    p_rank = {
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': 50,
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'random_state': 42, 'force_row_wise': True, 'n_jobs': 1,
    }
    ds = lgb.Dataset(X_base, label=y, feature_name=sn)
    m_rank = lgb.train(p_rank, ds, num_boost_round=50)
    imp = m_rank.feature_importance(importance_type='gain')
    ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
    
    sel_cols = [r[0] for r in ranked[:n_feat]]
    sn_sel = [sanitize(c) for c in sel_cols]
    
    # 5-fold CV OOF predictions
    oof_preds = np.zeros(len(train_p))
    fold_models = []
    
    gkf = GroupKFold(n_splits=5)
    groups = train_p['subject_id'].values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_p, y, groups)):
        X_train = train_p.iloc[train_idx][sel_cols].fillna(0).values.astype(np.float64)
        y_train = y[train_idx]
        X_val = train_p.iloc[val_idx][sel_cols].fillna(0).values.astype(np.float64)
        
        # Scale positive weight
        tr_rate = float(y_train.mean())
        tr_spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
        
        params = {
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'random_state': 42 + fold * 7, 'force_row_wise': True, 'n_jobs': 1,
            'scale_pos_weight': tr_spw,
        }
        
        ds_train = lgb.Dataset(X_train, label=y_train, feature_name=sn_sel)
        model = lgb.train(params, ds_train, num_boost_round=cfg['ne'])
        fold_models.append(model)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
    
    # Test predictions (average of 5 fold models on full train)
    X_test = test_p[sel_cols].fillna(0).values.astype(np.float64)
    test_preds = np.zeros(len(X_test))
    for m in fold_models:
        test_preds += m.predict(X_test)
    test_preds /= len(fold_models)
    
    # Apply calibration to match train rate
    cal_test = test_preds + (train_rate - test_preds.mean())
    cal_oof = oof_preds + (train_rate - oof_preds.mean())
    
    # Pseudo-labeling step
    n_pseudo = 0
    X_train_pseudo = None
    y_train_pseudo = None
    
    # Select high-confidence pseudo-labels
    # For binary: confidence = max(pred, 1-pred)
    confidence = np.maximum(cal_test, 1 - cal_test)
    high_conf_mask = confidence > pseudo_threshold
    
    n_high_conf = high_conf_mask.sum()
    pseudo_pred = np.clip(cal_test, 0.005, 0.995)
    
    if n_high_conf > 0:
        # Convert to binary pseudo-labels (threshold at 0.5)
        pseudo_labels = (pseudo_pred >= 0.5).astype(float)
        
        # Weight: high-confidence samples get higher weight
        pseudo_weights = pseudo_weight * np.clip(
            (confidence[high_conf_mask] - pseudo_threshold) / (1.0 - pseudo_threshold) + 0.5,
            0.5, 3.0
        )
        
        X_train_pseudo = test_p.iloc[high_conf_mask][sel_cols].fillna(0).values.astype(np.float64)
        y_train_pseudo = pseudo_labels
        n_pseudo = int(n_high_conf)
        
        log.info(f"    High-confidence pseudo: {n_pseudo} samples "
                 f"(threshold={pseudo_threshold}, mean_conf={confidence[high_conf_mask].mean():.3f})")
        log.info(f"    Pseudo label distribution: {float((pseudo_labels==0).mean()):.1f}% 0, "
                 f"{float((pseudo_labels==1).mean()):.1f}% 1")
    
    return {
        'oof_preds': cal_oof,
        'test_preds': np.clip(cal_test, 0.005, 0.995),
        'fold_models': fold_models,
        'sel_cols': sel_cols,
        'train_p': train_p,
        'test_p': test_p,
        'y': y,
        'train_rate': train_rate,
        'n_pseudo': n_pseudo,
        'X_train_pseudo': X_train_pseudo,
        'y_train_pseudo': y_train_pseudo,
        'pseudo_weights': pseudo_weights if X_train_pseudo is not None else None,
        'dummy': None,  # placeholder
        'sn_sel': sn_sel,
        'cfg': cfg,
    }


def retrain_with_pseudo(train_info, X_pseudo, y_pseudo, pseudo_weights=None, 
                        pseudo_boost_ratio=0.1):
    """Retrain model with pseudo-labeled data."""
    train_p = train_info['train_p']
    test_p = train_info['test_p']
    y = train_info['y']
    sel_cols = train_info['sel_cols']
    sn_sel = train_info['sn_sel']
    cfg = train_info['cfg']
    train_rate = train_info['train_rate']
    
    n_train = len(train_p)
    n_pseudo = len(X_pseudo)
    effective_n_pseudo = int(n_pseudo * pseudo_boost_ratio)
    
    log.info(f"    Retraining with {effective_n_pseudo} pseudo samples "
             f"(boost_ratio={pseudo_boost_ratio}, ratio={effective_n_pseudo/n_train:.1%} of train)")
    
    # Combine train + pseudo
    X_full = np.vstack([
        train_p[sel_cols].fillna(0).values.astype(np.float64),
        X_pseudo[:effective_n_pseudo]
    ])
    y_full = np.concatenate([y, y_pseudo[:effective_n_pseudo]])
    
    if pseudo_weights is not None:
        w_full = np.concatenate([
            np.ones(n_train),
            pseudo_weights[:effective_n_pseudo]
        ])
    else:
        w_full = np.ones(len(y_full))
    
    # 5-fold CV with subject grouping
    gkf = GroupKFold(n_splits=5)
    subjects = train_p['subject_id'].values
    groups = np.concatenate([subjects, subjects[:effective_n_pseudo]])
    
    oof_preds = np.zeros(len(train_p))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y_full, groups)):
        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        
        tr_spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
        
        params = {
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'random_state': 42 + fold * 7, 'force_row_wise': True, 'n_jobs': 1,
            'scale_pos_weight': tr_spw,
        }
        
        if pseudo_weights is not None:
            w = w_full[train_idx]
            ds_train = lgb.Dataset(X_train, label=y_train, weight=w, feature_name=sn_sel)
        else:
            ds_train = lgb.Dataset(X_train, label=y_train, feature_name=sn_sel)
        
        model = lgb.train(params, ds_train, num_boost_round=cfg['ne'])
        
        val_preds = model.predict(X_val)
        # Only count original train samples in OOF
        is_train = [i < n_train for i in val_idx]
        if any(is_train):
            oof_preds[val_idx[is_train]] = val_preds[np.array(is_train)]
    
    # Test predictions
    X_test = test_p[sel_cols].fillna(0).values.astype(np.float64)
    test_preds = np.zeros(len(X_test))
    
    # Retrain on full data
    params = {
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'random_state': 42, 'force_row_wise': True, 'n_jobs': 1,
    }
    
    if pseudo_weights is not None:
        w_full_all = np.concatenate([np.ones(n_train), pseudo_weights[:effective_n_pseudo]])
        ds_full = lgb.Dataset(X_full, label=y_full, weight=w_full_all, feature_name=sn_sel)
    else:
        ds_full = lgb.Dataset(X_full, label=y_full, feature_name=sn_sel)
    
    m_full = lgb.train(params, ds_full, num_boost_round=cfg['ne'])
    test_preds = m_full.predict(X_test)
    
    # Calibration
    cal_test = test_preds + (train_rate - test_preds.mean())
    cal_oof = oof_preds + (train_rate - oof_preds.mean())
    
    return cal_oof, np.clip(cal_test, 0.005, 0.995)


def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("V104: Pseudo-Labeling Experiment")
    log.info("=" * 80)
    
    # Load features
    train = pd.read_parquet(DATA / "features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    test = test[train.columns.tolist()]
    train_labels = pd.read_csv(ROOT / "data_raw" / "ch2026_metrics_train.csv")
    oof_v53 = pd.read_csv(DATA / "oof_v53.csv")
    sub_v53 = pd.read_csv(SUBMIT / "submission_v53_swept_20260510_215247.csv")
    
    # Baseline
    def compute_baseline():
        entropies, shifts, skews = [], [], []
        from scipy.stats import skew as skew_func
        from scipy.stats import entropy as entropy_func
        for t in TARGETS:
            p = np.clip(oof_v53[t].values, 0.005, 0.995)
            entropies.append(entropy_func([p.mean(), 1-p.mean()], base=2))
            shifts.append(sub_v53[t].mean() - oof_v53[t].mean())
            skews.append(skew_func(p))
        avg_e = np.mean(entropies)
        avg_s = np.mean(shifts)
        avg_sk = np.mean(skews)
        lb = 0.0896*avg_e - 0.4205*avg_s + 0.1877*avg_sk + 0.4262*shifts[5] + 0.2194*shifts[6] + 0.7740
        losses = []
        for t in TARGETS:
            p = np.clip(oof_v53[t].values, 0.005, 0.995)
            losses.append(log_loss(train_labels[t].values, p, labels=[0,1]))
        return float(lb), float(np.mean(losses))
    
    baseline_lb, baseline_ll = compute_baseline()
    log.info(f"\nBaseline V53: LB={baseline_lb:.5f}, Train LL={baseline_ll:.5f}")
    
    # Pseudo-labeling thresholds to try
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
    pseudo_weights_list = [0.5, 1.0, 2.0]
    pseudo_boost_ratios = [0.05, 0.1, 0.2]
    
    all_results = []
    
    for threshold in thresholds:
        for pw in pseudo_weights_list:
            for boost in pseudo_boost_ratios:
                target_results = {}
                target_test_preds = {}  # track per-target test predictions for LB
                target_oof_preds = {}  # track per-target OOF predictions
                all_valid = True
                
                for target in TARGETS:
                    # Skip if train rate is 0 or 1 (no variance)
                    train_rate = train[target].mean()
                    if train_rate < 0.01 or train_rate > 0.99:
                        log.info(f"  {target}: skipping (rate={train_rate:.3f})")
                        target_results[target] = None
                        continue
                    
                    # Initial model
                    info = pseudo_labeling_experiment(
                        target, train, test, 
                        pseudo_threshold=threshold,
                        pseudo_weight=pw
                    )
                    
                    if info['n_pseudo'] == 0:
                        target_results[target] = {
                            'oof_ll': None, 'test_mean': None, 'pseudo_n': 0,
                        }
                        continue
                    
                    # Retrain with pseudo-labels
                    if info['X_train_pseudo'] is not None:
                        oof_retrained, test_retrained = retrain_with_pseudo(
                            info, 
                            info['X_train_pseudo'], 
                            info['y_train_pseudo'],
                            pseudo_weights=info['pseudo_weights'],
                            pseudo_boost_ratio=boost
                        )
                    else:
                        oof_retrained = info['oof_preds']
                        test_retrained = info['test_preds']
                    
                    
                    # Track test predictions for LB computation
                    target_test_preds[target] = test_retrained
                    target_oof_preds[target] = oof_retrained
                    
                    # Compute metrics
                    p_train = np.clip(test_retrained, 0.005, 0.995)
                    oof_clip = np.clip(oof_retrained, 0.005, 0.995)
                    
                    train_ll = log_loss(train[target].values, oof_clip, labels=[0,1])
                    oof_ll = log_loss(train[target].values, oof_clip, labels=[0,1])
                    
                    target_results[target] = {
                        'oof_ll': float(oof_ll),
                        'train_ll': float(train_ll),
                        'test_mean': float(p_train.mean()),
                        'pseudo_n': info['n_pseudo'],
                        'test_ll_improvement': float(train_ll - baseline_ll / len(TARGETS)),
                    }
                
                # Only include if all targets succeeded
                if all(v is not None and v.get('oof_ll') is not None for v in target_results.values()):
                    avg_oof_ll = np.mean([v['oof_ll'] for v in target_results.values()])
                    avg_train_ll = np.mean([v['train_ll'] for v in target_results.values()])
                    avg_pseudo_n = np.mean([v['pseudo_n'] for v in target_results.values()])
                    
                    # Build test predictions DataFrame for LB computation
                    pseudo_sub = pd.DataFrame({
                        'subject_id': test['subject_id'],
                        'lifelog_date': test['lifelog_date'],
                        'sleep_date': test['sleep_date'],
                    })
                    for t in TARGETS:
                        if t in target_test_preds and target_test_preds[t] is not None:
                            pseudo_sub[t] = np.clip(target_test_preds[t], 0.005, 0.995)
                        else:
                            pseudo_sub[t] = sub_v53[t].mean()  # fallback to V53 mean
                    
                    # Compute predicted LB
                    predicted_lb = compute_lb(pseudo_sub, oof_v53)
                    
                    result = {
                        'threshold': threshold,
                        'pseudo_weight': pw,
                        'pseudo_boost_ratio': boost,
                        'avg_oof_ll': float(avg_oof_ll),
                        'avg_train_ll': float(avg_train_ll),
                        'avg_pseudo_n': float(avg_pseudo_n),
                        'predicted_lb': float(predicted_lb),
                        'n_pseudo_per_target': {t: v.get('pseudo_n', 0) if isinstance(v, dict) else 0 for t, v in target_results.items()},
                        'targets': target_results,
                        'oof_improvement': float(baseline_ll - avg_oof_ll),
                        'train_improvement': float(baseline_ll - avg_train_ll),
                    }
                    all_results.append(result)
                    
                    log.info(f"  T={threshold:.2f} pw={pw} boost={boost}: "
                             f"OOF={avg_oof_ll:.4f} (Δ={baseline_ll-avg_oof_ll:+.4f}) "
                             f"train={avg_train_ll:.4f} (Δ={baseline_ll-avg_train_ll:+.4f}) "
                             f"pseudo_n={avg_pseudo_n:.0f}")
    
    # Sort by OOF improvement
    all_results.sort(key=lambda x: x['oof_improvement'], reverse=True)
    
    log.info(f"\n{'='*80}")
    log.info("TOP 5 Pseudo-labeling Configs (by OOF improvement)")
    log.info(f"{'='*80}")
    for i, r in enumerate(all_results[:5]):
        log.info(f"  #{i+1}: T={r['threshold']:.2f} pw={r['pseudo_weight']} boost={r['pseudo_boost_ratio']} "
                 f"OOF={r['avg_oof_ll']:.4f} train={r['avg_train_ll']:.4f} "
                 f"ΔOOF={r['oof_improvement']:+.4f} Δtrain={r['train_improvement']:+.4f}")
    
    # Save results
    output = {
        'version': 'V104_pseudo_label',
        'timestamp': datetime.now().isoformat(),
        'baseline': {'lb': float(baseline_lb), 'train_ll': float(baseline_ll)},
        'all_results': all_results,
    }
    
    result_path = EXPERIMENTS / "v104_results.json"
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"\n✅ Results saved: {result_path}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
