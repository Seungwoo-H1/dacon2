"""
V53 CV baseline: compute OOF log-loss using same feature set and configs.
GroupKFold with 5 folds for stable comparison.
"""

import sys, gc, logging, json, time, re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

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
    'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum',}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
    'wHr_hr_median','wHr_hr_count'}

V53_CONFIGS = {
    'Q1': {'cfg': 'deep', 'n_feat': 20},
    'Q2': {'cfg': 'deep', 'n_feat': 15},
    'Q3': {'cfg': 'v48', 'n_feat': 8},
    'S1': {'cfg': 'wide', 'n_feat': 20},
    'S2': {'cfg': 'deep', 'n_feat': 20},
    'S3': {'cfg': 'safety', 'n_feat': 20},
    'S4': {'cfg': 'wide', 'n_feat': 20},
}

CFGS = {
    'wide': {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5},
    'deep': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15},
    'v48': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'safety': {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20},
}


def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)


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


def logloss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("V53 CV Baseline - GroupKFold 5-fold")
    
    # Load data
    train = pd.read_parquet(DATA / "features.parquet")
    
    # Personalization
    feat_cols = get_feature_cols(train)
    train_p, zscore_cols = add_personalization(train, feat_cols)
    all_cols = feat_cols + zscore_cols
    log.info(f"  Train: {train.shape} -> {train_p.shape}")
    log.info(f"  Features: {len(all_cols)} total")
    
    # GroupKFold using subject_id
    groups = train['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    
    target_losses = {}
    
    for target in TARGETS:
        log.info(f"\n  --- {target} ---")
        config = V53_CONFIGS[target]
        cfg_name = config['cfg']
        n_feat = config['n_feat']
        base_cfg = CFGS[cfg_name]
        
        leak_cols = remove_leak(all_cols, target)
        y = train_p[target].values.astype(np.float64)
        
        # Feature ranking on FULL data (same as submission)
        y_full = y
        X_full = train_p[leak_cols].fillna(0).values.astype(np.float64)
        spw = max(((y_full == 0).sum()) / max((y_full == 1).sum(), 1), 0.1)
        
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
            'learning_rate': base_cfg['lr'], 'n_estimators': min(base_cfg['ne'], 100),
            'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
            'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': base_cfg['mc'], 'force_row_wise': True, 'n_jobs': 1,
        }
        sn = [sanitize(c) for c in leak_cols]
        ds = lgb.Dataset(X_full, label=y_full, feature_name=sn, params={'verbose': '-1'})
        model_rank = lgb.train(params_rank, ds, num_boost_round=params_rank['n_estimators'])
        imp = model_rank.feature_importance(importance_type='gain')
        ranked = sorted(zip(leak_cols, imp), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked[:n_feat]]
        del model_rank, ds, X_full
        gc.collect()
        
        log.info(f"  Config: {cfg_name}, n_feat={n_feat}, features={sel_cols[:5]}...")
        
        # GroupKFold OOF
        oof_preds = np.zeros(len(y))
        fold_losses = []
        fold_num = 0
        
        for train_idx, val_idx in gkf.split(train_p, y, groups):
            X_tr = train_p.iloc[train_idx][sel_cols].fillna(0).values.astype(np.float64)
            y_tr = y[train_idx]
            X_val = train_p.iloc[val_idx][sel_cols].fillna(0).values.astype(np.float64)
            y_val = y[val_idx]
            
            spw_fold = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
            params_tr = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'num_leaves': base_cfg['nl'], 'max_depth': base_cfg['md'],
                'learning_rate': base_cfg['lr'], 'n_estimators': base_cfg['ne'],
                'subsample': base_cfg['ss'], 'colsample_bytree': base_cfg['cb'],
                'reg_alpha': base_cfg['ra'], 'reg_lambda': base_cfg['rl'],
                'min_child_samples': base_cfg['mc'], 'random_state': 42, 'scale_pos_weight': spw_fold,
                'force_row_wise': True, 'n_jobs': 1,
            }
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=[sanitize(c) for c in sel_cols], params={'verbose': '-1'})
            m = lgb.train(params_tr, ds_tr, num_boost_round=base_cfg['ne'])
            preds_val = m.predict(X_val)
            oof_preds[val_idx] = preds_val
            fl = logloss(y_val, preds_val)
            fold_losses.append(fl)
            fold_num += 1
            del m, ds_tr
            gc.collect()
        
        avg_loss = np.mean(fold_losses)
        target_losses[target] = avg_loss
        log.info(f"  Fold losses: {[f'{fl:.4f}' for fl in fold_losses]}")
        log.info(f"  OOF avg logloss: {avg_loss:.4f}")
        del oof_preds, fold_losses
        gc.collect()
    
    # Summary
    avg_cal = np.mean(list(target_losses.values()))
    log.info(f"\n{'='*60}")
    log.info(f"CV BASELINE RESULTS")
    log.info(f"{'='*60}")
    for t in TARGETS:
        log.info(f"  {t}: {target_losses[t]:.4f}")
    log.info(f"  AVG: {avg_cal:.4f}")
    log.info(f"  Original V53 reported avg_cal_loss_v53: 0.5479")
    log.info(f"  Time: {time.time()-t_start:.0f}s")
    
    # Save
    meta = {
        'version': 'V53_cv_baseline',
        'method': 'GroupKFold_5fold',
        'per_target_logloss': target_losses,
        'avg_logloss': avg_cal,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    meta_path = ROOT / 'experiments' / 'v53_cv_baseline.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved: {meta_path}")


if __name__ == "__main__":
    main()
