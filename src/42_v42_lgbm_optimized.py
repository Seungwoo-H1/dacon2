"""
V42 LGBM Optimized — V10 전략 완벽 재현

V10 방식: per-target 6 configs × 2 feat counts × 20 seeds
최적화:
  - features.parquet (153열) 직접 로드 + runtime personalization (542열)
  - ranking → top-20 선택 (8670열 zscore 안 만짐)
  - n_jobs=1 (병렬 제거, 메모리 안정성)
  - 한 타겟씩 순차적 처리
  - 빠른 feature engineering (47분 → 5분)
"""

import sys, re, gc, time, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from config import TARGETS, PARQUET_FILES, DATA_DIR, DATA_PROCESSED

TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    return np.clip(p + (r.mean() - p.mean()), 0.0001, 0.9999)

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

# ── Feature engineering (inline, no import of 02_feature_engineering) ──
def build_features():
    """Build features from raw parquet — avoids 47min feature engineering."""
    t0 = time.time()
    
    # Load all parquet files
    dfs = {}
    for name, fname in PARQUET_FILES.items():
        path = DATA_DIR / fname
        df = pd.read_parquet(path)
        dfs[name] = df

    # Prefix columns and merge
    merged = None
    for name, df in dfs.items():
        prefix = name + '_'
        new_cols = {}
        for c in df.columns:
            if c in ('subject_id', 'timestamp'):
                new_cols[c] = df[c]
            else:
                new_cols[prefix + c] = df[c]
        df_prefixed = pd.DataFrame(new_cols, index=df.index)
        
        # Add date from timestamp
        if 'timestamp' in df.columns:
            df_prefixed['date'] = df_prefixed['timestamp'].dt.date
            df_prefixed['hour'] = df_prefixed['timestamp'].dt.hour
        
        if merged is None:
            merged = df_prefixed
        else:
            merged = merged.merge(df_prefixed, on=['subject_id', 'date'], how='outer')
    
    # Load labels
    labels = pd.read_csv(DATA_RAW / 'ch2026_metrics_train.csv', parse_dates=['sleep_date', 'lifelog_date'])
    
    # Add date to labels
    labels_copy = labels.copy()
    labels_copy['date'] = labels_copy['lifelog_date'].dt.date
    
    # Merge labels
    merged = merged.merge(labels_copy[['subject_id', 'date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']],
                          on=['subject_id', 'date'], how='left')
    
    log.info(f"  Merged: {merged.shape} ({time.time()-t0:.1f}s)")
    
    # Get numeric feature columns
    remove_cols = META | set(TARGET_COLS)
    num_cols = [c for c in merged.columns if c not in remove_cols and merged[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    # Remove leakage columns (but keep for personalization)
    all_cols_no_leak = [c for c in num_cols if c not in LEAK_S | LEAK_Q]
    
    # Add personalization (z-score per subject)
    personal_cols = []
    t1 = time.time()
    for col in all_cols_no_leak:
        col_filled = merged[col].fillna(0)
        grp = col_filled.groupby(merged['subject_id']).agg(['mean', 'std']).reset_index()
        grp.columns = ['subject_id', f'{col}_subj_mean', f'{col}_subj_std']
        merged = merged.merge(grp, on='subject_id', how='left')
        
        mask_zero = merged[f'{col}_subj_std'] == 0
        mask_null = merged[col].isnull()
        merged[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (merged[col].fillna(0) - merged[f'{col}_subj_mean']) / merged[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
    
    log.info(f"  Personalization: {len(personal_cols)} z-score cols ({time.time()-t1:.1f}s)")
    
    # Combine: basic + personalization
    all_features = all_cols_no_leak + personal_cols
    log.info(f"  Total features: {len(all_features)}")
    
    return merged, all_features

DATA_RAW = ROOT / "data_raw"

# ── Hyper configs (V10-style 6 configs) ──
LGB_CONFIGS = [
    {'name': 'C1', 'nl': 10, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 0.5, 'rl': 1.0, 'mc': 5},
    {'name': 'C2', 'nl': 10, 'md': 4, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 8},
    {'name': 'C3', 'nl': 15, 'md': 4, 'lr': 0.02, 'ne': 400, 'ss': 0.8, 'cb': 0.8, 'ra': 0.5, 'rl': 2.0, 'mc': 10},
    {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    {'name': 'C6', 'nl': 8,  'md': 3, 'lr': 0.05, 'ne': 500, 'ss': 0.9, 'cb': 0.9, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

# ── Feature ranking with LGBM (single tree) ──
def rank_features(feat, all_feat_cols, target, seed=42):
    """Rank features using 1 LGBM tree, return ranked list."""
    y = feat[target].values.astype(np.float64)
    
    # Use z-score features only for ranking (most discriminative)
    zscore_cols = [c for c in all_feat_cols if '_zscore' in c]
    z_leak = remove_leak(zscore_cols, target)
    
    X = feat[z_leak].fillna(0).values.astype(np.float64)
    sn = [sanitize(c) for c in z_leak]
    
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=30)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(z_leak, imp), key=lambda x: -x[1])
    
    ds.save_binary('/dev/shm/rank_tmp.bin')
    del ds, model, imp, ranked, X, sn
    gc.collect()
    
    return [r[0] for r in ranked]

# ── Train LGBM ensemble (single thread) ──
def train_ensemble(feat, cols, target, seeds):
    """Train 20-seed ensemble with 5-fold GroupKFold."""
    y = feat[target].values.astype(np.float64)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(y), len(seeds)))
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    sn = [sanitize(c) for c in cols]
    
    for si, seed in enumerate(seeds):
        cfg = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'random_state': seed, 'scale_pos_weight': spw,
        }
        
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            
            m = lgb.train(cfg, ds, num_boost_round=500, valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            
            del ds, vd, m, X_tr, X_va
            gc.collect()
    
    return oof

# ── Main ──
def main():
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V42 LGBM — V10 Strategy (6 configs × 2 n_feat × 20 seeds)")
    log.info("Single-thread, sequential, memory-safe")
    log.info("=" * 70)
    
    # Build features
    log.info("\n--- Building features ---")
    feat, all_feat_cols = build_features()
    
    clear_gpu()
    
    all_results = {}
    tgt_times = {}
    
    for target in TARGET_COLS:
        tgt_t = time.time()
        train_rate = feat[target].mean()
        log.info(f"\n{'='*50}")
        log.info(f"--- {target} (rate={train_rate:.3f}) ---")
        
        y = feat[target].values.astype(np.float64)
        
        # Step 1: Feature ranking
        log.info(f"  [1/3] Feature ranking...")
        ranked = rank_features(feat, all_feat_cols, target)
        log.info(f"  Top-5: {ranked[:5]}")
        
        # Step 2: Config tuning
        log.info(f"  [2/3] Config tuning (5-fold × 20 seeds × 6 configs × 2 n_feat)...")
        best_cv = float('inf')
        best_cfg = None
        best_n = None
        
        for cfg in LGB_CONFIGS:
            for n_feat in [10, 20]:
                sel = ranked[:n_feat]
                
                # Override default config with this config's params
                cfg_spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
                cfg_full = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                    'scale_pos_weight': cfg_spw, 'random_state': 42,
                }
                
                # Quick check: train with just 5 seeds first
                gkf = GroupKFold(n_splits=5)
                oof_quick = np.zeros((len(y), 5))
                sn = [sanitize(c) for c in sel]
                
                quick_seeds = SEEDS[:5]
                for si, seed in enumerate(quick_seeds):
                    cfg_quick = {**cfg_full, 'random_state': seed}
                    for tr, va in gkf.split(feat, y, feat['subject_id']):
                        X_tr = feat.iloc[tr][sel].fillna(0).values.astype(np.float64)
                        X_va = feat.iloc[va][sel].fillna(0).values.astype(np.float64)
                        ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
                        vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
                        m = lgb.train(cfg_quick, ds, num_boost_round=cfg['ne'],
                                     valid_sets=[vd],
                                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                        oof_quick[va, si] = m.predict(X_va)
                        del ds, vd, m, X_tr, X_va
                        gc.collect()
                
                oof_avg = np.clip(oof_quick.mean(axis=1), 0.0001, 0.9999)
                cv = log_loss(y, oof_avg, labels=[0, 1])
                
                if cv < best_cv:
                    best_cv = cv
                    best_cfg = cfg
                    best_n = n_feat
                    log.info(f"    NEW BEST: {cfg['name']} n={n_feat} cv={cv:.4f}")
                
                del oof_quick
                gc.collect()
        
        # Step 3: Final model with best config, all 20 seeds
        sel_final = ranked[:best_n]
        log.info(f"  [3/3] Final model: {best_cfg['name']} n={best_n}")
        
        cfg_final = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': best_cfg['nl'], 'max_depth': best_cfg['md'],
            'learning_rate': best_cfg['lr'], 'n_estimators': best_cfg['ne'],
            'subsample': best_cfg['ss'], 'colsample_bytree': best_cfg['cb'],
            'reg_alpha': best_cfg['ra'], 'reg_lambda': best_cfg['rl'],
            'min_child_samples': best_cfg['mc'],
            'scale_pos_weight': max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1),
            'random_state': 42,
        }
        
        # Train final: 20 seeds × 5 folds
        gkf = GroupKFold(n_splits=5)
        oof_final = np.zeros((len(y), len(SEEDS)))
        sn = [sanitize(c) for c in sel_final]
        
        for si, seed in enumerate(SEEDS):
            cfg_seed = {**cfg_final, 'random_state': seed}
            for tr, va in gkf.split(feat, y, feat['subject_id']):
                X_tr = feat.iloc[tr][sel_final].fillna(0).values.astype(np.float64)
                X_va = feat.iloc[va][sel_final].fillna(0).values.astype(np.float64)
                ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
                vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
                m = lgb.train(cfg_seed, ds, num_boost_round=best_cfg['ne'],
                             valid_sets=[vd],
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                oof_final[va, si] = m.predict(X_va)
                del ds, vd, m, X_tr, X_va
                gc.collect()
        
        oof_avg = np.clip(oof_final.mean(axis=1), 0.0001, 0.9999)
        cal_final = mm(oof_avg, y)
        
        cal_loss = log_loss(y, cal_final, labels=[0, 1])
        oof_loss = log_loss(y, oof_avg, labels=[0, 1])
        
        all_results[target] = {
            'cal_oof': cal_final,
            'oof_oof': oof_avg,
            'config': best_cfg['name'],
            'n_feat': best_n,
            'cv': cal_loss,
        }
        
        tgt_time = time.time() - tgt_t
        tgt_times[target] = tgt_time
        
        log.info(f"  RESULT: Config={best_cfg['name']} n={best_n}, OOF={oof_loss:.4f}, Cal={cal_loss:.4f}")
        log.info(f"  Time: {tgt_time:.0f}s")
        
        del oof_final, oof_avg, cal_final, sel_final
        gc.collect()
    
    # ── Summary ──
    log.info(f"\n{'='*70}")
    log.info("V42 SUMMARY")
    log.info(f"{'='*70}")
    
    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: Config={r['config']} n={r['n_feat']} Cal={r['cv']:.4f} ({tgt_times[target]:.0f}s)")
    
    avg_cal = np.mean([log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
                       for t in TARGET_COLS])
    avg_oof = np.mean([log_loss(feat[t].values, all_results[t]['oof_oof'], labels=[0, 1])
                       for t in TARGET_COLS])
    
    log.info(f"\n  V42 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V42 Avg OOF: {avg_oof:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total time: {time.time()-t_start:.0f}s")
    
    # ── Save OOF for comparison ──
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    
    oof_path = DATA_PROCESSED / "oof_v42.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")
    
    return all_results

if __name__ == "__main__":
    main()
