"""
V43 — V10 재현 (memory-safe, sequential)

핵심:
1. features.parquet (153열) → personalization 추가 (141→576열) → 메모리 30MB 이하
2. leakage 제거 → 279열
3. ranking → top-20 선택
4. tuning: 6 configs × 2 feat counts × 5 seeds (quick check) → best config 선택
5. final: best config × 20 seeds (5-fold OOF)
6. test prediction
"""

import sys, re, gc, time, warnings, json, importlib.util, logging
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
from config import TARGETS, DATA_PROCESSED

DATA_RAW = ROOT / "data_raw"
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mm(p, r):
    return np.clip(p + (r.mean() - p.mean()), 0.0001, 0.9999)

# ── Leakage ──
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

# ── Configs ──
CONFIGS = [
    {'name': 'C1', 'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    {'name': 'C2', 'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C3', 'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C4', 'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    {'name': 'C5', 'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    {'name': 'C6', 'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001]

# ── Personalization ──
def add_personalization(df, feature_cols):
    """Add per-subject z-score features."""
    personal_cols = []
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col] - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols

# ── Feature ranking ──
def rank_features(feat, feat_cols, target, seed=42):
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': 50, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=50)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

# ── Train with specific config ──
def train_with_config(feat, cols, target, seeds, cfg, n_folds=5):
    y = feat[target].values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]
    
    cfg_full = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
        'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
        'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
        'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
        'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
        'min_child_samples': cfg['mc'],
    }
    
    for si, seed in enumerate(seeds):
        cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
        for tr, va in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va][cols].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_tr, label=y[tr], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
            gc.collect()
    
    return oof

# ── Main ──
def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V43 — V10 Reproduction (memory-safe, sequential)")
    log.info("=" * 70)
    
    # 1. Load features
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Loaded: {feat.shape}, memory: {feat.memory_usage(deep=True).sum()/1024**2:.1f}MB")
    
    # 2. Personalization
    log.info("\n--- 2. Personalization ---")
    t0 = time.time()
    feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS)
                 and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    feat, zscore_cols = add_personalization(feat, feat_cols)
    log.info(f"  After personalization: {feat.shape}, memory: {feat.memory_usage(deep=True).sum()/1024**2:.1f}MB")
    log.info(f"  Z-score cols: {len(zscore_cols)} ({time.time()-t0:.1f}s)")
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Target rates: {train_rate}")
    
    all_results = {}
    
    for target in TARGET_COLS:
        tgt_t = time.time()
        log.info(f"\n{'='*50}")
        log.info(f"--- {target} (rate={train_rate[target]:.3f}) ---")
        
        y = feat[target].values.astype(np.float64)
        
        # 3. Remove leakage
        leak_cols = remove_leak(feat_cols + zscore_cols, target)
        log.info(f"  Leak-free: {len(leak_cols)}")
        
        # 4. Feature ranking
        log.info("  [Ranking]...")
        ranked = rank_features(feat, leak_cols, target)
        log.info(f"  Top-10: {ranked[:10]}")
        
        # 5. Tuning: 6 configs × 2 feat counts × 5 seeds
        log.info("  [Tuning] 6 configs × 2 n_feat × 5 seeds...")
        best_score = float('inf')
        best_cfg = None
        best_n = None
        
        for n_feat in [10, 20]:
            sel = ranked[:n_feat]
            for cfg in CONFIGS:
                oof = train_with_config(feat, sel, target, SEEDS[:5], cfg, n_folds=5)
                oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
                cv = log_loss(y, oof_avg, labels=[0, 1])
                pred_shift = abs(oof_avg.mean() - y.mean())
                score = cv + 0.1 * pred_shift
                
                if score < best_score:
                    best_score = score
                    best_cfg = cfg
                    best_n = n_feat
                    log.info(f"    BEST: {cfg['name']} n={n_feat} cv={cv:.4f} shift={pred_shift:.4f}")
                
                del oof, oof_avg
                gc.collect()
        
        # 6. Final model with best config, 20 seeds
        sel_final = ranked[:best_n]
        log.info(f"  [Final] {best_cfg['name']} n={best_n}, 20 seeds...")
        oof_final = train_with_config(feat, sel_final, target, SEEDS, best_cfg, n_folds=5)
        oof_avg_final = np.clip(oof_final.mean(axis=1), 0.0001, 0.9999)
        cal_final = mm(oof_avg_final, y)
        
        cal_loss = log_loss(y, cal_final, labels=[0, 1])
        oof_loss = log_loss(y, oof_avg_final, labels=[0, 1])
        
        all_results[target] = {
            'cal_oof': cal_final,
            'oof_oof': oof_avg_final,
            'config': best_cfg['name'],
            'n_feat': best_n,
            'cv': cal_loss,
        }
        
        log.info(f"  RESULT: {best_cfg['name']} n={best_n}, OOF={oof_loss:.4f}, Cal={cal_loss:.4f}")
        log.info(f"  Time: {time.time()-tgt_t:.0f}s")
        
        del oof_final, oof_avg_final, cal_final, sel_final
        gc.collect()
    
    # 7. Summary
    log.info(f"\n{'='*70}")
    log.info("V43 SUMMARY")
    log.info(f"{'='*70}")
    
    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['config']} n={r['n_feat']} Cal={r['cv']:.4f}")
    
    avg_cal = np.mean([log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
                       for t in TARGET_COLS])
    log.info(f"\n  V43 Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")
    
    # 8. Save OOF
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    
    oof_path = DATA_PROCESSED / "oof_v43.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")
    
    # 9. Test prediction with best configs
    log.info(f"\n--- Test prediction ---")
    test = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    log.info(f"  Test loaded: {test.shape}")
    
    # Personalize test (same subject_id distribution, z-score from train stats)
    test_feat_cols = [c for c in test.columns if c not in META | set(TARGET_COLS)
                      and test[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    test = test.copy()
    for col in test_feat_cols:
        col_filled = test[col].fillna(0)
        # Use train stats for personalization (merge with train subject_id stats)
        train_zscore_mean = feat[[f'{col}_subj_mean', f'{col}_subj_std']].dropna(how='all')
        if len(train_zscore_mean) > 0:
            test = test.merge(train_zscore_mean[['subject_id', f'{col}_subj_mean', f'{col}_subj_std']],
                            on='subject_id', how='left')
            mask_zero = test[f'{col}_subj_std'] == 0
            mask_null = test[col].isnull()
            test[f'{col}_zscore'] = np.where(
                mask_zero | mask_null, 0.0,
                (test[col] - test[f'{col}_subj_mean']) / test[f'{col}_subj_std']
            )
    gc.collect()
    
    # Build test predictions
    test_preds = {t: np.zeros(len(test)) for t in TARGET_COLS}
    
    for target in TARGET_COLS:
        best_cfg = all_results[target]['config']
        best_n = all_results[target]['n_feat']
        sel = ranked_map[target][:best_n]  # Need to save ranked cols
        
        # Find best config
        cfg_best = None
        for cfg in CONFIGS:
            if cfg['name'] == best_cfg:
                cfg_best = cfg
                break
        
        leak_cols = remove_leak(test_feat_cols + [c for c in test_feat_cols if '_zscore' in c], target)
        sel = ranked_map[target][:best_n]
        
        cfg_full = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg_best['nl'], 'max_depth': cfg_best['md'],
            'learning_rate': cfg_best['lr'], 'n_estimators': cfg_best['ne'],
            'subsample': cfg_best['ss'], 'colsample_bytree': cfg_best['cb'],
            'reg_alpha': cfg_best['ra'], 'reg_lambda': cfg_best['rl'],
            'min_child_samples': cfg_best['mc'],
        }
        
        sn = [sanitize(c) for c in sel]
        all_preds = np.zeros(len(test))
        
        for seed in SEEDS:
            cfg_seed = {**cfg_full, 'random_state': seed, 'scale_pos_weight': train_rate[target] / (1 - train_rate[target]) if train_rate[target] < 1 else 1}
            X_test = test[sel].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_test, feature_name=sn, params={'verbose': '-1'})
            
            # Need to reload models — but we don't have them. Train from scratch.
            del ds
            
            # Retrain on ALL data with this seed
            X_all = feat[sel].fillna(0).values.astype(np.float64)
            y_all = feat[target].values.astype(np.float64)
            ds_all = lgb.Dataset(X_all, label=y_all, feature_name=sn, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds_all, num_boost_round=cfg_best['ne'])
            all_preds += m.predict(X_test)
            del m, ds_all
            gc.collect()
        
        all_preds /= len(SEEDS)
        cal_preds = mm(all_preds, train_rate[target])
        test_preds[target] = cal_preds
        log.info(f"  {target}: test_mean={cal_preds.mean():.4f}, train_rate={train_rate[target]:.3f}")
    
    # Save test submission
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_df = pd.DataFrame({
        'subject_id': test['subject_id'].values,
        'sleep_date': test['sleep_date'].values,
        'lifelog_date': test['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        sub_df[target] = test_preds[target]
    
    sub_path = SUBMIT_DIR / f'submission_v43_{timestamp}.csv'
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    log.info(f"\n✅ Submission saved: {sub_path}")
    
    return all_results

if __name__ == "__main__":
    main()
