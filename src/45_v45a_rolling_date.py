"""
V45a — Rolling + Date Feature Engineering

V10에 rolling(expanding) feature와 date/period feature를 추가하여
feature richness를 높이는 실험.

Improvements over V10:
1. Rolling mean/std over time (7-day, 3-day windows)
2. Expanding (cumulative) mean/std
3. Diff (day-over-day change) and pct_change
4. EMA (exponential moving average)
5. Date features: dayofweek, is_weekend, month, date_diff
6. Feature count: top-30, top-50, top-100
7. More seeds: 30 seeds instead of 20
"""

import sys, re, gc, time, json, logging, warnings
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
DATA_PROCESSED = ROOT / "data_processed"
SUBMIT_DIR = ROOT / "submissions"

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

# ── Seeds ──
SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001,
         6000, 7123, 8001, 9000, 10000, 11111, 12000, 13001, 14000, 15001,
         16000, 17123, 18001, 19000, 20000, 21111, 22000, 23001, 24000]

# ── Configs ──
CONFIGS = {
    'C1': {'nl': 8, 'md': 3, 'lr': 0.02, 'ne': 200, 'ss': 0.6, 'cb': 0.6, 'ra': 2.0, 'rl': 5.0, 'mc': 15},
    'C2': {'nl': 10, 'md': 3, 'lr': 0.03, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C3': {'nl': 12, 'md': 4, 'lr': 0.03, 'ne': 200, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C4': {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10},
    'C5': {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 300, 'ss': 0.7, 'cb': 0.7, 'ra': 0.5, 'rl': 2.0, 'mc': 8},
    'C6': {'nl': 6, 'md': 2, 'lr': 0.02, 'ne': 200, 'ss': 0.5, 'cb': 0.5, 'ra': 5.0, 'rl': 10.0, 'mc': 20},
}

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

# ── Rolling Feature Engineering ──
def add_rolling_features(df, feature_cols, target_col='sleep_date', subject_col='subject_id', window_days=[3, 7, 14]):
    """
    Add rolling/expanding/diff/ema features per subject ordered by date.
    Uses transform to keep same index.
    """
    df = df.sort_values([subject_col, target_col]).copy().reset_index(drop=True)
    new_cols = []
    
    for col in feature_cols:
        # Skip if not numeric
        if df[col].dtype not in [np.float64, np.int64, float, int, np.float32, np.int32]:
            continue
        
        grp = df.groupby(subject_col)[col]
        
        # Rolling mean/std
        for w in window_days:
            rm = grp.rolling(window=w, min_periods=1, center=False).mean()
            rs = grp.rolling(window=w, min_periods=1, center=False).std().fillna(0)
            col_rm = f'{col}_r{w}m'
            col_rs = f'{col}_r{w}s'
            df[col_rm] = rm.to_numpy()
            df[col_rs] = rs.to_numpy()
            new_cols.extend([col_rm, col_rs])
        
        # Expanding mean/std (cumulative)
        em = grp.expanding(min_periods=1).mean()
        es = grp.expanding(min_periods=1).std().fillna(0)
        col_em = f'{col}_expm'
        col_es = f'{col}_exps'
        df[col_em] = em.to_numpy()
        df[col_es] = es.to_numpy()
        new_cols.extend([col_em, col_es])
        
        # Diff (day-over-day change)
        d = grp.diff().fillna(0)
        col_d = f'{col}_diff'
        df[col_d] = d.to_numpy()
        new_cols.append(col_d)
        
        # Pct change
        pc = grp.pct_change().fillna(0)
        col_pc = f'{col}_pctc'
        df[col_pc] = pc.to_numpy()
        new_cols.append(col_pc)
        
        # EMA (half-life = 3 days)
        ema = grp.ewm(halflife=3, min_periods=1).mean()
        col_ema = f'{col}_ema3'
        df[col_ema] = ema.to_numpy()
        new_cols.append(col_ema)
        
        # EMA (half-life = 7 days)
        ema7 = grp.ewm(halflife=7, min_periods=1).mean()
        col_ema7 = f'{col}_ema7'
        df[col_ema7] = ema7.to_numpy()
        new_cols.append(col_ema7)
        
        gc.collect()
    
    return df, new_cols

def add_date_features(df):
    """Add date/period features."""
    date_col = df['date'] if 'date' in df.columns else pd.to_datetime(df['sleep_date'])
    
    df['dayofweek'] = pd.to_datetime(date_col).dt.dayofweek
    df['is_weekend'] = (pd.to_datetime(date_col).dt.dayofweek >= 5).astype(int)
    df['month'] = pd.to_datetime(date_col).dt.month
    df['is_monday'] = (pd.to_datetime(date_col).dt.dayofweek == 0).astype(int)
    df['is_friday'] = (pd.to_datetime(date_col).dt.dayofweek == 4).astype(int)
    
    # Day of year
    df['dayofyear'] = pd.to_datetime(date_col).dt.dayofyear
    
    # Date diff from first observation per subject
    first_dates = df.groupby('subject_id')['sleep_date'].transform('min')
    df['date_diff'] = (pd.to_datetime(df['sleep_date']) - pd.to_datetime(first_dates)).dt.days
    
    # Lag features for date
    df['date_lag1'] = df.groupby('subject_id')['sleep_date'].transform(
        lambda x: x.shift(1).diff().dt.days
    ).fillna(1).astype(int)
    
    return df

def add_personalization(df, feature_cols):
    """Per-subject z-score."""
    zscore_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        grp = df[col].fillna(0).groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        mask_zero = df[f'{col}_subj_std'] == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col] - df[f'{col}_subj_mean']) / df[f'{col}_subj_std']
        )
        zscore_cols.append(f'{col}_zscore')
        gc.collect()
    return df, zscore_cols

def rank_features(feat, feat_cols, target, seed=42, n_trees=50):
    """Quick ranking."""
    y = feat[target].values.astype(np.float64)
    X = feat[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03,
        'n_estimators': n_trees, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 3.0,
        'scale_pos_weight': spw, 'random_state': seed,
        'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': 1,
    }
    sn = [sanitize(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn, params={'verbose': '-1'})
    model = lgb.train(params, ds, num_boost_round=n_trees)
    imp = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del model, ds
    gc.collect()
    return [r[0] for r in ranked]

def simple_mm(p, r):
    """Mean-match calibration."""
    shift = r - p.mean()
    return np.clip(p + shift, 0.0001, 0.9999)

# ── Main ──
def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("V45a — Rolling + Date Feature Engineering")
    log.info("=" * 70)
    
    # 1. Load features
    log.info("\n--- 1. Load features ---")
    feat = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    log.info(f"  Train: {feat.shape}")
    
    # 2. Sort by subject + date for rolling
    feat = feat.sort_values(['subject_id', 'sleep_date']).copy()
    
    # 3. Add date features
    log.info("\n--- 2. Date features ---")
    feat = add_date_features(feat)
    date_cols = ['dayofweek', 'is_weekend', 'month', 'is_monday', 'is_friday',
                 'dayofyear', 'date_diff', 'date_lag1']
    feat = feat.dropna(subset=['date_diff'])  # date_diff requires sorted per subject
    
    # 4. Add rolling features
    log.info("\n--- 3. Rolling features ---")
    t0 = time.time()
    feat_cols = [c for c in feat.columns if c not in META | set(TARGET_COLS) | set(date_cols)
                 and feat[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]
    
    feat, rolling_cols = add_rolling_features(feat, feat_cols)
    log.info(f"  Rolling features added: {len(rolling_cols)}")
    log.info(f"  After rolling: {feat.shape}")
    log.info(f"  Time: {time.time()-t0:.0f}s")
    
    # 5. Personalization
    log.info("\n--- 4. Personalization ---")
    t0 = time.time()
    all_feat_cols = feat_cols + rolling_cols + date_cols
    feat, zscore_cols = add_personalization(feat, all_feat_cols)
    log.info(f"  Z-score cols: {len(zscore_cols)}")
    log.info(f"  After personalization: {feat.shape}")
    log.info(f"  Time: {time.time()-t0:.0f}s")
    
    all_cols = all_feat_cols + zscore_cols
    feat = feat.fillna(0)
    
    train_rate = {t: feat[t].mean() for t in TARGET_COLS}
    log.info(f"  Train rates: {train_rate}")
    
    # 6. Tuning + Final for each target
    all_results = {}
    feat_counts = [30, 50, 100]
    
    for target in TARGET_COLS:
        log.info(f"\n{'='*60}")
        log.info(f"--- {target} (rate={train_rate[target]:.3f}) ---")
        tgt_t = time.time()
        
        # Remove leakage
        leak_free = remove_leak(all_cols, target)
        log.info(f"  Leak-free cols: {len(leak_free)}")
        
        # Feature ranking
        log.info("  Ranking...")
        ranked = rank_features(feat, leak_free, target)
        top10 = ranked[:10]
        log.info(f"  Top-10: {top10[:5]}...")
        
        # Multi-window tuning
        best_score = float('inf')
        best_cfg = None
        best_n = None
        best_oof = None
        
        for n_feat in feat_counts:
            if n_feat > len(ranked):
                continue
            sel = ranked[:n_feat]
            
            y = feat[target].values
            spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
            
            for name, cfg in CONFIGS.items():
                cfg_full = {
                    'objective': 'binary', 'metric': 'binary_logloss',
                    'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
                    'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
                    'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
                    'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
                    'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
                    'min_child_samples': cfg['mc'],
                }
                
                gkf = GroupKFold(n_splits=5)
                oof_all = np.zeros(len(feat))
                
                for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(feat, y, feat['subject_id'])):
                    X_tr = feat.iloc[tr_idx][sel].values
                    X_va = feat.iloc[va_idx][sel].values
                    y_tr, y_va = y[tr_idx], y[va_idx]
                    
                    sn = [sanitize(c) for c in sel]
                    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn, params={'verbose': '-1'})
                    ds_va = lgb.Dataset(X_va, label=y_va, feature_name=sn, reference=ds_tr, params={'verbose': '-1'})
                    
                    params = {**cfg_full, 'scale_pos_weight': spw}
                    m = lgb.train(params, ds_tr, num_boost_round=cfg['ne'],
                                  valid_sets=[ds_va],
                                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
                    oof_all[va_idx] += m.predict(X_va) / 5  # average 5 folds
                
                cv_loss = log_loss(y, oof_all, labels=[0, 1])
                shift = abs(oof_all.mean() - train_rate[target])
                score = cv_loss + 0.3 * shift
                
                if score < best_score:
                    best_score = score
                    best_cfg = name
                    best_n = n_feat
                    best_oof = oof_all.copy()
                    log.info(f"    NEW BEST: {name} n={n_feat} cv={cv_loss:.4f} shift={shift:.4f}")
        
        log.info(f"  FINAL: {best_cfg} n={best_n} cv={log_loss(y, best_oof, labels=[0,1]):.4f}")
        
        # Final model: 30 seeds on ALL data
        log.info(f"  Final training: {best_cfg} n={best_n}, 30 seeds...")
        cfg_best = CONFIGS[best_cfg]
        cfg_full = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg_best['nl'], 'max_depth': cfg_best['md'],
            'learning_rate': cfg_best['lr'], 'n_estimators': cfg_best['ne'],
            'subsample': cfg_best['ss'], 'colsample_bytree': cfg_best['cb'],
            'reg_alpha': cfg_best['ra'], 'reg_lambda': cfg_best['rl'],
            'min_child_samples': cfg_best['mc'],
        }
        
        sel = ranked[:best_n]
        sn = [sanitize(c) for c in sel]
        X_all = feat[sel].values
        y_all = y
        
        all_preds = np.zeros(len(feat))
        for seed_i, seed in enumerate(SEEDS):
            ds = lgb.Dataset(X_all, label=y_all, feature_name=sn, params={'verbose': '-1'})
            params = {**cfg_full, 'random_state': seed, 'scale_pos_weight': spw}
            m = lgb.train(params, ds, num_boost_round=cfg_best['ne'])
            all_preds += m.predict(X_all)
            if (seed_i + 1) % 10 == 0:
                log.info(f"    seed {seed_i+1}/{len(SEEDS)}")
            del m, ds
            gc.collect()
        
        all_preds /= len(SEEDS)
        cal_preds = simple_mm(all_preds, train_rate[target])
        cal_loss = log_loss(y, cal_preds, labels=[0, 1])
        
        all_results[target] = {
            'config': best_cfg, 'n_feat': best_n, 'cv': float(log_loss(y, best_oof, labels=[0,1])),
            'cal': float(cal_loss), 'cal_oof': cal_preds,
        }
        log.info(f"  {target}: Cal={cal_loss:.4f} | Time: {time.time()-tgt_t:.0f}s")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("V45a SUMMARY")
    log.info(f"{'='*70}")
    
    for target in TARGET_COLS:
        r = all_results[target]
        log.info(f"  {target}: {r['config']} n={r['n_feat']} OOF={r['cv']:.4f} Cal={r['cal']:.4f}")
    
    avg_cal = np.mean([all_results[t]['cal'] for t in TARGET_COLS])
    log.info(f"\n  V45a Avg Cal: {avg_cal:.4f}")
    log.info(f"  V10 Avg Cal: 0.6038")
    log.info(f"  Δ: {avg_cal - 0.6038:+.4f}")
    log.info(f"  Total: {time.time()-t_start:.0f}s")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'subject_id': feat['subject_id'].values,
        'sleep_date': feat['sleep_date'].values,
        'lifelog_date': feat['lifelog_date'].values,
    })
    for target in TARGET_COLS:
        oof_df[target] = all_results[target]['cal_oof']
    
    oof_path = DATA_PROCESSED / "oof_v45a.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path}")
    
    # Save meta
    meta = {
        'version': 'v45a', 'avg_cal': float(avg_cal),
        'results': {t: {k: v for k, v in r.items() if k != 'cal_oof'} for t, r in all_results.items()},
        'feature_count': len(all_cols),
        'rolling_cols': len(rolling_cols),
        'zscore_cols': len(zscore_cols),
    }
    with open(DATA_PROCESSED / "v45a_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    main()
