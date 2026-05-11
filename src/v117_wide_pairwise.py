"""V117: Improved V54 strategy — wider pairwise, better ensemble

Key findings:
- V116 AVG = 0.54761 (50 seeds, pairwise + personalization + iso cal)
- V54 file = 0.53971
- Gap: 0.0079 (mainly Q1: +0.029, S2: +0.021)
- V116 uses pairwise on top-10 features only
- V54 likely uses pairwise on top-15 or all features

V117 improvements:
1. pairwise on top-15 features (instead of top-10)
2. broader n_feat search [5,8,10,12,15,20]
3. ensemble top-k models per target
4. cross-strategy ensemble (combine pairwise + base + trans)
5. V115 + V116 ensemble (personalization only vs personalization + pairwise)
"""
import sys, gc, time, warnings, logging, json, re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/projects/dacon2')
DATA = ROOT / "data_processed"
SUBMIT = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
TARGET_COLS = TARGETS
META = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

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

CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V48 = {'nl': 15, 'md': 4, 'lr': 0.03, 'ne': 500, 'ss': 0.7, 'cb': 0.7, 'ra': 1.0, 'rl': 3.0, 'mc': 10}
CFG_SAFETY = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFG_EXTRA_DEEP = {'nl': 25, 'md': 6, 'lr': 0.01, 'ne': 2000, 'ss': 0.6, 'cb': 0.5, 'ra': 0.1, 'rl': 1.0, 'mc': 25}
CFG_V53WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_V53DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFG_V53SAFE = {'nl': 10, 'md': 3, 'lr': 0.02, 'ne': 1000, 'ss': 0.6, 'cb': 0.6, 'ra': 3.0, 'rl': 10.0, 'mc': 20}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP, 'v48': CFG_V48, 'safety': CFG_SAFETY,
        'extra_deep': CFG_EXTRA_DEEP, 'v53wide': CFG_V53WIDE, 'v53deep': CFG_V53DEEP, 'v53safe': CFG_V53SAFE}

# Use a subset of 25 seeds for speed (12 seeds was V115, 50 seeds V116)
# V116 (50 seeds) didn't improve over V115 (8 seeds), so 25 should be enough
SEEDS = list(range(1, 26))

def sanitize(n):
    return re.sub(r'[^a-zA-Z0-9_]','_',n)

def mean_match(pred, target_mean):
    shift = target_mean - pred.mean()
    return np.clip(pred + shift, 0.0001, 0.9999)

def remove_leak(cols, target):
    if target.startswith('S'):
        return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'):
        return [c for c in cols if c not in LEAK_Q]
    return cols

def get_feature_cols(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and df[c].dtype in [np.float64, np.int64, float, int, bool, np.bool_]]

def add_personalization(df, feature_cols):
    personal_cols = []
    df = df.copy()
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
            (df[col].fillna(0) - df[f'{col}_subj_mean']) / np.maximum(df[f'{col}_subj_std'], 1e-8))
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols

def add_pairwise_interactions(feat, top_features):
    """Generate pairwise interactions on given top features."""
    feat = feat.copy()
    added = []
    for i in range(min(len(top_features), 15)):
        for j in range(i+1, min(len(top_features), 15)):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns:
                continue
            col_prod = f'{f1}_x_{f2}'
            feat[col_prod] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(col_prod)
            s1 = feat[f1].std()
            s2 = feat[f2].std()
            if s1 > 0 and s2 > 0:
                col_ratio = f'{f1}_div_{f2}'
                feat[col_ratio] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
                added.append(col_ratio)
    for f in top_features[:8]:
        if f in feat.columns:
            col_sq = f'{f}_sq'
            feat[col_sq] = feat[f].fillna(0) ** 2
            added.append(col_sq)
    return feat, added

def add_transformed_features(feat, top_features):
    feat = feat.copy()
    added = []
    for f in top_features[:15]:
        if f not in feat.columns:
            continue
        vals = feat[f].fillna(0).values
        vals_abs = np.abs(vals) + 1e-8
        feat[f'{f}_log'] = np.sign(vals) * np.log1p(vals_abs)
        added.append(f'{f}_log')
        feat[f'{f}_sqrt'] = np.sign(vals) * np.sqrt(vals_abs)
        added.append(f'{f}_sqrt')
        feat[f'{f}_abs'] = np.abs(vals)
        added.append(f'{f}_abs')
    return feat, added

def rank_features_importance(feat, feat_cols, target, seed=42):
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

def train_cv_model(feat, cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    sn = [sanitize(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
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
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            X_tr = feat.iloc[tr_i][cols].fillna(0).values.astype(np.float64)
            X_va = feat.iloc[va_i][cols].fillna(0).values.astype(np.float64)
            ds = lgb.Dataset(X_tr, label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_va, label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_seed, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_va)
            del ds, vd, m, X_tr, X_va
            gc.collect()
    return oof

def isotonic_calibrate(oof_preds, y_true):
    iso = IsotonicRegression(out_of_bounds='clip')
    try:
        iso.fit(oof_preds, y_true)
        return iso.predict(oof_preds), True
    except Exception:
        return oof_preds, False

# ============================================================
# Load data
# ============================================================
t_start = time.time()
log.info("Loading data...")

feat = pd.read_parquet(DATA / "features.parquet")
feat_test = pd.read_parquet(DATA / "test_features.parquet")
for df in [feat, feat_test]:
    for c in ['sleep_date', 'lifelog_date', 'date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

feat_cols_raw = get_feature_cols(feat)
log.info(f"Base features: {len(feat_cols_raw)}")

# Personalization
feat, zscore_cols = add_personalization(feat, feat_cols_raw)
all_cols = feat_cols_raw + zscore_cols
log.info(f"After personalization: {feat.shape}, zscore_cols: {len(zscore_cols)}")

y_train = {t: feat[t].values for t in TARGETS}
train_rates = {t: feat[t].mean() for t in TARGETS}

# Feature ranking
log.info("\n--- Feature ranking ---")
ranked_lgb = {}
for target in TARGETS:
    leak_cols = remove_leak(all_cols, target)
    ranked = rank_features_importance(feat, leak_cols, target)
    ranked_lgb[target] = ranked
    log.info(f"  {target} top-10: {ranked[:10]}")

# ============================================================
# Load V116 OOF for ensemble
# ============================================================
log.info("\n--- Loading V116 OOF for ensemble ---")
v116_file = sorted(Path(DATA).glob('oof_v116_*.csv'))[-1]
v116_oof = pd.read_csv(v116_file)
log.info(f"  V116 file: {v116_file.name}")

# ============================================================
# V117 experiments: pairwise top-15, broader n_feat search
# ============================================================
log.info(f"\n--- V117: pairwise top-15, n_feat={list(range(5,25,5))} ---")

# Store all models per target for ensemble
all_target_models = {}  # target -> list of (oof, ll, method_str)

for target in TARGETS:
    tgt_t = time.time()
    y = feat[target].values.astype(np.float64)
    
    log.info(f"\n  {target}:")
    
    # ===== Strategy A: pairwise top-15 (wider than V116's top-10) =====
    log.info(f"    Strategy A: pairwise top-15")
    top_features = ranked_lgb[target][:15]
    feat_pair15, pair_added = add_pairwise_interactions(feat, top_features)
    all_cols_p15 = get_feature_cols(feat_pair15)
    all_cols_p15 = [c for c in all_cols_p15 if c not in META | set(TARGETS)]
    ranked_p15 = rank_features_importance(feat_pair15, remove_leak(all_cols_p15, target), target)
    
    for n_feat in [5, 8, 10, 12, 15, 20]:
        sel_cols = ranked_p15[:n_feat]
        oof = train_cv_model(feat_pair15, sel_cols, y, SEEDS, CFG_WIDE, n_folds=5)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        iso_cal, ok = isotonic_calibrate(oof_avg, y)
        if ok:
            iso_cal = mean_match(iso_cal, train_rates[target])
            ll = log_loss(y, iso_cal, labels=[0, 1])
        else:
            ll = log_loss(y, oof_avg, labels=[0, 1])
            iso_cal = oof_avg
        
        all_target_models.setdefault(target, []).append((iso_cal, ll, f"pair15_{CFG_WIDE['nl']}n{n_feat}"))
        if n_feat <= 10:
            log.info(f"      n_feat={n_feat:2d} pair15_wide: LL={ll:.5f}")
    
    gc.collect()
    
    # ===== Strategy B: pairwise top-15 + extra_deep config =====
    log.info(f"    Strategy B: pairwise top-15 + extra_deep")
    for n_feat in [8, 10, 15, 20]:
        sel_cols = ranked_p15[:n_feat]
        oof = train_cv_model(feat_pair15, sel_cols, y, SEEDS, CFG_EXTRA_DEEP, n_folds=5)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        iso_cal, ok = isotonic_calibrate(oof_avg, y)
        if ok:
            iso_cal = mean_match(iso_cal, train_rates[target])
            ll = log_loss(y, iso_cal, labels=[0, 1])
        else:
            ll = log_loss(y, oof_avg, labels=[0, 1])
            iso_cal = oof_avg
        
        all_target_models.setdefault(target, []).append((iso_cal, ll, f"pair15_{CFG_EXTRA_DEEP['nl']}n{n_feat}"))
        if n_feat <= 15:
            log.info(f"      n_feat={n_feat:2d} pair15_exdeep: LL={ll:.5f}")
    
    gc.collect()
    
    # ===== Strategy C: pairwise top-15 + safety config =====
    log.info(f"    Strategy C: pairwise top-15 + safety")
    for n_feat in [8, 10, 15, 20]:
        sel_cols = ranked_p15[:n_feat]
        oof = train_cv_model(feat_pair15, sel_cols, y, SEEDS, CFG_SAFETY, n_folds=5)
        oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
        iso_cal, ok = isotonic_calibrate(oof_avg, y)
        if ok:
            iso_cal = mean_match(iso_cal, train_rates[target])
            ll = log_loss(y, iso_cal, labels=[0, 1])
        else:
            ll = log_loss(y, oof_avg, labels=[0, 1])
            iso_cal = oof_avg
        
        all_target_models.setdefault(target, []).append((iso_cal, ll, f"pair15_{CFG_SAFETY['nl']}n{n_feat}"))
    
    gc.collect()

# ============================================================
# Top-k ensemble per target
# ============================================================
log.info(f"\n{'='*70}")
log.info("TOP-K ENSEMBLE PER TARGET")
log.info(f"{'='*70}")

best_per_target = {}
for target in TARGETS:
    models = all_target_models[target]
    models_sorted = sorted(enumerate(models), key=lambda x: x[1][1])
    
    best_ll = float('inf')
    best_k = 1
    best_combo = models[:1]
    
    for k in range(1, len(models_sorted) + 1):
        top_k = models_sorted[:k]
        combined = np.zeros(len(y_train[target]))
        for idx, (oof, ll, method) in top_k:
            combined += oof
        n = len(top_k)
        combined /= n
        combined = mean_match(combined, train_rates[target])
        
        ll = log_loss(y_train[target], combined, labels=[0, 1])
        if ll < best_ll:
            best_ll = ll
            best_k = k
            best_combo = [m for _, m in top_k]
    
    best_per_target[target] = {
        'best_ll': best_ll,
        'best_k': best_k,
        'models': best_combo,
    }
    log.info(f"  {target}: top-{best_k} ensemble LL={best_ll:.5f}")
    for oof_v, ll_val, method in best_combo:
        log.info(f"    {method:40s} LL={ll_val:.5f}")

avg_ll = np.mean([best_per_target[t]['best_ll'] for t in TARGETS])
log.info(f"\n  AVG: {avg_ll:.5f}")

# ============================================================
# Cross-strategy ensemble: V116 + V117
# ============================================================
log.info(f"\n{'='*70}")
log.info("CROSS-STRATEGY ENSEMBLE (V116 + V117)")
log.info(f"{'='*70}")

for target in TARGETS:
    models = all_target_models[target]
    models_sorted = sorted(enumerate(models), key=lambda x: x[1][1])
    
    # Try ensemble with V116 prediction
    v116_pred = v116_oof[target].values
    best_ll = float('inf')
    best_k = 1
    best_combo = models[:1]
    best_weight = 1.0  # V116 weight
    
    for k in range(1, min(len(models_sorted) + 1, 15)):
        top_k = models_sorted[:k]
        for w_v116 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            n = len(top_k)
            combined = np.zeros(len(y_train[target]))
            if w_v116 > 0:
                combined += w_v116 * v116_pred
            for idx, (oof, ll, method) in top_k:
                combined += (1 - w_v116) * oof / max(n, 1)
            combined = mean_match(combined, train_rates[target])
            
            ll = log_loss(y_train[target], combined, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_k = k
                best_combo = [(oof, ll_v, method) for _, (oof, ll_v, method) in top_k]
                best_weight = w_v116
    
    log.info(f"  {target}: V116+w={best_weight:.1f} + top-{best_k} V117 -> LL={best_ll:.5f}")

# ============================================================
# Summary
# ============================================================
log.info(f"\n{'='*70}")
log.info("V117 SUMMARY")
log.info(f"{'='*70}")

avg_cal = np.mean([
    log_loss(feat[t].values, best_per_target[t]['models'][0][0], labels=[0, 1])
    for t in TARGETS
])
for t in TARGETS:
    ll = best_per_target[t]['best_ll']
    log.info(f"  {t}: LL={ll:.5f} (top-{best_per_target[t]['best_k']})")
log.info(f"  AVG: {avg_cal:.5f}")
log.info(f"  V116:  0.54761")
log.info(f"  V54:   0.53971")

# ============================================================
# Save
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

oof_df = pd.DataFrame({t: best_per_target[t]['models'][0][0] for t in TARGETS})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v117_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

exp_log = {
    'version': 'V117',
    'timestamp': ts,
    'seeds': len(SEEDS),
    'strategy': 'pairwise top-15 + broader n_feat + cross-strategy ensemble',
    'results': {t: {
        'best_ll': round(best_per_target[t]['best_ll'], 5),
        'best_k': best_per_target[t]['best_k'],
        'models': [m[2] for m in best_per_target[t]['models']],
    } for t in TARGETS},
    'avg_ll': round(avg_cal, 5),
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v117_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"\nSaved: {oof_path}")
log.info(f"Done in {time.time()-t_start:.0f}s")
