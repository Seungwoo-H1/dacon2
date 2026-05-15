"""V121: V115 pipeline with test predictions + submission (fast config)

Fast config: wide+deep configs only, n_feat [10, 15, 20], seeds [42, 7, 999, 777]
Expected: OOF ~0.54, test predictions generated, submission saved.
Time: ~60-90 min (vs V115's ~120 min, V118's SIGKILL)
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

# Fast config: wide+deep only, 4 seeds, 3 n_feat
CFG_WIDE = {'nl': 30, 'md': 3, 'lr': 0.05, 'ne': 300, 'ss': 0.8, 'cb': 0.8, 'ra': 2.0, 'rl': 5.0, 'mc': 5}
CFG_DEEP = {'nl': 20, 'md': 5, 'lr': 0.02, 'ne': 1000, 'ss': 0.7, 'cb': 0.6, 'ra': 0.5, 'rl': 2.0, 'mc': 15}
CFGS = {'wide': CFG_WIDE, 'deep': CFG_DEEP}

SEEDS = [42, 7, 999, 777]

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

def add_personalization(df, feature_cols, fit_stats=None, for_test=False):
    personal_cols = []
    df = df.copy()
    all_stats = {}
    for col in feature_cols:
        col_filled = df[col].fillna(0)
        grp = col_filled.groupby(df['subject_id']).agg(['mean', 'std'])
        grp.columns = [f'{col}_subj_mean', f'{col}_subj_std']
        grp = grp.reset_index()
        df = df.merge(grp, on='subject_id', how='left')
        if not for_test:
            all_stats[col] = {'mean': grp[f'{col}_subj_mean'], 'std': grp[f'{col}_subj_std']}
        if fit_stats is not None and col in fit_stats:
            subj_mean = fit_stats[col]['mean']
            subj_std = fit_stats[col]['std']
        else:
            subj_mean = df[f'{col}_subj_mean']
            subj_std = df[f'{col}_subj_std']
        mask_zero = subj_std == 0
        mask_null = df[col].isnull()
        df[f'{col}_zscore'] = np.where(
            mask_zero | mask_null, 0.0,
            (df[col].fillna(0) - subj_mean) / np.maximum(subj_std, 1e-8))
        personal_cols.append(f'{col}_zscore')
        gc.collect()
    return df, personal_cols, all_stats

def add_pairwise_interactions(feat, top_features):
    feat = feat.copy()
    added = []
    for i in range(min(len(top_features), 10)):
        for j in range(i+1, min(len(top_features), 10)):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feat.columns or f2 not in feat.columns:
                continue
            col_prod = f'{f1}_x_{f2}'
            feat[col_prod] = feat[f1].fillna(0) * feat[f2].fillna(0)
            added.append(col_prod)
            if feat[f1].std() > 0 and feat[f2].std() > 0:
                col_ratio = f'{f1}_div_{f2}'
                feat[col_ratio] = feat[f1].fillna(0) / (feat[f2].fillna(0) + 1e-8)
                added.append(col_ratio)
    for f in top_features[:5]:
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

def train_cv_model(feat, feat_tst, cols, y, seeds, cfg, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros((len(y), len(seeds)))
    test_preds = np.zeros((len(feat_tst), len(seeds)))
    sn = [sanitize(c) for c in cols]
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    X_full = feat[cols].fillna(0).values.astype(np.float64)
    X_test = feat_tst[cols].fillna(0).values.astype(np.float64)
    for si, seed in enumerate(seeds):
        cfg_full = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'verbose': -1, 'force_row_wise': True, 'n_jobs': 1,
            'num_leaves': cfg['nl'], 'max_depth': cfg['md'],
            'learning_rate': cfg['lr'], 'n_estimators': cfg['ne'],
            'subsample': cfg['ss'], 'colsample_bytree': cfg['cb'],
            'reg_alpha': cfg['ra'], 'reg_lambda': cfg['rl'],
            'min_child_samples': cfg['mc'],
            'random_state': seed, 'scale_pos_weight': spw,
        }
        for tr_i, va_i in gkf.split(feat, y, feat['subject_id']):
            ds = lgb.Dataset(X_full[tr_i], label=y[tr_i], feature_name=sn, params={'verbose': '-1'})
            vd = lgb.Dataset(X_full[va_i], label=y[va_i], feature_name=sn, reference=ds, params={'verbose': '-1'})
            m = lgb.train(cfg_full, ds, num_boost_round=cfg['ne'],
                         valid_sets=[vd],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = m.predict(X_full[va_i])
            test_preds[:, si] = m.predict(X_test)
            del ds, vd, m
            gc.collect()
    return oof, test_preds

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
feat, zscore_cols, fit_stats = add_personalization(feat, feat_cols_raw)
feat_test_z, _, _ = add_personalization(feat_test, feat_cols_raw, fit_stats=fit_stats, for_test=True)
all_cols = feat_cols_raw + zscore_cols
log.info(f"After personalization: train={feat.shape}, test={feat_test_z.shape}")

y_train = {t: feat[t].values for t in TARGETS}
train_rates = {t: feat[t].mean() for t in TARGETS}

# ============================================================
# Feature ranking
# ============================================================
log.info("\n--- Feature ranking ---")
ranked_lgb = {}
for target in TARGETS:
    leak_cols = remove_leak(all_cols, target)
    ranked = rank_features_importance(feat, leak_cols, target)
    ranked_lgb[target] = ranked
    log.info(f"  {target} top-5: {ranked[:5]}")

# ============================================================
# Strategy experiments with test predictions
# ============================================================
log.info(f"\n--- Multi-strategy experiments (test preds) ---")
log.info(f"  2 configs × 3 n_feat × {len(SEEDS)} seeds × 3 strategies × 7 targets")

all_results = {}

for target in TARGETS:
    tgt_t = time.time()
    y = feat[target].values.astype(np.float64)
    best_cal = float('inf')
    best_oof = None
    best_test = None
    best_config_str = None
    best_iso = None
    
    # Strategy A: Base
    base_cols = remove_leak(all_cols, target)
    ranked_base = rank_features_importance(feat, base_cols, target)
    
    for cfg_name, cfg in CFGS.items():
        for n_feat in [10, 15, 20]:
            sel_cols = ranked_base[:n_feat]
            oof, test_p = train_cv_model(feat, feat_test_z, sel_cols, y, SEEDS, cfg, n_folds=5)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            iso = IsotonicRegression(out_of_bounds='clip')
            try:
                iso.fit(oof_avg, y)
                iso_cal_oof = iso.predict(oof_avg)
                iso_cal_oof = mean_match(iso_cal_oof, train_rates[target])
                iso_cal_test = iso.predict(test_avg)
                iso_cal_test = mean_match(iso_cal_test, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
            except Exception:
                iso_cal_oof = mean_match(oof_avg, train_rates[target])
                iso_cal_test = mean_match(test_avg, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
                iso = None
            
            config_str = f"base_{cfg_name}_n{n_feat}"
            if ll < best_cal:
                best_cal = ll
                best_oof = iso_cal_oof.copy()
                best_test = iso_cal_test.copy() if iso is not None else iso_cal_oof.copy()
                best_config_str = config_str
                best_iso = iso
    
    gc.collect()
    
    # Strategy B: Pairwise
    top_features = ranked_lgb[target][:10]
    feat_pair = add_pairwise_interactions(feat.copy(), top_features)[0]
    feat_pair_z, _, _ = add_personalization(feat_pair, feat_cols_raw + zscore_cols, fit_stats=fit_stats)
    feat_test_pair, _ = add_pairwise_interactions(feat_test_z.copy(), top_features)
    feat_test_pair_z, _, _ = add_personalization(feat_test_pair, feat_cols_raw + zscore_cols, fit_stats=fit_stats, for_test=True)
    all_cols_pair = get_feature_cols(feat_pair_z)
    all_cols_pair = [c for c in all_cols_pair if c not in META | set(TARGETS)]
    ranked_pair = rank_features_importance(feat_pair_z, remove_leak(all_cols_pair, target), target)
    
    for cfg_name, cfg in CFGS.items():
        for n_feat in [10, 15, 20]:
            sel_cols = ranked_pair[:n_feat]
            oof, test_p = train_cv_model(feat_pair_z, feat_test_pair_z, sel_cols, y, SEEDS, cfg, n_folds=5)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            iso = IsotonicRegression(out_of_bounds='clip')
            try:
                iso.fit(oof_avg, y)
                iso_cal_oof = iso.predict(oof_avg)
                iso_cal_oof = mean_match(iso_cal_oof, train_rates[target])
                iso_cal_test = iso.predict(test_avg)
                iso_cal_test = mean_match(iso_cal_test, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
            except Exception:
                iso_cal_oof = mean_match(oof_avg, train_rates[target])
                iso_cal_test = mean_match(test_avg, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
                iso = None
            
            config_str = f"pair_{cfg_name}_n{n_feat}"
            if ll < best_cal:
                best_cal = ll
                best_oof = iso_cal_oof.copy()
                best_test = iso_cal_test.copy() if iso is not None else iso_cal_oof.copy()
                best_config_str = config_str
                best_iso = iso
    
    gc.collect()
    
    # Strategy C: Transformed
    feat_trans = add_transformed_features(feat.copy(), ranked_lgb[target][:15])[0]
    feat_trans_z, _, _ = add_personalization(feat_trans, feat_cols_raw + zscore_cols, fit_stats=fit_stats)
    feat_test_trans, _ = add_transformed_features(feat_test_z.copy(), ranked_lgb[target][:15])
    feat_test_trans_z, _, _ = add_personalization(feat_test_trans, feat_cols_raw + zscore_cols, fit_stats=fit_stats, for_test=True)
    all_cols_trans = get_feature_cols(feat_trans_z)
    all_cols_trans = [c for c in all_cols_trans if c not in META | set(TARGETS)]
    ranked_trans = rank_features_importance(feat_trans_z, remove_leak(all_cols_trans, target), target)
    
    for cfg_name, cfg in CFGS.items():
        for n_feat in [10, 15, 20]:
            sel_cols = ranked_trans[:n_feat]
            oof, test_p = train_cv_model(feat_trans_z, feat_test_trans_z, sel_cols, y, SEEDS, cfg, n_folds=5)
            oof_avg = np.clip(oof.mean(axis=1), 0.0001, 0.9999)
            test_avg = np.clip(test_p.mean(axis=1), 0.0001, 0.9999)
            
            iso = IsotonicRegression(out_of_bounds='clip')
            try:
                iso.fit(oof_avg, y)
                iso_cal_oof = iso.predict(oof_avg)
                iso_cal_oof = mean_match(iso_cal_oof, train_rates[target])
                iso_cal_test = iso.predict(test_avg)
                iso_cal_test = mean_match(iso_cal_test, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
            except Exception:
                iso_cal_oof = mean_match(oof_avg, train_rates[target])
                iso_cal_test = mean_match(test_avg, train_rates[target])
                ll = log_loss(y, iso_cal_oof, labels=[0, 1])
                iso = None
            
            config_str = f"trans_{cfg_name}_n{n_feat}"
            if ll < best_cal:
                best_cal = ll
                best_oof = iso_cal_oof.copy()
                best_test = iso_cal_test.copy() if iso is not None else iso_cal_oof.copy()
                best_config_str = config_str
                best_iso = iso
    
    gc.collect()
    
    all_results[target] = {
        'best_method': best_config_str,
        'cal_oof': best_oof,
        'cal_loss': best_cal,
        'test_preds': best_test,
        'iso_model': best_iso,
    }
    log.info(f"  {target}: {best_config_str:30s} Cal={best_cal:.5f} (time: {time.time()-tgt_t:.0f}s)")

# ============================================================
# Summary
# ============================================================
log.info(f"\n{'='*70}")
log.info("V121 SUMMARY")
log.info(f"{'='*70}")

avg_cal = np.mean([
    log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
    for t in TARGETS
])
for t in TARGETS:
    ll = log_loss(feat[t].values, all_results[t]['cal_oof'], labels=[0, 1])
    test_pred = all_results[t]['test_preds']
    log.info(f"  {t}: {all_results[t]['best_method']:30s} Cal={ll:.5f} test_mean={test_pred.mean():.4f} test_std={test_pred.std():.4f}")
log.info(f"  AVG: {avg_cal:.5f}")

# ============================================================
# Generate submission
# ============================================================
log.info(f"\n{'='*70}")
log.info("GENERATING SUBMISSION")
log.info(f"{'='*70}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save OOF
oof_df = pd.DataFrame({t: all_results[t]['cal_oof'] for t in TARGETS})
oof_df.insert(0, 'subject_id', feat['subject_id'].values)
oof_df.insert(1, 'sleep_date', feat['sleep_date'].values)
oof_df.insert(2, 'lifelog_date', feat['lifelog_date'].values)
oof_path = DATA / f'oof_v121_{ts}.csv'
oof_df.to_csv(oof_path, index=False)

# Save submission
sub_df = pd.DataFrame({t: all_results[t]['test_preds'] for t in TARGETS})
sub_df.insert(0, 'subject_id', feat_test['subject_id'].values)
sub_df.insert(1, 'sleep_date', feat_test['sleep_date'].values)
sub_df.insert(2, 'lifelog_date', feat_test['lifelog_date'].values)
sub_path = SUBMIT / f'submission_v121_{ts}.csv'
sub_df.to_csv(sub_path, index=False)

log.info(f"\nSaved:")
log.info(f"  OOF: {oof_path}")
log.info(f"  Submission: {sub_path}")

# Experiment log
exp_log = {
    'version': 'V121',
    'timestamp': ts,
    'configs': list(CFGS.keys()),
    'seeds': SEEDS,
    'strategy': 'base+pairwise+transformed, test preds',
    'results': {t: {
        'best_method': all_results[t]['best_method'],
        'cal_loss': round(all_results[t]['cal_loss'], 5),
        'test_mean': round(all_results[t]['test_preds'].mean(), 4),
        'test_std': round(all_results[t]['test_preds'].std(), 4),
    } for t in TARGETS},
    'avg_cal_loss': round(avg_cal, 5),
    'total_time_s': round(time.time() - t_start, 0),
}
with open(EXPERIMENTS / f'v121_{ts}.json', 'w') as f:
    json.dump(exp_log, f, indent=2, default=str)

log.info(f"  Log: {EXPERIMENTS / f'v121_{ts}.json'}")
log.info(f"Done in {time.time()-t_start:.0f}s")
