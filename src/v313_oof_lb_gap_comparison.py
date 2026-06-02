"""
V312 → V313 OOF-LB Gap Comparison

V312: 15 seeds, C=500, OOF=0.61448
V313: 30 seeds, C=500, OOF=0.59512

Key question: Which has smaller OOF-LB gap?

Theoretical analysis:
- More seeds → better ensemble averaging → smaller variance
- Higher C → meta trusts students more → lower OOF but risk of gap increase
- 30 seeds should have LOWER variance than 15 seeds
- But higher C=500 (vs V308's C=10) means more fitting to students

Expected OOF-LB gaps:
- V308 (15 seeds, C=10): gap +0.01658
- V312 (15 seeds, C=500): gap ~0.015-0.025?
- V313 (30 seeds, C=500): gap ~0.012-0.020?

If V313 gap is SMALLER than V308's gap despite lower OOF, 
then V313 is a clear winner.
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
TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id', 'lifelog_date', 'sleep_date', 'date'}

LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count','wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count','wPedo_pedo_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean','wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum','wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean','wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum','wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}

CFGS = {'wide': {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5}, 'deep': {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15}, 'v48': {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10}, 'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20}}

V53_SWEEP = {'Q1': {'cfg': 'deep', 'n_feat': 19}, 'Q2': {'cfg': 'deep', 'n_feat': 14}, 'Q3': {'cfg': 'v48', 'n_feat': 11}, 'S1': {'cfg': 'wide', 'n_feat': 21}, 'S2': {'cfg': 'deep', 'n_feat': 19}, 'S3': {'cfg': 'safety', 'n_feat': 23}, 'S4': {'cfg': 'wide', 'n_feat': 20}}

SEED = 42
N_FOLDS = 5


def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat_df, feat_cols, target, seed=SEED):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(),1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,'num_leaves':20,'max_depth':5,'learning_rate':0.05,'n_estimators':50,'scale_pos_weight':spw,'random_state':seed,'force_row_wise':True,'n_jobs':1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X; gc.collect()
    return [r[0] for r in ranked]

def generate_zscore(train_df, test_df):
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8: std = 1e-8
            zc = f'{col}_zscore'
            test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64) - mean) / std
            train_df[zc] = (vals - mean) / std
    return test_df

def run_experiment(n_seeds, meta_C, name):
    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df = generate_zscore(train_df, test_df)
    
    test_feat_cols = get_feature_cols(test_df)
    train_feat_cols = get_feature_cols(train_df)
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    target_oofs = {}
    per_seed_all = {t: [] for t in TARGETS}
    student_oofs_all = {t: [] for t in TARGETS}
    
    for t in TARGETS:
        log.info(f"\n  {t} ({name}, {n_seeds} seeds, C={meta_C}):")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        ranked = rank_features(train_df, feat_cols_clean, t)
        sel_cols = ranked[:n_feat]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols): sel_cols = sel_cols_test
        cfg = CFGS[cfg_name]
        
        per_seed_oofs = []
        for si in range(n_seeds):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                spw = max(((y_tr==0).sum()) / max((y_tr==1).sum(),1), 0.1)
                params = {**cfg, 'scale_pos_weight':spw, 'random_state':seed, 'force_row_wise':True, 'n_jobs':1, 'verbose':-1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                seed_oof[va_idx] = m.predict(X_va)
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            per_seed_oofs.append(seed_oof)
            student_oofs_all[t].append(log_loss(y, seed_oof))
        
        stacked = np.column_stack(per_seed_oofs)
        meta = LogisticRegression(C=meta_C, max_iter=1000, random_state=SEED)
        meta.fit(stacked, y)
        train_pred = meta.predict_proba(stacked)[:, 1]
        ll = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        target_oofs[t] = ll
        per_seed_all[t] = per_seed_oofs
    
    avg_oof = np.mean(list(target_oofs.values()))
    student_avg = np.mean([np.mean(student_oofs_all[t]) for t in TARGETS])
    student_std = np.mean([np.std(student_oofs_all[t]) for t in TARGETS])
    gap_meta_student = student_avg - avg_oof
    
    return avg_oof, student_avg, student_std, gap_meta_student, per_seed_all, student_oofs_all, target_oofs

# Run V308 config (15 seeds, C=10) - for gap reference
log.info("="*70)
log.info("Running V308 config: 15 seeds, C=10")
v308_oof, v308_student_avg, v308_student_std, v308_gap, _, _, v308_targets = run_experiment(15, 10.0, "V308")
log.info(f"\nV308 config: OOF={v308_oof:.5f}, Student avg={v308_student_avg:.5f}, gap={v308_gap:.5f}")
log.info(f"V308 actual LB: 0.63893, actual gap: +0.01658")

# Run V312 config (15 seeds, C=500)
log.info("\n" + "="*70)
log.info("Running V312 config: 15 seeds, C=500")
v312_oof, v312_student_avg, v312_student_std, v312_gap, _, _, v312_targets = run_experiment(15, 500.0, "V312")
log.info(f"\nV312 config: OOF={v312_oof:.5f}, Student avg={v312_student_avg:.5f}, gap={v312_gap:.5f}")
log.info(f"Predicted LB (V308 gap): {v312_oof + 0.01658:.5f}")
log.info(f"Predicted LB (V312 gap): {v312_oof + v312_gap:.5f}")

# Run V313 config (30 seeds, C=500)
log.info("\n" + "="*70)
log.info("Running V313 config: 30 seeds, C=500")
v313_oof, v313_student_avg, v313_student_std, v313_gap, _, _, v313_targets = run_experiment(30, 500.0, "V313")
log.info(f"\nV313 config: OOF={v313_oof:.5f}, Student avg={v313_student_avg:.5f}, gap={v313_gap:.5f}")
log.info(f"Predicted LB (V308 gap): {v313_oof + 0.01658:.5f}")
log.info(f"Predicted LB (V313 gap): {v313_oof + v313_gap:.5f}")

# Comparison
log.info(f"\n{'='*70}")
log.info("=== GAP ANALYSIS SUMMARY ===")
log.info("="*70)
log.info(f"{'Config':<15} {'OOF':<10} {'Student':<10} {'Gap':<10} {'Pred LB(V308)':<15} {'Pred LB(act)':<15}")
log.info(f"{'V308 (15,10)':<15} {'0.62235':<10} {'0.77000':<10} {'0.14765':<10} {'0.63893':<15} {'0.63893':<15}")
log.info(f"{'V312 (15,500)':<15} {v312_oof:<10.5f} {v312_student_avg:<10.5f} {v312_gap:<10.5f} {v312_oof+0.01658:<15.5f} {v312_oof+v312_gap:<15.5f}")
log.info(f"{'V313 (30,500)':<15} {v313_oof:<10.5f} {v313_student_avg:<10.5f} {v313_gap:<10.5f} {v313_oof+0.01658:<15.5f} {v313_oof+v313_gap:<15.5f}")
log.info(f"\n{'V308 LB':<15} {'0.63893':<10} {'0.63893':<10} {'vs V308':<10} {'0.63893':<15} {'0.63893':<15}")
log.info(f"{'V312':<15} {'0.62235':<10} {'0.62235':<10} {'vs V308':<10} {'{:+.5f}'.format(v312_oof+0.01658-0.63893):<15} {'{:+.5f}'.format(v312_oof+v312_gap-0.63893):<15}")
log.info(f"{'V313':<15} {'0.62235':<10} {'0.62235':<10} {'vs V308':<10} {'{:+.5f}'.format(v313_oof+0.01658-0.63893):<15} {'{:+.5f}'.format(v313_oof+v313_gap-0.63893):<15}")
