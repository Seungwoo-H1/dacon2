"""
V393 — Trimmed Mean Ensemble

Hypothesis: V308 uses equal averaging of 15 seeds. Some seeds may be outliers
(poor calibration) that drag down ensemble performance. Trimmed mean removes
extreme predictions before averaging → more robust ensemble.

V393: Replace equal averaging with trimmed mean (remove top/bottom 2 of 15 seeds).
Also try removing top/bottom 1 and 3.
This doesn't change the meta-learner at all — only changes the aggregation method.

Key insight from V308:
- Seed student OOF range: ~0.622~0.629 (very tight)
- But individual seed predictions may have extreme values in tails
- Trimmed mean on predictions (not OOF) may help calibration

Methods tested:
- Trim 0 (V308 baseline): equal average
- Trim 1: remove best + worst seed
- Trim 2: remove 2 best + 2 worst seeds
- Trim 3: remove 3 best + 3 worst seeds

Expected:
- Student avg: slightly better (0.688-0.692)
- Meta OOF: similar
- Risk: Low (simple change, no structural impact)
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

CFGS = {
    'wide':   {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 300,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'deep':   {'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_samples': 15},
    'v48':    {'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 500,
               'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_samples': 10},
    'safety': {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1000,
               'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

V53_SWEEP = {
    'Q1':  {'cfg': 'deep',   'n_feat': 19},
    'Q2':  {'cfg': 'deep',   'n_feat': 14},
    'Q3':  {'cfg': 'v48',    'n_feat': 11},
    'S1':  {'cfg': 'wide',   'n_feat': 21},
    'S2':  {'cfg': 'deep',   'n_feat': 19},
    'S3':  {'cfg': 'safety', 'n_feat': 23},
    'S4':  {'cfg': 'wide',   'n_feat': 20},
}

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C = 10.0


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


def generate_test_zscore(train_df, test_df):
    log.info("Generating test z-score features...")
    train_feat_cols = [c for c in train_df.columns
                       if c not in META_COLS | set(TARGETS)
                       and not c.endswith('_zscore')
                       and np.issubdtype(train_df[c].dtype, np.number)]
    test_feat_cols = [c for c in test_df.columns
                      if c not in META_COLS | set(TARGETS)
                      and not c.endswith('_zscore')
                      and np.issubdtype(test_df[c].dtype, np.number)]
    common_cols = set(train_feat_cols) & set(test_feat_cols)
    log.info(f"Common base columns for z-score: {len(common_cols)}")
    zscore_cols = []
    for col in common_cols:
        train_vals = train_df[col].fillna(0).values.astype(np.float64)
        test_vals = test_df[col].fillna(0).values.astype(np.float64)
        mean = np.mean(train_vals)
        std = np.std(train_vals, ddof=0)
        if std < 1e-8:
            std = 1e-8
        zc_name = f'{col}_zscore'
        test_df[zc_name] = (test_vals - mean) / std
        zscore_cols.append(zc_name)
    log.info(f"Generated {len(zscore_cols)} z-score features for test")
    return test_df, zscore_cols


def trimmed_mean(preds, trim_count):
    """Trim top/bottom trim_count predictions and average."""
    n = len(preds)
    if n <= 2 * trim_count:
        return np.mean(preds, axis=0)
    sorted_preds = np.sort(preds, axis=0)
    trimmed = sorted_preds[trim_count:n-trim_count, :]
    return np.mean(trimmed, axis=0)


def main():
    global t_start
    t_start = time.time()
    
    log.info("=" * 70)
    log.info("V393 — Trimmed Mean Ensemble")
    log.info("Hypothesis: Trimmed mean → robust aggregation → better calibration")
    log.info("V308: OOF=0.62235, LB=0.63893")
    log.info("=" * 70)
    
    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    
    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    
    test_df, zscore_cols = generate_test_zscore(train_df, test_df)
    
    train_base = [c for c in train_df.columns
                  if c not in META_COLS | set(TARGETS)
                  and not c.endswith('_zscore')
                  and np.issubdtype(train_df[c].dtype, np.number)]
    
    for col in train_base:
        if col in test_df.columns:
            vals = train_df[col].fillna(0).values.astype(np.float64)
            mean = np.mean(vals)
            std = np.std(vals, ddof=0)
            if std < 1e-8:
                std = 1e-8
            zc = f'{col}_zscore'
            train_df[zc] = (vals - mean) / std
    
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    
    log.info(f"Train: {len(train_feat_cols)} features")
    log.info(f"Test:  {len(test_feat_cols)} features")
    
    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)
    
    V308_OOF = {
        'Q1': 0.67096, 'Q2': 0.62299, 'Q3': 0.61939,
        'S1': 0.57915, 'S2': 0.61564, 'S3': 0.60994, 'S4': 0.63839
    }
    
    # Test trim counts
    trim_counts = [0, 1, 2, 3]
    
    all_results_by_trim = {tc: [] for tc in trim_counts}
    all_student_oofs = []
    all_per_seed_test = {}
    all_per_seed_oofs = {}
    
    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        cfg_name = V53_SWEEP[t]['cfg']
        cfg = CFGS[cfg_name]
        
        log.info(f"    Config: {cfg_name}, n_feat: {n_feat}")
        
        # V308-style ranking
        y_rank = train_df[t].values.astype(np.float64)
        X_rank = train_df[feat_cols_clean].fillna(0).values.astype(np.float64)
        spw_rank = max(((y_rank == 0).sum()) / max((y_rank == 1).sum(), 1), 0.1)
        params_rank = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
            'scale_pos_weight': spw_rank, 'random_state': SEED, 'force_row_wise': True, 'n_jobs': 1
        }
        sn_rank = [sanitize_col(c) for c in feat_cols_clean]
        ds_rank = lgb.Dataset(X_rank, label=y_rank, feature_name=sn_rank)
        m_rank = lgb.train(params_rank, ds_rank, num_boost_round=50)
        imp_rank = m_rank.feature_importance(importance_type='gain')
        ranked_all = sorted(zip(feat_cols_clean, imp_rank), key=lambda x: -x[1])
        sel_cols = [r[0] for r in ranked_all[:n_feat]]
        sel_cols_test = [c for c in sel_cols if c in test_feat_cols]
        if len(sel_cols_test) != len(sel_cols):
            sel_cols = sel_cols_test
        
        # Train models per fold
        per_seed_oofs = []
        per_seed_test = []
        student_oofs = []
        
        for si in range(N_SEEDS):
            seed = SEED + si * 7
            seed_oof = np.zeros(n_train)
            seed_test = np.zeros(n_test)
            
            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**cfg, 'scale_pos_weight': spw, 'random_state': seed,
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                sn = [sanitize_col(c) for c in sel_cols]
                ds = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
                m = lgb.train(params, ds, num_boost_round=cfg['n_estimators'])
                
                seed_oof[va_idx] = m.predict(X_va)
                seed_test += m.predict(test_df[sel_cols].fillna(0).values.astype(np.float64))
            
            seed_oof = np.clip(seed_oof, 0.001, 0.999)
            seed_test /= N_FOLDS
            per_seed_oofs.append(seed_oof)
            per_seed_test.append(seed_test)
            student_oofs.append(log_loss(y, seed_oof))
            
            if si < 5 or si % 3 == 0:
                log.info(f"    Seed {si:2d}: OOF={log_loss(y, seed_oof):.5f}")
        
        all_student_oofs.extend(student_oofs)
        all_per_seed_test[t] = per_seed_test
        all_per_seed_oofs[t] = per_seed_oofs
        
        # Test trimmed mean for each trim count
        stacked_oofs = np.array(per_seed_oofs)  # (15, 450)
        stacked_test = np.column_stack(per_seed_test)
        
        for tc in trim_counts:
            # Trimmed mean OOF
            if tc == 0:
                tm_oof = np.mean(stacked_oofs, axis=0)
            else:
                tm_oof = trimmed_mean(stacked_oofs, tc)
            tm_oof_ll = log_loss(y, tm_oof)
            
            # Trimmed mean test
            if tc == 0:
                tm_test = np.mean(stacked_test, axis=1)  # equal avg, shape (250,)
            else:
                # For test, trim per-sample across seeds
                tm_test = np.zeros(n_test)
                for i in range(n_test):
                    vals = stacked_test[i, :]  # shape (15,) — per test sample across seeds
                    sorted_vals = np.sort(vals)
                    if tc > 0 and tc * 2 < len(vals):
                        trimmed = sorted_vals[tc:-tc]
                    else:
                        trimmed = sorted_vals
                    tm_test[i] = np.mean(trimmed)
            
            # Trimmed mean student (OOF)
            tm_student_ll = log_loss(y, np.clip(tm_oof, 0.001, 0.999))
            
            all_results_by_trim[tc].append({
                'target': t, 'tm_oof': tm_oof_ll, 'tm_test': tm_test, 'tm_student': tm_student_ll
            })
        
        log.info(f"    Equal avg student OOF: {np.mean(student_oofs):.5f}")
    
    # Summary
    avg_student = np.mean(all_student_oofs)
    
    log.info(f"\n{'='*70}")
    log.info("V393 RESULTS")
    log.info(f"{'='*70}")
    
    for tc in trim_counts:
        avg_tm_oof = np.mean([r['tm_oof'] for r in all_results_by_trim[tc]])
        avg_tm_student = np.mean([r['tm_student'] for r in all_results_by_trim[tc]])
        predicted_lb = avg_tm_oof + 0.01658
        
        label = "Equal" if tc == 0 else f"Trim-{tc}"
        log.info(f"\n  {label}:")
        log.info(f"    AVG TM OOF: {avg_tm_oof:.5f} (Δ: {avg_tm_oof-0.62235:+.5f})")
        log.info(f"    AVG TM student: {avg_tm_student:.5f}")
        log.info(f"    Predicted LB: {predicted_lb:.5f} (Δ: {predicted_lb-0.63893:+.5f})")
        
        beats = predicted_lb < 0.63893
        log.info(f"    Beats V308: {beats}")
    
    log.info(f"{'='*70}")
    
    # Pick best trim count
    best_tc = 0
    best_pred_lb = float('inf')
    for tc in trim_counts:
        avg_tm_oof = np.mean([r['tm_oof'] for r in all_results_by_trim[tc]])
        pred_lb = avg_tm_oof + 0.01658
        if pred_lb < best_pred_lb:
            best_pred_lb = pred_lb
            best_tc = tc
    
    log.info(f"\n  Best trim count: {best_tc} (Predicted LB: {best_pred_lb:.5f})")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    meta_data = {
        'version': 'V393',
        'name': 'Trimmed Mean Ensemble',
        'v308_avg_oof': 0.62235,
        'v308_lb': 0.63893,
        'student_avg_oof': round(float(avg_student), 5),
        'best_trim_count': best_tc,
        'results_by_trim': {},
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }
    
    for tc in trim_counts:
        avg_tm_oof = np.mean([r['tm_oof'] for r in all_results_by_trim[tc]])
        avg_tm_student = np.mean([r['tm_student'] for r in all_results_by_trim[tc]])
        meta_data['results_by_trim'][str(tc)] = {
            'avg_oof': round(float(avg_tm_oof), 5),
            'delta_vs_v308': round(float(avg_tm_oof - 0.62235), 5),
            'avg_student': round(float(avg_tm_student), 5),
            'predicted_lb': round(float(avg_tm_oof + 0.01658), 5),
            'beats_v308': bool(avg_tm_oof + 0.01658 < 0.63893)
        }
    
    meta_path = EXPERIMENTS / f'v393_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    
    # Create submission with best trim count
    log.info(f"\nV393: Creating submission with trim={best_tc}...")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    
    for t in TARGETS:
        r = next(rr for rr in all_results_by_trim[best_tc] if rr['target'] == t)
        sub[t] = r['tm_test']
    
    sub_path = SUBMIT / f"submission_v393_trimmed_mean_trim{best_tc}_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")
    meta_data['submission_file'] = str(sub_path)
    
    if best_pred_lb < 0.63893:
        log.info(f"V393 predicted LB: {best_pred_lb:.5f} (Δ: {best_pred_lb-0.63893:+.5f})")
    else:
        log.info(f"V393 predicted LB: {best_pred_lb:.5f} (Δ: {best_pred_lb-0.63893:+.5f}) — does NOT beat V308")
    
    return meta_data


if __name__ == '__main__':
    main()
