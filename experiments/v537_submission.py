"""V537 Submission Generator - Fine-Grained Per-Target Ridge Alpha

V537 found:
- Q2 α=0.06 (best α), gap=-0.01458 (vs308 baseline α=0.05: -0.01370)
- S2 α=0.03 (best α), gap=-0.07483 (vs308 baseline α=0.05: -0.07423)
- S3 α=10.0 (edge), gap=-0.00500
- Others: same or similar to V534/H1
- avg_gap=-0.03016, vs308=7/7, expected LB≈0.65251
"""
import sys, re, gc, logging, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT = Path('/home/mwoo423/.openclaw/workspace')
DATA = ROOT / 'data_processed'
SUBMIT = ROOT / 'submissions'
EXPERIMENTS = ROOT / 'experiments'
SUBMIT.mkdir(exist_ok=True)
EXPERIMENTS.mkdir(exist_ok=True)

TARGETS = ['Q1','Q2','Q3','S1','S2','S3','S4']
META_COLS = {'subject_id','lifelog_date','sleep_date','date'}
LEAK_S = {'wLight_w_light_mean','wLight_w_light_std','wLight_w_light_min','wLight_w_light_max','wLight_w_light_count',
          'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count',
          'wPedo_pedo_step_step_mean','wPedo_pedo_step_sum','wPedo_pedo_step_frequency_mean',
          'wPedo_pedo_step_frequency_sum','wPedo_pedo_running_step_mean','wPedo_pedo_running_step_sum',
          'wPedo_pedo_walking_step_mean','wPedo_pedo_walking_step_sum','wPedo_pedo_distance_mean',
          'wPedo_pedo_distance_sum','wPedo_pedo_speed_mean','wPedo_pedo_speed_sum',
          'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max','wHr_hr_median','wHr_hr_count'}
SEED = 42
N_FOLDS = 5
N_SEEDS = 13
V308_GAPS = {'Q1': 0.113, 'Q2': 0.079, 'Q3': 0.124, 'S1': 0.020, 'S2': 0.097, 'S3': 0.017, 'S4': 0.039}

# V537 best alphas from fine-grained sweep
BEST_ALPHAS = {'Q1': 0.001, 'Q2': 0.06, 'Q3': 0.001, 'S1': 0.01, 'S2': 0.03, 'S3': 10.0, 'S4': 0.003}

# V534 config (same as V536/H1)
V534_CONFIG = {
    'Q1':  {'n_feat': 3,  'xgb_cfg': 'q_narrow',  'lgbm_cfg': 'wide',    'n_est': 600},
    'Q2':  {'n_feat': 10, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 800},
    'Q3':  {'n_feat': 7,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 500},
    'S1':  {'n_feat': 3,  'xgb_cfg': 'q_strong',  'lgbm_cfg': 'wide',    'n_est': 500},
    'S2':  {'n_feat': 7,  'xgb_cfg': 's_strong',  'lgbm_cfg': 'wide_strong', 'n_est': 500},
    'S3':  {'n_feat': 23, 'xgb_cfg': 'q_strong',  'lgbm_cfg': 'safety',  'n_est': 1000},
    'S4':  {'n_feat': 20, 'xgb_cfg': 'q_deep',    'lgbm_cfg': 'wide',    'n_est': 300},
}
XGB_CFGS = {
    'q_narrow':  {'max_depth': 4, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 3},
    'q_deep':    {'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'min_child_weight': 5},
    'q_strong':  {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_weight': 5},
    's_strong':  {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_alpha': 10.0, 'reg_lambda': 20.0, 'min_child_weight': 10},
}
LGBM_CFGS = {
    'wide':      {'num_leaves': 30, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_samples': 5},
    'wide_strong': {'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 10.0, 'min_child_samples': 10},
    'safety':    {'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 10.0, 'min_child_samples': 20},
}

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
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
              'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
              'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1}
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X; gc.collect()
    return [r[0] for r in ranked]

def train_one(seed, X_tr, y_tr, X_va, X_test, feat_names, learner, n_est, **mp):
    if learner == 'xgb':
        params = {**mp, 'random_state': seed, 'n_jobs': 1, 'verbosity': 0}
        ds_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        ds_va = xgb.DMatrix(X_va, feature_names=feat_names)
        ds_te = xgb.DMatrix(X_test, feature_names=feat_names)
        m = xgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(ds_va), m.predict(ds_te)
    else:
        spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
        params = {**mp, 'scale_pos_weight': spw, 'random_state': seed,
                 'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
        sn = [sanitize_col(c) for c in feat_names]
        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sn)
        m = lgb.train(params, ds_tr, num_boost_round=n_est)
        return m.predict(X_va), m.predict(X_test)


def main():
    log.info("=" * 70)
    log.info("V537 Submission: Fine-Grained Per-Target Ridge Alpha")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / 'features.parquet')
    test_df = pd.read_parquet(DATA / 'test_features.parquet')
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    # Z-score
    train_base = [c for c in train_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    test_base = [c for c in test_df.columns if c not in META_COLS | set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    common_base = set(train_base) & set(test_base)
    for col in sorted(common_base):
        tv = train_df[col].fillna(0).values.astype(np.float64)
        ev = test_df[col].fillna(0).values.astype(np.float64)
        m, s = np.mean(tv), np.std(tv, ddof=0)
        if s < 1e-8: s = 1e-8
        zc = f'{col}_zscore'
        train_df[zc] = (tv - m) / s
        test_df[zc] = (ev - m) / s

    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(train_df)
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Pre-rank features
    log.info('Pre-ranking features...')
    ranked_features = {}
    for target in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, target)
        ranked_features[target] = rank_features(train_df, feat_cols_clean, target)

    # Generate predictions for all targets
    n_train = len(train_df)
    n_test = len(test_df)
    all_seed_oofs = {t: [] for t in TARGETS}
    all_test_preds = {t: [] for t in TARGETS}
    train_means = {}

    for target in TARGETS:
        bc = V534_CONFIG[target]
        sel_cols = ranked_features[target][:bc['n_feat']]
        feat_names = [c for c in sel_cols if c in test_feat_cols]
        if len(feat_names) != len(sel_cols): sel_cols = feat_names
        n_est = bc['n_est']
        xgb_params = {'n_estimators': n_est, **XGB_CFGS[bc['xgb_cfg']]}
        lgbm_params = {'n_estimators': n_est, **LGBM_CFGS[bc['lgbm_cfg']]}
        y = train_df[target].values.astype(np.float64)
        train_means[target] = np.mean(y)
        X_test_full = test_df[sel_cols].fillna(0).values.astype(np.float64)

        log.info(f'\n{target}: n_est={n_est}, n_feat={bc["n_feat"]}')
        xgb_oofs, lgbm_oofs = [], []
        xgb_tests, lgbm_tests = [], []

        for si in range(N_SEEDS):
            seed = SEED + si * 11
            oof_xgb = np.zeros(n_train)
            oof_lgbm = np.zeros(n_train)
            test_xgb = np.zeros(n_test)
            test_lgbm = np.zeros(n_test)

            for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, train_df['subject_id'].values)):
                X_tr = train_df[sel_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                X_va = train_df[sel_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                y_tr = y[tr_idx]
                pvxgb, ttxgb = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'xgb', n_est, **xgb_params)
                pvlgbm, ttlgbm = train_one(seed, X_tr, y_tr, X_va, X_test_full, sel_cols, 'lgbm', n_est, **lgbm_params)
                oof_xgb[va_idx] = pvxgb; oof_lgbm[va_idx] = pvlgbm
                test_xgb += ttxgb; test_lgbm += ttlgbm

            oof_xgb = np.clip(oof_xgb, 0.001, 0.999)
            oof_lgbm = np.clip(oof_lgbm, 0.001, 0.999)
            test_xgb /= N_FOLDS; test_lgbm /= N_FOLDS
            xgb_oofs.append(oof_xgb); lgbm_oofs.append(oof_lgbm)
            xgb_tests.append(test_xgb); lgbm_tests.append(test_lgbm)

            if si % 4 == 0 or si == N_SEEDS - 1:
                log.info(f'    seed {si:2d}: xgb_oof_mean={np.mean(oof_xgb):.4f}, lgbm_oof_mean={np.mean(oof_lgbm):.4f}')

        # Blend weight
        ll_xgb = np.mean([log_loss(y, np.clip(xo, 0.001, 0.999)) for xo in xgb_oofs])
        ll_lgbm = np.mean([log_loss(y, np.clip(lo, 0.001, 0.999)) for lo in lgbm_oofs])
        inv_xgb = 1.0 / max(ll_xgb, 1e-10); inv_lgbm = 1.0 / max(ll_lgbm, 1e-10)
        wx = inv_xgb / (inv_xgb + inv_lgbm)

        blended_oofs = [wx * xoof + (1-wx) * loof for xoof, loof in zip(xgb_oofs, lgbm_oofs)]
        blended_tests = [wx * xt + (1-wx) * lt for xt, lt in zip(xgb_tests, lgbm_tests)]
        all_seed_oofs[target] = blended_oofs
        all_test_preds[target] = blended_tests
        log.info(f'  {target}: wx={wx:.3f}, xgb_ll={ll_xgb:.5f}, lgbm_ll={ll_lgbm:.5f}')

    # ================================================================
    # Generate submission with V537 best alphas
    # ================================================================
    log.info('\nGenerating submission with V537 best alphas...')
    sub_df = pd.DataFrame({'subject_id': test_df['subject_id'].values})
    target_gaps = {}

    for t in TARGETS:
        alpha = BEST_ALPHAS[t]
        n_seeds = len(all_seed_oofs[t])
        oofs_2d = np.column_stack(all_seed_oofs[t])
        y_true = train_df[t].values

        # Compute gap
        avg_pred = np.mean(oofs_2d, axis=1)
        std_pred = np.std(oofs_2d, axis=1)
        X_meta = np.column_stack([avg_pred, std_pred])
        meta = Ridge(alpha=alpha)
        meta.fit(X_meta, y_true)
        train_pred = meta.predict(X_meta)
        pmin, pmax = train_pred.min(), train_pred.max()
        if pmax - pmin < 1e-10:
            train_proba = np.ones_like(train_pred) * 0.5
        else:
            train_proba = (train_pred - pmin) / (pmax - pmin)
        train_proba = np.clip(train_proba, 0.001, 0.999)
        meta_ll = log_loss(y_true, train_proba)
        avg_student = np.mean([log_loss(y_true, np.clip(oofs_2d[:, si], 0.001, 0.999)) for si in range(n_seeds)])
        gap = avg_student - meta_ll
        target_gaps[t] = gap

        # Apply to test
        test_preds_2d = np.column_stack(all_test_preds[t])
        test_avg_pred = np.mean(test_preds_2d, axis=1)
        test_std_pred = np.std(test_preds_2d, axis=1)
        X_test_meta = np.column_stack([test_avg_pred, test_std_pred])
        test_pred = meta.predict(X_test_meta)
        test_pmin, test_pmax = train_pred.min(), train_pred.max()
        if test_pmax - test_pmin < 1e-10:
            test_proba = np.ones_like(test_pred) * 0.5
        else:
            test_proba = (test_pred - test_pmin) / (test_pmax - test_pmin)
        test_proba = np.clip(test_proba, 0.001, 0.999)
        sub_df[t] = test_proba

        log.info(f'  {t}: α={alpha:.4f}, gap={gap:+.5f} (vs308={"✅" if gap < V308_GAPS[t] else "❌"})')

    avg_gap = sum(target_gaps.values()) / 7
    vs308 = sum(1 for t in TARGETS if target_gaps[t] < V308_GAPS[t])
    improvement_v534 = -0.02754 - avg_gap
    improvement_v536h1 = -0.02996 - avg_gap

    # Save submission
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_str = '_'.join([f'{t}{BEST_ALPHAS[t]:.3f}' for t in TARGETS])
    sub_path = SUBMIT / f'submission_v537_fine_alpha_{alpha_str}_{ts}.csv'
    sub_df.to_csv(sub_path, index=False)
    log.info(f'\n📁 Submission saved: {sub_path}')

    # Save result JSON
    result = {
        'version': 'V537',
        'hypothesis': 'fine_per_target_ridge_alpha',
        'name': 'Fine-Grained Per-Target Ridge Alpha',
        'avg_gap': float(avg_gap),
        'improvement_vs_v534': float(improvement_v534),
        'improvement_vs_v536_h1': float(improvement_v536h1),
        'vs308': int(vs308),
        'expected_lb': float(0.62235 - avg_gap),
        'best_alphas': {k: float(v) for k, v in BEST_ALPHAS.items()},
        'target_gaps': {k: float(v) for k, v in target_gaps.items()},
        'submission_file': str(sub_path.name),
        'timestamp': ts,
    }
    result_path = EXPERIMENTS / f'v537_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f'📁 Result saved: {result_path}')

    # ================================================================
    # Summary
    # ================================================================
    log.info('\n' + '=' * 70)
    log.info('🏆 V537 FINAL SUMMARY')
    log.info('=' * 70)
    log.info(f'  Config: Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20')
    log.info(f'  Meta: Ridge (per-target α)')
    log.info(f'  Best alphas: {BEST_ALPHAS}')
    log.info(f'  Target gaps: {target_gaps}')
    log.info(f'  avg_gap: {avg_gap:+.5f}')
    log.info(f'  vs V534 (-0.02754): +{improvement_v534:+.5f}')
    log.info(f'  vs V536 H1 (-0.02996): +{improvement_v536h1:+.5f}')
    log.info(f'  vs308: {vs308}/7')
    log.info(f'  Expected LB: {0.62235 - avg_gap:.5f}')
    log.info(f'  Submission: {sub_path.name}')
    log.info('=' * 70)

    return result


if __name__ == '__main__':
    import json
    result = main()
