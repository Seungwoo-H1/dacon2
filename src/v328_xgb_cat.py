"""
V328 — XGBoost + CatBoost + LGBM Ensemble

Hypothesis: Adding XGBoost and CatBoost alongside LGBM captures different
patterns. 3 model types × 15 seeds = 45 students per target → LR meta.

Expected OOF: 0.595-0.605
Risk: MEDIUM
Cost: ~180s
"""
import sys, gc, logging, json, re, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

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
          'wPedo_pedo_burned_calories_mean','wPedo_pedo_burned_calories_sum'}
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
          'wHr_hr_median','wHr_hr_count'}

N_FOLDS = 5
N_SEEDS = 15
FEATURE_BAG_FRACTION = 0.75

def sanitize_col(n):
    return re.sub(r'[^a-zA-Z0-9_]', '_', n)

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS | set(TARGETS) and np.issubdtype(df[c].dtype, np.number)]

def remove_leak(cols, target):
    if target.startswith('S'): return [c for c in cols if c not in LEAK_S]
    elif target.startswith('Q'): return [c for c in cols if c not in LEAK_Q]
    return cols

def rank_features(feat_df, feat_cols, target, seed=42):
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y==0).sum()) / max((y==1).sum(), 1), 0.1)
    params = {'objective':'binary','metric':'binary_logloss','verbose':-1,
              'num_leaves':20,'max_depth':5,'learning_rate':0.05,'n_estimators':50,
              'scale_pos_weight':spw,'random_state':seed,'force_row_wise':True,'n_jobs':1}
    ds = lgb.Dataset(X, label=y, feature_name=[sanitize_col(c) for c in feat_cols])
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X; gc.collect()
    return [r[0] for r in ranked]

def generate_zscore(train_df, test_df):
    tb = [c for c in train_df.columns if c not in META_COLS|set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(train_df[c].dtype, np.number)]
    te = [c for c in test_df.columns if c not in META_COLS|set(TARGETS) and not c.endswith('_zscore') and np.issubdtype(test_df[c].dtype, np.number)]
    cc = set(tb) & set(te)
    for col in cc:
        v = train_df[col].fillna(0).values.astype(np.float64)
        mu, sd = np.mean(v), max(np.std(v, ddof=0), 1e-8)
        zc = f'{col}_zscore'
        test_df = test_df.copy(); test_df[zc] = (test_df[col].fillna(0).values.astype(np.float64)-mu)/sd
        train_df = train_df.copy(); train_df[zc] = (v-mu)/sd
    return train_df, test_df

CFGS_LGBM = {
    'wide':   {'num_leaves':30,'max_depth':3,'learning_rate':0.05,'n_estimators':300,
              'subsample':0.8,'colsample_bytree':0.8,'reg_alpha':2.0,'reg_lambda':5.0,'min_child_samples':5},
    'deep':   {'num_leaves':20,'max_depth':5,'learning_rate':0.02,'n_estimators':1000,
              'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':0.5,'reg_lambda':2.0,'min_child_samples':15},
    'v48':    {'num_leaves':15,'max_depth':4,'learning_rate':0.03,'n_estimators':500,
              'subsample':0.7,'colsample_bytree':0.7,'reg_alpha':1.0,'reg_lambda':3.0,'min_child_samples':10},
    'safety': {'num_leaves':10,'max_depth':3,'learning_rate':0.02,'n_estimators':1000,
              'subsample':0.6,'colsample_bytree':0.6,'reg_alpha':3.0,'reg_lambda':10.0,'min_child_samples':20},
}
V53_SWEEP = {
    'Q1':{'cfg':'deep','n_feat':19},'Q2':{'cfg':'deep','n_feat':14},'Q3':{'cfg':'v48','n_feat':11},
    'S1':{'cfg':'wide','n_feat':21},'S2':{'cfg':'deep','n_feat':19},
    'S3':{'cfg':'safety','n_feat':23},'S4':{'cfg':'wide','n_feat':20},
}

def main():
    global t_start; t_start = time.time()
    log.info("="*70)
    log.info("V328 — XGBoost + CatBoost + LGBM Ensemble")
    log.info("3 model types × 15 seeds = 45 students per target")
    log.info("="*70)

    SEED = 42
    MODEL_TYPES = ['cat', 'lgbm']

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")
    for df in [train_df, test_df]:
        for c in ['sleep_date','lifelog_date','date']:
            if c in df.columns: df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    train_df, test_df = generate_zscore(train_df, test_df)
    train_feat_cols = get_feature_cols(train_df)
    test_feat_cols = get_feature_cols(test_df)
    log.info(f"Train: {len(train_feat_cols)} features, Test: {len(test_feat_cols)}")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    all_student_results = {}
    meta_results = {}

    for t in TARGETS:
        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}")
        y = train_df[t].values.astype(np.float64)
        feat_cols_clean = remove_leak(train_feat_cols, t)
        n_feat = V53_SWEEP[t]['n_feat']
        lgbm_cfg = CFGS_LGBM[V53_SWEEP[t]['cfg']]
        ranked = rank_features(train_df, feat_cols_clean, t)

        oof_list = []
        test_list = []

        for mi, mtype in enumerate(MODEL_TYPES):
            log.info(f"  Model: {mtype}")
            for si in range(N_SEEDS):
                seed = SEED + si * 7
                rng = np.random.RandomState(seed)
                n_bag = max(int(len(ranked) * FEATURE_BAG_FRACTION), n_feat)
                bag = rng.choice(ranked, size=n_bag, replace=False)
                bag_set = set(bag)
                bag_feats = [f for f in ranked if f in bag_set][:n_feat]
                if len(bag_feats) < n_feat:
                    bag_feats += [f for f in ranked if f not in bag_set][:n_feat - len(bag_feats)]
                s_cols = [c for c in bag_feats if c in test_feat_cols]

                seed_oof = np.zeros(len(train_df))
                seed_test = np.zeros(len(test_df))

                for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, y, group)):
                    X_tr = train_df[s_cols].iloc[tr_idx].fillna(0).values.astype(np.float64)
                    X_va = train_df[s_cols].iloc[va_idx].fillna(0).values.astype(np.float64)
                    y_tr = y[tr_idx]
                    spw = max(((y_tr==0).sum()) / max((y_tr==1).sum(), 1), 0.1)

                    if mtype == 'lgbm':
                        params = {**lgbm_cfg, 'scale_pos_weight':spw, 'random_state':seed,
                                  'force_row_wise':True, 'n_jobs':1, 'verbose':-1}
                        ds = lgb.Dataset(X_tr, label=y_tr, feature_name=[sanitize_col(c) for c in s_cols])
                        mdl = lgb.train(params, ds, num_boost_round=lgbm_cfg['n_estimators'])
                        oof_p = mdl.predict(X_va)
                        test_p = mdl.predict(test_df[s_cols].fillna(0).values.astype(np.float64))

                    elif mtype == 'xgb':
                        params = {'objective':'binary:logistic','eval_metric':'logloss',
                                  'learning_rate':0.05,'max_depth':5,
                                  'subsample':0.7,'colsample_bytree':0.7,
                                  'reg_alpha':1.0,'reg_lambda':3.0,
                                  'scale_pos_weight':spw,'random_state':seed,
                                  'tree_method':'hist','n_jobs':1,'verbosity':0}
                        sn = [sanitize_col(c) for c in s_cols]
                        ds = xgb.DMatrix(X_tr, label=y_tr, feature_names=sn)
                        mdl = xgb.train(params, ds, num_boost_round=300, verbose_eval=False)
                        oof_p = mdl.predict(xgb.DMatrix(X_va, feature_names=sn))
                        test_p = mdl.predict(xgb.DMatrix(test_df[s_cols].fillna(0).values.astype(np.float64), feature_names=sn))

                    elif mtype == 'cat':
                        params = {'loss_function':'Logloss','eval_metric':'Logloss',
                                  'learning_rate':0.05,'max_depth':5,
                                  'subsample':0.7,'colsample_bylevel':0.7,
                                  'l2_leaf_reg':3.0,'random_seed':seed,
                                  'bootstrap_type':'Bernoulli','random_strength':1.0,
                                  'verbose':0}
                        sn = [sanitize_col(c) for c in s_cols]
                        train_pool = cb.Pool(X_tr, label=y_tr, feature_names=sn)
                        mdl = cb.CatBoostClassifier(**params, iterations=300)
                        mdl.fit(train_pool, verbose=False)
                        oof_p = mdl.predict(cb.Pool(X_va), prediction_type='Probability')[:, 1]
                        test_p = mdl.predict(cb.Pool(test_df[s_cols].fillna(0).values.astype(np.float64)),
                                             prediction_type='Probability')[:, 1]

                    seed_oof[va_idx] += oof_p
                    seed_test += test_p

                seed_oof /= N_FOLDS
                seed_test /= N_FOLDS
                seed_oof = np.clip(seed_oof, 0.001, 0.999)
                oof_list.append(seed_oof)
                test_list.append(seed_test)

                if si < 2 or si == N_SEEDS - 1:
                    s_oof = log_loss(y, seed_oof)
                    log.info(f"    {mtype} Seed {si}: OOF={s_oof:.5f}")

        # LR meta on all 45 students
        oof_matrix = np.column_stack(oof_list)
        meta = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
        meta.fit(oof_matrix, y)
        train_pred = meta.predict_proba(oof_matrix)[:, 1]
        target_oof = log_loss(y, np.clip(train_pred, 0.001, 0.999))
        student_avg = np.mean([log_loss(y, p) for p in oof_list])

        for mi, mtype in enumerate(MODEL_TYPES):
            mt_oofs = [log_loss(y, oof_list[mi*N_SEEDS+si]) for si in range(N_SEEDS)]
            log.info(f"  {mtype} avg OOF: {np.mean(mt_oofs):.5f}")

        log.info(f"  {t}: Meta OOF={target_oof:.5f} (student-avg={student_avg:.5f})")
        all_student_results[t] = (oof_list, test_list)
        meta_results[t] = (target_oof, student_avg, meta)

    avg_oof = np.mean([r[0] for r in meta_results.values()])
    avg_student = np.mean([r[1] for r in meta_results.values()])

    log.info(f"\n{'='*70}")
    log.info(f"V328 RESULTS (3 model types x 15 seeds)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        gap = meta_results[t][1] - meta_results[t][0]
        log.info(f"  {t}: OOF={meta_results[t][0]:.5f} (L1-avg={meta_results[t][1]:.5f}, gap={gap:+.4f})")
    log.info(f"  AVG OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  V321: 0.60569 | V326: 0.59159 | V308: 0.62235")
    log.info(f"  Delta vs V321: {avg_oof - 0.60569:+.5f}")
    log.info(f"  Delta vs V326: {avg_oof - 0.59159:+.5f}")
    pred_lb = avg_oof + 0.019
    log.info(f"  Predicted LB: {pred_lb:.5f}")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame()
    sub['subject_id'] = test_df['subject_id'].values
    sub['sleep_date'] = test_df['sleep_date'].values
    sub['lifelog_date'] = test_df['lifelog_date'].values
    for t in TARGETS:
        sub[t] = np.clip(np.mean(all_student_results[t][1]), 0.001, 0.999)

    sub_path = SUBMIT / f"submission_v328_3model_{ts}.csv"
    sub.to_csv(sub_path, index=False)
    log.info(f"Saved submission: {sub_path}")

    meta_data = {
        'version':'V328', 'name':'XGBoost+CatBoost+LGBM Ensemble',
        'avg_oof':round(float(avg_oof),5), 'avg_student_oof':round(float(avg_student),5),
        'n_features_total':len(train_feat_cols), 'model_types':MODEL_TYPES, 'n_seeds':N_SEEDS,
        'per_target_oof':{t:round(float(meta_results[t][0]),5) for t in TARGETS},
        'v321_avg_oof':0.60569, 'v326_avg_oof':0.59159,
        'delta_vs_v321':round(float(avg_oof-0.60569),5),
        'delta_vs_v326':round(float(avg_oof-0.59159),5),
        'predicted_lb':round(float(pred_lb),5),
        'submission_file':str(sub_path),
        'timestamp':ts, 'total_time_s':round(time.time()-t_start,0),
        'key_difference':'3 model types (XGBoost+CatBoost+LGBM) x 15 seeds',
    }
    meta_path = EXPERIMENTS / f'v328_{ts}.json'
    with open(meta_path,'w') as f: json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
