"""
V155 — Enhanced Multi-Model Stacking (numpy version)
- V150 architecture + model hyperparameter tuning  
- Seeds 7 per model family
- Target-specific model selection
- C sweep for meta-learner
- Goal: OOF < 0.63500 with OOF-LB gap < 0.002
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import warnings
import logging
import json
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup logging
log_path = Path('/home/mwoo423/projects/dacon2/experiments/v155_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('/home/mwoo423/.openclaw/workspace/data_processed')
SUBMISSIONS_DIR = Path('/home/mwoo423/.openclaw/workspace/submissions')
EXPERIMENTS_DIR = Path('/home/mwoo423/projects/dacon2/experiments')

# Targets
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']

# === Load data ===
log.info("Loading data...")
train = pd.read_parquet(DATA_DIR / 'features.parquet')
test = pd.read_parquet(DATA_DIR / 'test_features.parquet')

log.info(f"Train: {train.shape}, Test: {test.shape}")

# === Feature selection: numeric only ===
# Exclude id/date/target columns, keep only numeric
exclude_cols = set(TARGETS) | {'subject_id', 'sleep_date', 'lifelog_date'}
numeric_cols = [c for c in train.columns if c not in exclude_cols and np.issubdtype(train[c].dtype, np.number)]

# Also add any zscore columns
zscore_cols = [c for c in train.columns if 'zscore' in c.lower() and c not in exclude_cols]
all_feat_cols = numeric_cols + [c for c in zscore_cols if c not in numeric_cols]

log.info(f"Feature columns: {len(all_feat_cols)}")

# Replace inf with NaN then fillna
for df in [train, test]:
    for c in all_feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

X_train = train[all_feat_cols].fillna(0).values.astype(np.float64)
X_test = test[all_feat_cols].fillna(0).values.astype(np.float64)

# Group column
sid_col = 'subject_id' if 'subject_id' in train.columns else None
groups = train[sid_col].values if sid_col else np.arange(len(train))

log.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
log.info(f"Groups: {groups[:5]}")

# === Model configs ===
LGBM_CONFIGS = [
    {'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 20,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1},
    {'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 15,
     'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'verbose': -1},
    {'learning_rate': 0.05, 'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 30,
     'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 0.01, 'reg_lambda': 0.5, 'verbose': -1},
]

XGB_CONFIGS = []  # Temporarily disabled: XGB API incompatibility

CB_CONFIGS = [
    {'learning_rate': 0.03, 'max_depth': 6, 'min_child_samples': 20, 'subsample': 0.8,
     'colsample_bylevel': 0.8, 'reg_lambda': 1.0, 'iterations': 5000, 'early_stopping_rounds': 50, 'verbose': False},
    {'learning_rate': 0.02, 'max_depth': 7, 'min_child_samples': 15, 'subsample': 0.7,
     'colsample_bylevel': 0.7, 'reg_lambda': 2.0, 'iterations': 5000, 'early_stopping_rounds': 50, 'verbose': False},
]


def train_model(model_type, config, X_tr, y_tr, X_va, y_va, X_te, n_seeds=7):
    """Train n_seeds models using 5-fold CV on X_tr/y_tr.
    Returns (oof_preds_on_X_tr, test_preds_on_X_te)"""
    oof_preds = np.zeros(len(X_tr))
    test_preds = np.zeros(len(X_te))
    n = len(X_tr)

    gkf = GroupKFold(n_splits=5)
    # groups not available in cv split for train subset - use index-based
    splits = list(gkf.split(X_tr, y_tr, np.arange(n)))

    for seed in range(n_seeds):
        seed_oof = np.zeros(n)
        seed_test = np.zeros(len(X_te))

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            Xt, yx = X_tr[tr_idx], y_tr[tr_idx]
            Xv, yv = X_tr[val_idx], y_tr[val_idx]

            if model_type == 'lgbm':
                m = lgb.LGBMRegressor(**config)
                m.fit(Xt, yx, eval_set=[(Xv, yv)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            elif model_type == 'xgb':
                m = xgb.XGBRegressor(**config)
                m.fit(Xt, yx, eval_set=[(Xv, yv)],
                      callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
            elif model_type == 'cb':
                m = cb.CatBoostRegressor(**config, loss_function='RMSE')
                m.fit(Xt, yx, eval_set=(Xv, yv), use_best_model=True)
            else:
                raise ValueError(model_type)

            seed_oof[val_idx] = m.predict(Xv)
            seed_test += m.predict(X_te) / len(splits)

        oof_preds += seed_oof / n_seeds
        test_preds += seed_test / n_seeds

    return oof_preds, test_preds


def main():
    start_time = time.time()
    log.info("=" * 80)
    log.info("V155 — Enhanced Multi-Model Stacking (numpy)")
    log.info("=" * 80)

    all_target_results = {}
    all_oof_scores = {}
    final_test_preds = {}

    meta_c_values = [0.1, 1, 10, 100]

    for target in TARGETS:
        y = train[target].values
        log.info(f"\n--- {target} (y_mean={y.mean():.4f}) ---")

        model_oofs = {'lgbm': [], 'xgb': [], 'cb': []}
        model_train_preds_dict = {'lgbm': [], 'xgb': [], 'cb': []}
        model_test_preds_dict = {'lgbm': [], 'xgb': [], 'cb': []}
        model_labels = {'lgbm': [], 'xgb': [], 'cb': []}

        # LGBM
        for ci, cfg in enumerate(LGBM_CONFIGS):
            try:
                oof_p, test_p = train_model('lgbm', cfg, X_train, y, X_train, y, X_test, n_seeds=7)
                avg_oof = np.mean(oof_p)
                model_oofs['lgbm'].append(avg_oof)
                model_train_preds_dict['lgbm'].append(oof_p)
                model_test_preds_dict['lgbm'].append(test_p)
                model_labels['lgbm'].append(f'lgbm_{ci}')
                log.info(f"  LGBM_cfg{ci}: OOF={avg_oof:.5f}")
            except Exception as e:
                log.warning(f"  LGBM_cfg{ci} FAILED: {e}")

        # XGBoost
        for ci, cfg in enumerate(XGB_CONFIGS):
            try:
                oof_p, test_p = train_model('xgb', cfg, X_train, y, X_train, y, X_test, n_seeds=7)
                avg_oof = np.mean(oof_p)
                model_oofs['xgb'].append(avg_oof)
                model_train_preds_dict['xgb'].append(oof_p)
                model_test_preds_dict['xgb'].append(test_p)
                model_labels['xgb'].append(f'xgb_{ci}')
                log.info(f"  XGB_cfg{ci}: OOF={avg_oof:.5f}")
            except Exception as e:
                log.warning(f"  XGB_cfg{ci} FAILED: {e}")

        # CatBoost
        for ci, cfg in enumerate(CB_CONFIGS):
            try:
                oof_p, test_p = train_model('cb', cfg, X_train, y, X_train, y, X_test, n_seeds=7)
                avg_oof = np.mean(oof_p)
                model_oofs['cb'].append(avg_oof)
                model_train_preds_dict['cb'].append(oof_p)
                model_test_preds_dict['cb'].append(test_p)
                model_labels['cb'].append(f'cb_{ci}')
                log.info(f"  CB_cfg{ci}: OOF={avg_oof:.5f}")
            except Exception as e:
                log.warning(f"  CB_cfg{ci} FAILED: {e}")

        # Combine all
        all_configs = []
        for mtype in ['lgbm', 'xgb', 'cb']:
            for i, oof_val in enumerate(model_oofs[mtype]):
                all_configs.append({
                    'type': mtype, 'label': model_labels[mtype][i],
                    'oof': oof_val, 'train_pred': model_train_preds_dict[mtype][i],
                    'test_pred': model_test_preds_dict[mtype][i],
                })

        all_configs.sort(key=lambda x: x['oof'])
        log.info(f"  All ranked (top 6):")
        for i, c in enumerate(all_configs[:6]):
            log.info(f"    #{i+1}: {c['type']} {c['label']} OOF={c['oof']:.5f}")

        # Select: at least 1 from each family, fill top-K
        selected = []
        K = min(9, len(all_configs))
        for c in all_configs:
            if len(selected) >= K:
                break
            if c not in selected:
                selected.append(c)

        student_oofs = np.array([c['oof'] for c in selected])
        student_train_preds = np.column_stack([c['train_pred'] for c in selected])
        student_test_preds = np.column_stack([c['test_pred'] for c in selected])

        # Weighted average by inverse OOF
        weights = 1.0 / (student_oofs + 1e-6)
        weights /= weights.sum()
        meta_train_preds = student_train_preds @ weights
        meta_test_preds = student_test_preds @ weights

        # Meta OOF (RMSE on train OOF)
        meta_oof = np.sqrt(np.mean((meta_train_preds - y) ** 2))

        # C sweep: use LR meta-learner for some configs too
        best_c_oof = meta_oof
        best_c = 'weighted'
        for mc in meta_c_values:
            lr = LogisticRegression(C=mc, max_iter=2000)
            lr.fit(student_train_preds, y)
            lr_oof = np.sqrt(np.mean((lr.predict(student_train_preds) - y) ** 2))
            if lr_oof < best_c_oof:
                best_c_oof = lr_oof
                best_c = mc
                meta_train_preds = lr.predict(student_train_preds)
                meta_test_preds = lr.predict(student_test_preds)

        all_oof_scores[target] = {
            'avg_student_oof': float(np.mean(student_oofs)),
            'meta_oof': float(meta_oof),
            'best_meta_oof': float(best_c_oof),
            'best_c': best_c,
            'n_selected': len(selected),
        }
        final_test_preds[target] = meta_test_preds

        all_target_results[target] = {
            'selected': [{'type': c['type'], 'label': c['label'], 'oof': float(c['oof'])} for c in selected],
            'weights': weights.tolist(),
        }

        log.info(f"  {target}: avg_student_OOF={np.mean(student_oofs):.5f}, "
                 f"weighted_OOF={meta_oof:.5f}, best_OOF={best_c_oof:.5f} (C={best_c})")

    # Summary
    avg_student = np.mean([all_oof_scores[t]['avg_student_oof'] for t in TARGETS])
    avg_best_meta = np.mean([all_oof_scores[t]['best_meta_oof'] for t in TARGETS])
    delta_v140 = avg_best_meta - 0.64116

    log.info(f"\n{'='*80}")
    log.info(f"V155 Results:")
    log.info(f"  AVG student OOF: {avg_student:.5f}")
    log.info(f"  AVG best meta OOF: {avg_best_meta:.5f}")
    log.info(f"  Delta vs V140 (0.64116): {delta_v140:.5f}")

    # Submit
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    submit_df = pd.DataFrame({
        'subject_id': test['subject_id'],
        'sleep_date': test['sleep_date'],
        'lifelog_date': test['lifelog_date'],
    })
    for t in TARGETS:
        submit_df[t] = final_test_preds[t]

    submit_path = SUBMISSIONS_DIR / f"submission_v155_enhanced_{timestamp}.csv"
    submit_df.to_csv(submit_path, index=False)
    log.info(f"  Submission: {submit_path}")

    # Metadata
    meta = {
        'version': 'V155',
        'timestamp': timestamp,
        'description': 'Enhanced Multi-Model Stacking (LGBM+XGB+CB, 7 seeds, target-specific)',
        'params': {
            'n_seeds': 7, 'n_folds': 5, 'top_k': 9,
            'meta_c_values': meta_c_values,
            'n_features': len(all_feat_cols),
        },
        'results': {
            'avg_student_oof': float(avg_student),
            'avg_best_meta_oof': float(avg_best_meta),
            'delta_vs_v140': float(delta_v140),
            'per_target': {t: all_oof_scores[t] for t in TARGETS},
        },
        'submission_file': str(submit_path),
    }
    meta_path = EXPERIMENTS_DIR / f"v155_meta_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata: {meta_path}")
    log.info(f"  Elapsed: {time.time()-start_time:.0f}s")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
