"""
V420 — LOO-Date CV: Data Leakage Check

Hypothesis: GroupKFold based on subject_id may allow temporal leakage.
If a subject has multiple dates, the model sees predictions from other
dates of the same subject during training → data leakage.

V420 approach:
1. Use LOO-Date CV instead of GroupKFold
2. For each test date, train on all data EXCEPT that date
3. Compare OOF with V413's GroupKFold — if LOO-Date OOF is much higher,
   then GroupKFold has leakage

This is a diagnostic experiment, not necessarily for submission.
"""
import sys, gc, logging, json, re, time, warnings, math
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss
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
LEAK_Q = {'wHr_hr_mean','wHr_hr_std','wHr_hr_min','wHr_hr_max',
          'wHr_hr_median','wHr_hr_count'}

CFG = {'num_leaves': 20, 'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 2500,
       'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 8.0, 'reg_lambda': 30.0,
       'min_child_samples': 35}  # narrow config

SEED = 42
N_SEEDS = 5  # Fewer seeds for speed (diagnostic)


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


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V420 — LOO-Date CV: Data Leakage Check")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")

    # Check date distribution
    train_df['date'] = pd.to_datetime(train_df['lifelog_date']).dt.normalize()
    unique_dates = train_df['date'].nunique()
    log.info(f"Train shape: {train_df.shape}, Unique dates: {unique_dates}")
    log.info(f"Subjects: {train_df['subject_id'].nunique()}")

    # Group by subject and date
    subj_dates = train_df.groupby('subject_id')['date'].apply(set).to_dict()
    for sid, dates in subj_dates.items():
        if len(dates) > 1:
            log.info(f"  Subject {sid}: {len(dates)} dates")

    feat_cols = get_feature_cols(train_df)
    log.info(f"Total features: {len(feat_cols)}")

    # Check leakage columns
    leak_removed = remove_leak(feat_cols, 'Q1')
    log.info(f"Features after leak removal (Q1): {len(leak_removed)}")
    leaked = set(feat_cols) - set(leak_removed)
    log.info(f"Leaked features: {leaked}")

    # Run GroupKFold OOF for V413 baseline
    from sklearn.model_selection import GroupKFold
    import traceback

    group_oofs = {}
    date_oofs = {}

    for t_idx, target in enumerate(TARGETS):
        t_start_local = time.time()
        feat_cols_t = remove_leak(feat_cols, target)
        n_feat = V413_NFEAT.get(target, 19)

        X_all = train_df[feat_cols_t + [target]].fillna(0).values.astype(np.float64)
        y_all = X_all[:, -1]
        X_all = X_all[:, :-1]

        groups_arr = train_df['subject_id'].values

        # GroupKFold OOF
        gkf = GroupKFold(n_splits=5)
        gkf_preds = np.zeros(len(train_df))
        gkf_folds = []

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups_arr)):
            x_train, y_train = X_all[tr_idx], y_all[tr_idx]
            x_val = X_all[val_idx]
            spw = max(((y_train == 0).sum()) / max((y_train == 1).sum(), 1), 0.1)
            params = {**CFG, 'scale_pos_weight': spw, 'random_state': SEED,
                      'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
            ds_train = lgb.Dataset(x_train, label=y_train)
            ds_val = lgb.Dataset(x_val, label=y_all[val_idx], reference=ds_train)
            model = lgb.train(params, ds_train, num_boost_round=CFG['n_estimators'],
                valid_sets=[ds_val], callbacks=[lgb.early_stopping(200, verbose=False)])
            gkf_preds[val_idx] = model.predict(x_val)

        gkf_oof = log_loss(y_all, gkf_preds)
        group_oofs[target] = gkf_oof

        # LOO-Date OOF (simplified: by-subject leave-one-date-out)
        # For each subject, leave one date out at a time
        date_preds = np.zeros(len(train_df))
        date_mask = np.zeros(len(train_df), dtype=bool)
        date_labels = np.zeros(len(train_df))

        groups = train_df.groupby('subject_id')
        for sid, group_idx in groups.indices.items():
            dates_in_subj = train_df.loc[group_idx, 'date'].unique()
            for d in dates_in_subj:
                mask_train = group_idx[~(train_df.loc[group_idx, 'date'] == d).values]
                mask_val = group_idx[(train_df.loc[group_idx, 'date'] == d).values]
                if len(mask_val) == 0 or len(mask_train) < 10:
                    continue

                x_tr, y_tr = X_all[mask_train], y_all[mask_train]
                x_val = X_all[mask_val]
                spw = max(((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1), 0.1)
                params = {**CFG, 'scale_pos_weight': spw, 'random_state': SEED + int(re.sub(r'[^0-9]', '', sid) or sid[:2]),
                          'force_row_wise': True, 'n_jobs': 1, 'verbose': -1}
                ds_train = lgb.Dataset(x_tr, label=y_tr)
                model = lgb.train(params, ds_train, num_boost_round=CFG['n_estimators']//3)
                date_preds[mask_val] = model.predict(x_val)
                date_mask[mask_val] = True
                date_labels[mask_val] = y_all[mask_val]

        date_oof = log_loss(date_labels[date_mask], date_preds[date_mask])
        date_oofs[target] = date_oof

        log.info(f"\n{target}:")
        log.info(f"  GroupKFold OOF: {gkf_oof:.5f}")
        log.info(f"  LOO-Date OOF:   {date_oof:.5f}")
        log.info(f"  Δ (Group-Date): {date_oof - gkf_oof:+.5f}")
        log.info(f"  ({time.time()-t_start_local:.0f}s)")

    avg_gkf = np.mean(list(group_oofs.values()))
    avg_date = np.mean(list(date_oofs.values()))

    log.info(f"\n{'='*70}")
    log.info("V420 Results:")
    log.info(f"  GroupKFold AVG OOF: {avg_gkf:.5f}")
    log.info(f"  LOO-Date AVG OOF:   {avg_date:.5f}")
    log.info(f"  Δ: {avg_date - avg_gkf:+.5f}")
    log.info(f"  If Δ > 0.01, GroupKFold has significant temporal leakage")
    log.info(f"{'='*70}")

    # Save for analysis
    result = {
        'version': 'V420',
        'name': 'LOO-Date CV: Data Leakage Check',
        'unique_dates': unique_dates,
        'avg_groupkf_oof': round(float(avg_gkf), 5),
        'avg_date_oof': round(float(avg_date), 5),
        'delta': round(float(avg_date - avg_gkf), 5),
        'per_target': {t: {'groupkf': round(float(v), 5), 'loo_date': round(float(date_oofs[t]), 5)}
                      for t, v in group_oofs.items()},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    meta_path = EXPERIMENTS / f'v420_{result["timestamp"]}.json'
    with open(meta_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved analysis: {meta_path}")
    log.info(f"Total time: {time.time()-t_start:.0f}s")


# Need to import V413_NFEAT
V413_NFEAT = {'Q1': 19, 'Q2': 19, 'Q3': 15, 'S1': 21, 'S2': 19, 'S3': 23, 'S4': 20}

if __name__ == '__main__':
    main()
