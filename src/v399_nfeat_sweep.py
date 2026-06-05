"""
V399 — Per-Target Feature Count Sweep with Per-Target Meta C

Hypothesis: V392 uses fixed n_feat per target from V53_SWEEP, but those
sweeps were done without per-target meta C. The optimal n_feat may differ
when using per-target C=10/100.

V392: fixed n_feat (Q→19/14/11, S→21/19/23/20) + per-target C → meta 0.617
V399: sweep n_feat per target + per-target C → find true optimal

Sweep ranges:
- Q targets: try n_feat = 8, 12, 16, 20, 25
- S targets: try n_feat = 12, 16, 20, 24, 28

This is a systematic sweep, so we only test the best config per target.
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

SEED = 42
N_FOLDS = 5
N_SEEDS = 15
META_C_PER_TARGET = {
    'Q1': 10.0, 'Q2': 10.0, 'Q3': 10.0,
    'S1': 100.0, 'S2': 100.0, 'S3': 100.0, 'S4': 100.0,
}

# Per-target feature count sweep
N_FEAT_SWEEP = {
    'Q1':  [10, 14, 19, 24, 29],
    'Q2':  [8, 12, 14, 19, 24],
    'Q3':  [8, 11, 14, 19, 24],
    'S1':  [15, 18, 21, 24, 27],
    'S2':  [14, 17, 19, 22, 25],
    'S3':  [15, 19, 23, 27, 31],
    'S4':  [14, 17, 20, 23, 26],
}


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


def rank_features(feat_df, feat_cols, target, seed=SEED):
    """Rank features by LGBM gain importance."""
    y = feat_df[target].values.astype(np.float64)
    X = feat_df[feat_cols].fillna(0).values.astype(np.float64)
    spw = max(((y == 0).sum()) / max((y == 1).sum(), 1), 0.1)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
        'num_leaves': 20, 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 50,
        'scale_pos_weight': spw, 'random_state': seed, 'force_row_wise': True, 'n_jobs': 1
    }
    sn = [sanitize_col(c) for c in feat_cols]
    ds = lgb.Dataset(X, label=y, feature_name=sn)
    m = lgb.train(params, ds, num_boost_round=50)
    imp = m.feature_importance(importance_type='gain')
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    del m, ds, X
    gc.collect()
    return [r[0] for r in ranked]


def run_target(target, train_df, test_df, feat_cols_clean, n_feat, cfg_name, meta_c, gkf, group, n_train, n_test, SEED=SEED):
    """Run full pipeline for one target with given n_feat. Returns (meta_oof, student_oof)."""
    ranked = rank_features(train_df, feat_cols_clean, target)
    sel_cols = ranked[:n_feat]

    test_feat_cols = [c for c in test_df.columns
                      if c not in META_COLS | set(TARGETS)
                      and np.issubdtype(test_df[c].dtype, np.number)]
    sel_cols_test = [c for c in sel_cols if c in test_feat_cols]

    if len(sel_cols_test) != len(sel_cols):
        missing = set(sel_cols) - set(sel_cols_test)
        sel_cols = sel_cols_test

    y = train_df[target].values.astype(np.float64)
    cfg = CFGS[cfg_name]

    # Level 0: N_SEEDS LGBM models
    per_seed_oofs = []
    test_preds = np.zeros(n_test)

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
            seed_test += m.predict(test_df[sel_cols_test].fillna(0).values.astype(np.float64))

        seed_oof = np.clip(seed_oof, 0.001, 0.999)
        seed_test /= N_FOLDS
        per_seed_oofs.append(seed_oof)
    del test_preds  # not used in this function

    # Level 1: Stack → meta-learner
    stacked = np.column_stack(per_seed_oofs)
    meta = LogisticRegression(C=meta_c, max_iter=1000, random_state=SEED)
    meta.fit(stacked, y)

    meta_oof = log_loss(y, np.clip(meta.predict_proba(stacked)[:, 1], 0.001, 0.999))
    student_oof = np.mean([log_loss(y, np.clip(p, 0.001, 0.999)) for p in per_seed_oofs])
    return meta_oof, student_oof, per_seed_oofs, sel_cols_test, cfg_name


def main():
    global t_start
    t_start = time.time()

    log.info("=" * 70)
    log.info("V399 — Per-Target Feature Count Sweep + Per-Target Meta C")
    log.info("Hypothesis: Optimal n_feat differs when using per-target C")
    log.info("V392: fixed n_feat → meta 0.617, student 0.692")
    log.info("V399: sweep n_feat per target → find true optimal")
    log.info("=" * 70)

    train_df = pd.read_parquet(DATA / "features.parquet")
    test_df = pd.read_parquet(DATA / "test_features.parquet")

    for df in [train_df, test_df]:
        for c in ['sleep_date', 'lifelog_date', 'date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')

    train_feat_cols = get_feature_cols(train_df)
    log.info(f"Train: {len(train_feat_cols)} features")

    group = train_df['subject_id'].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    n_train = len(train_df)
    n_test = len(test_df)

    best_config = {}
    for t in TARGETS:
        feat_cols_clean = remove_leak(train_feat_cols, t)
        y = train_df[t].values.astype(np.float64)

        # Determine cfg based on target type
        if t.startswith('Q'):
            cfg_name = 'deep' if t in ['Q1','Q2'] else 'v48'
        else:
            cfg_name = 'wide' if t in ['S1','S4'] else 'deep' if t == 'S2' else 'safety'

        n_feats_to_try = N_FEAT_SWEEP[t]
        meta_c = META_C_PER_TARGET[t]

        log.info(f"\n{'='*60}")
        log.info(f"Target: {t}, cfg: {cfg_name}, meta C: {meta_c}")

        best_oof = float('inf')
        best_n_feat = None
        best_student = None
        results = []

        for nf in n_feats_to_try:
            meta_oof, student_oof, _, _, _ = run_target(
                t, train_df, test_df, feat_cols_clean, nf, cfg_name, meta_c,
                gkf, group, n_train, n_test
            )
            results.append((nf, meta_oof, student_oof))
            log.info(f"  n_feat={nf:3d}: meta={meta_oof:.5f}, student={student_oof:.5f}")

            # Pick best meta OOF that doesn't exceed V308 student by more than 0.02
            if meta_oof < best_oof and student_oof <= 0.72:
                best_oof = meta_oof
                best_n_feat = nf
                best_student = student_oof

        if best_n_feat is None:
            log.warning(f"  → No config met student<=0.72 constraint, picking n_feat with lowest meta OOF")
            # Pick n_feat with lowest meta OOF regardless
            ranked_results = sorted(results, key=lambda x: x[1])
            best_n_feat = ranked_results[0][0]
            best_oof = ranked_results[0][1]
            best_student = ranked_results[0][2]
            log.warning(f"  → Best fallback: n_feat={best_n_feat}, meta={best_oof:.5f}, student={best_student:.5f}")
        else:
            log.info(f"  → Best: n_feat={best_n_feat}, meta={best_oof:.5f}, student={best_student:.5f}")
        best_config[t] = {
            'n_feat': best_n_feat,
            'meta_oof': best_oof,
            'student_oof': best_student,
            'cfg': cfg_name,
            'C': meta_c,
        }

    # Now run full pipeline with best configs
    log.info(f"\n{'='*70}")
    log.info("Running full pipeline with best configs...")
    log.info(f"{'='*70}")

    per_target_oof = {}
    per_target_student = {}
    all_meta_oofs = {}
    all_student_oofs = {}

    for t in TARGETS:
        bc = best_config[t]
        feat_cols_clean = remove_leak(train_feat_cols, t)
        nf, cfg_name, meta_c = bc['n_feat'], bc['cfg'], bc['C']

        meta_oof, student_oof, _, _, _ = run_target(
            t, train_df, test_df, feat_cols_clean, nf, cfg_name, meta_c,
            gkf, group, n_train, n_test
        )
        per_target_oof[t] = meta_oof
        per_target_student[t] = student_oof
        log.info(f"{t}: meta={meta_oof:.5f}, student={student_oof:.5f}, n_feat={nf}, C={meta_c}")

    avg_oof = np.mean(list(per_target_oof.values()))
    avg_student = np.mean(list(per_target_student.values()))

    log.info(f"\n{'='*70}")
    log.info(f"V399 RESULTS (Per-Target Feature Count Sweep)")
    log.info(f"{'='*70}")
    for t in TARGETS:
        nf = best_config[t]['n_feat']
        log.info(f"  {t}: meta={per_target_oof[t]:.5f}, student={per_target_student[t]:.5f}, n_feat={nf}, C={best_config[t]['C']}")
    log.info(f"  AVG Meta OOF: {avg_oof:.5f}")
    log.info(f"  AVG Student OOF: {avg_student:.5f}")
    log.info(f"  V308 AVG OOF: 0.62235")
    log.info(f"  V308 AVG Student: 0.69212")
    log.info(f"  Δ Meta vs V308: {avg_oof - 0.62235:+.5f}")
    log.info(f"  Δ Student vs V308: {avg_student - 0.69212:+.5f}")

    v308_gap = 0.63893 - 0.62235
    predicted_lb = avg_oof + v308_gap
    log.info(f"  V308 OOF-LB gap: {v308_gap:.5f}")
    log.info(f"  Predicted LB: {predicted_lb:.5f}")
    log.info(f"  Beats V308? {predicted_lb < 0.63893}")

    gap = avg_student - avg_oof
    v308_student_gap = 0.69212 - 0.62235
    log.info(f"  Student-Meta Gap: {gap:.5f} (V308: {v308_student_gap:.5f})")
    log.info(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta_data = {
        'version': 'V399',
        'name': 'Per-Target Feature Count Sweep + Per-Target Meta C',
        'avg_meta_oof': round(float(avg_oof), 5),
        'avg_student_oof': round(float(avg_student), 5),
        'v308_avg_oof': 0.62235,
        'v308_avg_student': 0.69212,
        'v308_lb': 0.63893,
        'delta_vs_v308_meta': round(float(avg_oof - 0.62235), 5),
        'delta_vs_v308_student': round(float(avg_student - 0.69212), 5),
        'predicted_lb': round(float(predicted_lb), 5),
        'student_meta_gap': round(float(gap), 5),
        'v308_gap': round(float(v308_gap), 5),
        'n_seeds': N_SEEDS,
        'best_config': {t: {k: v for k, v in best_config[t].items()} for t in TARGETS},
        'per_target_meta_oof': {t: round(float(per_target_oof[t]), 5) for t in TARGETS},
        'per_target_student_oof': {t: round(float(per_target_student[t]), 5) for t in TARGETS},
        'timestamp': ts,
        'total_time_s': round(time.time() - t_start, 0),
    }

    meta_path = EXPERIMENTS / f'v399_{ts}.json'
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    log.info(f"Saved meta: {meta_path}")

    log.info(f"\nTotal time: {time.time()-t_start:.0f}s")
    return avg_oof, meta_data


if __name__ == '__main__':
    main()
