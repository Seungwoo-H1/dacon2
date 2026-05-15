"""
16_v11_tiny.py — V11: Small candidate set + fast tuning

Approach:
1. Rank ALL 4860 features per target using a fast LGBM model
2. Take top 50 ranked features  
3. For tuning, only use top 10, 15, 20, 25, 30 of those
4. 10 seeds × 3 configs × 5 fold sets = 150 trainings per target
5. Calibrate + compare vs V10
"""

import sys
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import TARGETS, DATA_PROCESSED, MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_COLS = TARGETS
META = {"subject_id", "lifelog_date", "sleep_date", "date"}

V10_SCORES = {
    'Q1': 0.6338, 'Q2': 0.6034, 'Q3': 0.6119,
    'S1': 0.5680, 'S2': 0.6022, 'S3': 0.5835, 'S4': 0.6240,
}
V10_AVG = 0.6038

LEAKAGE_S = {
    'wLight_w_light_mean', 'wLight_w_light_std', 'wLight_w_light_min',
    'wLight_w_light_max', 'wLight_w_light_count',
    'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max',
    'wHr_hr_median', 'wHr_hr_count',
    'wPedo_pedo_step_mean', 'wPedo_pedo_step_sum',
    'wPedo_pedo_step_frequency_mean', 'wPedo_pedo_step_frequency_sum',
    'wPedo_pedo_running_step_mean', 'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean', 'wPedo_pedo_walking_step_sum',
    'wPedo_pedo_distance_mean', 'wPedo_pedo_distance_sum',
    'wPedo_pedo_speed_mean', 'wPedo_pedo_speed_sum',
    'wPedo_pedo_burned_calories_mean', 'wPedo_pedo_burned_calories_sum',
}
LEAKAGE_Q = {'wHr_hr_mean', 'wHr_hr_std', 'wHr_hr_min', 'wHr_hr_max', 'wHr_hr_median', 'wHr_hr_count'}

LGB_CFGS = [
    dict(nl=10, md=3, lr=0.02, ne=200, ss=0.7, cst=0.7, ra=1.0, rl=3.0, mc=10),
    dict(nl=6, md=2, lr=0.02, ne=150, ss=0.5, cst=0.5, ra=5.0, rl=10.0, mc=20),
    dict(nl=15, md=4, lr=0.03, ne=300, ss=0.7, cst=0.7, ra=1.0, rl=3.0, mc=10),
]

SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3037, 4096, 5001]


def remove_leakage(cols, t):
    set_l = LEAKAGE_S if t.startswith('S') else LEAKAGE_Q
    return [c for c in cols if c not in set_l]


def get_base(df):
    return [c for c in df.columns
            if c not in META | set(TARGET_COLS)
            and not c.endswith('_zscore')
            and df[c].dtype in [np.float64, np.int64, float, int, bool]]


def main():
    log.info("=" * 70)
    log.info("V11 Tiny")
    log.info("=" * 70)

    df = pd.read_parquet(DATA_PROCESSED / "features_v11.parquet")
    sid = df['subject_id'].values.values if hasattr(df['subject_id'].values, 'values') else df['subject_id'].values
    base = get_base(df)
    X = df[base].fillna(0).values

    log.info(f"Features: {len(base)}, Shape: {X.shape}, Subject IDs: {len(np.unique(sid))}")

    # Build col→index map
    col2idx = {c: i for i, c in enumerate(base)}

    # Per-target ranking
    all_rankings = {}
    all_Y = {}
    all_gkf = {}
    all_spw = {}

    for t in TARGET_COLS:
        leak_free = remove_leakage(base, t)
        leak_idx = [col2idx[c] for c in leak_free]
        y = df[t].values
        X_leak = X[:, leak_idx]

        n_pos = max((y == 1).sum(), 1)
        n_neg = (y == 0).sum()
        spw = n_neg / n_pos

        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
            'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.03,
            'n_estimators': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 3.0,
            'scale_pos_weight': spw, 'random_state': 42,
            'min_child_samples': 10, 'force_row_wise': True, 'n_jobs': -1,
        }
        ds = lgb.Dataset(X_leak, label=y, params={'verbose': '-1'})
        mdl = lgb.train(params, ds, num_boost_round=30)
        imp = mdl.feature_importance(importance_type="gain")
        ranked = sorted(zip(leak_free, imp), key=lambda x: -x[1])

        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X_leak, y, sid))

        all_rankings[t] = ranked
        all_Y[t] = y
        all_gkf[t] = splits
        all_spw[t] = spw

        log.info(f"{t}: {len(leak_free)} leak-free, top5={[r[0] for r in ranked[:5]]}")

    # Tuning
    results = {}
    for t in TARGET_COLS:
        y = all_Y[t]
        spw = all_spw[t]
        splits = all_gkf[t]
        rate = y.mean()
        ranked = all_rankings[t]

        log.info(f"\n  {t}:")

        # Try top-50 features, then subsets
        candidates = {}
        for n in [5, 10, 15, 20, 25, 30, 40, 50]:
            sel = [r[0] for r in ranked[:n] if r[1] > 0]
            sel_idx = [col2idx[c] for c in sel]
            candidates[n] = X[:, sel_idx]

        best_cv = float('inf')
        best = None

        for n, Xs in candidates.items():
            for ci, cfg in enumerate(LGB_CFGS):
                oof = run_cv(Xs, y, splits, SEEDS, cfg, spw)
                oof_avg = oof.mean(1)
                cv = log_loss(y, oof_avg, labels=[0, 1])

                if cv < best_cv:
                    best_cv = cv
                    best = dict(n=n, cfg=cfg, cv=cv, oof=oof, sel=[r[0] for r in ranked[:n] if r[1] > 0])
                    log.info(f"    n={n} c={ci}: {cv:.4f}")

        if best is None:
            continue

        # Calibrate
        cal = np.clip(best['oof'].mean(1) + (rate - best['oof'].mean(1).mean()), 0.0001, 0.9999)
        cal_loss = log_loss(y, cal, labels=[0, 1])
        v10 = V10_SCORES[t]
        d = cal_loss - v10
        marker = "✅" if d < 0 else "❌"

        log.info(f"    BEST: n={best['n']} V10={v10:.4f} V11={cal_loss:.4f} {d:+.4f} {marker}")

        results[t] = dict(cal_oof=cal_loss, v10=v10, delta=d, n_features=best['n'])

    # Summary
    log.info(f"\n{'='*70}")
    log.info("V11 FINAL")
    log.info(f"{'Target':<6} {'V10':<10} {'V11':<10} {'Δ':<8} {'Win'}")
    log.info("-" * 70)

    avg11 = 0; cnt = 0
    for t in TARGET_COLS:
        if t not in results:
            continue
        v10 = V10_SCORES[t]
        v11 = results[t]['cal_oof']
        d = v11 - v10
        w = "V11" if d < 0 else "V10"
        log.info(f"{t:<6} {v10:<10.4f} {v11:<10.4f} {d:+.4f} {w}")
        avg11 += v11; cnt += 1

    avg11 /= cnt if cnt else 1
    d_avg = avg11 - V10_AVG
    w = "V11" if d_avg < 0 else "V10"
    log.info("-" * 70)
    log.info(f"{'AVG':<6} {V10_AVG:<10.4f} {avg11:<10.4f} {d_avg:+.4f} {w}")
    log.info(f"{'🎉 V11!' if d_avg < 0 else 'V10 wins'}")

    # Save
    for name, data in [("final", results)]:
        p = MODEL_DIR / f'v11_{name}_results.json'
        with open(p, 'w') as f:
            json.dump(data, f, indent=2, default=float)
        log.info(f"Saved: {p}")

    meta = dict(version='v11_tiny', avg_v10=float(V10_AVG), avg_v11=float(avg11), beat_v10=bool(d_avg < 0),
                per_target={t: {k: v for k, v in results[t].items()} for t in results})
    with open(MODEL_DIR / 'v11_tiny_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    log.info("Done!")


def run_cv(X, y, splits, seeds, cfg, spw):
    n = len(y)
    oof = np.zeros((n, len(seeds)))
    for si, seed in enumerate(seeds):
        prms = dict(
            objective='binary', metric='binary_logloss', verbose=-1,
            num_leaves=cfg['nl'], max_depth=cfg['md'], learning_rate=cfg['lr'],
            n_estimators=cfg['ne'], subsample=cfg['ss'], colsample_bytree=cfg['cst'],
            reg_alpha=cfg['ra'], reg_lambda=cfg['rl'],
            min_child_samples=cfg['mc'], force_row_wise=True, n_jobs=-1,
            scale_pos_weight=spw, random_state=seed,
        )
        for fi, (tr_i, va_i) in enumerate(splits):
            tr = lgb.Dataset(X[tr_i], label=y[tr_i], params={'verbose': '-1'})
            va = lgb.Dataset(X[va_i], label=y[va_i], feature_name=tr.feature_name, reference=tr, params={'verbose': '-1'})
            mdl = lgb.train(prms, tr, num_boost_round=cfg['ne'], valid_sets=[va],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            oof[va_i, si] = mdl.predict(X[va_i])
    return oof


if __name__ == "__main__":
    main()
