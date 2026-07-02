"""End-to-end pipeline that reproduces the best submission.

Recipe (the verified honest best, avg log-loss ≈ 0.599):
  1.  G features = [subject_rate, seasonal, daytime sensor aggs]  (+ TST for S1).
  2.  Per target, blend recency P with LightGBM G; adopt the blend only if it
      beats pure recency on BOTH 3-block and 5-block forward CV (anti-overfit).
  3.  Refit on all train, predict the 250 test rows.
  4.  Override Q1/Q2/Q3 with SHORT-halflife recency (the R53 temporal finding).
  5.  Write the submission.

This is the R8 → R50 (TST on S1) → R53 (Q recency) lineage as one clean pass.
The shipped ``submission_best.csv`` additionally consolidates a few equivalent
seeds via a clipped robust mean (R86); that step is documented in the README and
changes the score only at the 4th decimal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# LightGBM 4.x sets feature_names_in_ = ['Column_0', ...] even when fitted on a
# plain ndarray, so every ndarray predict trips sklearn's name-mismatch warning.
# False positive — silence just that message.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from . import config as C
from . import data, features, model, sleep
from .recency import recency_predict
from .validation import BLOCKS_3, BLOCKS_5, forward_folds, log_loss_binary


def _gbm_matrix(y, subj, train_idx, query_idx, seasonal, sensors, sleep_feats):
    """Assemble the LightGBM design matrix for given rows (fold-safe subject_rate)."""
    rate = features.subject_rate(y, subj, train_idx, subj[query_idx])
    parts = [rate[:, None], seasonal[query_idx], sensors[query_idx]]
    if sleep_feats is not None:
        parts.append(sleep_feats[query_idx])
    return np.column_stack(parts)


def _oof_p_and_g(y, subj, dates, seasonal, sensors, sleep_feats, folds):
    """Out-of-fold recency (P) and LightGBM (G) predictions over one block scheme."""
    P = np.zeros(len(y))
    G = np.zeros(len(y))
    for train_idx, val_idx in folds:
        P[val_idx] = recency_predict(y[train_idx], subj[train_idx], dates[train_idx],
                                     subj[val_idx], dates[val_idx])
        x_tr = _gbm_matrix(y, subj, train_idx, train_idx, seasonal, sensors, sleep_feats)
        x_va = _gbm_matrix(y, subj, train_idx, val_idx, seasonal, sensors, sleep_feats)
        m = model.make_lgbm().fit(x_tr, y[train_idx])
        G[val_idx] = m.predict_proba(x_va)[:, 1]
    return P, G


def _cv_score(y, P, G, w, s, folds):
    yt, pt = [], []
    for _, val_idx in folds:
        pt.append(model.blend(P[val_idx], G[val_idx], w, s))
        yt.append(y[val_idx])
    return log_loss_binary(np.concatenate(yt), np.concatenate(pt))


def _choose_blend(y, P3, G3, P5, G5, folds3, folds5):
    """Pick (w, s) only if it beats pure recency on BOTH block schemes."""
    base3 = _cv_score(y, P3, G3, 1.0, 1.0, folds3)
    base5 = _cv_score(y, P5, G5, 1.0, 1.0, folds5)
    best_w, best_s, best = 1.0, 1.0, base3
    for w in C.BLEND_WEIGHTS:
        for s in C.SHRINK_FACTORS:
            l3 = _cv_score(y, P3, G3, w, s, folds3)
            l5 = _cv_score(y, P5, G5, w, s, folds5)
            if l3 <= base3 + 1e-9 and l5 <= base5 + 1e-9 and l3 < best:
                best_w, best_s, best = w, s, l3
    return best_w, best_s, base3


def run(out_path=None, verbose=True):
    """Build the submission and return it as a DataFrame."""
    lab = data.load_labels()
    sub = data.load_submission_template()
    subj = lab["subject_id"].values
    dates = lab["lifelog_date"].values.astype("datetime64[D]")
    q_subj = sub["subject_id"].values
    q_dates = sub["lifelog_date"].values.astype("datetime64[D]")

    seas_tr = features.seasonal_features(lab)
    seas_te = features.seasonal_features(sub)
    sens_tr, _ = features.sensor_features(lab)
    sens_te, _ = features.sensor_features(sub)
    med = pd.DataFrame(sens_tr).median()
    sens_tr = np.nan_to_num(pd.DataFrame(sens_tr).fillna(med).values)
    sens_te = np.nan_to_num(pd.DataFrame(sens_te).fillna(med).values)

    folds3 = forward_folds(subj, BLOCKS_3)
    folds5 = forward_folds(subj, BLOCKS_5)

    out = sub.copy()
    for t in C.TARGETS:
        y = lab[t].values
        sleep_tr = sleep_te = None
        if t in C.SLEEP_FEATURE_TARGETS:
            cols = C.SLEEP_FEATURE_TARGETS[t]
            sleep_tr = sleep.load_sleep_features(lab, cols)
            sleep_te = sleep.load_sleep_features(sub, cols)

        P3, G3 = _oof_p_and_g(y, subj, dates, seas_tr, sens_tr, sleep_tr, folds3)
        P5, G5 = _oof_p_and_g(y, subj, dates, seas_tr, sens_tr, sleep_tr, folds5)
        w, s, base = _choose_blend(y, P3, G3, P5, G5, folds3, folds5)

        # Refit on all train, predict test.
        p_te = recency_predict(y, subj, dates, q_subj, q_dates)
        x_tr = _gbm_matrix(y, subj, np.arange(len(y)), np.arange(len(y)), seas_tr, sens_tr, sleep_tr)
        x_te_rate = features.subject_rate(y, subj, np.arange(len(y)), q_subj)
        parts = [x_te_rate[:, None], seas_te, sens_te]
        if sleep_te is not None:
            parts.append(sleep_te)
        g_te = model.make_lgbm().fit(x_tr, y).predict_proba(np.column_stack(parts))[:, 1]
        out[t] = model.blend(p_te, g_te, w, s)
        if verbose:
            tag = "BLEND" if (w < 1.0 or s < 1.0) else "pureP"
            print(f"  {t}: w={w} shrink={s} cv={base:.4f} [{tag}]", flush=True)

    # R53: override Q targets with short-halflife temporal recency.
    for t, (hl, alpha) in C.Q_RECENCY_CFG.items():
        out[t] = np.clip(recency_predict(lab[t].values, subj, dates, q_subj, q_dates, hl, alpha),
                         C.CLIP_EPS, 1 - C.CLIP_EPS)
        if verbose:
            print(f"  {t}: overridden with recency(halflife={hl}, alpha={alpha})", flush=True)

    if out_path is not None:
        out.to_csv(out_path, index=False)
        if verbose:
            print(f"wrote {out_path}  nan={out[C.TARGETS].isna().sum().sum()}")
    return out
