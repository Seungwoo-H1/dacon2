"""Data-free unit tests for the core model logic.

Unlike scripts/test_pipeline.py (which requires the private competition data),
this runs anywhere: it checks the math of recency personalization, the
blend/shrink step, forward-fold construction, and the fold-safe subject rate
on tiny synthetic inputs.

Run:  python scripts/test_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.features import seasonal_features, subject_rate
from src.model import blend, shrink
from src.recency import recency_predict
from src.validation import BLOCKS_3, BLOCKS_5, forward_folds, log_loss_binary


def days(*offsets: int) -> np.ndarray:
    return np.array([np.datetime64("2024-06-01") + np.timedelta64(o, "D") for o in offsets])


def test_recency() -> None:
    y = np.array([1.0, 0.0])
    subj = np.array(["a", "b"])
    src_dates = days(0, 0)

    # Unseen subject falls back to the global mean.
    p = recency_predict(y, subj, src_dates, np.array(["c"]), days(0))
    assert np.isclose(p[0], 0.5), p

    # Hand-computed: subject 'a' neighbor y=1 at gap 7, halflife 7 -> w=0.5;
    # global mean 0.5, alpha 1 -> (0.5*1 + 1*0.5) / (0.5 + 1) = 2/3.
    p = recency_predict(y, subj, src_dates, np.array(["a"]), days(7), halflife=7.0, alpha=1.0)
    assert np.isclose(p[0], 2 / 3), p

    # Huge halflife + alpha=0 collapses to the subject's own mean.
    y2 = np.array([1.0, 1.0, 0.0, 0.0])
    subj2 = np.array(["a"] * 4)
    p = recency_predict(y2, subj2, days(0, 10, 20, 30), np.array(["a"]), days(100),
                        halflife=1e9, alpha=0.0)
    assert np.isclose(p[0], 0.5), p


def test_blend_shrink() -> None:
    p = np.array([0.0, 0.25, 0.5, 1.0])
    # s=1 is a no-op except for the safety clip at the extremes.
    out = shrink(p, 1.0)
    assert np.allclose(out[1:3], [0.25, 0.5])
    assert out[0] == C.CLIP_EPS and out[3] == 1 - C.CLIP_EPS
    # s=0.5 halves the distance to 0.5.
    assert np.allclose(shrink(np.array([0.1, 0.9]), 0.5), [0.3, 0.7])
    # w=1 ignores the GBM leg entirely.
    g = np.array([0.99, 0.99])
    assert np.allclose(blend(np.array([0.3, 0.6]), g, 1.0, 1.0), [0.3, 0.6])


def test_forward_folds() -> None:
    subj = np.array(["a"] * 20 + ["b"] * 20)  # sorted by (subject, day)
    for spec in (BLOCKS_3, BLOCKS_5):
        folds = forward_folds(subj, spec)
        assert len(folds) == len(spec)
        covered = []
        for train, val in folds:
            assert len(np.intersect1d(train, val)) == 0
            # Within each subject, every train row precedes every val row.
            for sid in ("a", "b"):
                tr = train[subj[train] == sid]
                va = val[subj[val] == sid]
                assert len(va) > 0 and tr.max() < va.min()
            covered += val.tolist()
        assert max(covered) == 39, "last block must reach each subject's final day"


def test_subject_rate() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    subj = np.array(["a", "a", "b", "b"])
    train_idx = np.array([0, 1])  # only subject 'a' visible
    # Unseen-in-train subject gets the train global mean (leakage-safe).
    r = subject_rate(y, subj, train_idx, np.array(["b"]), alpha=1.0)
    assert np.isclose(r[0], 1.0), r
    # Seen subject: (sum + alpha*gm) / (n + alpha) = (2 + 1*1) / (2 + 1) = 1.
    r = subject_rate(y, subj, train_idx, np.array(["a"]), alpha=1.0)
    assert np.isclose(r[0], 1.0), r
    # Mixed train: gm=0.5, subject 'b' rate = (0 + 0.5) / (2 + 1) = 1/6.
    r = subject_rate(y, subj, np.arange(4), np.array(["b"]), alpha=1.0)
    assert np.isclose(r[0], 1 / 6), r


def test_seasonal_and_logloss() -> None:
    df = pd.DataFrame({"lifelog_date": pd.to_datetime(["2024-06-01", "2024-06-03"])})
    f = seasonal_features(df)
    assert f.shape == (2, 9)
    assert f[0, -1] == 1.0 and f[1, -1] == 0.0  # Sat is weekend, Mon is not

    y = np.array([0, 1, 1, 0])
    assert log_loss_binary(y, y.astype(float)) < 1e-9  # perfect + clip-safe
    assert np.isfinite(log_loss_binary(y, np.array([1.0, 0.0, 1.0, 0.0])))


def main() -> None:
    test_recency()
    test_blend_shrink()
    test_forward_folds()
    test_subject_rate()
    test_seasonal_and_logloss()
    print("OK: all unit tests passed")


if __name__ == "__main__":
    main()
