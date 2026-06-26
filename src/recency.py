"""Temporal recency personalization — the core transferable signal.

For each query (subject, day) we take a time-decayed average of that subject's
known labels, shrunk toward the subject's global mean:

    pred = (Σ_n 0.5^(|gap_n| / halflife) · y_n  +  alpha · global_mean)
           ─────────────────────────────────────────────────────────────
           (Σ_n 0.5^(|gap_n| / halflife)        +  alpha)

A long halflife collapses to the (shrunk) subject mean — a stable baseline. A
short halflife exploits day-to-day autocorrelation, which is what lifts the Q
targets on the time-interleaved test days (the R53 finding).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def recency_predict(y: np.ndarray, src_subj: np.ndarray, src_dates: np.ndarray,
                    q_subj: np.ndarray, q_dates: np.ndarray,
                    halflife: float = C.RECENCY_HALFLIFE,
                    alpha: float = C.RECENCY_ALPHA) -> np.ndarray:
    """Predict each query row from same-subject source labels (dates as datetime64[D])."""
    global_mean = y.mean()
    out = np.zeros(len(q_subj))
    for j in range(len(q_subj)):
        mask = src_subj == q_subj[j]
        if not mask.any():
            out[j] = global_mean
            continue
        gap = np.abs((q_dates[j] - src_dates[mask]).astype("timedelta64[D]").astype(float))
        w = 0.5 ** (gap / halflife)
        out[j] = (np.sum(w * y[mask]) + alpha * global_mean) / (np.sum(w) + alpha)
    return out
