"""The blended per-target model: recency (P) + regularized LightGBM (G).

Final prediction per target:  shrink( w·P + (1-w)·G  toward 0.5 )

The blend weight ``w`` and shrink ``s`` are chosen by an anti-overfit rule: a
blend is adopted only if it beats pure recency on BOTH a 3-block and a 5-block
forward temporal CV (see validation.py). Otherwise the target falls back to
pure recency. This is what kept the minimal recipe from overfitting.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from . import config as C


def make_lgbm() -> "lgb.LGBMClassifier":
    return lgb.LGBMClassifier(**C.LGBM_PARAMS)


def shrink(p: np.ndarray, s: float) -> np.ndarray:
    """Pull probabilities toward 0.5 by factor ``s`` (s=1 is a no-op)."""
    return np.clip(0.5 + s * (p - 0.5), C.CLIP_EPS, 1 - C.CLIP_EPS)


def blend(p_recency: np.ndarray, p_gbm: np.ndarray, w: float, s: float) -> np.ndarray:
    """Weighted recency/GBM blend followed by shrink-toward-0.5."""
    return shrink(w * p_recency + (1 - w) * p_gbm, s)
