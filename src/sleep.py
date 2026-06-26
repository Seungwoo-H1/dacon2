"""Sleep / TST feature access.

The minute-grid sleep-window detector lives in ``scripts/build_sleep_features.py``
(kept as-is from the research phase) and writes ``data_processed/sleep_v3.parquet``
with TST, sleep efficiency, WASO, arousals, etc. plus per-subject z-scores.

These objective sleep features are added surgically — only to Q1 and S1, the two
targets where they improved temporal cross-validation by a robust margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_sleep_features(keys: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Median-imputed sleep feature matrix aligned to the given (subject, day) keys.

    Raises a clear error if the parquet is missing so the user knows to run the
    build script first.
    """
    if not C.SLEEP_FEATURES.exists():
        raise FileNotFoundError(
            f"{C.SLEEP_FEATURES} not found. Run: python scripts/build_sleep_features.py"
        )
    sv = pd.read_parquet(C.SLEEP_FEATURES)
    sv["lifelog_date"] = pd.to_datetime(sv["lifelog_date"])
    sv["sleep_date"] = pd.to_datetime(sv["sleep_date"])
    merged = keys[["subject_id", "sleep_date", "lifelog_date"]].merge(
        sv, on=["subject_id", "sleep_date", "lifelog_date"], how="left")
    feats = merged[columns].apply(pd.to_numeric, errors="coerce")
    return np.nan_to_num(feats.fillna(feats.median()).values)
