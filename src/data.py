"""Data loading: competition labels, submission template, and raw sensor streams."""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_labels() -> pd.DataFrame:
    """Train labels, sorted by (subject, day) so temporal logic is positional-safe."""
    df = pd.read_csv(C.TRAIN_LABELS, parse_dates=["lifelog_date", "sleep_date"])
    return df.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)


def load_submission_template() -> pd.DataFrame:
    """The 250 test (subject_id, sleep_date, lifelog_date) keys to predict."""
    return pd.read_csv(C.SUBMISSION_TEMPLATE, parse_dates=["lifelog_date", "sleep_date"])


def load_sensor(stem: str, column: str) -> pd.DataFrame:
    """Load one raw sensor parquet with just (subject_id, timestamp, <column>)."""
    df = pd.read_parquet(C.SENSOR_DIR / f"ch2025_{stem}.parquet",
                         columns=["subject_id", "timestamp", column])
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
