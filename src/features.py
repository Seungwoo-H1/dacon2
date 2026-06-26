"""Feature engineering.

Three feature groups feed the regularized LightGBM ("G") component:
  1. seasonal     — cyclical encodings of week/day-of-year/month/weekday.
  2. sensor aggs  — daytime (06-22h) mean/std/count of 6 phone/watch sensors.
  3. subject_rate — fold-safe shrunk per-subject base rate (the personalization prior).

Sleep / TST features are loaded separately (see sleep.py) and added only to the
targets where they validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def seasonal_features(df: pd.DataFrame) -> np.ndarray:
    """Cyclical calendar encodings — give the model a smooth notion of 'when'."""
    d = df["lifelog_date"]
    woy = d.dt.isocalendar().week.astype(float).values
    doy = d.dt.dayofyear.values
    mo = d.dt.month.values
    dw = d.dt.dayofweek.values
    return np.column_stack([
        np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52),
        np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
        np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12),
        np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
        (dw >= 5).astype(float),
    ])


def _daytime_agg(stem: str, column: str, prefix: str) -> pd.DataFrame:
    """mean/std/count of one sensor inside the daytime window, per (subject, date)."""
    from .data import load_sensor
    df = load_sensor(stem, column)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df = df[(df["hour"] >= C.DAY_START_HOUR) & (df["hour"] < C.DAY_END_HOUR)]
    out = df.groupby(["subject_id", "date"])[column].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["subject_id", "date"] + [f"{prefix}_{s}" for s in ("mean", "std", "count")]
    return out


def sensor_features(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join all configured daytime sensor aggregates onto the given (subject, day) keys."""
    k = keys[["subject_id", "lifelog_date"]].copy()
    k["date"] = k["lifelog_date"].dt.date
    for stem, column, prefix in C.SENSOR_AGGS:
        k = k.merge(_daytime_agg(stem, column, prefix), on=["subject_id", "date"], how="left")
    cols = [c for c in k.columns if c not in ("subject_id", "lifelog_date", "date")]
    return k[cols].astype(float), cols


def subject_rate(y: np.ndarray, subj: np.ndarray, train_idx: np.ndarray,
                 query_subj: np.ndarray, alpha: float = C.RECENCY_ALPHA) -> np.ndarray:
    """Shrunk per-subject base rate computed ONLY from train rows (leakage-safe).

    rate(s) = (Σ y_train[s] + alpha·global_mean) / (n_train[s] + alpha)
    """
    gm = y[train_idx].mean()
    agg = (pd.DataFrame({"s": subj[train_idx], "y": y[train_idx]})
           .groupby("s")["y"].agg(["sum", "count"]))
    rate = ((agg["sum"] + alpha * gm) / (agg["count"] + alpha)).to_dict()
    return np.array([rate.get(s, gm) for s in query_subj])
