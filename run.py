"""Reproduce the best submission.

Usage:
    python scripts/build_sleep_features.py        # one-time: writes data_processed/sleep_v3.parquet
    python run.py                                 # writes submissions/submission_reproduced.csv

The result is byte-for-byte deterministic (fixed seeds, no randomness beyond
LightGBM's seeded sampling) and matches the shipped submission_best.csv to the
4th decimal of the leaderboard metric.
"""
from __future__ import annotations

from src import config as C
from src.pipeline import run

if __name__ == "__main__":
    out_path = C.SUBMISSIONS / "submission_reproduced.csv"
    print("Reproducing best submission (avg log-loss target ~0.599)...")
    run(out_path=out_path, verbose=True)
    print("Done.")
