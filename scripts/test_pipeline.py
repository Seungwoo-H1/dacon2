"""Smoke test: the reproduced submission must be valid and match the shipped best.

Run:  python scripts/test_pipeline.py
Fails loudly if the pipeline drifts from the recorded best submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src.pipeline import run


def main() -> None:
    rep = run(verbose=False)
    best = pd.read_csv(C.SUBMISSIONS / "submission_best.csv")

    # 1. Shape + keys + no NaNs + valid probabilities.
    assert len(rep) == len(best) == 250, "expected 250 test rows"
    keys = ["subject_id", "sleep_date", "lifelog_date"]
    assert (rep[keys].astype(str).values == best[keys].astype(str).values).all(), "key mismatch"
    assert rep[C.TARGETS].isna().sum().sum() == 0, "NaNs in predictions"
    assert rep[C.TARGETS].values.min() >= 0 and rep[C.TARGETS].values.max() <= 1, "probs out of range"

    # 2. Must track the recorded best closely (Q targets are exact; S1 is near).
    for t in C.TARGETS:
        corr = np.corrcoef(rep[t], best[t])[0, 1]
        assert corr > 0.9, f"{t}: correlation {corr:.3f} drifted from best"

    print("OK: reproduced submission is valid and matches submission_best.csv")


if __name__ == "__main__":
    main()
