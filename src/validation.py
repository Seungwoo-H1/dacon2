"""Forward temporal cross-validation.

Random K-fold is optimistically biased here (~0.02) because it scrambles the
day-to-day adjacency the model exploits, AND because the test set is partly a
future extension. The faithful validator trains on each subject's earliest days
and validates on later days — mirroring the extension half of the test.

Two block schemes are used together as an anti-overfit gate: a candidate blend
must beat the baseline on BOTH before it is adopted.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from . import config as C

# (lo, hi) fractions of each subject's chronological timeline used as validation;
# everything strictly before `lo` is the training window for that fold.
BLOCKS_3 = [(0.60, 0.74), (0.74, 0.87), (0.87, 1.001)]
BLOCKS_5 = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.001)]


def log_loss_binary(y: np.ndarray, p: np.ndarray) -> float:
    return log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])


def forward_folds(subj: np.ndarray, spec: list[tuple[float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build (train_idx, val_idx) folds from chronological per-subject blocks.

    Assumes rows are already sorted by (subject, day) — see data.load_labels.
    """
    folds = []
    for lo, hi in spec:
        train, val = [], []
        for sid in np.unique(subj):
            idx = np.where(subj == sid)[0]
            n = len(idx)
            a, b = int(n * lo), int(n * hi)
            val += idx[a:b].tolist()
            train += idx[:a].tolist()
        folds.append((np.array(sorted(train)), np.array(sorted(val))))
    return folds
