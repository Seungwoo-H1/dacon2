"""Central configuration for the ETRI Lifelog 2024 (DACON ch2026) pipeline.

All tunables live here so the rest of the code stays declarative. Values are the
ones that produced the best verified leaderboard submission (avg log-loss 0.5988).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed"
SUBMISSIONS = ROOT / "submissions"
SENSOR_DIR = DATA_RAW / "ch2025_data_items"

TRAIN_LABELS = DATA_RAW / "ch2026_metrics_train.csv"
SUBMISSION_TEMPLATE = DATA_RAW / "ch2026_submission_sample.csv"
SLEEP_FEATURES = DATA_PROCESSED / "sleep_v3.parquet"

# ---------------------------------------------------------------- task
# 7 binary targets: Q1-Q3 self-reported (fatigue / stress / sleep-quality
# deviation from the subject's own average); S1-S4 objective sleep-guideline
# adherence (measured by an under-mattress sensor in the original study).
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEED = 42

# Daytime window (hours) for phone/watch sensor aggregation. Night is excluded
# because the sleep signal is captured separately by the TST features.
DAY_START_HOUR = 6
DAY_END_HOUR = 22

# Sensors aggregated into daytime mean/std/count features. (parquet stem, column, prefix)
SENSOR_AGGS = [
    ("mScreenStatus", "m_screen_use", "screen"),
    ("mActivity", "m_activity", "act"),
    ("wHr", "heart_rate", "hr"),
    ("wPedo", "step", "step"),
    ("wLight", "w_light", "wl"),
    ("mACStatus", "m_charging", "chg"),
]

# ---------------------------------------------------------------- model
# Recency personalization: pred = (Σ 0.5^(gap/halflife)·y_neighbor + alpha·mean) / (Σw + alpha)
# Long halflife ≈ subject mean (stable baseline used by the LGBM blend and S targets).
RECENCY_HALFLIFE = 21.0
RECENCY_ALPHA = 3.0

# R53 improvement: Q targets use a SHORT halflife to exploit day-to-day
# autocorrelation on the time-interleaved test days. (halflife, alpha) per Q target.
Q_RECENCY_CFG = {"Q1": (1.0, 1.0), "Q2": (2.0, 1.0), "Q3": (1.0, 1.0)}

# Regularized LightGBM (the "G" component): deliberately small + heavily
# regularized because the day-level feature signal is weak and overfits easily.
LGBM_PARAMS = dict(
    n_estimators=250,
    num_leaves=8,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_samples=25,
    reg_lambda=5.0,
    reg_alpha=1.0,
    random_state=SEED,
    verbose=-1,
)

# Anti-overfit blend search grid: weight on recency P, and shrink toward 0.5.
BLEND_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
SHRINK_FACTORS = [1.0, 0.95, 0.9, 0.85, 0.8]

# Sleep (TST) features added surgically only to the targets where they validated.
SLEEP_FEATURE_TARGETS = {
    "Q1": ["tst", "tst_z", "se", "waso_z", "sleep_light"],
    "S1": ["tst", "tst_z", "waso", "n_arousal", "sol"],
}

CLIP_EPS = 1e-6
