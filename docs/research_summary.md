# Research Summary

~140 experiments across feature engineering, validation design, modeling, ensembling,
pseudo-labeling, calibration, and external data. This document records the meaningful
ones, why they were tried, what happened, and the final ranking of ideas.

The competition metric is **average binary log-loss** (lower better). All "LB" numbers
are real leaderboard scores; "CV" numbers are internal estimates.

---

## Verified leaderboard anchors (ground truth)

| Submission | LB | Description |
|---|---|---|
| R8 (`exp_R8_best_blend.py`) | 0.603 | recency + regularized LightGBM, anti-overfit per-target blend |
| R53 (`exp_R53_synthesis.py`) | 0.59915 | R8 + short-halflife recency on Q + TST on S1 |
| **R86 (`submission_best.csv`)** | **0.5987807602** | robust mean of R53-equivalent seeds (the shipped best) |
| R96 / R97 | > 0.5988 | R86 + TabPFN-on-Q + weather blend — **worse**, rejected |

**Headline:** complexity beyond R53/R86 consistently *raised* log-loss. The data's
honest ceiling from the released sensors is ≈ 0.59.

---

## What worked (ranked)

1. **Temporal recency personalization** — time-decayed average of a subject's own labels.
   The dominant transferable signal. *Why:* targets have within-subject day-to-day
   autocorrelation (lag-1 ≈ 0.21–0.30).
2. **Short-halflife recency on Q1/Q2/Q3** (R53). *Why:* the deviation targets are ≈ coin-flip
   per subject, but consecutive days correlate; a halflife of 1–2 days exploits the
   time-interleaved test rows. Random-CV had hidden this by scrambling adjacency.
3. **Anti-overfit blend gate** — adopt a recency↔GBM blend only if it beats pure recency on
   *both* a 3-block and 5-block forward CV. Kept the model from chasing noise.
4. **Surgical TST feature on S1** — phone-derived total-sleep-time partly recovers the sleep
   duration target. Added to S1 only; S2–S4 got zero gain.
5. **Robust-mean consolidation (R86)** — averaging a few equivalent seeds + safe clipping;
   marginal but stabilizing.

## What failed (ranked by how convincing the false signal was)

1. **TabPFN-on-Q (R94/R95)** — passed *nested* cross-subject CV (−0.014 to −0.017 on Q) but
   lost on the real LB. The strongest false positive of the project.
2. **Weather / calendar features (R96)** — Q3 day-of-week + moon-phase correlations validated
   cross-subject (−0.045 on Q3!) yet lost on LB. *Root cause:* date-level features must be
   validated by *time*, not by subject — all 10 subjects share the same calendar, so
   subject-nesting never controlled the within-period confound.
3. **Stacking / winner-stacks / hill-climb ensembles** — large random-CV gains, all refuted on
   forward-holdout. Random-CV is +0.02 optimistic on engineered features.
4. **Extra model classes** (CatBoost, XGBoost, HistGBM) — ≈ LightGBM; no diversity benefit.
5. **Deep models** (FT-Transformer, RTDL, 1D-CNN/TraM-style) — near base-rate; the limit is
   signal, not capacity.
6. **Pseudo-labeling (R35)** — hurt; the unlabeled extension half has a different label
   relationship to features.
7. **Calibration / isotonic / temperature scaling** — no gain; the recency blend is already
   well-calibrated (shrinking toward 0.5 or subject-rate only hurt).
8. **Per-target alpha / halflife micro-tuning** — overfit the validator; default config was best.
9. **Balancing / anti-balance constraint (R73)** — refuted; within-subject rates persist.

---

## Validation findings

- **Random K-fold:** +0.02 optimistic (scrambles temporal adjacency the model exploits).
- **LOO (leave-one-day-out):** interleaved-like; tracks LB ± 0.005 for *minimal* models but is
  optimistic for engineered features.
- **Forward block-holdout:** transfer-faithful (mirrors the extension half). The validator that
  actually ordered models correctly.
- **Subject-nested CV:** correct for *subject-varying* features, **invalid for date-level
  features** (the weather trap above).
- **Bottom line:** pick the validator whose split geometry matches the test set's, then trust
  the LB anchors over any internal number.

## Why the ceiling is ~0.59

- **Q1–Q3** are deviations from each subject's own mean → per-subject base rate ≈ 0.5,
  uninformative; only autocorrelation helps, and only on near-neighbor test days.
- **S2–S4** (efficiency / latency / WASO) were measured by a Withings under-mattress sensor
  that is **absent from the released phone/watch sensors** — physically not recoverable.
- The test set is ~half future-extension with no nearby labeled day, where neither recency nor
  day-level features generalize.

## Final ranking of ideas (for the next competition)

1. Temporal personalization + a validator that matches the test's time structure. *(do first)*
2. Domain-adaptation framing for the extension split.
3. Objective sleep features from same-device external data as a prior (not a join).
4. Everything else (stacking, deep nets, exotic features) — only after 1–3, and only if it
   survives forward-time CV **and** a held-out probe.
