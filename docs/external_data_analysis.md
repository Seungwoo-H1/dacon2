# External Data Analysis — ETRI Lifelog 2018–2020 (historical)

> **Status: analysis only.** Per the task scope, these datasets are **not** used in the
> final solution. This document assesses their quality and how they *could* help a future
> competition.

The repo's `data_raw/` contains historical ETRI lifelog sleep data predating the 2024
competition cohort:

| File | Shape | Device era |
|---|---|---|
| `user_info_2019_2018_updated.csv` | 50 × 7 | demographics (gender/age/height/weight, study dates) |
| `user_info_2020.csv` | 22 × 8 | demographics (+ handedness) |
| `user_sleep_2019_2018.csv` | 736 × 13 | **Actigraph** sleep metrics |
| `user_sleep_2020.csv` | 615 × 23 | **Withings-style** sleep metrics |
| `README_2019.txt` | — | 50 subjects, 700 days, E4 wrist sensor (PPG/EDA/temp), IMU, GPS |

## Data quality

**2019/2018 (Actigraph), 46 subjects, 736 nights**
- Always-present: `sleep_score, total_sleep_time, time_in_bed`.
- **44.6% missing** on `waso, wakeupcount, aal, movement_index, fragmentation_index` — only
  a subset of devices reported them.
- TST mean ≈ 486 min (8.1 h) but **std ≈ 305 min** — implausibly large; some rows mix
  daytime naps / mis-segmented sessions. Needs cleaning before use.

**2020 (Withings-style), 22 subjects, 615 nights**
- Rich and **complete (0% missing)** on `lightsleepduration, deepsleepduration,
  remsleepduration, hr_average/min/max, rr_*, snoring, sleep_score`.
- This is the **same sensor family** (Withings) behind the competition's S2–S4 sleep-adherence
  targets — the most relevant external source we have.

## Schema differences vs the 2024 competition

| Aspect | 2024 competition | 2018–2020 historical |
|---|---|---|
| Subjects | `id01`–`id10` | `1`–`50` (2019) / `user01`–`user24` (2020) — **no overlap** |
| Target form | 7 binary (deviation / guideline-adherence) | raw continuous sleep metrics, no competition labels |
| Sleep device | Withings under-mattress | Actigraph (2019) / Withings (2020) |
| Extra sensors | phone + watch (12 items) | E4 wrist (PPG/EDA/temp), IMU, GPS |

**Subject overlap with the competition: ∅ (empty).** Confirmed programmatically.

## Potential usefulness

- ❌ **Direct label join** — impossible. Different people, different years, no shared keys.
- ❌ **Personalization transfer** — impossible. Per-subject priors don't carry across cohorts.
- ✅ **Distribution prior** — the 2020 Withings durations/efficiency could anchor sensible
  thresholds/priors for S2–S4 (where the competition sensors can't measure directly).
- ✅ **Pretraining a sleep estimator** — a model mapping raw signals → sleep stages/TST,
  pretrained here, could improve the phone-derived `sleep_v3` features (better TST → better S1).
- ✅ **Feature-engineering reference** — which Withings fields most predict `sleep_score`
  informs which derived features are worth computing from phone sensors.

## Risks

- **Cohort / device shift** — different population, years, hardware; naive transfer would
  inject bias. Any prior must be used softly (regularization target), not as hard features.
- **Data-quality (2019)** — 45% missingness and noisy TST; requires filtering.
- **Leakage illusion** — tempting to treat as "more training data", but with no subject or
  label correspondence it would only add noise to a personalization-driven model.

## Estimated leaderboard impact

- As a **direct training signal: ≈ 0** (likely negative — cohort shift, no labels).
- As a **sleep-feature prior for S1 / S2**: plausibly a *small* gain (~0.001–0.003) **if** it
  improves phone-derived TST/efficiency estimates — but S2–S4 remain capped by the missing
  measuring device. It would not move the solution off the ~0.59 ceiling.

## Proposed integration strategy (future competitions)

1. Clean the 2020 Withings set (drop noisy/short sessions; keep complete-field nights).
2. Fit a small supervised model: raw-signal-derived features → Withings sleep stages / TST.
   Use it to **denoise the phone-based `sleep_v3` features**, not as direct inputs.
3. Use 2020 duration/efficiency distributions as **soft priors** (shrinkage targets) for S1/S2.
4. Keep the join path (`exp_R29_external_join.py`) reserved strictly for *same-cohort* public
   labels, should they ever become available.
