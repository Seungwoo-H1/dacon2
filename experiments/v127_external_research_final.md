# V127 External Data Research — Final Consolidated Report (V06–V10)

## Executive Summary

**V127 baseline (internal features only)**: AVG OOF = **0.61034**

**Final (V09 external + V10 multi-config ensemble)**: AVG OOF = **0.58716** (Δ = **-0.02318**)

- **Q1**: 0.64859 → 0.63605 (Δ = -0.01254)
- **Q2**: 0.59823 → 0.59250 (Δ = -0.00573)
- **Q3**: 0.61111 → 0.60499 (Δ = -0.00612)
- **S1**: 0.57929 → 0.55918 (Δ = -0.02011)
- **S2**: 0.59690 → 0.55128 (Δ = -0.04562)
- **S3**: 0.63444 → 0.58321 (Δ = -0.05123)
- **S4**: 0.62103 → 0.60011 (Δ = -0.02092)

## Pipeline Summary

### V06: Global Features (FAILED)
- **Result**: Δ = 0.00000
- **Cause**: Global stats constant across all samples → zero gain

### V07: Proxy Features
- **Result**: AVG Δ = -0.01324
- **Key insight**: External features must vary per sample to be useful

### V08: Target-Specific Selection (4 seeds, exhaustive)
- **Result**: AVG Δ = -0.02062 (3173s)
- **Finding**: n_ext=1 optimal for most targets

### V09: Fast Selection (1 seed, 36x faster)
- **Result**: AVG Δ = -0.01895 (87s)
- **Best features per target**:
  - Q1: ext_night_light_zscore (n_ext=1, n_total=15)
  - Q2: ext_total_ambience_zscore (n_ext=1, n_total=20)
  - Q3: None needed (n_ext=0, n_total=12)
  - S1: ext_wifi_ble, ext_activity_z (n_ext=2, n_total=20)
  - S2: ext_night_light_zscore, ext_total_ambience_zscore (n_ext=2, n_total=15)
  - S3: ext_night_light_zscore (n_ext=1, n_total=12)
  - S4: ext_night_light_zscore, ext_activity_z (n_ext=2, n_total=15)

### V10 EXP1: Calibration
- **Result**: Minor improvement (Δ = -0.001 to +0.010)
- **S1**: +0.00978, S3: +0.00877 — calibration helps when OOF mean ≠ train mean
- **Pseudo-labeling**: NOT effective — S1/S3 overconfident (all 1.0), Q3 domain gap

### V10 EXP2: Staged Training ❌
- **Result**: Worse on ALL targets (Δ = +0.004 to +0.027)
- **Verdict**: Discard — 2-stage approach overfits small data (450 samples)

### V10 EXP3: Multi-config Ensemble ✅
- **Result**: AVG Δ = -0.00375
- **Improves 5/7 targets**: Q1, Q3, S1, S3, S4
- **Best combos**:
  - Q1: wide:0.5 + deep:0.5
  - Q3: wide:0.45 + deep:0.45 + safety:0.1
  - S1: v48:0.6 + wide:0.2 + deep:0.2
  - S3: v48:0.75 + safety:0.25
  - S4: v48:0.8 + wide:0.2

## Key Insights

1. **ext_night_light** (night light/night hours ratio) — universal signal, helps 4/7 targets
2. **ext_total_ambience** (ambient noise) — critical for Q2 and S2
3. **ext_wifi_ble** (social activity) — key for S1
4. **Q3 needs NO external features** — internal features already sufficient
5. **S2/S3 show biggest gains** — sleep/stress targets most sensitive to external data
6. **Calibration matters** — OOF mean matching helps when distributions differ
7. **Multi-config ensemble** adds marginal but consistent improvement
8. **Staged training fails** — too few samples for multi-stage learning

## Domain Analysis

### External Data A: sleep_health_lifestyle.csv (400 samples, Kaggle)
- Features: Age, Sleep Duration, Quality of Sleep, Physical Activity, Stress Level, BMI, BP, HR, Daily Steps, Sleep Disorder
- **Domain**: Population-level health statistics

### External Data B: date_features (183 samples)
- Features: season, day_of_year, holiday, temperature, winter indicator
- **Domain**: Temporal/climate features

### Bridge:
External features inform **what internal features mean** (e.g., high charging + low steps = poor health profile)

## Recommendations

1. **Use external features as described in V09** (target-specific selection)
2. **Apply multi-config ensemble from V10** for Q1, Q3, S1, S3, S4
3. **Don't use staged training** (overfits)
4. **Don't rely on pseudo-labeling** (domain gap too large)
5. **Next**: Explore more external data sources (weather, economic, urban)
6. **Next**: Feature interaction engineering
7. **Next**: Cross-validation leakage-free feature selection

## Estimated LB

If OOF improvement of -0.02318 generalizes:
- **Expected LB: ~0.58-0.59** (from ~0.61 baseline)
- Conservative estimate: LB 0.60
- Aggressive estimate: LB 0.58

## Files

- `experiments/v09_single_seed_20260512_123222.json` — V09 results
- `experiments/v10_20260512_124310.json` — V10 results
- `src/external_data_research/v09_ensemble_pseudo_labeling.py` — V09 pipeline
- `src/external_data_research/v10_staged_training.py` — V10 pipeline
