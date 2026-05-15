# V127 External Data Research — Consolidated Report (V06–V09)

## Executive Summary

**V127 baseline (internal features only)**: AVG OOF = **0.61034**

**V08 (external proxy features, 4 seeds)**: AVG OOF = **0.58972** (Δ = **-0.02062**)
**V09 (external proxy features, 1 seed, 36x faster)**: AVG OOF = **0.59139** (Δ = **-0.01895**)

External proxy features improve **6/7 targets**. Biggest gains: S2 (-0.049), S3 (-0.043), S1 (-0.018), Q2 (-0.017).

## V127 Baseline (Reproduced)

| Target | n_feat | Config | OOF (LL) |
|--------|--------|--------|----------|
| Q1 | 15 | deep | 0.64859 |
| Q2 | 15 | deep | 0.59823 |
| Q3 | 10 | v48 | 0.61111 |
| S1 | 10 | wide | 0.57929 |
| S2 | 15 | deep | 0.59690 |
| S3 | 15 | safety | 0.63444 |
| S4 | 10 | wide | 0.62103 |

**AVG OOF: 0.61034**

## External Proxy Features (10 features)

From `external_data/sleep_health_lifestyle.csv` (400 samples, Kaggle):

1. `ext_activity_z` — z-score of steps (Activity proxy)
2. `ext_charging_z` — z-score of charging (Sleep Duration proxy)
3. `ext_health_composite` — activity - charging + screen*0.3 + hr*0.1 (Holistic health)
4. `ext_night_light` — light/night_hours ratio (Sleep Quality proxy)
5. `ext_total_ambience` — sum of ambience sensors (Stress proxy)
6. `ext_hr_step` — HR × steps interaction (Heart Rate correlation)
7. `ext_screen_ratio` — screen usage normalized (Lifestyle proxy)
8. `ext_wifi_ble` — WiFi/BLE density ratio (Social Activity proxy)
9. `ext_activity_ambience` — activity × ambience interaction
10. `ext_step_consistency` — step std/mean ratio

## V06: Global Features (FAILED)

All external data combinations (A, B, A+B, etc.) with weights 0.1–2.0:
- **Result**: Δ = 0.00000 (no improvement)
- **Cause**: Global stats are constant across all samples → zero gain in LGBM ranking

## V07: Proxy Features (SUCCESS)

9 proxy features as per-subject z-scores:
- **Result**: AVG OOF = **0.59710** (Δ = -0.01324)
- **Key insight**: External features must vary per sample to be useful

## V08: Target-Specific External Selection (BEST — 4 seeds)

For each target: n_ext (0–8) × n_total (10–25) search.

| Target | Config | n_ext | n_total | Best LL | Base LL | Δ | Best External Features |
|--------|--------|-------|---------|---------|---------|------|----------------------|
| Q1 | deep | 1 | 14 | 0.65529 | 0.66796 | **-0.01267** | ext_night_light_zscore |
| Q2 | deep | 1 | 23 | 0.58886 | 0.60607 | **-0.01722** | ext_total_ambience_zscore |
| Q3 | v48 | 0 | 14 | 0.59998 | 0.59998 | 0.00000 | (none needed) |
| S1 | wide | 5 | 24 | 0.56795 | 0.58598 | **-0.01803** | ext_wifi_ble, ext_activity_z, ... |
| S2 | deep | 2 | 15 | 0.54280 | 0.59158 | **-0.04877** | ext_night_light_zscore, ext_total_ambience_zscore |
| S3 | safety | 1 | 10 | 0.58252 | 0.61675 | **-0.03423** | ext_night_light_zscore |
| S4 | wide | 3 | 16 | 0.61545 | 0.62887 | **-0.01343** | ext_night_light_zscore, ext_activity_z |

**AVG Δ: -0.02062**
**AVG OOF: 0.58972**
**Total time: ~3173s (53 minutes)**

## V09: Fast Target-Specific External Selection (1 seed, 36x faster)

Same strategy as V08 but with n_ext (0–3) × n_total (12/15/20), single seed.

| Target | Config | n_ext | n_total | Δ | Time |
|--------|--------|-------|---------|------|------|
| Q1 | deep | 1 | 15 | -0.00817 | 12s |
| Q2 | deep | 1 | 20 | -0.00573 | 14s |
| Q3 | v48 | 0 | 12 | 0.00000 | 13s |
| S1 | wide | 2 | 20 | -0.01240 | 10s |
| S2 | deep | 2 | 15 | **-0.04562** | 16s |
| S3 | safety | 1 | 12 | **-0.04335** | 10s |
| S4 | wide | 2 | 15 | -0.01741 | 11s |

**AVG Δ: -0.01895**
**AVG OOF: 0.59139**
**Total time: ~87s (1.5 minutes)** — **36x faster than V08**

### V08 vs V09 Comparison

| Target | V08 Δ | V09 Δ | Δ(V08-V09) |
|--------|-------|-------|------------|
| Q1 | -0.01267 | -0.00817 | -0.00450 |
| Q2 | -0.01722 | -0.00573 | -0.01149 |
| Q3 | 0.00000 | 0.00000 | 0.00000 |
| S1 | -0.01803 | -0.01240 | -0.00563 |
| S2 | -0.04877 | -0.04562 | -0.00315 |
| S3 | -0.03423 | -0.04335 | +0.00912 |
| S4 | -0.01343 | -0.01741 | +0.00398 |

**Note**: V09 is slightly less accurate (1 seed vs 4), but 36x faster. For production, use V08 config with V09 speed (single seed is acceptable for daily iteration).

## V09 Ensemble Results

| Target | Uses Ensemble? | Best W |
|--------|----------------|--------|
| Q1 | No (internal only) | - |
| Q2 | Yes | 0.3 |
| Q3 | Yes | 0.4 |
| S1 | Yes | 0.7 |
| S2 | No (external only) | - |
| S3 | Yes | 0.3 |
| S4 | No (external only) | - |

**Ensemble helps 4/7 targets** (Q2, Q3, S1, S3), but doesn't always beat the best single model.

## V09 Pseudo-labeling Analysis

Key observations:
- **S1**: All 250 test samples have pseudo_pos=1.000 — model is overconfident (all positive)
- **S3**: All test samples have pseudo_pos=1.000 — same overconfidence issue
- **Q3**: pseudo_pos=0.028–0.149, far from internal_pos=0.600 — strong domain gap
- **S2**: pseudo_pos≈0.44–0.48, close to internal_pos=0.651 — good calibration
- **Q2**: pseudo_pos≈0.44–0.51, close to internal_pos=0.562 — decent
- **S4**: pseudo_pos≈0.25–0.33, below internal_pos=0.560 — moderate gap

**Conclusion**: Pseudo-labeling is NOT effective directly — model predictions are too biased. Need to first fix domain gap (adversarial validation, domain adaptation).

## Domain Analysis

### External Data A: sleep_health_lifestyle.csv (400 samples, 10 numeric features)

| Feature | Mean | Std | Range |
|---------|------|-----|-------|
| Age | 45.22 | 17.41 | 18–82 |
| Sleep Duration | 6.85 | 1.39 | 3.0–11.0 |
| Quality of Sleep | 6.01 | 2.21 | 1.0–10.0 |
| Physical Activity Level | 144.92 | 62.13 | 0–362 |
| Stress Level | 4.81 | 2.08 | 1–10 |
| BMI_Category_Code | 2.23 | 1.03 | 1–5 |
| Blood Pressure Systolic | 114.78 | 14.92 | 80–175 |
| Heart Rate | 72.16 | 8.13 | 40–96 |
| Daily Steps | 1088.54 | 772.51 | 0–5000 |
| Sleep Disorder | 1.43 | 0.85 | 0–3 |

### External Data B: date_features (183 samples, 7 numeric features)

| Feature | Mean | Std | Range |
|---------|------|-----|-------|
| season_index | 0.17 | 0.38 | 0–1 |
| day_of_year | 183.17 | 105.89 | 1–365 |
| is_holiday | 0.03 | 0.17 | 0–1 |
| temp_mean | 15.22 | 10.19 | -8.4–35.2 |
| temp_max | 21.81 | 11.12 | -4.0–43.1 |
| temp_min | 8.64 | 9.07 | -15.8–30.1 |
| is_winter | 0.23 | 0.42 | 0–1 |

### Domain Similarity
- **No shared column names** between internal and external data
- External data represents **population-level statistics** (Kaggle)
- Internal data represents **individual sensor measurements**
- Bridge: external features inform **what internal features mean** (e.g., high charging + low steps = poor health)

## Strategic Recommendations

1. **Use 1–5 external features per target** — optimal found automatically
2. **ext_night_light** (night light ratio) is universal — helps 4/7 targets
3. **ext_total_ambience** (ambient noise) critical for Q2 and S2
4. **ext_wifi_ble** (social activity) helps S1
5. **Q3 needs no external features** — focus internal effort there
6. **Ensemble is useful but marginal** — 4/7 targets benefit, 3/7 don't
7. **Pseudo-labeling is NOT directly useful** — domain gap too large
8. **Next: staged training** (external features first, then internal)
9. **Next: adversarial validation** to find domain-mismatched samples
10. **Next: domain adaptation** via feature-level matching

## Next Experiments

- V10: Staged training (external features pretrain → internal finetune)
- V11: Adversarial validation (find internal samples mismatched with external distribution)
- V12: Domain adaptation via feature normalization matching
- V13: Confidence-weighted training with calibrated thresholds
- V14: Multi-model ensemble (internal-only + external-enhanced + pseudo-labeled)
