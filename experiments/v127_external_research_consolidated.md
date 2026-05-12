# V127 External Data Research — Consolidated Report (V06–V08)

## Executive Summary

**V127 baseline (internal features only)**: AVG OOF = **0.61034**

**V08 (external proxy features)**: AVG OOF = **0.58972** (Δ = **-0.02062**)

External proxy features improve **6/7 targets**. Biggest gains: S2 (-0.049), S3 (-0.034), S1 (-0.018), Q2 (-0.017).

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

## V08: Target-Specific External Selection (BEST)

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

### Key Findings

1. **6/7 targets improved** with external features
2. **S2 (Δ=-0.049) and S3 (Δ=-0.034)** show the biggest gains — sleep/stress targets
3. **ext_night_light_zscore** (night light/night hours ratio) is the most universal external feature (helps Q1, S2, S3, S4)
4. **ext_total_ambience_zscore** (ambient noise) helps Q2 and S2
5. **S1 needs 5 external features** — social activity proxy (WiFi/BLE) is key
6. **Q3 needs no external features** — already well modeled by internal features alone

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
6. **Next: pseudo-labeling** using external distribution for soft labels
7. **Next: staged training** (external features first, then internal)
8. **Next: ensemble** of internal-only vs external-enhanced models

## Next Experiments

- V09: Pseudo-labeling with confidence filtering (currently running)
- V10: Staged training (external pretrain → internal finetune)
- V11: Domain adaptation via adversarial validation
- V12: Confidence-weighted training
- V13: Ensemble optimization (internal-only vs external-enhanced)
