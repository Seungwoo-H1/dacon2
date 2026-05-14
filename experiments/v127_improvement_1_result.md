# V128: Group-wise Target Encoding (Leave-One-Out) + Enhanced Feature Engineering

**Experiment ID**: V128 / V127 Improvement #1  
**Date**: 2026-05-13  
**Status**: ❌ Not an improvement — reverted

---

## Summary

| Feature Set | AVG OOF | vs V127 |
|---|---|---|
| V127 baseline | 0.53731 | — |
| **V128 base** | **0.64920** | **+0.11189** ❌ |
| V128 full_internal | 0.65185 | +0.11454 ❌ |
| V128 full_external | 0.65627 | +0.11896 ❌ |

**Verdict**: All V128 configurations are significantly worse than V127 baseline.  
**Decision**: Do not adopt. Keep V127 as is.

---

## Per-Target Breakdown (V128 base vs V127)

| Target | V127 OOF | V128 Base | Δ |
|---|---|---|---|
| Q1 | 0.53731 | 0.68734 | **+0.15003** |
| Q2 | 0.53731 | 0.65599 | **+0.11868** |
| Q3 | 0.53731 | 0.62183 | **+0.08452** |
| S1 | 0.53731 | 0.61644 | **+0.07913** |
| S2 | 0.53731 | 0.63850 | **+0.10119** |
| S3 | 0.53731 | 0.63951 | **+0.10220** |
| S4 | 0.53731 | 0.68479 | **+0.14748** |

**Worst degradation**: Q1 (+0.15), S4 (+0.147)  
**Least affected**: S1 (+0.079), Q3 (+0.085)

---

## What Was Tested

### New Features Added

1. **Group-wise Leave-One-Out Target Encoding**
   - For each subject, computed mean of target labels from ALL OTHER subjects
   - LOO mean, LOO std, LOO count per target (21 new features)
   - Intended to capture peer-influence / cohort effects
   
2. **Cross-Group Interactions**
   - Activity × Environment interactions (activity * ambience, activity * wifi, etc.)
   - 28 new interaction features

3. **Subject-Level Temporal Trends**
   - Linear slope per subject for 6 base features (wPedo, mActivity, mScreen, wHr, mLight, mAmbience)
   - Recent trend (last 3 vs first 3 samples)
   - 56 new trend features

4. **Subject-Level Aggregate Stats**
   - Mean, std, min, max per subject for 9 base features (36 new features)

5. **Deviation from Subject Mean (Personalized Residual)**
   - Z-score-like deviation per feature relative to subject's own mean (9 new features)

### Feature Set Configurations

- **V128 base**: V127 base features only (269 cols) — no LOO, no enhanced
- **V128 full_internal**: Base + LOO + Enhanced (349 cols)
- **V128 full_external**: Base + LOO + Enhanced + External (357 cols)

### Pipeline Structure

Same V127 ensemble structure:
- 3 strategies: V115_base (0.40), V123_pair (0.25), V121_p+t (0.35)
- 5-fold GroupKFold (by subject_id)
- 4 seeds: [42, 7, 999, 777]
- Isotonic calibration + mean matching

---

## Root Cause Analysis

### Why LOO Target Encoding Hurt

1. **Small dataset (450 samples, 10 subjects)**: With only ~45 samples per subject on average, the LOO mean is computed from very few other subjects. The estimate is extremely noisy.

2. **High leakage risk**: LOO target encoding encodes target information from other subjects. Even though it's "leave-one-out", the signal from 9 other subjects is strong enough to create patterns that look like signal in CV but don't generalize.

3. **The model learned a shortcut**: Instead of learning meaningful features, the LGBM models latched onto the LOO target means as a proxy — effectively memorizing which subject tends to have which target, rather than learning the relationship between features and targets.

4. **This explains why the degradation is uniform**: All targets degrade similarly, suggesting the LOO features create a systematic bias rather than target-specific overfitting.

### Why Enhanced Features Hurt

1. **Noise amplification**: With only 450 samples, adding ~90 new features (trends, aggregates, deviations) dilutes the signal-to-noise ratio. The LGBM models with limited depth can't distinguish signal from noise.

2. **Temporal trends are unreliable**: With few observations per subject (median ~45 total, many subjects have <10 per session), slope estimates are dominated by noise.

3. **Subject aggregates don't add information**: The original V127 features already capture per-session behavior. Aggregating per subject adds little new signal but increases feature count.

---

## Key Lessons

1. **LOO target encoding is dangerous on small datasets**: When the number of groups is small (<20), LOO estimates are too noisy to be useful. Only consider when group count > 50.

2. **More features ≠ better on small data**: With 450 samples, adding 90 new features (20% increase) hurts performance. The effective sample-to-feature ratio drops below optimal.

3. **Personalization via deviation features is not helping**: Computing z-scores per subject doesn't add discriminative power beyond the existing features.

4. **Cross-validation leakage from LOO encoding**: Even mathematically correct LOO can cause CV to overestimate performance in practice, because the LOO statistics encode structural patterns (e.g., subject demographics) that leak into test predictions.

---

## Recommendation

**Keep V127 as-is.** The next improvements should focus on:

1. **Better ensemble weights** — optimize per-target weights instead of fixed 0.35/0.25/0.40
2. **Additional seeds** — more diverse models (V53 used 50 seeds)
3. **Calibration improvements** — explore better methods beyond isotonic regression
4. **Feature selection** — more aggressive per-target feature reduction (already in V53 sweep)
5. **External data** — new external data sources with higher sample overlap
6. **Stacking/Blending** — use CV predictions from multiple models as meta-features

---

## Files

- **Script**: `experiments/v128_groupwise_target_encode.py`
- **Result**: `experiments/v128_20260513_223907.json`
- **Pipeline**: `experiments/v128_groupwise_target_encode.py`
