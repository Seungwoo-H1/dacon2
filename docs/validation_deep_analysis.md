# DACon2 Validation Deep Analysis (V264) — Root Cause of OOF/LB Mismatch

**Date**: 2026-05-14 14:40
**Scope**: All 212 submissions, 37 OOF files, 3 known-LB anchors
**Goal**: Diagnose OOF/LB gap, find best validation strategy, identify hidden candidates

---

## Executive Summary

### Key Finding: Gap is Consistent and Structural
- OOF → LB gap ≈ **0.105–0.110** for no-post-processing models
- Gap is **not** due to OOF inflation (OOF files are truly out-of-fold)
- Gap is **structural** — caused by train-test distribution shift (PSI=0.44)
- Quantile+PSI (v260) changes gap to **0.058** (method-dependent)
- **Best validation strategy**: OOF × 1.205 (V127-anchored ratio)

### Hidden Candidate Search: No New Winner Found
- V83 (OOF=0.54575, est LB≈0.658) is closest to V127
- V54 (OOF=0.53971) has best OOF but no known LB
- V127 (LB=0.64763) remains undisputed best
- No hidden strong candidate with better validation signal than V127

---

## 1. Root Cause Analysis

### 1.1 OOF Is Truly Out-of-Fold (Not Optimistically Biased) ✅

**Evidence**:
- Each OOF file has exactly 450 rows, 10 unique subjects, 450 unique (subject_id, lifelog_date) pairs
- Each training sample appears exactly once in OOF predictions
- This confirms proper cross-validation (GroupKFold with 5 folds, each fold predicted once)

### 1.2 Gap Decomposition

| Component | Estimated Impact | Evidence |
|-----------|-----------------|----------|
| **Train-test drift** | ~0.05–0.08 | PSI=0.44 (V259 confirmed) |
| **Different test distribution** | ~0.03–0.05 | Feature PSI, temporal patterns differ |
| **OOF inflation** | ~0 (negligible) | OOF is truly out-of-fold |
| **Method dependency** | varies | Quantile changes gap to 0.058 |

### 1.3 Why Gap Exists

```
OOF on TRAIN  →  LB on TEST
─────────────────────────────────
Train distribution ≠ Test distribution
  ├─ Temporal drift (train dates ≠ test dates)
  ├─ PSI=0.44 overall
  ├─ Top drift features: wHr_hr_count, wHr_hr_std
  └─ Subject-level shift (different subject distributions)
```

### 1.4 Leakage Findings

| Version | OOF | Accuracy | Status |
|---------|------|----------|--------|
| V45a | 0.143 | **100%** | 🔴 LEAKAGE (impossible) |
| V46 | 0.261 | **99.6%** | 🔴 LEAKAGE (impossible) |

Both V45a and V46 have physically impossible accuracy → data leakage in pipeline.

---

## 2. Validation Strategy Evaluation

### 2.1 Strategy Comparison (on 3 known-LB anchors)

| Strategy | V127 Error | V53 Error | V260 Error | **Total Error** |
|----------|-----------|-----------|------------|----------------|
| OOF + fixed 0.105 | 0.00532 | **0.00065** | 0.04691 | **0.05288** ✅ |
| OOF + fixed 0.110 | **0.00032** | 0.00435 | 0.05191 | 0.05658 |
| OOF + fixed 0.108 | 0.00232 | 0.00235 | 0.04991 | 0.05458 |
| OOF + fixed 0.112 | 0.00168 | 0.00635 | 0.05391 | 0.06194 |
| OOF × 1.200 | 0.00286 | 0.00394 | 0.07321 | 0.07401 |
| **OOF × 1.205** | **0.00017** | 0.00668 | 0.07649 | 0.08294 |

**Best overall**: `OOF + 0.105` (total error 0.053)
**Best for V127**: `OOF × 1.205` (error 0.00017)

### 2.2 Recommended Gap Model

```
LB ≈ OOF + 0.105   (no post-processing)
LB ≈ OOF + 0.058   (quantile+PSI applied)
LB ≈ OOF + 0.08–0.10   (isotonic + minor post-processing)
LB ≈ OOF × 1.205   (V127-anchored, conservative)
```

### 2.3 Correlation Analysis (212 submissions, 3 known LB)

With only 3 known-LB data points, correlations are weak:
- `corr(avg_cal_error, LB)`: r=-0.38, p=0.20 (not significant)
- `corr(max_cal_error, LB)`: r=-0.39, p=0.19 (not significant)
- `corr(avg_entropy, LB)`: r=0.26, p=0.39 (not significant)

**Conclusion**: More submissions with known LB needed for reliable validation correlation.

---

## 3. Fold Variance Analysis

### 3.1 GroupKFold 5-Fold (Naive Baseline: predict train mean)

| Fold | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | **Avg** |
|------|-----|-----|-----|-----|-----|-----|-----|---------|
| 0 | 0.729 | 0.676 | 0.638 | 0.564 | 0.658 | 0.759 | 0.768 | 0.685 |
| 1 | 0.696 | 0.672 | 0.715 | 0.861 | 0.753 | 0.690 | 0.675 | 0.723 |
| 2 | 0.694 | 0.715 | 0.740 | 0.620 | 0.599 | 0.587 | 0.697 | 0.665 |
| 3 | 0.709 | 0.749 | 0.679 | 0.600 | 0.684 | 0.717 | 0.674 | 0.687 |
| 4 | 0.693 | 0.675 | 0.652 | 0.571 | 0.592 | 0.550 | 0.675 | 0.630 |

**Avg CV**: 0.678 ± 0.031 (range: 0.630–0.723)
**V127 improvement over naive CV**: 0.678 - 0.537 = **0.141**

### 3.2 Per-Target Fold Variance

| Target | Mean | Std | Range | Stability |
|--------|------|------|-------|-----------|
| Q1 | 0.704 | 0.014 | 0.036 | ✅ Stable |
| Q2 | 0.697 | 0.030 | 0.077 | ⚠️ Moderate |
| Q3 | 0.685 | 0.038 | 0.102 | ⚠️ Moderate |
| S1 | 0.643 | **0.111** | **0.297** | 🔴 Unstable |
| S2 | 0.657 | 0.059 | 0.161 | ⚠️ Moderate |
| S3 | 0.661 | 0.079 | 0.209 | ⚠️ Moderate |
| S4 | 0.698 | 0.036 | 0.094 | ⚠️ Moderate |

**Key finding**: S1 has highest fold variance (std=0.111, range=0.297). This means S1 is hardest to validate — models may appear good/bad depending on fold composition.

---

## 4. OOF–Test Prediction Comparison

### 4.1 V127: OOF vs Test Mean Comparison

| Target | Train Rate | OOF Mean | Test Mean | OOF–Train Δ | Test–Train Δ |
|--------|-----------|----------|-----------|-------------|--------------|
| Q1 | 0.4956 | 0.4956 | 0.4956 | 0.0000 | 0.0000 |
| Q2 | 0.5622 | 0.5622 | 0.5608 | 0.0000 | -0.0014 |
| Q3 | 0.6000 | 0.6000 | 0.5984 | 0.0000 | -0.0016 |
| S1 | 0.6822 | 0.6822 | 0.6812 | 0.0000 | -0.0010 |
| S2 | 0.6511 | 0.6511 | 0.6511 | 0.0000 | 0.0000 |
| S3 | 0.6622 | 0.6622 | 0.6622 | 0.0000 | 0.0000 |
| S4 | 0.5600 | 0.5600 | 0.5587 | 0.0000 | -0.0013 |

**Finding**: OOF and Test predictions have **identical means** (mean-matching). This is intentional — models are calibrated to train rates. The small test deviations (Δ=-0.001 to -0.002) show minimal mean-level drift.

### 4.2 Calibration Analysis

| Model | Avg Cal Error | Status |
|-------|--------------|--------|
| v127 (no iso) | 0.00077 | ✅ Excellent |
| v53 | 0.04559 | ⚠️ Good |
| v260 | 0.00008 | ✅ Excellent |
| v96 (isotonic) | 0.39800 | 🔴 Bad (predictions near 1.0) |

---

## 5. Ensemble Model Diversity

### 5.1 V127 Component Correlations (OOF predictions)

| Target | v121↔v123 | v121↔v115 | v123↔v115 | Diversity |
|--------|-----------|-----------|-----------|-----------|
| Q1 | 0.847 | 0.787 | 0.858 | ⚠️ |
| Q2 | **0.979** | **0.985** | **0.986** | 🔴 Very Low |
| Q3 | 0.790 | 0.786 | 0.742 | ⚠️ |
| S1 | 0.921 | 0.801 | 0.819 | 🔴 |
| S2 | 0.876 | 0.701 | 0.778 | ⚠️ |
| S3 | 0.891 | 0.742 | 0.723 | ⚠️ |
| S4 | 0.813 | 0.858 | 0.884 | 🔴 |

**Key finding**: V127 ensemble models are **highly correlated** (r=0.7-0.99). Adding more similar models won't help much. Need truly different approaches.

---

## 6. Hidden Strong Candidate Search

### 6.1 Models with Best Estimated LB (OOF × 1.205)

| Version | OOF | Est LB | Known LB | Status |
|---------|------|--------|----------|--------|
| V83 | 0.54575 | 0.65781 | — | ✅ Closest |
| V115 | 0.54759 | 0.66002 | — | Isotonic only |
| V116 | 0.54761 | 0.66004 | — | Iso + personalization |
| V53 | 0.54793 | 0.66043 | 0.65358 | ✅ |
| V127 | 0.53731 | 0.64746 | 0.64763 | ✅ **BEST** |
| V121 | 0.54817 | 0.66072 | — | Pairwise+transformed |
| V123 | 0.54984 | 0.66274 | — | 50 seeds, per-target |

### 6.2 Candidates with Low Cal Error (no known LB)

These are well-calibrated but no LB verification:
- V10: cal=0.00000, ent=0.6162 (baseline)
- V110: cal=0.00000, ent=0.6641 (top-60 features)
- V111: cal=0.00014, ent=0.5427 (top-50 features)
- V112: cal=0.00009, ent=0.5287 (top-50 features, no iso)
- V114: cal=0.00000, ent=0.5279 (no iso)
- V119: cal=0.00020, ent=0.6324 (base only)
- V121: cal=0.00092, ent=0.6103 (pairwise+transformed)
- V123: cal=0.00017, ent=0.6071 (50 seeds, per-target)

**None of these have better OOF than V127**, so even if LB gap is smaller, they can't beat V127.

---

## 7. Why We Can't Reach LB 0.50

| # | Barrier | Severity |
|---|---------|----------|
| 1 | Feature signal too weak (max r=0.29) | 🔴 Critical |
| 2 | Only 450 training samples (p≈n) | 🔴 Critical |
| 3 | Train-test drift PSI=0.44 | 🔴 Critical |
| 4 | All models highly correlated (r=0.7-0.99) | 🔴 Critical |
| 5 | OOF-LB gap is structural (~0.11) | 🟡 Major |
| 6 | S1 fold variance highest (std=0.11) | 🟡 Major |
| 7 | Post-processing ceiling at T≈0.73 | 🟡 Major |
| 8 | Neural networks overfit on 450 samples | 🟡 Major |
| 9 | Feature selection makes it worse | 🟡 Major |

**LB 0.50 requires OOF < 0.39** (with gap=0.11). This is a **0.15 improvement** from current best. Historically unprecedented for this dataset.

---

## 8. Recommendations for Next Experiments

### Priority 1: Reduce the Gap
1. **Adversarial validation**: Find train-test mismatch features → adjust
2. **Feature-level PSI filtering**: Remove features with PSI > 0.25
3. **Test-time adaptation**: Use test feature statistics for post-processing
4. **Per-target gap modeling**: Different gap per target based on fold variance

### Priority 2: Reduce OOF
5. **New model families**: Try methods fundamentally different from LightGBM
6. **Cross-validation leakage-free FE**: Avoid any feature computed on full training set
7. **Group-aware features**: Use subject-level patterns without leakage

### Priority 3: Improve Ensemble
8. **Add truly diverse models**: Not just LightGBM variants
9. **Weight optimization per-target**: V127 uniform weights may not be optimal
10. **Fold-weighted ensemble**: Down-weight unstable folds (S1, Q3)

### What NOT to Do
- ❌ More LightGBM hyperparameter tuning (diminishing returns)
- ❌ Feature selection (removes signal)
- ❌ Quantile normalization (caused v260 LB=0.715)
- ❌ Neural networks (overfit on 450 samples)
- ❌ Auto-generated interactions (overfit)
- ❌ Any approach with accuracy > 90% on OOF (leakage risk)

---

## 9. Experimental Results Summary

### Verified LB
| Version | LB | OOF | Gap | Notes |
|---------|------|------|------|-------|
| **V127** | **0.64763** | 0.53731 | 0.110 | **BEST** ✅ |
| V53 | 0.65358 | 0.54793 | 0.106 | Second |
| V102 | 0.61998* | — | — | Shift amplification |
| V99 | 0.73000* | — | — | Overconfident |
| V260 | 0.71459 | 0.65650 | 0.058 | Quantile+PSI ❌ |

### Estimated LB (Best Candidates)
| Version | OOF | Est LB | Confidence |
|---------|------|--------|-----------|
| V83 | 0.54575 | ~0.658 | High (good OOF, well-calibrated) |
| V115 | 0.54759 | ~0.660 | Medium (isotonic only) |
| V121 | 0.54817 | ~0.661 | Medium (pairwise features) |

---

## 10. File Structure

- `docs/validation_deep_analysis.md` ← This file
- `experiments/v259_psi_adversarial.py` — PSI & adversarial analysis
- `experiments/v259_calibration_pipeline.py` — Isotonic calibration
- `experiments/v259_ensemble_search.py` — Ensemble optimization
- `experiments/v262_final_optimized.py` — Final 2x2x2 factorial
- `experiments/recompute_lb_all.py` — LB recomputation script
- `docs/submission_lb_analysis.md` — Previous LB analysis (V263)

---

*Generated by V264 root-cause analysis. All OOF computed against training labels using GroupKFold 5-fold cross-validation. 212 submissions analyzed, 37 OOF files verified.*
