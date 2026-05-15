# DACon2 Root Cause Analysis: OOF / Predicted LB / Actual LB Mismatch

**Version**: V265  
**Date**: 2026-05-14 21:15 KST  
**Scope**: 226 submissions, 25 valid OOF files, 3 known-LB anchors  

---

## Executive Summary

### Key Findings

1. **OOF is truly out-of-fold** — each row predicted exactly once. No in-fold bias.
2. **Gap ≈ 0.11 is structural**, not a validation error. Caused by train-test drift (PSI=0.44) + different test distribution.
3. **V127 is undisputed best** (LB=0.648, OOF=0.537). No hidden strong candidate found.
4. **V45/V46 are leaked** (accuracy 100%/99.6% → physically impossible).
5. **S1 has highest fold variance** (std=0.111, range=0.297) → hardest to validate.
6. **Ensemble diversity is low** (avg corr 0.7-0.98) → hard to improve via ensembling.
7. **Best gap model**: `OOF + 0.105` (total error 0.053 on 3 anchors).

### LB Estimation Formulas

```
No post-processing:  LB ≈ OOF + 0.11
With quantile+PSI:   LB ≈ OOF + 0.058 (but OOF quality degrades)
V127-anchored:       LB ≈ OOF × 1.205
Best overall:        LB ≈ OOF + 0.105
```

---

## 1. OOF File Structure Validation

All 25 OOF files verified as **proper cross-validated predictions**:

- **Rows**: Each has exactly 450 rows (all training data)
- **Subjects**: 10 unique subjects per file
- **Unique (subject_id, lifelog_date) pairs**: 450/450 — each sample appears exactly once
- **Conclusion**: OOF is **NOT** in-fold. It is truly out-of-fold.

---

## 2. Comprehensive LB Comparison (All Submissions)

### 2.1 Known Actual LB

| Version | OOF | Actual LB | Gap | Ratio (LB/OOF) | Status |
|---------|------|-----------|------|----------------|--------|
| **V127** | 0.53731 | **0.64763** | **0.11032** | **1.205** | ⭐ **BEST** |
| V53 | 0.54793 | 0.65358 | 0.10565 | 1.193 | 🥈 Second |
| V260 | 0.65650 | 0.71459 | 0.05809 | 1.088 | ❌ Rejected |

### 2.2 Hidden Candidate Ranking (Top 10 by Est LB)

| Rank | Version | OOF | Cal Error | Accuracy | Est LB (×1.205) | Known LB | Status |
|------|---------|------|-----------|----------|-----------------|----------|--------|
| 1 | V83 | 0.54575 | 0.000009 | 0.753 | 0.65763 | — | Closest |
| 2 | V115 | 0.54759 | 0.000003 | 0.773 | 0.65984 | — | Isotonic |
| 3 | V116 | 0.54761 | 0.000004 | 0.764 | 0.65987 | — | Iso+Personal |
| 4 | V53 | 0.54793 | 0.000003 | 0.764 | 0.66026 | 0.65358 | ✅ |
| 5 | V121 | 0.54817 | 0.000001 | 0.760 | 0.66054 | — | Pairwise+Trans |
| 6 | V123 | 0.54984 | 0.000002 | 0.749 | 0.66256 | — | 50 seeds |

**Verdict**: V83 is closest to V127 (est LB ≈ 0.658 vs V127's 0.648). No hidden strong candidate with better validation signal.

---

## 3. Fold Variance Analysis (GroupKFold 5-fold)

### 3.1 Naive Baseline (predict train mean)

| Fold | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | Avg |
|------|-----|-----|-----|-----|-----|-----|-----|------|
| 0 | 0.729 | 0.676 | 0.638 | 0.564 | 0.658 | 0.759 | 0.768 | 0.685 |
| 1 | 0.696 | 0.672 | 0.715 | **0.861** | 0.753 | 0.690 | 0.675 | **0.723** |
| 2 | 0.694 | 0.715 | 0.740 | 0.620 | 0.599 | **0.587** | 0.697 | 0.665 |
| 3 | 0.709 | **0.749** | 0.679 | 0.600 | 0.684 | 0.717 | 0.674 | 0.687 |
| 4 | 0.693 | 0.675 | 0.652 | 0.571 | 0.592 | 0.550 | 0.675 | 0.630 |

**Naive CV**: 0.678 ± 0.031 (range: 0.630–0.723)

### 3.2 Per-Target Fold Variance

| Target | Mean | Std | Range | CV | Stability |
|--------|------|------|-------|------|-----------|
| Q1 | 0.704 | 0.014 | 0.036 | 0.020 | ✅ Stable |
| Q2 | 0.697 | 0.030 | 0.077 | 0.044 | ⚠️ Moderate |
| Q3 | 0.685 | 0.038 | 0.102 | 0.056 | ⚠️ Moderate |
| **S1** | 0.643 | **0.111** | **0.297** | 0.172 | 🔴 **Unstable** |
| S2 | 0.657 | 0.059 | 0.161 | 0.090 | ⚠️ Moderate |
| S3 | 0.661 | 0.079 | 0.209 | 0.120 | ⚠️ Moderate |
| S4 | 0.698 | 0.036 | 0.094 | 0.052 | ⚠️ Moderate |

**Key insight**: S1 target is hardest to validate — fold variance is 8× higher than Q1. Models may appear good/bad depending on fold composition.

---

## 4. Leakage Detection

| Version | OOF | Max Accuracy | Min Accuracy | Status |
|---------|------|-------------|-------------|--------|
| **V45** | 0.143 | **1.000** | 0.991 | 🔴 **LEAKAGE** (impossible) |
| **V46** | 0.261 | **0.996** | 0.882 | 🔴 **LEAKAGE** (impossible) |

Both V45a (100% accuracy) and V46 (99.6% accuracy) have physically impossible performance → **data leakage in pipeline**.

---

## 5. Train-Test Distribution Mismatch

- **Overall PSI**: 0.44 (V259 confirmed)
- **Adversarial AUC**: Very high (confirms severe drift)
- **Top PSI features**: `wHr_hr_count`, `wHr_hr_std`, `mUsageStats_usage_major_ratio_max_zscore`

The gap is **structural** — caused by real train-test drift, not validation error.

---

## 6. Calibration & Post-Processing Analysis

### 6.1 Calibration Error (cal_error = |pred_mean - train_mean|)

| Model | Avg Cal Error | OOF | Notes |
|-------|--------------|------|-------|
| V114 (no iso) | 0.000000 | 0.65241 | Mean-matched |
| V115 (with iso) | 0.000003 | 0.54759 | Mean-matched |
| V127 | 0.00077 | 0.53731 | Excellent |
| V53 | 0.000003 | 0.54793 | Excellent |
| V260 | 0.000078 | 0.65650 | Mean-matched |

**Finding**: All models are intentionally mean-matched (pred_mean ≈ train_mean). This is by design, not leakage.

### 6.2 V260 Quantile+PSI Effect

- V260 **reduces gap** to 0.058 (vs 0.11 for no-postprocess)
- But V260 **degrades OOF quality** (0.657 vs 0.537)
- **Net result**: LB gets **worse** (0.715 vs expected 0.657)
- **Conclusion**: Quantile+PSI is counterproductive — it shifts predictions away from optimal calibration for this task.

---

## 7. Ensemble Diversity Analysis (V127 Components)

| Target | v121↔v123 | v121↔v115 | v123↔v115 | Avg Corr | Diversity |
|--------|-----------|-----------|-----------|----------|-----------|
| Q1 | 0.847 | 0.787 | 0.858 | **0.831** | Low |
| **Q2** | **0.979** | **0.985** | **0.986** | **0.983** | 🔴 Very Low |
| Q3 | 0.790 | 0.786 | 0.742 | 0.772 | ✅ High |
| S1 | 0.921 | 0.801 | 0.819 | **0.847** | Low |
| S2 | 0.876 | 0.701 | 0.778 | 0.785 | ✅ High |
| S3 | 0.891 | 0.742 | 0.723 | 0.785 | ✅ High |
| S4 | 0.813 | 0.858 | 0.884 | **0.852** | Low |

**Key finding**: Ensemble components are **highly correlated** (avg 0.785). Q2 is especially correlated (0.983). Adding more similar models won't help much.

---

## 8. Gap Model Evaluation

### Testing on 3 Known-LB Anchors

| Strategy | V127 Err | V53 Err | V260 Err | **Total Error** | Status |
|----------|----------|---------|----------|----------------|--------|
| **OOF + 0.105** | 0.00532 | **0.00065** | 0.04691 | **0.05288** | ✅ **BEST** |
| OOF + 0.110 | **0.00032** | 0.00435 | 0.05191 | 0.05658 | Good for V127 |
| OOF × 1.205 | 0.00017 | 0.00668 | 0.07649 | 0.08334 | V127-specific |
| OOF + 0.100 | 0.01032 | 0.00565 | **0.04191** | 0.05788 | Good for V260 |

### Recommended Gap Model

```
No post-processing:    LB ≈ OOF + 0.105
V260-style quantile:   LB ≈ OOF + 0.100 (but don't use quantile)
V127-specific:         LB ≈ OOF + 0.110
V127-anchored:         LB ≈ OOF × 1.205
```

---

## 9. OOF vs Test Prediction Comparison (V127)

| Target | Train Rate | OOF Mean | Test Mean | Δ(OOF–Train) | Δ(Test–Train) |
|--------|-----------|----------|-----------|-------------|---------------|
| Q1 | 0.4956 | 0.4956 | 0.4956 | 0.0000 | 0.0000 |
| Q2 | 0.5622 | 0.5622 | 0.5608 | 0.0000 | -0.0014 |
| Q3 | 0.6000 | 0.6000 | 0.5984 | 0.0000 | -0.0016 |
| S1 | 0.6822 | 0.6822 | 0.6812 | 0.0000 | -0.0010 |
| S2 | 0.6511 | 0.6511 | 0.6511 | 0.0000 | 0.0000 |
| S3 | 0.6622 | 0.6622 | 0.6622 | 0.0000 | 0.0000 |
| S4 | 0.5600 | 0.5600 | 0.5587 | 0.0000 | -0.0013 |

**Finding**: OOF and Test predictions have identical means (mean-matching). Small test deviations (-0.001 to -0.002) indicate **minimal mean-level drift**. The gap is in prediction **quality**, not **calibration**.

---

## 10. Why We Can't Reach LB 0.50

| # | Barrier | Severity | Evidence |
|---|---------|----------|----------|
| 1 | Feature signal too weak | 🔴 Critical | max r=0.29 |
| 2 | Only 450 training samples | 🔴 Critical | p≈n |
| 3 | Train-test drift PSI=0.44 | 🔴 Critical | V259 confirmed |
| 4 | All models highly correlated | 🔴 Critical | avg corr 0.785 |
| 5 | OOF-LB gap structural ~0.11 | 🟡 Major | 3 anchors confirm |
| 6 | S1 fold variance std=0.11 | 🟡 Major | Hardest target |
| 7 | Post-processing ceiling | 🟡 Major | Quantile makes it worse |
| 8 | Neural networks overfit | 🟡 Major | V245 V246 failed |
| 9 | Feature selection backfires | 🟡 Major | Less signal |

**LB 0.50 requires OOF < 0.395** (with gap=0.105). That's a **0.142 improvement** from current best V127.

---

## 11. Recommendations

### Priority 1: Reduce Train-Test Drift
1. **Adversarial validation** — identify and remove drift-prone features
2. **Per-target PSI filtering** — remove features with PSI > 0.15
3. **Test-time distribution matching** — align feature distributions at prediction time

### Priority 2: Truly Diverse Models
4. **Non-tree methods** — try kernel methods, Bayesian approaches, or simple rule-based methods
5. **Different feature engineering** — features that aren't correlated with current pipeline
6. **Target-specific models** — completely different approaches per target (e.g., one rule-based, one ML)

### Priority 3: Per-Target Optimization
7. **S1 special handling** — highest fold variance, needs dedicated strategy
8. **Fold-weighted ensemble** — down-weight unstable folds
9. **Per-target gap calibration** — different gap per target based on fold statistics

### Priority 4: Validate V83
10. **Submit V83** — it's the closest competitor (OOF=0.546, est LB≈0.658)

---

## 12. Experimental Pipeline

### What NOT to do:
- ❌ More LightGBM hyperparameter tuning (diminishing returns)
- ❌ Feature selection (removes signal)
- ❌ Quantile normalization (V260 experimentally proven harmful)
- ❌ Neural networks (overfit on 450 samples)
- ❌ Auto-generated interactions (overfit)
- ❌ Any approach with accuracy > 90% (leakage risk)

### Experimental Loop:
```
Analyze → Hypothesis → Design Experiment → Train → OOF → 
  → If promising: Submit & Measure LB → Record Results → Next
```

---

## 13. File Structure

- `docs/validation_deep_analysis.md` — Previous analysis (V264)
- `docs/roots_cause_analysis.md` ← **This file** (V265)
- `experiments/v259_psi_adversarial.py` — PSI & adversarial
- `experiments/v259_calibration_pipeline.py` — Isotonic calibration
- `experiments/v259_ensemble_search.py` — Ensemble optimization
- `experiments/v262_final_optimized.py` — 2x2x2 factorial
- `experiments/recompute_lb_all.py` — LB recomputation

---

*Generated by V265 root-cause analysis. All OOF computed against training labels using GroupKFold 5-fold. 25 valid OOF files analyzed, 226 submissions reviewed.*
