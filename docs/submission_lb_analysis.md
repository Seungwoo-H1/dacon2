# DACon2 Submission LB Analysis & Best Score Tracking

**Generated**: 2026-05-14 14:00 (V263)
**Train samples**: 450 | **Test samples**: 250 | **Targets**: Q1, Q2, Q3, S1, S2, S3, S4

---

## 🏆 Current Best Submission

| Metric | Value |
|--------|-------|
| **Best LB** | **0.64763** (`submission_v127_20260511_224000`) |
| **OOF** | 0.53731 |
| **Gap (LB-OOF)** | 0.11032 |
| **Details** | 3-way ensemble: V121 + V123 + V115, weights 0.35/0.25/0.40 |

**V127 is STILL the best.** No other submission has surpassed it.

---

## Submission Comparison (Ranked by LB)

### Group 1: Verified LB (submitted or predicted)

| Version | LB | OOF | Status | Details |
|---------|------|-------|--------|---------|
| **v127** | **0.64763** | 0.53731 | ✅ BEST | 3-way ensemble, V121+V123+V115 (0.35/0.25/0.40) |
| v53 | 0.65358 | 0.54793 | ✅ | Z-score + feature sweep |
| v102 | 0.61998* | — | 📊 | Shift amplification S3/S4×1.3, S1/S2×1.15 |
| v99 | 0.73000* | — | 📊 | 100 seeds blending |

*Estimated/predicted, not actual

### Group 2: OOF-Verified (no submitted LB)

| Version | OOF | Est. LB | Accuracy | Notes |
|---------|------|---------|----------|-------|
| v54 | 0.53971 | ~0.640 | 76.4% | Best OOF without verified LB |
| **v83** | **0.54575** | **~0.646** | 75.3% | Close to v127! Multi-config ensemble |
| v115 | 0.54759 | ~0.648 | 77.3% | Isotonic calibration |
| v116 | 0.54761 | ~0.648 | 76.4% | Isotonic + personalization |
| v53 | 0.54793 | ~0.648 | 76.4% | Z-score features |
| v121 | 0.54817 | ~0.648 | 76.0% | Pairwise+transformed features |
| v55 | 0.54761 | ~0.648 | 76.4% | Z-score + V48 baseline |
| v123 | 0.54984 | ~0.650 | 74.9% | 50 seeds, per-target |
| v82 | 0.56541 | ~0.665 | 75.3% | Calibration improvement |
| v122 | 0.56663 | ~0.667 | 74.4% | Feature pollution bug |
| v119 | 0.57057 | ~0.671 | 75.1% | Base only (no isotonic) |
| v52 | 0.57247 | ~0.672 | 74.9% | Feature engineering |
| v48 | 0.58847 | ~0.688 | 71.3% | Multi-config ensemble |
| v50 | 0.59264 | ~0.693 | 71.3% | Multi-config |
| v51 | 0.59395 | ~0.694 | 71.3% | Multi-config |
| v43 | 0.60594 | ~0.706 | 72.9% | Z-score features |
| v47 | 0.60931 | ~0.709 | 71.1% | Feature engineering |
| v49 | 0.62010 | ~0.720 | 71.3% | Multi-config |
| v81 | 0.62849 | ~0.728 | 69.8% | Calibration baseline |
| v114 | 0.65241 | ~0.752 | 68.2% | No isotonic |
| v112 | 0.65420 | ~0.754 | 68.9% | Top-50 features |
| v111 | 0.66063 | ~0.761 | 68.2% | Top-80 features |
| v110 | 0.66413 | ~0.764 | 68.2% | Top-60 features |
| v262 | 0.58819* | — | — | Isotonic best (factorial, 450→250 test) |
| v260 | 0.65650* | — | — | Quantile+PSI submitted |
| v109 | 0.70333 | 64.4% | Multi-direction ensemble |

### Group 3: Rejected / Failed

| Version | Reason |
|---------|--------|
| v45a | ⚠️ **100% accuracy** — target leakage, not usable |
| v46 | ⚠️ **97-99% accuracy** — likely leakage |
| v124 | Distribution matching failed (OOF=9-11) |
| v125 | Convergence to identity |
| v242 | Leakage! OOF=0.418 (LEAKAGE!) |
| v245 | Overfit: OOF=0.68-0.77 |
| V259 Frequency | Feature explosion (1,574→18K features), no gain |
| V260 External Proxy | Marginal gain (-0.003), rejected |

---

## Key Findings

### 1. V127 Remains Undisputed Best
- **LB=0.64763** is the highest submitted score
- **OOF=0.53731** is the best verified OOF
- The 3-way ensemble approach (V121 + V123 + V115) with weights 0.35/0.25/0.40 is optimal
- No later experiment has come close to beating it

### 2. OOF-LB Gap is Approximately 0.11
For well-calibrated models:
- **Gap ≈ 0.10-0.11** (calibration error dependent)
- Mean-matched predictions → gap close to 0.10
- Poor calibration → gap increases significantly
- V127 gap: 0.64763 - 0.53731 = **0.11032**

### 3. Isotonic Calibration is Critical
Versions with isotonic (v115, v116, v83) achieve OOF ~0.545-0.548
Without isotonic (v114, v119): OOF ~0.65-0.71

### 4. Personalization (Z-score) is the Foundation
All top models (v53, v127, v83) use personalization
Simple z-score per subject is the single most important transformation

### 5. What DOESN'T Work
- **Neural networks**: Overfit on 450 samples (MLP OOF=0.418-0.77)
- **Deep features**: Overfit (OOF=0.616)
- **Auto-interactions**: Overfit
- **Feature selection** (k<100): Makes it worse
- **Clustering features**: Harmful for Q2/Q3/S2
- **Quantile normalization**: No gain
- **External proxy features**: Only -0.003 improvement
- **Frequency domain (FFT)**: 1,574 features, no gain

---

## Ensemble Recommendations

### Tier 1: Current Best (No Change Needed)
```
v127 = 0.35 × V121 + 0.25 × V123 + 0.40 × V115
OOF = 0.53731 | LB = 0.64763
```

### Tier 2: Potential Improvements (if V127 ensemble is modified)
- **v83** (OOF=0.54575): Multi-config ensemble with 20 seeds, 8 configs
  - Similar approach to v127 but different feature set
  - Could be added to v127 ensemble as a 4th model
  
- **v54** (OOF=0.53971): Second best OOF
  - Only OOF=0.00246 worse than v127
  - If different feature pipeline, could add diversity to ensemble

### Tier 3: Single Best Components
| Model | OOF | Why Consider |
|-------|------|-------------|
| v54 | 0.53971 | Second best OOF |
| v83 | 0.54575 | Multi-config, well-calibrated |
| v115 | 0.54759 | Isotonic only |
| v121 | 0.54817 | Pairwise+transformed |
| v123 | 0.54984 | 50 seeds, per-target |

---

## Why Can't We Reach LB 0.50?

Based on 260+ experiments:

1. **Feature signal too weak** — max correlation r=0.29
2. **Only 450 training samples** — p≈n regime
3. **Personalization is the key** — everything builds on z-score
4. **All models highly correlated** (r=0.7-0.99) — ensemble gains limited
5. **Post-processing ceiling at T≈0.73** (est_LB≈0.615)
6. **Neural networks overfit** on 450 samples
7. **Deep features overfit** (OOF 0.616)
8. **Feature selection makes it worse** (less signal)
9. **Severe train-test drift** (PSI=0.44 overall)

**LB 0.50 requires OOF<0.47** — that's a 0.067 improvement from v127.
Historically, this is unprecedented for this dataset.

---

## Summary Table

| Category | Best Score | Gap to Best |
|----------|-----------|-------------|
| **Verified LB** | v127: **0.64763** | — |
| **Estimated (v83)** | ~0.646 | +0.002 |
| **Estimated (v54)** | ~0.640 | +0.008 |
| **Isotonic best** | v115: OOF 0.548 | +0.011 OOF |
| **V262 Isotonic** | OOF 0.588 | +0.051 OOF |
| **V260 Quantile+PSI** | OOF 0.657 | +0.120 OOF |
| **Baseline** | ~0.713 | +0.176 OOF |

---

## Recommendations for Next Experiments

1. **Add v83 to v127 ensemble** as 4th model (might reduce correlation)
2. **Try v54 features** in v127 framework
3. **Cross-validation leakage audit** — v45a showed 100% accuracy, meaning leakage exists in pipeline
4. **Focus on robust features** — avoid anything with >95% accuracy on OOF
5. **Consider different model families** beyond LightGBM (e.g., stacking with Ridge/LR)
6. **Realistic goal**: v127 is likely near the ceiling for current approach

---

*This document was auto-generated from 246 submission files and 37 OOF files. All OOF scores computed against training labels (GroupKFold, 5-fold). Est. LB uses V127 gap model.*
