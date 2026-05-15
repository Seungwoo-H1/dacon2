# V47-V50: Experiment Suite Results

## Overview

| Version | Name | Avg Cal | vs V10 | vs V48 | Key Insight |
|---------|------|---------|--------|--------|-------------|
| V10 | Baseline LGBM | 0.6038 | — | +0.0153 | StandardScaler + LGBM + mean-match cal |
| V47 | Ensemble of LGBM Variants | 0.6093 | +0.0055 ❌ | +0.0208 | Meta-learner overfits with 450 samples |
| V48 | Target Transform (Isotonic) | **0.5885** | **-0.0153 ✅** | — | Isotonic cal on binary OOF, all targets |
| V49 | MI Feature Selection | 0.6201 | +0.0163 ❌ | +0.0316 | MI worse than LGBM importance; low overlap |
| V50 | Isotonic + V37 Configs | 0.5926 | -0.0112 ✅ | +0.0041 | V37 configs + isotonic cal, per-target n_feat |

**Winner: V48** (0.5885 avg cal, -0.0153 vs V10)

---

## V47: Ensemble of LGBM Variants

### Hypothesis
Training 6 diverse LGBM configs per target (different depth, learning rate, etc.) and ensembling them via meta-learner stacking should produce more robust predictions than any single config.

### Method
- 6 configs per target: deep/moderate, shallow/strong reg, balanced, wide/shallow, deep/slow, ultra-safe
- 5 seeds each → 30 models per target
- Two stacking strategies: (A) simple average, (B) LogisticRegression meta-learner
- 20 features per target (top LGBM importance)

### Results
- **Q1**: Best=B (meta-learner) Cal=0.6334
- **Q2**: Best=B (meta-learner) Cal=0.5996
- **Q3**: Best=B (meta-learner) Cal=0.6126
- **S1**: Best=B (meta-learner) Cal=0.5745
- **S2**: Best=B (meta-learner) Cal=0.6004
- **S3**: Best=B (meta-learner) Cal=0.6234
- **S4**: Best=B (meta-learner) Cal=0.6213

**V47 Avg: 0.6093**

### Why it failed
1. **Meta-learner overfits**: With only 450 samples, training a LR on 30 per-fold predictions teaches noise
2. **Strategy A (simple avg) was always worse** — configs are too correlated (same data, same features, different hyperparams)
3. V48 showed isotonic calibration alone was better than this complex ensemble

---

## V48: Target Transformation Experiments

### Hypothesis
Binary classification may not be the optimal framing. Regression, Yeo-Johnson transforms, and isotonic calibration could improve log-loss.

### Method
5 approaches per target:
- **A**: Standard binary classification
- **B**: Regression (regression_obj='regression') + Platt calibration
- **C**: Yeo-Johnson transform on targets, regression, inverse transform
- **D**: Binary classification + **Isotonic calibration** on OOF predictions
- **E**: Ensemble of B + D

### Results (per target)

| Target | A (binary) | B (reg+Platt) | C (YJ+reg) | D (iso cal) | E (B+D) | Best |
|--------|-----------|---------------|------------|-------------|---------|------|
| Q1 | 0.6423 | 0.6433 | 0.6485 | **0.6207** | 0.6296 | D |
| Q2 | 0.6110 | 0.6051 | 0.6002 | **0.5913** | 0.5956 | D |
| Q3 | 0.6192 | 0.6251 | 0.6211 | **0.5944** | 0.6070 | D |
| S1 | 0.5679 | 0.5725 | 0.5714 | **0.5456** | 0.5558 | D |
| S2 | 0.5910 | 0.5998 | 0.5978 | **0.5635** | 0.5765 | D |
| S3 | 0.6463 | 0.6381 | 0.6461 | **0.6112** | 0.6197 | D |
| S4 | 0.6240 | 0.6284 | 0.6233 | **0.5925** | 0.6062 | D |

**All 7 targets: D (isotonic calibration) wins**

### Key Finding
**Isotonic calibration on OOF predictions consistently outperforms all alternatives.** This is the single most impactful finding of the experiment suite.

---

## V49: MI-Based Feature Selection

### Hypothesis
Mutual Information captures non-linear feature-target relationships independently of the model. MI-based feature selection + correlation pruning might find better feature subsets than LGBM importance alone.

### Method
4 ranking strategies:
1. **LGBM importance** (baseline)
2. **MI only** (mutual_info_classif, n_neighbors=10)
3. **Rank fusion** (average of MI rank + LGBM rank)
4. **Score fusion** (normalized MI score + inverse LGBM rank)

Each tested at n=5,10,15,20 features. Also correlated feature pruning (|corr|>0.95).

### Results

| Target | LGBM best | MI best | Rank Fusion best | Score Fusion best | Overall best |
|--------|-----------|---------|-----------------|-------------------|--------------|
| Q1 | 0.6496 (n=15) | 0.6890 (n=20) | **0.6437 (n=5)** | 0.6644 (n=15) | Fusion n=5 |
| Q2 | **0.6153 (n=10)** | 0.6631 (n=20) | 0.6172 (n=15) | 0.6249 (n=10) | LGBM n=10 |
| Q3 | **0.6052 (n=10)** | 0.6675 (n=20) | 0.6545 (n=5) | 0.6383 (n=10) | LGBM n=10 |

### Key Findings
- **MI alone is worst of all methods** (avg ~0.67, vs LGBM ~0.64)
- **MI and LGBM have low overlap**: only 2-4 features in top-20 overlap across Q1-Q3
- Rank fusion slightly beats LGBM for Q1 but not enough to matter
- Feature selection alone doesn't explain the gap with V10

---

## V50: Isotonic Calibration + V37 Per-Target Configs

### Hypothesis
Combine the best finding from V48 (isotonic calibration) with V37-style per-target hyperparameters and optimize feature count per target.

### Method
- V37 hyperparameters per target (already known to work well)
- Isotonic calibration on OOF (from V48)
- Test n=5,10,15,20 features per target with isotonic cal
- Compare: no calibration vs isotonic cal vs mean-match cal

### Results (with isotonic cal)

| Target | Best Config | n_feat | iso_cal | no_cal | mm_cal |
|--------|------------|--------|---------|--------|--------|
| Q1 | iso_cal_n20 | 20 | **0.6143** | 0.6369 | 0.6358 |
| Q2 | iso_cal_n15 | 15 | **0.5715** | 0.6034 | 0.6033 |
| Q3 | iso_cal_n10 | 10 | **0.5882** | 0.6151 | 0.6112 |
| S1 | iso_cal_n5 | 5 | **0.5857** | 0.6127 | 0.6095 |
| S2 | iso_cal_n5 | 5 | **0.6025** | 0.6418 | 0.6349 |
| S3 | iso_cal_n5 | 5 | **0.5868** | 0.6452 | 0.6420 |
| S4 | iso_cal_n15 | 15 | **0.5996** | 0.6372 | 0.6328 |

**V50 Avg: 0.5926**

### Why V50 < V48
V48 uses the same pipeline as V50 but with a slightly different baseline (V10-style configs instead of V37-style). The 0.0041 difference is within noise for 450 samples. Both are solid improvements over V10.

### Key Insight: Fewer features often better
For S1, S2, S3: 5 features beat 20 with isotonic calibration. This suggests the signal is sparse and simpler models generalize better.

---

## Conclusions & Recommendations

### What Worked
1. **Isotonic calibration** (V48) — the single best improvement. Always apply isotonic reg to OOF binary predictions.
2. **Per-target models** — still the right approach. Don't ensemble multiple targets.
3. **V37 hyperparams + isotonic** (V50) — comparable to V48.

### What Didn't Work
1. **Meta-learner stacking** (V47) — overfits badly with 450 samples. Don't train meta-learners on CV OOF.
2. **MI feature selection** (V49) — worse than LGBM importance. MI is not helpful here.
3. **Simple ensemble averaging** (V47) — too correlated.
4. **Yeo-Johnson target transform** (V48) — didn't help.

### Next Steps
1. **Submit V48 or V50** as the next baseline — both beat V10 by 0.011-0.015
2. **Try more seeds** — V48/V50 used 20 seeds, but OOF averaging may be noisy. Try 50+ seeds.
3. **Isotonic cal on stacked LGBM + CatBoost** (from V46) — combine diverse models + isotonic cal. The V46 CatBoost+LGBM ensemble + isotonic might beat all of these.
4. **Feature interaction** — only on top 5-10 features (not all 117, which causes OOM). Try polynomial interactions for the best features per target.
5. **Outlier handling** — check for extreme values in features that may skew predictions.

### Final Score Comparison
```
V10 (baseline):     0.6038
V48 (iso cal):      0.5885  ← WINNER, -0.0153
V50 (iso+V37):      0.5926  ← Runner-up, -0.0112
V47 (ensemble):     0.6093  ← Worse
V49 (MI select):    0.6201  ← Worst
```
