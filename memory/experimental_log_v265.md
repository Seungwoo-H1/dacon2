
# DACon2 Research Log - V265 (2026-05-14)

## Root-Cause Analysis: OOF/LB Mismatch

### Key Findings

1. **OOF IS TRULY OUT-OF-FOLD**: Each row predicted exactly once → not in-fold biased
2. **Gap ≈ 0.11 is STRUCTURAL**: train-test drift (PSI=0.44) + different test distribution
3. **V127 undisputed best** (LB=0.648, OOF=0.537) — no hidden strong candidate
4. **V45/V46 leaked** (accuracy 100%/99.6%)
5. **S1 highest fold variance** (std=0.111) — hardest target to validate
6. **Ensemble diversity LOW** (avg corr 0.785) — hard to improve via ensembling
7. **Best gap model**: LB ≈ OOF + 0.105

### Gap Model Evaluation (3 known-LB anchors)

| Strategy | V127 Err | V53 Err | V260 Err | Total |
|----------|----------|---------|----------|-------|
| OOF + 0.105 | 0.00532 | 0.00065 | 0.04691 | 0.05288 ✅ |
| OOF + 0.110 | 0.00032 | 0.00435 | 0.05191 | 0.05658 |
| OOF × 1.205 | 0.00017 | 0.00668 | 0.07649 | 0.08334 |

### Hidden Candidate Ranking

| Rank | Version | OOF | Est LB | Known LB | Status |
|------|---------|------|--------|----------|--------|
| 1 | V83 | 0.54575 | 0.658 | — | Closest |
| 2 | V115 | 0.54759 | 0.660 | — | Isotonic |
| 3 | V53 | 0.54793 | 0.660 | 0.654 | ✅ |
| 4 | V121 | 0.54817 | 0.661 | — | Pairwise |

### PSI Analysis

| PSI Level | Count | Features |
|-----------|-------|----------|
| High (>0.25) | 4 | wHr_hr_std(12.88), wHr_hr_count(10.83), wHr_hr_mean(0.83), mGps_gps_count_mean(0.29) |
| Moderate (0.1-0.25) | 17 | mGps_gps_count_std, mWifi_wifi_avg_rssi_max, ... |
| Low (<0.1) | 53 | Most features |

**Key insight**: PSI features are predictive — removing them makes OOF worse.
The gap is structural, not fixable by feature removal alone.

### Temperature Scaling Experiment

Best T on OOF: **0.86**
- V127 OOF: 0.53731
- V266 OOF: 0.53517
- Δ OOF: -0.00214

T<1 sharpens predictions (pushes toward 0 or 1).
Expected LB improvement: +0.007 (V266 est LB ≈ 0.640 vs V127's 0.648)

### Recommendations

1. **Submit V266** (T=0.86 temperature scaling) — test on actual LB
2. **Submit V83** — closest competitor, needs LB verification
3. **Adversarial validation** for next iteration
4. **Truly diverse models** — non-tree methods

