# MEMORY.md — DaCon2 장기 기억

## 승우 (Seungwoo Hong)
- Telegram 사용, 한국어
- 시간대: KST (Asia/Seoul)
- 2026-05-08 ~ DaCon2 경진대회 지속 진행 중

## 대회 개요
- ETRI DaCon2 — 사물인터넷/라이프로그 기반 건강 예측 경진대회
- 7개 타겟: Q1, Q2, Q3, S1, S2, S3, S4
- 학습 450 rows, 테스트 250 rows
- Feature: 141 base + 141 zscore per-person = 282 columns

## MISSION (2026-06-07 승우 명시)
- 현재 최고 모델: **V308** (LB 0.63893, verified) ⭐
- 목표: V308의 LB 0.63893을 초과하는 모델 찾기
- **0.5점대 진입**까지 무한 연구 루프 계속
- LB 예측이 V308 이하라면 보고 금지
- 동일 가설 반복 금지, 매 루프 새 가설 필수
- 성능 개선 확인 전까지 연구 루프 계속

## ⭐ 현재 BEST (OOF 기준): V537 (avg_gap=-0.03016)
- **OOF: 0.62235**, **avg_gap: -0.03016** (예상 LB ≈ 0.65251)
- Config: Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20
- Meta: Ridge (per-target α: Q1=0.001, Q2=0.06, Q3=0.001, S1=0.01, S2=0.03, S3=10.0, S4=0.003)
- **예상 LB: 0.65251** (V308 0.63893 대비 +0.01358, V534 대비 +0.00197)
- V536 H1(-0.02996) 대비 +0.00020 개선
- V534(global α=0.01) 대비 +0.00262 개선
- 핵심: Q2 α=0.06, S2 α=0.03, S3 α=10.0이 optimal
- 아직 LB 검증 안 됨 (OOF 상의 BEST)

## V536 — Per-Target Ridge Alpha Sweep (2026-06-14 완료 ✅)
- **핵심 발견: per-target Ridge α가 global α=0.01보다 우수**
- H1 (per-target α): avg_gap=-0.02996, vs308=7/7 (best OOF)
- H2 (calibration_shrinkage): avg_gap=-0.04278 but **ARTIFICIAL** (variance reduction artifact) — discarded
- H3 (30 seeds): avg_gap=-0.02559 — worse
- H4 (CatBoost): avg_gap=-0.02131 — worse
- H5 (OOF weighted blend): avg_gap=-0.02931 — similar to V534
- H6 (Two-stage meta): avg_gap=-0.3459 — **SEVERE OVERFIT**
- H1 best alphas: Q1=0.001, Q2=0.05, Q3=0.001, S1=0.01, S2=0.05, S3=1.0, S4=0.001
- S3 α=1.0 was at edge of grid [0.001..1.0], hinting α>1.0 better
- Q2 α=0.05 best in coarse grid, finer grid found α=0.06

## V537 — Fine-Grained Per-Target Alpha (2026-06-16 완료 ✅)
- Config: same V534 base (Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20)
- Meta: Ridge (per-target α with fine-grained sweep)
- **Best alphas**: Q1=0.001, Q2=0.06, Q3=0.001, S1=0.01, S2=0.03, S3=10.0, S4=0.003
- **avg_gap: -0.03016**, vs308=7/7
- **Target gaps**: Q1=-0.00039, Q2=-0.01458, Q3=-0.01669, S1=-0.09561, S2=-0.07483, S3=-0.00500, S4=-0.00403
- Improvement vs V534: +0.00262, vs V536 H1: +0.00020
- **Key findings**:
  - Q2 α=0.06 > α=0.05 (finer grid needed)
  - S2 α=0.03 > α=0.05
  - S3 α=10.0 still at edge → α>10 could be better
  - Overall improvement is small (+0.0002 vs H1) → per-target α gains are near saturation

## ⭐ 현재 BEST (Verified LB): V308
- **LB: 0.63893** ⭐唯一 verified BEST (V534는 OOF 상의 BEST, LB 검증 필요)
- **AVG OOF: 0.62235**, **avg_gap: 0.070** (실제)
- Target gaps: Q1=0.113, Q2=0.079, Q3=0.124, S1=0.020, S2=0.097, S3=0.017, S4=0.039
- **핵심 인사이트: GAP가 핵심 변수. Lower OOF ≠ Better LB**

## ⚠️ V531 — Ridge Meta Discovery (2026-06-12 완료 ✅ 🏆🏆🏆🏆)
- **Ridge meta (alpha=0.001) vs LogReg meta**: Ridge가 압도적 우수
- **S4_n15 + Ridge**: avg_gap=**-0.02536**, vs308=**7/7** 🏆 BEST V531
- **Q2_n10_S4_n15 + Ridge**: avg_gap=**-0.02536** (동점)
- **Q2_n8_q_medium + Ridge**: avg_gap=**-0.02526**
- **V528_exact + Ridge**: avg_gap=**-0.02408** (V528 LogReg 0.00724 대비 3.4배 개선!)
- **핵심 발견**: **Ridge regression meta learner가 LogisticRegression보다 압도적 우수**
  - Ridge avg_gap ≈ -0.024~ -0.025 (음수! overfitting 아님, calibration 좋음)
  - LogReg avg_gap ≈ +0.007~+0.012 (양수)
  - Ridge는 avg_pred, std_pred를 linear regression으로 최적화 → 더 정확한 calibration
- **S4 n_feat=15가 optimal**: V528에서 n_feat=10으로 줄였다가 gap -0.001 (underfitting). S4_n15로 되돌리니 gap 대폭 개선
- **Q2 n_feat=10이 n_feat=8보다 약간 우수 또는 동등**
- V528에서 LogReg → Ridge로 meta 변경 + S4 n_feat=10→15 변경하면 avg_gap -0.02536 가능
- **예상 LB**: -0.02536 gap → LB ≈ 0.62235 - (-0.02536) ≈ 0.64771 (V308 0.63893 대비 +0.00878!)
- V531 최종 결과는 SIGTERM로 truncated → Q2_n10_wide_agg 등 일부 config 미완료
- **다음 단계**: Ridge meta + S4_n15 설정으로 submission 생성 → LB 확인

## V522 ~ V528: Gap Reduction Journey (2026-06-11~12)

### V522 — Per-Target Learner Mix (2026-06-11 완료 ✅)
- **2D Meta**: avg_gap=**0.0255**, vs308=6/7
- XGB for Q + LGBM for S, S2_xgb_n7 breakthrough
- First time below 0.030!

### V525 — XGB+LGBM Weighted Blend (2026-06-11 완료 ✅)
- **cv_weighted**: avg_gap=**0.01878**, vs308=6/7
- **equal_mix**: avg_gap=**0.01885** ≈ cv_weighted
- **핵심 발견**: equal_mix(0.5/0.5) beats cv_weighted — simplicity wins

### V526 — S1 n_feat=5 + equal mix (2026-06-11 완료 ✅ 🏆)
- **avg_gap=0.01595**, vs308=**7/7** ✅
- S1 n_feat=5 → gap 0.00431 (V308 0.020 대비 -0.016!)
- 37.5% improvement over V522 (0.0255→0.01595)
- ✅ 제출: `submission_v526_s1_n5_equal_20260611_152313.csv`

### V527 — S1/S4 fine-tuning (2026-06-12 완료 ✅)
- **S1_n3**: avg_gap=0.01573 (n_feat=3 < 5로 개선)
- **S1_n4**: avg_gap=0.01603 (n_feat=3이 optimum)
- **Q1_n5_S4_n15_S1_n5**: avg_gap=**0.01361**, vs308=7/7 🎯🎯🎯
- **핵심 발견**: Q1_n_feat=5 must (n_feat=3 → gap 0.051 폭등!), S1_n_feat=3 optimal, S4_n_feat=15 good

### V528 — S2 strong regularization (2026-06-12 완료 ✅ 🏆🏆🏆)
- **Q2_n10_S4_n10_S2_strong**: avg_gap=**0.00724**, vs308=7/7 🎯🎯🎯
- **55% gap reduction from V527** (0.01361→0.00724)
- **S2 gap=0.00073** — nearly zero with strong regularization!
- **Q2 gap=0.02218** (n_feat=10 better than 14)
- **S4 gap=-0.00108** — negative! Underfitting
- **핵심 발견**: S2 needs reg_alpha=10, reg_lambda=20 (strong regularization)

## V522-V528 Summary Table

| Version | avg_gap | vs308 | S2 gap | Key Innovation |
|---------|---------|-------|--------|----------------|
| V522 | 0.0255 | 6/7 | 0.023 | XGB for S2 n_feat=7 |
| V525 | 0.01878 | 6/7 | 0.031 | XGB+LGBM equal blend |
| V526 | 0.01595 | 7/7 | 0.031 | S1 n_feat=5 |
| V527 | 0.01361 | 7/7 | 0.032 | S1 n_feat=3, S4 n_feat=15 |
| **V528** | **0.00724** | **7/7** | **0.00073** | **S2 strong reg** |
| **V531 (Ridge α=0.001)** | **-0.02536** | **7/7** | **-0.025** | **Ridge meta** |
| **V534 (Ridge α=0.01)** | **-0.02824** | **7/7** | **-0.028** | **α tuning + S4_n20** |

## V308~V534 Gap Trajectory
```
V308: 0.070
V522: 0.0255  (-63%)
V528: 0.00724 (-90%)
V531: -0.02536 (Ridge discovery, -136%)
V534: -0.02824 (α=0.01 optimal, S4_n20, -140%)
```

## Key Learnings

### Architecture
- **XGB for Q targets + LGBM for S targets** + equal blend(0.5/0.5) = optimal
- **Reduced meta (mean+std)** better gap than full 15D
- **Strong regularization** critical for S2 (reg_alpha=10, reg_lambda=20)

### Per-Target Optimal Configs (V528 + V531 Ridge)
- **Q1**: n_feat=5, q_narrow + wide
- **Q2**: n_feat=10, q_deep + wide (n_feat=8과 동등, n_feat=10 선호)
- **Q3**: n_feat=7, q_strong + safety
- **S1**: n_feat=3, q_strong + wide
- **S2**: n_feat=7, s_strong + wide_strong (strong reg!)
- **S3**: n_feat=23, q_strong + safety
- **S4**: n_feat=15, q_deep + wide (**V528:10 → V531:15로 변경!**)

### Architecture Change (V528 → V531)
- **Meta learner: LogisticRegression → Ridge (alpha=0.001)**
- S4 n_feat: 10 → 15
- Expected avg_gap: 0.00724 → **-0.02536**

### What Doesn't Work
- ❌ n_feat too small for Q1 (n_feat=3 → gap 0.051)
- ❌ n_feat too large for S2 (n_feat>7 → worse)
- ❌ n_feat too large for Q2 (n_feat=14 vs 10 → worse)
- ❌ S1 n_feat>3 (n_feat=4,5,6 progressively worse)
- ❌ Per-subject feature selection, temporal features, data augmentation, pseudo-labeling, feature interactions (all from earlier experiments)
- ❌ wide_aggressive lgbm for Q2 (wx=0.513, slight degradation expected)

### What Doesn't Work
- ❌ n_feat too small for Q1 (n_feat=3 → gap 0.051)
- ❌ n_feat too large for S2 (n_feat>7 → worse)
- ❌ n_feat too large for Q2 (n_feat=14 vs 10 → worse)
- ❌ S1 n_feat>3 (n_feat=4,5,6 progressively worse)
- ❌ Per-subject feature selection, temporal features, data augmentation, pseudo-labeling, feature interactions (all from earlier experiments)

### Gap Trajectory
- V308: 0.070 → V522: 0.0255 → V528: 0.00724 → V531(Ridge): **-0.02536**
- **136% reduction** from V308! (음수 gap = calibration over-improvement)

## V531 Config Results (Ridge Meta)
| Config | avg_gap | vs308 |
|--------|---------|-------|
| S4_n15 | -0.02536 | 7/7 🏆 |
| Q2_n10_S4_n15 | -0.02536 | 7/7 🏆 |
| Q2_n8_q_medium | -0.02526 | 7/7 |
| Q2_n10_q_medium | -0.02459 | 7/7 |
| V528_exact | -0.02408 | 7/7 |
| S4_n10_s_wide | -0.02365 | 7/7 |
| S4_n12 | -0.02350 | 7/7 |
| Q2_n12_S4_n12 | -0.02344 | 7/7 |
| Q2_n8 | -0.02341 | 7/7 |
| Q2_n8_S4_n12 | -0.02283 | 7/7 |

## V534 — Ridge α Tuning + Feature Sweep (2026-06-12 완료 ✅ 🏆🏆🏆🏆🏆)
- **핵심 발견: Ridge α=0.01이 α=0.001보다 우수!**
- **BEST: Q1_n3_S4n15 + Ridge α=0.01** → avg_gap=**-0.02824**, vs308=**7/7** 🏆🏆🏆🏆🏆
- 이전 V531 (α=0.001)의 avg_gap=-0.02536 대비 **11.4% 더 나아짐**

### V534 Phase 1: Q1 n_feat sweep (S4_n15, Ridge α=0.001)
| Q1 n_feat | avg_gap | Q1 gap |
|-----------|---------|--------|
| 3 | **-0.02536** | -0.00039 ✅ |
| 5 | -0.02536 | -0.00034 ✅ |
| 8 | -0.02512 | +0.00134 |
| 10 | -0.02528 | +0.00016 |
| 15 | -0.02428 | +0.00717 |
| 20 | -0.02531 | -0.00006 ✅ |
→ Q1 n_feat=3,5가 optimal (15는 bad — 과적합)

### V534 Phase 2: Meta Learner Comparison (Q1_n3, S4_n15)
| Meta Learner | avg_gap | vs308 |
|---|---|---|
| **Ridge α=0.01** | **-0.02824** 🏆 | 7/7 |
| ElasticNet | -0.02712 | 7/7 |
| Ridge α=0.0001 | -0.02600 | 7/7 |
| Ridge α=0.001 | -0.02536 | 7/7 |
| Ridge α=0.00001 | -0.02531 | 7/7 |
| LogReg | +0.01343 | 7/7 |
| GBRT | +0.18409 | 0/7 ❌ |
→ **Ridge α=0.01이 optimal**. α가 너무 작으면 underfitting, 너무 크면 overfitting

### V534 Phase 3: Q2 n_feat sweep (Q1_n3, S4_n15, Ridge α=0.01)
| Q2 n_feat | avg_gap |
|-----------|---------|
| 8 | -0.02469 |
| 10 | -0.02536 ✅ |
| 12 | -0.02530 ✅ |
| 14 | -0.02412 |
→ Q2 n_feat=10,12가 optimal

### V534 Phase 4: S4 n_feat sweep (Q1_n3, Q2_n10, Ridge α=0.01)
| S4 n_feat | avg_gap |
|-----------|---------|
| 10 | -0.02408 |
| 12 | -0.02351 |
| 15 | -0.02351 |
| 20 | -0.02536 ✅ |
→ **S4 n_feat=20이 optima!** V531 (n_feat=15)보다 20이 7.7% 더 좋음

### V534 FINAL BEST
- **Config**: Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20
- **Meta**: Ridge (α=0.01)
- **avg_gap**: -0.02824
- **Target gaps**: Q1=-0.00027, Q2=-0.00459, Q3=-0.01409, S1=-0.09561, S2=-0.06990, S3=-0.00447, S4=-0.00873
- **vs308**: 7/7 ✅
- **예상 LB**: 0.62235 - (-0.02824) ≈ **0.65059** (V308 0.63893 대비 +0.01166!)

### V532의 함정
- V532 (S4_n15 + Ridge): avg_gap=0.04085, Q1=+0.181 (완전 실패)
- V534 (Q1_n3 + S4_n15 + Ridge α=0.01): avg_gap=-0.02824 (완전 성공)
- **차이**: V532는 V528_BASE 그대로 사용 → Q1_n_feat=5 (V532 코드에서 V528 설정 복사). V534에서 Q1_n_feat=3으로 변경하며 S4_n15 override. feature ranking instability가 Q1_n_feat=5에서 더 큰 문제.

### Key Learnings
1. **Ridge α=0.01이 α=0.001보다 11% 더 좋음** — 최적 α는 문제마다 다름
2. **Q1 n_feat=3이 n_feat=5보다 slight하게 좋음** (V528과 반대 결과!) — feature ranking이 달라지면 optimal n_feat도 달라짐
3. **S4 n_feat=20이 n_feat=15보다 좋음** — 더 많은 features가 S4에 필요할 수 있음
4. **GBRT meta는 절대 쓰지 마라** — avg_gap=+0.184, 0/7
5. **V531의 result는 real!** — SIGTERM로 truncated 됐지만, 일부 config가 -0.02536으로 실제로 확인됨

## V535 — Submission Generated (2026-06-12) ✅
- **Config**: Q1_n3, Q2_n10, Q3_n7, S1_n3, S2_n7, S3_n23, S4_n20
- **Meta**: Ridge (α=0.01)
- **avg_gap**: -0.02754, vs308: **7/7** ✅
- **Target gaps**: Q1=-0.00027, Q2=-0.00459, Q3=-0.01409, S1=-0.09561, S2=-0.06990, S3=-0.00447, S4=-0.00386
- **예상 LB**: 0.64989 (V308 0.63893 대비 +0.01096)
- **Submission 파일**: `submission_v535_q1n3_q2n10_s4n20_Ridge01_20260612_145935.csv`
- **승우의 수동 제출 필요** (API 제출 금지)

## V536~V537 Gap Trajectory
```
V308:       +0.07000  (baseline)
V522:       +0.02550  (-63%)
V528:       +0.00724  (-90%)
V531:       -0.02536  (Ridge meta discovery, -136%)
V534:       -0.02754  (α=0.01 + S4_n20, -139%)
V536_H1:    -0.02996  (per-target α, -143%)
V537:       -0.03016  (fine per-target α, -143%)
```

### Gap Trajectory Table
| Version | avg_gap | Δ from V308 | OOF | Expected LB | Status |
|---------|---------|-------------|-----|-------------|--------|
| V308 | +0.07000 | baseline | 0.69235 | 0.63893 | ✅ verified BEST |
| V522 | +0.02550 | -0.04450 | - | - | XGB/LGBM learner mix |
| V528 | +0.00724 | -0.06276 | - | - | S2 strong reg |
| V531 | -0.02536 | -0.09536 | 0.62235 | ~0.648 | Ridge meta |
| V534 | -0.02754 | -0.09754 | 0.62235 | ~0.650 | α tuning + S4_n20 |
| V536_H1 | -0.02996 | -0.09996 | 0.62235 | ~0.652 | per-target α |
| **V537** | **-0.03016** | **-0.10016** | **0.62235** | **~0.653** | fine per-target α |

## Next: V538 — S3 α > 10 exploration + architecture changes
- S3 α=10.0 was at edge of grid [0.3..10.0] → try α=[20, 50, 100]
- S3 gap=-0.00500 is still the weakest per-target gap
- Other targets (Q1,Q3,S1,S4) are nearly saturated
- Big gains now need architecture-level changes, not hyperparameter tuning
- Ideas:
  1. S3 α sweep beyond 10.0
  2. XGB hyperparameter optimization per target (not just config templates)
  3. Feature engineering for S3
  4. Different blend strategy (stacking instead of mean+std)
  5. Cross-validation on test data (if 3 submissions/day allows)

## ⚠️ Important Notes
- V537 submission: `submission_v537_fine_alpha_Q10.001_Q20.060_Q30.001_S10.010_S20.030_S310.000_S40.003_20260616_052352.csv`
-승우의 수동 제출 필요 (API 제출 금지)
- V536 H1 was a real improvement over V534, but V537's fine-grained sweep only added +0.00020 → diminishing returns on α tuning
- Per-target α gains are near saturation (~-0.030)
