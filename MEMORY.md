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

## ⭐ 현재 BEST: V308 (Verified LB)
- **LB: 0.63893** ⭐唯一 verified BEST
- **AVG OOF: 0.62235**, **avg_gap: 0.070** (실제)
- Target gaps: Q1=0.113, Q2=0.079, Q3=0.124, S1=0.020, S2=0.097, S3=0.017, S4=0.039
- **핵심 인사이트: GAP가 핵심 변수. Lower OOF ≠ Better LB**

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

## Key Learnings

### Architecture
- **XGB for Q targets + LGBM for S targets** + equal blend(0.5/0.5) = optimal
- **Reduced meta (mean+std)** better gap than full 15D
- **Strong regularization** critical for S2 (reg_alpha=10, reg_lambda=20)

### Per-Target Optimal Configs (V528 Best)
- **Q1**: n_feat=5, q_narrow + wide
- **Q2**: n_feat=10, q_deep + wide
- **Q3**: n_feat=7, q_strong + safety
- **S1**: n_feat=3, q_strong + wide
- **S2**: n_feat=7, s_strong + wide_strong (strong reg!)
- **S3**: n_feat=23, q_strong + safety
- **S4**: n_feat=10, q_deep + wide

### What Doesn't Work
- ❌ n_feat too small for Q1 (n_feat=3 → gap 0.051)
- ❌ n_feat too large for S2 (n_feat>7 → worse)
- ❌ n_feat too large for Q2 (n_feat=14 vs 10 → worse)
- ❌ S1 n_feat>3 (n_feat=4,5,6 progressively worse)
- ❌ Per-subject feature selection, temporal features, data augmentation, pseudo-labeling, feature interactions (all from earlier experiments)

### Gap Trajectory
- V308: 0.070 → V522: 0.0255 → V528: **0.00724**
- **90% reduction** from V308!
