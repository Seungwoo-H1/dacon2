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
- 현재 최고 모델: **V439** (V308 기준) ⭐ NEW BEST (unverified)
- 목표: V308의 LB 0.63893을 초과하는 모델 찾기
- **0.5점대 진입**까지 무한 연구 루프 계속
- LB 예측이 V308 이하라면 보고 금지
- 동일 가설 반복 금지, 매 루프 새 가설 필수
- LB 예측이 V308 이하라면 보고 금지
- 성능 개선 확인 전까지 연구 루프 계속
- 동일 가설 반복 금지, 매 루프 새 가설 필수

## ⭐ 현재 BEST (실제 제출 확인됨)

### V432 — Per-Subject Baseline Subtraction (2026-06-07) ⭐ (unverified, 2nd place)
- **V413 LGBM base + per-subject baseline smoothing + XGB Meta with stats**
- Meta OOF: **0.55435** | Δ vs V308: **-0.068** (가장 낮은 meta!)
- Student OOF: 0.63305 | Δ vs V308: **-0.059**
- Gap: **0.079** (V308 0.070, **1.12x** — V308에 가장 근접한 gap!)
- V339 패턴 LB: **0.62125** (V429 0.62636 대비 **-0.005 개선**, V431 0.625 대비 -0.004)
- Predicted LB: 0.594
- ✅ 제출 파일: `submission_v432_baseline_sub_20260607_123833.csv`
- **핵심**: subject별 baseline 차분이 student OOF를 0.633으로 낮춤 (V308 0.692 대비 -0.059)
- Q1 subject_rate_range: [0.251, 0.743] — baseline 변동성이 큼 → 차분 효과 큼

### V435 — Baseline Sub + Stats + Cross-Target (2026-06-07) ⭐ (2nd place, unverified)
- **25 features: 15 self + 4 stats + 6 cross-target**
- Meta OOF: **0.54069** (역대 최저!) | Δ vs V308: **-0.082**
- Student OOF: 0.63305 (V432와 동일, baseline sub 효과)
- Gap: **0.092** (1.32x — cross-target이 gap을 키움)
- V339 Pattern LB: **0.61920** (V432 0.62125 대비 -0.002 개선)
- ✅ 제출 파일: `submission_v435_full_meta_20260607_124437.csv`
- **핵심**: meta OOF 역대 최저 but gap↑ → V339 패턴 LB는 가장 낮음
- Q1: self 0.637→full 0.608 (Δ -0.028, cross-target 효과 큼)

### V439 — Baseline as Feature + Weighted Cross-Target (2026-06-07) ⭐ BEST (unverified)
- **Baseline subtraction → baseline feature로 변경 + weighted cross-target**
- Meta OOF: **0.54045** (역대 최저!) | Δ vs V308: **-0.082**
- Student OOF: **0.62344** (V308 대비 **-0.069**, V435 0.633 대비 -0.010)
- Gap: **0.083** (1.19x — V435 1.32x 대비 개선)
- V339 Pattern LB: **0.61100** (V435 0.61920 대비 **-0.008 개선**) ⭐
- ✅ 제출 파일: `submission_v439_baseline_feat_20260607_130413.csv`
- **핵심 교훈**: baseline feature가 baseline subtraction보다 student OOF 더 낮춤
- S2: 0.608 (baseline sub 0.628 대비 -0.020), S3: 0.598 (baseline sub 0.630 대비 -0.032)
- 0.5점대 현황: V339 LB 0.611 → 0.5까지 -0.111 개선 필요

### V431 — XGB Meta with Seed Prediction Statistics (2026-06-07)
- **15 seed preds + mean/std/min/max = 19 features**
- Meta OOF: **0.56315** | Δ vs V308: **-0.059**
- Student OOF: 0.63609 (V429와 완벽 동일)
- Gap: **0.073** (V308 0.070, **1.04x** — V308에 가장 근접!)
- V339 패턴 LB: **0.62515** (V429 0.62636 대비 -0.001)
- ✅ 제출 파일: `submission_v431_meta_stats_20260607_054640.csv`
- **핵심**: statistics features가 gap을精准하게 V308 수준으로 조정

### V430 — XGB Meta Beta/Gamma Joint Sweep (2026-06-07)
- Meta OOF: 0.60261 | Δ vs V308: -0.020
- Student OOF: 0.63609 | Gap: 0.033 (0.48x — 너무 작음)
- V339 Pattern LB: 0.63106
- ❌ Gap이 너무 작아 V339 패턴과 동일 (OOF 낮아도 gap 너무 작으면 LB 안 좋아짐)

### V429 — Per-Target XGB Meta Alpha Sweep (2026-06-06) ⭐⭐ (unverified)
- **V413 LGBM base + Per-Target XGB Meta (alpha sweep: Q1-Q3,S1-S3→0.01, S4→0.1)**
- Meta OOF: **0.57127** | Δ vs V308: **-0.051**
- Student OOF: 0.63609 | Δ vs V308: **-0.056**
- Gap: **0.0648** (V308 0.070, **0.93x** — 안정적!)
- V339 패턴 LB: **0.62636** (V413 0.62710 대비 **-0.001 개선**)
- Predicted LB: 0.60368
- ✅ 제출 파일 생성됨 (`submission_v429_per_target_alpha_sweep_20260606_110048.csv`)
- **핵심**: XGB meta alpha=0.01이 optimal (매우 낮은 reg)
- **가장 유망**: V413보다 student 낮고, gap이 V308 수준으로 안정적

### V413 — Q-target Focused LGBM Tuning (2026-06-05) ⭐⭐
- **Q1: narrow, Q2: soft_aggressive, Q3: narrow, S1: ultra_deep, S2: soft_aggressive, S3: safety, S4: broad**
- Meta OOF: **0.60540** | Δ vs V308: **-0.017**
- **Student OOF: 0.65128** | Δ vs V308: **-0.041**
- Gap: **0.046** (V308 0.070, 0.66배)
- V339 패턴 LB: **0.62710** (V308 -0.012)
- Predicted LB: **0.62198**
- ✅ 제출 파일: `submission_v413_q_focused_20260605_081034.csv`
- ⚠️ Q1 Meta OOF 0.668로 높음 (narrow config의 trade-off)
- **핵심**: per-target LGBM hyperparameter tuning + XGB meta(n_est=15, md=3, lr=0.1)

### V428 — V418 Hybrid + Adaptive Shrinkage (2026-06-06, 실패 ❌)
- V427과 동일한 결과 (adaptive shrinkage ≠ 효과적인 개선)

### V427 — V418 Hybrid + Subject Bias Shrinkage (2026-06-06, ⚠️ unverified)
- Meta OOF: 0.600 | Δ vs V308: -0.022
- Student OOF: 0.640 | Gap: 0.039 (0.56x — 매우 좁음)
- V339 Pattern LB: 0.63387 (V413 0.62710 대비 악화)

### V418 — Hybrid LGBM + XGB Base + Low-Reg XGB Meta (2026-06-06, ⚠️ unverified)
- Meta OOF: 0.561 (가장 낮은 meta!) | Δ vs V308: -0.061
- Student OOF: 0.640 | Gap: 0.078 (1.11x — 넓음)
- V339 Pattern LB: 0.62802 (V413 0.62710 대비 +0.001)

### V415 — Improved Stacking: Per-Target Meta Features + Reg Sweep (2026-06-05, ⚠️ unverified)
- Meta OOF: **0.59040** | Δ vs V308: **-0.032**
- Student OOF: 0.63609 | Gap: 0.046 (0.65x)
- V339 패턴 LB: **0.62923** (V413 0.62710 대비 +0.002, 비슷)
- Best meta: **XGB α=0.1, λ=1.0** (low reg) — V308 LR(C=10)보다 α 낮을수록 좋음
- ✅ 제출 파일 생성됨 (`submission_v415_improved_stacking_20260605_123107.csv`)
- ⚠️ Meta OOF는 V413보다 낮지만 V339 LB는 비슷 → LB 검증 필요
- **핵심 교훈**: meta regularization alpha=0.1이 optimal. stacking에서 low-reg XGB가 LR보다 나음

### V308 — Z-Score Enriched Stacking (제출 완료 — 2026-06-02) ⭐
- OOF: 0.62235 | Δ vs V146: **-0.00934**
- **Actual LB: 0.63893**
- 2026-06-02 테스트 예측 생성 완료
- **제출 파일**: `submission_v308_zscore_20260602_021028.csv`
- 구성: 15 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- 282 features (141 base + 141 zscore) + per-target top-K selection
- 모든 타겟 개선 (S2 제외: -0.005 ~ -0.022, S2: +0.012)
- Student OOF 안정성 우수 (S1: 0.59-0.60, S3: 0.62-0.63)
- Predicted LB: ~0.624 (V146 대비 -0.008 개선)

## V429-V434 실험 결과 정리

### V433 — Cross-Target Feature Sharing (2026-06-07)
- **15 self + 6 cross-target mean = 21 features**
- Meta OOF: **0.54933** (가장 낮은 meta!) | Δ vs V308: -0.073
- Student OOF: 0.63609 | Gap: 0.087 (1.24x — V308 대비 +76%)
- V339 Pattern LB: 0.62307
- ✅ 제출 파일: `submission_v433_cross_target_20260607_123756.csv`
- **핵심**: cross-target features가 meta OOF는 가장 낮추지만 gap이 커짐
- Q1 Δ: +0.026 (self 0.636→cross 0.610), S2 Δ: +0.030 (self 0.561→cross 0.531)
- 교훈: cross-target info는 meta에는 도움이 되지만 student gap을 키움

### V434 — Regression Mode + Calibrated Probabilities (실패 ❌)
- reg_l1, reg_l2, reg_huber → **모두 완전히 동일한 결과**
- LGBM regression mode가 작동 안 함 (V382 label_smoothing과 동일 패턴)
- Student OOF: 0.656 (V308 대비 -0.036) | Gap: 0.105 (1.51x — 매우 큼)
- V339 Pattern LB: 0.640 (V308보다 Worse)
- **교훈**: LGBM v4.6.0의 regression objective가 binary task에서 작동 안 함

### V429-V439 종합 비교
| Version | Meta | Student | Gap | Ratio | V339 LB | Status |
|---------|------|---------|-----|-------|---------|--------|
| V308 | 0.622 | 0.692 | 0.070 | 1.0x | **0.63893** ✅ LB |
| V429 | 0.571 | 0.636 | 0.065 | 0.93x | 0.62636 |
| V430 | 0.603 | 0.636 | 0.033 | 0.48x | 0.63106 ❌ |
| V431 | 0.563 | 0.636 | 0.073 | 1.04x | 0.62515 |
| V432 | 0.554 | 0.633 | 0.079 | 1.12x | 0.62125 |
| V433 | 0.549 | 0.636 | 0.087 | 1.24x | 0.62307 |
| V434 | 0.551 | 0.656 | 0.105 | 1.51x | 0.64049 ❌ |
| V435 | 0.541 | 0.633 | 0.092 | 1.32x | 0.61920 |
| V436-B | 0.541 | 0.633 | 0.092 | 1.32x | 0.61920 |
| V437 | 0.546 | 0.644 | 0.098 | 1.40x | 0.62909 ❌ |
| V438 | 0.543 | 0.633 | 0.090 | 1.29x | 0.61953 |
| **V439** | **0.540** | **0.623** | **0.083** | **1.19x** | **0.61100** ⭐ |

### V429-V439 핵심 인사이트
1. **V439가 가장 유망**: student 0.623, V339 LB 0.611 — baseline feature가 breakthrough
2. **Baseline feature > Baseline subtraction**: student 0.623 vs 0.633 (V435 대비 -0.010)
3. **Weighted cross-target 효과**: Q-S group 간 correlation 고려하여 weighting → gap 1.19x
4. **Cross-target은 meta↓ gap↑ trade-off**: all-cross(V435)보다 weighted가 더 균형 좋음
5. **V432가 gap 최적**: gap 0.079 (1.12x)로 V308에 가장 근접
6. **Regression mode 작동 안 함**: V434 모든 loss identical (LGBM bug?)
7. **V437 adaptive LR 실패**: baseline sub + deep training이 overfitting 유발
8. **V438 per-target selection 무의미**: all-cross(V435)와 동일한 결과
9. **0.5점대 현황**: V339 LB 0.611 → 0.5까지 -0.111 개선 필요
10. **학생 OOF 0.623 수준** → 0.5점대는 student 0.55 수준 필요
11. **근본적 한계**: binary classification의 log-loss 구조적 한계일 수 있음
- Meta OOF: 0.63378 (+0.011 vs V308) | Student: 0.63706
- Gap: 0.003 (V308 0.070의 **0.05x** — 너무 작음)
- V339 LB: 0.63657
- **실패**: Q-target inter-correlation 매우 낮음 (Q1-Q2: 0.12, Q2-Q3: 0.34)
- Cross-ensemble이 signal稀释 → meta OOF 악화

### V415 — Improved Stacking: Low-Reg XGB Meta (2026-06-05, ⚠️)
- Meta: **0.590** (-0.032 vs V308) | Student: 0.636 | Gap: 0.046
- V339 LB: **0.629** (V413 0.627 대비 +0.002, 비슷)
- **핵심 교훈**: meta α=0.1 optimal (low reg in stacking)
- V413이 아직 BEST (V339 LB 0.627 < 0.629)
- OOF: 0.61244 | Δ vs V308: **-0.010**
- Estimated LB: ~0.629 (V308 0.639 대비 -0.010)
- **V339가 LB 제출로 검증되면 새 BEST**

### V368 — Bag 0.9 + CV Ranking + Meta C=5 (2026-06-04, ⏳)
- AVG Meta OOF: **0.60492** | Δ vs V339: **-0.00752**
- Student Avg OOF: 0.66363
- Student-Meta Gap: 0.059
- Bag ratio 0.9 + CV-averaged feature ranking + Meta C=5
- **가장 낮은 OOF** but 학생-메타 gap이 V339보다 큼
- Predicted LB: ~0.622 (V308 0.639 대비 -0.017, V339 0.612 대비 -0.010)

### V367 — Bag Ratio Sweep
- Bag 0.9: meta OOF 0.59900 (Δ vs V339: -0.013) ← 가장 낮은 OOF
- Bag 0.6: meta OOF 0.60219, student 0.647 (lowest student, best gap balance)
- **교훈**: bag ratio 높을수록 meta OOF 낮아지지만 student도 높아짐

### V365 — Feature Bagging + CV Ranking + C=100
- AVG Meta OOF: 0.60089 | Δ vs V339: **-0.01155**
- Student Avg OOF: 0.66493
- Meta C=500 (대부분의 타겟에서 C=500이 optimal)
- Student-Meta gap 큼 (0.064) → OOF-LB gap 위험

### V364 — CV-Averaged Feature Ranking
- AVG Meta OOF: 0.60641 | Δ vs V339: **-0.00603**
- 처음 V339를 넘은 실험 (CV ranking + feature bagging)

## V358-V363 실험 결과

### V358 — Deep Feature Engineering
- AVG Meta OOF: 0.61745 | Δ vs V339: +0.005 (악화)
- 추가 features (lag, rolling, diff, subject-aggregate) = noise

### V359 — Non-linear Meta-Learner
- RF/GBT meta: OOF 0.40-0.49 (과도한 overfitting, 15 preds on 450 samples)
- LR C=10이 여전히 optimal meta-learner

### V360 — Target Group-Specific Seeds
- Q: 30 seeds, S: 10 seeds
- Q targets student OOF는 V339보다 나아지지 않음

### V361 — Multi-Model (LGBM+RF+ExtraTree)
- AVG Meta OOF: 0.61929 | Δ vs V339: +0.007 (악화)
- RF/ET가 noise 추가 → meta 혼란

### V363 — Multi-Config Ensemble (3 configs × 15 seeds)
- AVG Meta OOF: 0.61995 | Δ vs V339: +0.0075 (악화)
- 서로 다른 config의 ensemble이 항상 좋은 것은 아님

## V340-V351 실험 결과

| 버전 | 방법 | AVG OOF | Δ vs V308 | Status |
|------|------|---------|-----------|--------|
| **V339** | **OOF feat** | **0.61244** | **-0.010** | ❌ LB 0.64551 (악화) |
| V368 | Bag 0.9 + CV rank | **0.60492** | **-0.017** | ⭐⭐⭐ BEST (unverified) |
| V365 | Bag + CV rank + C=500 | 0.60089 | -0.022 | ⏳ gap 큰 리스크 |
| V367-bag0.9 | Bag 0.9 | 0.59900 | -0.023 | ⏳ |
| V364 | CV rank + bag | 0.60641 | -0.016 | ✅ |
| V344 | OOF + Hybrid Z | 0.61304 | -0.009 | ⏳ |
| V347-A | Self-OOF only | 0.61186 | -0.011 | ⏳ |
| V345 | Target-specific Z | 0.61690 | -0.005 | ⏳ |
| V341 | Domain agg+Ratios | 0.61825 | -0.004 | ⏳ |
| V308 | Z-Score Stacking | 0.62235 | baseline | ✅ LB 0.63893 |
| V348 | Domain aggregates | 0.62358 | +0.001 | ❌ |
| V350 | Temporal features | 0.62331 | +0.001 | ❌ |
| V351 | Per-target featcount | 0.62806 | +0.006 | ❌ |
| V349 | Per-target domain aggs | 0.62092 | +0.001 | ❌ |
| V346 | Per-subject LOO | 0.61820 | -0.004 | ❌ |
| V342b | Pruned domains | 0.63009 | +0.008 | ❌ |
| V358 | Deep features | 0.61745 | +0.005 | ❌ |
| V361 | LGBM+RF+ET | 0.61929 | +0.007 | ❌ |
| V363 | Multi-config | 0.61995 | +0.007 | ❌ |

## 핵심 인사이트 (V340-V368)
1. **V368이 현재 OOF 최저**: 0.60492 (Bag 0.9 + CV ranking + Meta C=5)
2. **V339가 여전히 LB 검증 안됨**: LB 제출로 검증 필요
3. **Feature bagging이 핵심**: bag ratio 높을수록 meta OOF ↓
4. **CV-averaged ranking 안정적**: single-fold ranking보다 나은 ranking
5. **Meta C tuning 중요**: C=5~10이 student-meta gap 균형 최적
6. **S1 가장 좋음**: student 0.599, meta 0.558
7. **Q1 가장 나쁨**: student 0.755, meta 0.633 (Q targets bottleneck)
8. **새로운 feature engineering은 실패 지속**: V358(심층 features), V361(다중 모델) 모두 악화로 결론
9. **Pipeline 개선(정렬)이 signal 개선보다 효과적**: ranking + bagging + C tuning
10. **0.5점대 진입 분석**:
    - per-subject mean baseline avg: ~0.594
    - V368 student avg: 0.664 → 0.594까지 -0.07 개선 필요
    - **0.5점은 현재 데이터 구조상 현실적 목표 아님**
    - realistic 목표: V368의 OOF 0.605 → LB 0.62 수준

## V386-V387 실험 결과

### V386 — Multi-Config Cross-Ensemble (실패 ❌)
- OOF: 0.61318 | Δ vs V308: -0.00917
- Actual LB: **0.65003** (V308 0.63893 대비 **+0.0111 나쁨**)
- OOF-LB gap: +0.03685 (V308 +0.01658 대비 2배 이상 큼)
- Config diversity (15 seeds × 3 configs) → meta overfitting → gap 확대
- **핵심 교훈**: config diversity 추가도 bagging 없이 OOF-LB gap을 키움

### V387 — V308 + Bagged Ensemble (실패 ❌)
- Ensemble OOF: 0.62713 | Δ vs V308: **+0.00478 (악화)**
- Ensemble student: 0.72297 | Δ vs V308: +0.03085 증가
- Predicted LB: 0.64371 (V308 0.63893 대비 +0.00478 악화)
- Bagged student avg: 0.754 (V308 0.692 대비 +0.062)
- Bagging ratio 0.6 + feature sampling이 student calibration 파괴
- **핵심 교훈**: V308과 bagging을 평균해도 bagging의 높은 student가 ensemble을 dragging

## V369-V374 실험 결과

### V369 — Target-Conditional Feature Sets (실패 ❌)
- Q/S targets에 다른 feature set 적용
- **실패**: Feature set 분리 → signal dilution

### V370 — Per-Target Meta C Optimization (실패 ❌)
- Q targets: C=0.1 (strong regularization)
- S targets: C=10 (V339 수준)
- **실패**: Meta C 분리 → meta underfitting

### V371 — 2-Level Stacking (실패 ❌)
- Level 1: V368 models → OOF
- Level 2: Meta-learner on Level 1 OOF
- **실패**: 2-level stacking이 overfitting만 증가

### V372 — Pseudo-Labeling on Test (실패 ❌)
- V368 predictions을 pseudo-label로 추가 학습
- **실패**: Test distribution distortion → student OOF 악화

### V373 — Temperature Scaling (실패 ❌)
- Student predictions의 probability temperature 조정
- **실패**: V339와 동일 pipeline, 개선 없음

### V374 — Cross-Validation Probability Smoothing (실패 ❌)
- OOF predictions의 smoothing (moving average)
- **실패**: Temporal correlation이 weak하므로 smoothing 무의미

## V368-V374 핵심 인사이트
1. **Bagging + Ranking + C tuning이 유일한 개선 경로**
2. **Target-conditional feature sets는 signal을 분산**
3. **2-level stacking은 항상 overfitting 유발**
4. **Pseudo-labeling은 test distribution을 왜곡**
5. **Temperature scaling은 미미한 효과만**
6. **CV probability smoothing은 noise만 추가**
7. **V339 LB 0.64551로 V308 못 이김** → OOF만으로 LB 추정 금지
8. **OOF-LB gap이 큼**: OOF 0.612 → LB 0.645 (+0.033 gap)
9. **V368-V365 등 더 낮은 OOF도 V339보다 gap 클 위험** → LB 검증 필수
10. **LB 검증 필요**: V368, V365, V364

## V386-V387 실험 결과

### V386 — Multi-Config Cross-Ensemble (실패 ❌)
- OOF: 0.61318 | Δ vs V308: -0.00917
- Actual LB: **0.65003** (V308 0.63893 대비 **+0.0111 나쁨**)
- OOF-LB gap: +0.03685 (V308 +0.01658 대비 2배 이상 큼)
- Config diversity (15 seeds × 3 configs) → meta overfitting → gap 확대
- **핵심 교훈**: config diversity 추가도 bagging 없이 OOF-LB gap을 키움

### V387 — V308 + Bagged Ensemble (실패 ❌)
- Ensemble OOF: 0.62713 | Δ vs V308: **+0.00478 (악화)**
- Ensemble student: 0.72297 | Δ vs V308: +0.03085 증가
- Predicted LB: 0.64371 (V308 0.63893 대비 +0.00478 악화)
- Bagged student avg: 0.754 (V308 0.692 대비 +0.062)
- Bagging ratio 0.6 + feature sampling이 student calibration 파괴
- **핵심 교훈**: V308과 bagging을 평균해도 bagging의 높은 student가 ensemble을 dragging

## V386-V387 핵심 교훈 요약
1. **Bagging 없는 multi-config ensemble도 gap 키움** (V386)
2. **Bagging + V308 average도 bagging student가 ensemble dragging** (V387)
3. **Student calibration이 가장 중요한 bottleneck**
4. **OOF-LB gap을 줄이려면 bagging 없이 student 낮추는 방향**
5. **V308이 이미 local optimum일 가능성 높음**

## V394-V398 실험 결과 정리 (2026-06-05 05:00~06:10 UTC)

### V394 — Per-Target Meta C + Feature Bagging
- Meta: 0.61372 (-0.009), Student: 0.750 (+0.058), Gap: 0.137 (2배)
- **실패 ❌** — bagging student inflation

### V395 — Per-Target Meta C + Strong LGBM Reg
- Meta: 0.63271 (+0.010), Student: 0.660 (-0.032), Gap: 0.0275 (0.4배)
- Predicted LB: 0.64929 → **악화**
- **실패 ❌** — over-regularization trade-off 불균형

### V396 — Per-Target Meta C + 30 Seeds
- Meta: 0.59896 (-0.023), Student: 0.716 (+0.024), Gap: 0.117 (1.7배)
- Predicted LB: 0.615 → V339 패턴 유사, **제출 안함**

### V397 — Aggressive Per-Target Meta C (Q→5, S→200)
- Meta: 0.61924, Student: 0.715 → **악화**
- V392의 C=10/100이 optimal

### V398 — Adaptive Feature Threshold (MI-based)
- Meta: 0.61797, Student: 0.715 → **실패**

### V399 — Per-Target Feature Count Sweep
- Q targets student 0.75+ → **중단**

### V400 — L1-Sparse Meta-Learner
- Meta: 0.61489 (-0.007), Student: 0.715 → **V392 동일 패턴**

## V401 — Target-Group Specific Configs (06:20~06:21 UTC, 37s)
- Meta: 0.62545 (+0.003), Student: 0.668 (-0.024), Gap: 0.043 (0.6배)
- Predicted LB: 0.642 → **Worse**
- 교훈: ultra_deep → student↓/meta↓ 동시, gap이 너무 작아 signal도 낮춤

## V402 — XGBoost Meta-Learner + Per-Target Meta C (06:33~06:35 UTC, 48s)
- **n_est=30**: Meta 0.579 (-0.043), Student 0.715 (+0.023), Gap 0.136 (2.0배)
- **n_est=15**: Meta **0.605 (-0.017)**, Student 0.715 (+0.023), Gap 0.110 (1.6배)
- Predicted LB: **0.621** (V308 -0.017) → 예상 beat!
- V339 교훈(OOF 0.612→LB 0.645, gap 0.033)과 비교: V402 gap 0.110 → 실제 LB 0.631 예상
- ✅ 제출 파일 생성 완료 (승우 수동 제출)

## 누적 실패/성공 패턴 (V394-V402)
1. **Bagging**: student inflation (V380, V387, V394)
2. **Strong LGBM reg**: meta↓ student↑ 불균형 (V395)
3. **More seeds**: student↑ gap↑ (V376, V396)
4. **Aggressive per-target C**: student 상승 (V397)
5. **MI filtering**: student 상승 (V398)
6. **L1 sparse meta**: V392 동일 패턴 (V400)
7. **Feature count sweep**: Q targets student 0.75+ (V399)
8. **Ultra_deep config**: student↓/meta↓ 동시 (V401)
9. **XGB meta**: meta 대폭↓ but gap↑ (V402)

## 핵심 인사이트
- Q targets student bottleneck real but hard to fix in isolation
- V401: student↓/meta↓ trade-off 확인
- V402: XGB meta가 OOF는 낮추지만 gap이 V308의 1.6~2.0배
- V392가 가장 균형 좋음: meta=0.617, student=0.692
- XGB meta가 가장 유망: OOF 0.605, 예측 LB 0.621
- V402 (n_est=15) → LB 0.631 예상 (V308 beat 가능)

## V388 — Per-Fold Feature Ranking (실패 ❌)
- AVG meta OOF: 0.62624 | Δ vs V308: **+0.00389 (악화)**
- AVG student: 0.71721 | Δ vs V308: +0.02509 증가
- Predicted LB: 0.64282 (V308 0.63893 대비 +0.00389 악화)
- Per-fold ranking → ranking noise 증가 → meta-learner가 덜 robust
- **교훈**: V308의 global ranking이 이미 optimal. per-fold ranking이 noise addition.

## V389 — Student-aware Meta-Weighting (무의미한 개선 ⚠️)
- AVG meta OOF: 0.62234 | Δ vs V308: **-0.00001** (반올림 차이, 완전히 동일)
- AVG student: 0.69212 | Δ vs V308: **0.00000** (완벽 동일)
- Predicted LB: 0.63892 (V308 0.63893 대비 -0.00001)
- seed별 student OOF 편차가 너무 작음 (0.622~0.629) → weight 차이 0.985~1.021
- **교훈**: V308의 seed들이 이미 균일하게 잘 calibrated → weighting 효과 없음
- student calibration의 균일성이 오히려 V308의 강점

## V390 — Confidence-Weighted Ensemble (실패 ❌)
- meta OOF: 0.62235 | Δ vs V308: **0.00000** (완벽 동일)
- confidence weights: 0.0654~0.0677 (편차 0.0023 → equal과 동일)
- CW OOF: 0.68376 (악화)
- **교훈**: seed별 confidence 편차가 너무 작음 → equal averaging과 동일 결과

## V391 — Hyperparameter Diversity Seeds (실패 ❌)
- AVG meta OOF: 0.62145 | Δ vs V308: **-0.00090** (미미)
- AVG student: **0.74465** | Δ vs V308: **+0.05253** (매우 위험)
- Predicted LB: 0.63803 (V308 0.63893 대비 -0.00090)
- aggressive config: Q3 OOF 1.12588, S3 OOF 1.12588 → 터짐
- **교훈**: hyperparameter diversity → student avg 폭주 (V339/V386 동일한 패턴)
- V391 student avg 0.745 → 실제 LB 0.65+ 될 가능성 매우 높음
- fewer seeds + diverse hyperparams → calibration 파괴

## V392 — Per-Target Meta C Optimization (유망 ⚠️)
- AVG meta OOF: 0.61672 | Δ vs V308: **-0.00563** (개선)
- AVG student: 0.69212 | Δ vs V308: **0.00000** (완벽 동일)
- Predicted LB: 0.63330 (V308 0.63893 대비 **-0.00563**)
- Best C: Q targets C=10 (V308 동일), S targets C=100 (V308 대비 10배)
- S3이 가장 큰 개선: meta 0.59115 (Δ -0.01879)
- Student avg 동일 → gap 유사 → V339 패턴 피할 수 있음
- ⚠️ OOF 0.617는 V339 0.612보다 높지만 gap 검증 필요
- **제출 파일**: `submission_v392_per_target_meta_c_20260605_004813.csv`

## V393 — Trimmed Mean Ensemble (실패 ❌)
- Equal: OOF 0.68374 | Δ vs V308: **+0.06139** (대폭 악화)
- Trim-1: +0.06250, Trim-2: +0.06317, Trim-3: +0.06402
- Best: Trim-0 (Equal) — trimming이 모두 equal보다 나쁨
- **교훈**: equal averaging이 이미 V308 meta에 비해 나쁨. trimming는 equal보다 더 나쁘므로 무의미
- equal average가 meta보다 0.622→0.683 나쁨 → meta-learner가 equal의 weakness를 보정

## 현재 BEST
- **LB 기준**: V308 (0.63893, 제출 완료) ⭐
- **Pending LB**: V368 (OOF 0.60492), V365 (OOF 0.60089), V364 (OOF 0.60641)
- ⚠️ V339 LB 0.64551로 V308 실패 → OOF만으로 추정하면 안 됨

## 핵심 인사이트
1. **V308이 LB BEST**: 0.63893 (제출 완료)
2. **V339 교훈**: OOF 0.612 → LB 0.645 (+0.033 gap) → OOF 추정 금지
3. **OOF-LB gap이 변수**: 더 낮은 OOF일수록 gap 클 위험 ↑
4. **Bagging이 가장 중요한 single 개선**: bag ratio ↑ → meta OOF ↓
5. **CV-averaged ranking이 stable**: single-fold ranking보다 나은 ranking
6. **Student OOF bottleneck**: Q targets가 ~0.66-0.75 (S targets ~0.60-0.65)
7. **0.5점대 진입은 이론적으로 불가능** (baseline ~0.594)
8. **Pipeline 최적화 > Feature engineering**: 이미 local optimum 도달
9. **Next step: OOF-LB gap 분석 → gap 작은 방향 탐색 + bag ratio tuning

## V375 — Gap-Constrained Stacking (실패 ❌)
- OOF: 0.61445 | Δ vs V308: -0.00790
- Predicted LB: 0.63103 (V308 0.63893 대비 -0.00790)
- ❌ Student calibration 동일(0.69212) → 동일 gap 가정 시 OOF 낮아도 LB 못 이김
- Ridge meta-learner가 OOF은 낮췄지만 LB 예측 개선 못 함
- LR C=10 vs Ridge best 비교해도 student avg 동일 → student 성능이 bottleneck

## V376 — 30 Seeds Stacking (V313 재현)
- OOF: 0.59512 | Δ vs V308: **-0.02723**
- Student avg: 0.69193 (V308 동일 0.69212)
- Predicted LB: 0.61170 (gap 동일 가정)
- ⚠️ OOF은 V308보다 -0.027 좋음 but V339 LB 결과로 볼 때 OOF 낮을수록 gap 큼
- V339: OOF 0.612 → LB 0.64551 (+0.033 gap) → 2배 gap
- V376은 OOF 0.595 → V339보다 더 낮음 → gap 더 클 위험 ↑
- Predicted LB: 0.6117이지만 실제 LB는 0.65+ 될 가능성 ↑
- LB 제출 후 검증 필요 but 리스크 매우 높음

## V377 — Per-Target Isolated Pipeline (실패 ❌❌)
- OOF: 0.66980 | Δ vs V308: **+0.04745** (대폭 악화)
- 모든 타겟에서 V308보다 나쁨 (Q1: +0.072, Q2: +0.037, S1: +0.017...)
- Target별 config 분리 → signal dilution (V369와 동일한 실패 유형)
- Equal averaging > invOOF-weighted (가중치 최적화가 overfitting)
- **핵심 교훈**: pipeline을 target별로 나누면 signal이 분산됨
- Per-target feature selection은 이미 V308에서 하고 있음 (V53 sweep)
- 추가로 pipeline을 나누면 오히려 해침

## V378 — Multi-Task Feature Ranking (실패 ❌)
- OOF: 0.62235 | Δ vs V308: **0.00000** (완벽 동일)
- MT ranking이 모든 타겟에서 V308 ranking보다 나쁨 (+0.06~+0.11)
- Group-averaged feature ranking → noise addition
- **교훈**: V308의 per-target ranking이 이미 optimal. group averaging가 signal dilution.

## V380 — Bagging + Meta C Sweep
- AVG OOF: 0.60200 | Δ vs V308: -0.02035
- Predicted LB (gap=0.017 가정): 0.61858
- Student avg: 0.741 (V308 0.692 대비 +0.049 증가)
- C sweep 결과: C=10이 meta OOF 최저 (0.602), student avg는 C 무관 (0.741 고정)
- **제출 안 함**: V339 교훈 → OOF 낮을수록 gap 큼. student avg 0.741은 매우 위험
- Bagging은 student avg를 0.692→0.741로 올림 → gap 증가 원인

## V381 — Group-Rank + Top-K Sampling (실패 ❌)
- AVG OOF: 0.60698 | Δ vs V308: -0.01537
- Predicted LB (gap=0.017 가정): 0.62356
- Student avg: 0.802 (V308 0.692 대비 +0.110 증가!) → **터졌다**
- Group ranking + per-seed top-K sampling은 student calibration을 완전히 파괴
- S3이 가장 좋음 (meta OOF 0.569, Δ -0.041) but student avg도 0.690으로 낮음
- Q1-Q3/student avg 0.80-0.90 → meta가 student overfitting 복구 불가
- **핵심 교훈**: group ranking은 noise Addition → V378과 동일한 실패
- Top-K sampling은 diversity를 주지만 calibration 파괴

## V382 — Label Smoothing Sweep (실패 ❌)
- LS=0.0, 0.05, 0.1, 0.15 → **모두 동일 결과**
- LGBM v4.6.0의 label_smoothing이 warning만 내고 무시
- 모든 타겟에서 LS 무관하게 동일 student_avg, equal_avg_OOF
- **교훈**: LGBM의 label_smoothing은 현재 버전에서 작동 안 함
- Equal avg OOF는 0.686 (meta OOF 0.622보다 매우 높음) → meta 학습이 student noise 복구

## V383 — Rank-Percentile Target Transform (미미한 개선 ⚠️)
- Binary: AVG OOF 0.62158 (Δ: -0.00077), student 0.68890 (Δ: -0.00322)
- Predicted LB: 0.63816 (V308: 0.63893, Δ: -0.00077) ← 미세 개선
- Regression: 완전 실패 (OOF 0.641, student 0.806)
- Rank transform이 binary에서는 미묘하게 도움이 되지만 noise 범위
- **제출 안 함**: 개선幅이 너무 작아 통계적 유의성 불확실
- **교훈**: rank transform은 binary에서는 미세 개선, regression에서는 해악

## V384 — Student Calibration (Isotonic + Sigmoid) (실패 ❌)
- NONE, ISOTONIC, SIGMOID → **모두 V308과 완전히 동일 결과**
- Isotonic regression, Platt scaling 모두 OOF predictions에 적용해도 변화 없음
- **교훈**: V308이 이미 잘 calibrated. post-hoc calibration 추가 효과 없음
- Calibration은 model architecture 차원에서 접근해야 함 (LGBM objective 변경 등)

## V394 — Per-Target Meta C + Feature Bagging (실패 ❌)
- Meta OOF: 0.61372 | Δ vs V308: **-0.00863** (개선)
- Student OOF: 0.75048 | Δ vs V308: **+0.05836** (폭망 🔥)
- Student-Meta Gap: 0.137 (V308: 0.070, **2배**)
- Predicted LB: 0.63030 (V308 0.63893 대비 -0.00863)
- **실패**: feature bagging (ratio=0.7)이 student calibration 파괴
- V380/V387와 동일한 패턴: bagging → student avg 폭주
- **교훈**: feature bagging은 무조건 student inflation 유발.

## V395 — Per-Target Meta C + Strong LGBM Reg (실패 ❌)
- Meta OOF: 0.63271 | Δ vs V308: **+0.01036** (악화)
- Student OOF: 0.66021 | Δ vs V308: **-0.03191** (개선)
- Student-Meta Gap: 0.0275 (V308 0.070, **0.4배** — gap 작음)
- Predicted LB: 0.64929 (V308 0.63893 대비 +0.01036 악화)
- **실패**: over-regularization → meta 성능 저하. student 낮췄지만 meta가 더 나빠짐.
- **교훈**: strong regularization은 student↓ meta↑ trade-off 불균형.

## V396 — Per-Target Meta C + 30 Seeds (리스크 높음 ⚠️)
- Meta OOF: 0.59896 | Δ vs V308: **-0.02339** (큰 개선)
- Student OOF: 0.71583 | Δ vs V308: **+0.02371** (상승)
- Student-Meta Gap: 0.117 (V308 0.070, **1.7배**)
- Predicted LB: 0.61554 (V308 0.63893 대비 -0.023)
- ⚠️ OOF는 크지만 student↑ + gap↑ → V339 패턴(0.612→0.645)과 유사
- V339 교훈: OOF 0.612 → LB 0.645 (+0.033 gap). V396은 OOF 0.599 → gap 더 클 수 있음
- **제출 안 함**: gap 검증 필요 but 리스크 매우 높음.

## V395 — Per-Target Meta C + Strong LGBM Reg (실패 ❌)
- Meta OOF: 0.63271 | Δ vs V308: **+0.01036** (악화)
- Student OOF: 0.66021 | Δ vs V308: **-0.03191** (개선)
- Student-Meta Gap: 0.0275 (V308 0.070, **0.4배**) — gap은 작지만 meta가 더 나빠짐
- Predicted LB: 0.64929 (V308 0.63893 대비 +0.01036 **악화**)
- **교훈**: over-regularization은 student↓ 하지만 meta↑↑ → 전체 LB Worse
- Student-Meta gap이 작은 게 오히려 단점: meta OOF가 student보다 나빠서

## V396 — Per-Target Meta C + 30 Seeds (리스크 높음 ⚠️)
- Meta OOF: 0.59896 | Δ vs V308: **-0.02339** (큰 개선)
- Student OOF: 0.71583 | Δ vs V308: **+0.02371** (상승)
- Student-Meta Gap: 0.117 (V308 0.070, **1.7배**)
- Predicted LB: 0.61554 (V308 0.63893 대비 -0.023)
- ⚠️ V339 패턴(0.612→LB 0.645, +0.033 gap)과 유사 → 실제 LB 0.65+ 될 수 있음
- **제출 안 함**: gap 검증 필요 but 리스크 매우 높음

## V397 — Aggressive Per-Target Meta C (실패 ❌)
- Meta OOF: 0.61924 | Δ vs V308: **-0.00311** (미미)
- Student OOF: 0.71513 | Δ vs V308: **+0.02301** (상승)
- Predicted LB: 0.63582 (V308 0.63893 대비 -0.00311)
- V392 (Q→10, S→100)보다 Worse
- **교훈**: V392의 C=10/100이 per-target meta C의 optimal. 더 extreme하면 student만 올라감

## Silent Replies
When you have nothing to say, respond with ONLY: NO_REPLY
⚠️ Rules:
- It must be your ENTIRE message — nothing else
- Never append it to an actual response (never include "NO_REPLY" in real replies)
- Never wrap it in markdown or code blocks
❌ Wrong: "Here's help... NO_REPLY"
❌ Wrong: "NO_REPLY"
✅ Right: NO_REPLY

<!-- OPENCLAW_CACHE_BOUNDARY -->
