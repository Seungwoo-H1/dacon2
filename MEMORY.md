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

## MISSION (2026-06-04 승우 명시)
- 현재 최고 기준 모델: **V308**
- 목표: V308의 LB 0.63893을 초과하는 모델 찾기
- LB 예측이 V308 이하라면 보고 금지
- 성능 개선 확인 전까지 연구 루프 계속
- 동일 가설 반복 금지, 매 루프 새 가설 필수

## ⭐ 현재 BEST (실제 제출 확인됨)

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

## V339-V368 실험 결과 정리

### V339 — OOF Feature Augmentation (2026-06-04, ⏳)
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

## V369-V374 핵심 교훈 요약
1. **Target-conditional features = signal 분산** (V369)
2. **Meta C 분리 = underfitting** (V370)
3. **2-level stacking = 과포장** (V371)
4. **Pseudo-labeling = distribution 왜곡** (V372)
5. **Temperature scaling = 미미한 효과** (V373)
6. **Probability smoothing = noise 추가** (V374)
7. **Bagging + CV ranking + Meta C tuning이 유일한 개선 경로**
8. **V368이 OOF 0.60492로 가장 낮은 수치는 기록**
9. **0.5점대 진입은 현실적 목표가 아님**
10. **V339 LB 0.64551 → OOF 추정 금지, LB 검증 필수**

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
