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

## ⏳ Pending LB Submission (V308 초월 가능성 높음)

### V312 — Per-Target Meta C=500 (제출 완료 — 2026-06-02)
- OOF: 0.61448 | Δ vs V308: **-0.00787**
- Δ vs V146: **-0.01721**
- **제출 파일**: `submission_v312_per_target_C_20260602_024509.csv`
- 구성: 15 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=500)
- 모든 타겟에서 C=500이 optimal (C=10 → C=500으로 일괄 개선)
- 타겟별 개선: Q3(-0.020), S3(-0.036), S1(-0.007), S2(-0.007), S4(-0.008), Q1(-0.003)
- 핵심 발견: C=500으로 meta regularization 완화 → OOF 일괄 개선
- **예상 LB**: 0.631 (V308 0.63893 대비 -0.008)
- **리스크**: OOF-LB gap이 V308보다 클 가능성
- **V312는 아직 LB 미검증** — 승우님 수동 제출 필요
- **핵심 교훈: C=500이 C=10보다 훨씬 좋음. C가 클수록 meta가 student를 더 강하게 반영**

### V313 — 30 Seeds + C=500 (제출 완료 — 2026-06-02)
- OOF: 0.59512 | Δ vs V308: **-0.02723**
- Δ vs V312: **-0.01936**
- Δ vs V146: **-0.03657**
- **제출 파일**: `submission_v313_30seeds_C500_20260602_025815.csv`
- 구성: 30 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=500)
- **모든 7개 타겟 개선** (V312 대비):
  - Q1: -0.02335 | Q2: -0.01951 | Q3: -0.01153
  - S1: -0.02041 | S2: -0.02403 | S3: -0.01308 | S4: -0.02362
- Student avg OOF: 0.692 (V312와 동일) → calibration 유사
- **예상 LB (V308 gap 가정)**: 0.61170 (V308 0.63893 대비 **-0.027**)
- **V308을 크게 넘을 가능성 높음** — LB 제출로 검증 필요
- **핵심 발견: seeds 15→30 + C=500이 가장 강력한 조합**

### V310 — Z-Score + S2 Base-Only + Calibration (실패)
- OOF: 0.60036 (calibration 후)
- Δ vs V308: **-0.02199** (인위적 개선)
- Calibration OOF는 train data에 fitting → LB 일반화 불가
- Student OOF: 0.76~0.79 (meta 0.600 대비 gap 0.16~0.19)
- S2: 0.60117 (V308 0.61653 대비 -0.01536 개선)
- **제출 안 함**: calibration은 CV OOF만 낮출 뿐, 실제 LB에는 도움 안 됨
- **핵심 교훈: calibration은 OOF-LB correlation을 더 깨뜨림**

### V310b — Z-Score + S2 Base-Only (No Calibration) (실패)
- OOF: 0.62338 | Δ vs V308: **+0.00103** (거의 동일)
- S2: 0.62283 (V308 0.61653 대비 -0.00633 **악화**)
- S3: 0.60994 (V308 0.62331 대비 -0.01337 개선)
- Q3: 0.61939 (V308 0.63507 대비 -0.01568 개선)
- Student OOF: 0.76~0.79 (gap 0.14~0.16)
- S2 z-score removal은 S2에 오히려 해로움 (ranked features에 z-score 포함 안 됨)
- **제출 안 함**: V308과 거의 동률
- **교훈: S2 z-score features가 ranking에서 선택 안 되어도 base features만으로도 V308과 동률**

## V307-V309 결과 정리

### V307 — CV-only Z-Score Test (성공 ⭐)
- OOF: 0.61390 | Δ vs V146: **-0.01779**
- features_enhanced.parquet(132 per-subject z-score) 사용
- Z-score hypothesis 검증 — OOF 큰 개선
- 테스트 예측 없음 → V308로 이어짐

### V308 — Z-Score Enriched Stacking (제출 완료 — 2026-06-02)
- OOF: 0.62235 | Δ vs V146: **-0.00934**
- **Actual LB: 0.63893**
- V307의 Z-score를 full pipeline에 적용
- Global z-score 사용 (train mean/std 기반)
- OOF-LB gap: 0.01658 (V140: -0.00044 대비 큼)
- 모든 타겟 개선 (S2 제외: -0.005 ~ -0.022, S2: +0.012)
- S2 z-score features가 noise 역할

### V309 — Per-Subject Z-Score Stacking (실패)
- OOF: 0.62415 | Δ vs V308: **+0.00180** (악화)
- Δ vs V146: -0.00754 (개선)
- Student OOF: 0.79~0.82 (V308: 0.63~0.67, V309 gap 0.18)
- Per-subject z-score가 within-person variability만 포착
- Global z-score가 between-person signal도 포착 → 더 나은 generalization
- S2만 V308 대비 0.005 개선, 다른 타겟 대부분 악화
- **제출 안 함**: V308이 더 우수
- **핵심 교훈: global z-score > per-subject z-score**

## V140-V164 결과 정리

### V141 — Drift-Aware Stacking (테스트됨, LB=0.64031)
- OOF: 0.63678 → LB: 0.64031
- OOF-LB 갭: 0.00353 (V140 대비 8배 증가)
- fold drift weighting은 noise fitting으로 일반화 저하
- **제출 안 함**

### V142 — Stability-Weighted Stacking (OOF만 테스트)
- V142-A (feat sel only): OOF=0.63565
- V142-B (fold drift weights): OOF=0.63483
- OOF는 V140보다 낮지만, OOF-LB correlation 깨질 가능성 높음
- **제출 안 함**

### V143 — Multi-Config Stacking (실패)
- 4 configs × 3 seeds = 12 students
- OOF=0.63873 but student OOF 1.2~1.4 (성능 붕괴)
- **제출 안 함**

### V144 — Double-Blind Ensemble Stacking (실패)
- Pipeline A + Pipeline B → equal-weight ensemble
- OOF: 0.64987 | Δ: +0.00877 (악화)
- **제출 안 함**

### V145 — Heterogeneous Multi-Model Stacking (실패)
- 2 LGBM + 2 CatBoost + 2 XGBoost = 6 students
- OOF: 0.66032 | Δ: +0.01922 (심각 악화)
- 다른 model family는 V140의 정교한 LGBM 설계를 망가뜨림
- **제출 안 함**

### V155 — Heterogeneous Multi-Model (실패)
- OOF=0.46447 but severe overfitting (Student OOF 0.602 vs Meta OOF 0.464, gap -0.138)
- **NOT submitted**

### V156 — Group-Enriched Stacking (실패)
- Added 564 group features + 55 cross-domain interactions + multi-meta
- AVG OOF: 0.62768 (better) but student OOF very poor (Q1: 0.72-0.76)
- Group features are pure noise. Multi-meta averaging masked poor student predictions.
- **제출 안 함**

### V157 — Wider Feature Selection (실패)
- top-K×2 feature selection (same 141 base features)
- AVG OOF: 0.64317 | Δ vs V146: **+0.01148** (worse)
- Wider feature selection = more noise. V146's conservative selection is optimal.
- **제출 안 함**

### V158 — Pseudo-Labeling (실패)
- V146 meta output too conservative (|pred - 0.05 < 0.05 for all)
- No high-confidence predictions selected (threshold=0.55)
- Pseudo-labeling requires confident predictions to be useful
- **제출 안 함**

### V159 — Non-linear Meta-learner (실패)
- GBM meta-learner (8 leaves, depth=2, 50 trees)
- AVG OOF: 0.54298 | Δ vs V146: **-0.08871** (but severe overfitting)
- Student OOF: 0.63-0.80 | Meta OOF: 0.51-0.57 → gap 0.12-0.24
- Non-linear meta overfits with only 450 training samples
- Same pattern as V155: meta memorizes training patterns
- **제출 안 함**

### V160 — More Seeds Ensemble (성공 ⭐)
- 15 seeds vs V146's 5 seeds, same architecture (LR C=10 meta)
- **AVG OOF: 0.62240 | Δ vs V146: -0.00929**
- All 7 targets improved consistently:
  - Q1: 0.67694 → 0.67121 (-0.00573)
  - Q2: 0.62758 → 0.61828 (-0.00930)
  - Q3: 0.64119 → 0.63507 (-0.00612)
  - S1: 0.58833 → 0.57792 (-0.01041)
  - S2: 0.60366 → 0.59058 (-0.01308)
  - S3: 0.63244 → 0.62331 (-0.00913)
  - S4: 0.65171 → 0.64040 (-0.01131)
- **Key insight**: Ensemble diversity (more seeds) reduces variance without overfitting
- **Risk**: Low — same LR(C=10) meta, proven architecture
- **Expected LB**: Similar OOF-LB gap as V146 (~0.000-0.002)
- **Predicted LB**: ~0.620-0.625 (better than V146)
- **Submission candidate**: YES (OOF improvement consistent, low overfitting risk)
- **Status**: Not yet submitted — needs user approval for submission

### V161 — Iterative Pseudo-Labeling with V160 Seeds (실패)
- 15 seeds + pseudo-labeling (threshold=0.50, weight=0.5)
- Only 46 pseudo-labels generated (Q1:1, Q2:14, Q3:3, S1:5, S2:17, S3:2, S4:4)
- Too few pseudo-labels to affect training
- **Δ vs R1: +0.00000** (no change)
- V160 meta still too conservative for pseudo-labeling
- **제출 안 함**

### V162 — More Diverse Seeds (Random Range) (실패)
- Diverse seeds from range [42, 300): AVG OOF = 0.62534
- Regular seeds (step=7): AVG OOF = 0.62240
- Diverse seeds were **worse** (+0.00294 vs V160)
- LGBM internal randomness dominates over seed spacing
- **제출 안 함**

### V163 — Two-Level Stacking (Hierarchical) (실패)
- 15 seeds → 3 groups of 5 → 3 LR → 1 final LR
- AVG OOF: 0.63019 | Δ vs V160: **+0.00779** (worse)
- Hierarchical structure adds parameters without benefit
- 450 samples too few for 3 intermediate LR + 1 final LR
- **제출 안 함**

### V164 — Cross-Fold Feature Ranking (미완료)
- Cross-fold feature ranking takes too long
- Feature ranking was consistent across folds in sampling
- **Next**: skip if not proven beneficial

## 핵심 인사이트
- **V140이 BEST (LB=0.64072, OOF=0.64116, 갭=0.00044)**
- V140의 핵심: proper CV stacking, OOF≈LB (stable generalization)
- **fold drift weighting은 noise fitting** (V142 실패 — OOF↓했지만 LB↑)
- **config→target 매핑이 이미 최적** — 깨면 성능 붕괴 (V143 실패)
- **V142 교훈: OOF↓ ≠ LB↓**, OOF-LB correlation 유지가 최우선
- **meta C=10이 C=0.1보다 OOF 개선** (V146, -0.00941). OOF-LB gap 확인 필요.
- **C=500이 C=10보다 더 좋음** (V312, -0.00787 Δ vs V308). C가 클수록 meta가 student를 더 강하게 반영.
- Isotonic calibration: OOF에서는 강력하지만 LB에는 역효과
- **LB 0.50은 매우 어려움** (OOF <0.47 필요)
- 단순 feature engineering, calendar, naive ensemble 재탕 금지
- 다음 실험은 OOF-LB gap <0.002일 때만 제출
- **V156 교훈: group features = noise** — 추가 feature는 오히려 해로움
- **V157 교훈: wider feature selection = noise** — V146의 feature selection 이미 최적
- **V158 교훈: V146 meta output too conservative** — pseudo-labeling 무용
- **V159 교훈: non-linear meta = overfitting** — 450 samples에는 linear meta가 안전
- **V160 발견: 더 많은 seeds = 더 나은 ensemble** (Δ=-0.00929, 모든 타겟 개선)
- Seeds 증가(5→15)는 가장 low-risk한 개선 방법
- Non-linear meta는 training data에 과적합됨 (V155, V159 동일 패턴)
- **V308 교훈: global z-score > per-subject z-score**
- V309 (per-subject z-score): OOF 0.62415 vs V308 0.62235
- Student OOF V309: 0.79~0.82 (V308: 0.63~0.67)
- Global z-score가 between-person signal도 포착 → 더 나은 generalization
- **다음 실험: V308 아키텍처 유지하고 다른 개선점 탐색**
- **V313 핵심 발견: seeds 15→30 + C=500이 현재까지 가장 강력한 조합**
- **Student avg가 configs 간에 일정(0.692) → OOF-LB gap도 유사할 가능성**
- **V312와 V313 모두 V308을 넘을 가능성 높음 — LB 검증 대기 중**

### V313 — LB 확인 결과 (실패)
- **Actual LB: 0.6467217671** (V308 0.63893 대비 **-0.0078 나쁨**)
- OOF: 0.59512 → LB 0.6467, **OOF-LB gap: +0.0516** (V308 +0.01658 대비 3배)
- 원인: C=500 + 30 seeds 조합이 과도한 overfitting
- student avg OOF: 0.69193 (V308 0.69212와 동일) → student 성능은 이미 최적점 도달
- 핵심 교훈: student avg OOF는 configs 간에 거의 일정. Meta OOF만 낮출 뿐.
- **Student avg가 0.692에서 수렴 → student 성능 개선이瓶颈**

### V314 — Per-Target Meta C Sweeping + 20 Seeds (실패)
- OOF: 0.60669 | Δ vs V308: **-0.01566**
- 모든 타겟에서 C=500이最佳 (V312 재확인)
- Seeds 20도 15보다 약간 나아짐
- Student avg OOF: 0.69211 (V308/V312/V313과 동일)
- Student-Meta gap: 0.085 (V308: 0.070, V313: 0.097)
- OOF-LB gap이 V308 대비 5배 → 신뢰도 낮은 예측
- **LB 예측은 V308 이하일 가능성 높음 (과적합 리스크)**
- 제출 안 함
- 핵심 교훈: C=500과 seeds 증가만으로는 student bottleneck을 뚫을 수 없음

### V315 — Per-Target Feature Consensus Selection (실패 — 무변화)
- OOF: 0.62235 | Δ vs V308: **+0.00000 (완전히 동일)**
- 5 ranking runs 모두에서 100% consensus (all features identical across runs)
- top-5 features가 5 runs 동안 완전히 동일
- Feature ranking이 이미 매우 안정적 → consensus selection = single ranking
- Feature selection이 bottleneck이 아님
- 제출 안 함 — V308과 동일
- 핵심 교훈: feature ranking stability 높음, consensus feature selection 효과 없음

## 핵심 인사이트

## 파일 구조
- `/home/mwoo423/projects/dacon2/src/` — 모든 실험 스크립트
- `/home/mwoo423/projects/dacon2/submissions/` — 제출물 + 메타
- `/home/mwoo423/projects/dacon2/experiments/` — OOF, 로깅
- `/home/mwoo423/projects/dacon2/docs/` — 분석 문서
- `/home/mwoo423/.openclaw/workspace/data_processed/` — 전처리 데이터
- `/home/mwoo423/.openclaw/workspace/submissions/` — submissions (workspace 기준)

## 작업 원칙
1. 모든 실험은 메타 JSON 남기기
2. 신규 실험은 **V140 (LB=0.64072, OOF=0.64116)** 기준 비교
3. **OOF-LB correlation 유지가 최우선** (gap <0.002면 submit 가능)
4. Leak 제거 필수: LEAK_S, LEAK_Q, NIGHTTIME_LEAK, SLEEP_DIRECT_LEAK
5. CV/OOF = 리더보드 점수 아님. 오버피팅 주의
6. Submission은 3회 제한 — 실험은 로컬에서 다 하고 OOF-LB gap 작을 때만 제출
7. **V140의 stacking 구조(local optimum)에 가까운 것 같음** — 큰 개선을 원하면 아키텍처 전환 필요
8. 단순 feature engineering 반복 금지
9. **meta-learner C는 실험 필요**: C=0.1 → C=10으로 OOF 개선 확인됨
10. **C=500이 C=10보다 우월** (V312). **seeds 30이 15보다 우월** (V313)

## V316~V317 실험 결과 정리


## V316~V318 실험 결과 (LB 검증 완료)

### V316 — Per-Target Student Hyperparameter Optimization (실제 LB 실패 ❌)
- OOF: 0.61800 | Actual **LB: 0.650549** (V308 0.63893 대비 **+0.0116 악화!**)
- Per-target cfg 최적화가 train OOF만 낮춤 → test에서 calibration 붕괴
- **핵심 교훈: per-target student cfg customization은 test distribution에서 overfitting 유발**
- Student avg 0.692→0.651 개선은 train distribution에만 fit된 것
- **Per-target cfg 통일: 모든 target에 동일한 cfg 사용해야 test generalization 가능**
- OOF-LB gap 추정 방식이 flawed — per-target cfg로는 correlation 붕괴

### V317 — Gap-Balanced Student HP Optimization (실패)
- OOF: 0.61859 | Predicted LB: 0.63982 (V308 대비 +0.00089 악화)
- **제출 안 함** — predicted LB가 V308보다 나쁨
- **핵심 교훈: gap balancing vs OOF 트레이드오프가 있음**

### V318 — Forced Strong Regularization on Q1/Q3/S4 (실패)
- OOF: 0.63327 | Predicted LB: 0.64428 (V308 대비 +0.00535 악화)
- Gap 균일화는 완벽하지만 S4 OOF 폭등 (0.618→0.680)
- **핵심 교훈: Q1/Q3/S4는 inherently noisy. regularization trade-off fatal**

## V308 이후 실패 요약

| 실험 | OOF | Δ vs V308 | Actual/Pred LB | Δ vs V308 LB | Status |
|------|-----|-----------|----------------|--------------|--------|
| V312 | 0.61448 | -0.00787 | ? | ? | 미제출 |
| V313 | 0.59512 | -0.02723 | 0.6467 | +0.00777 | ❌ LB 악화 |
| V314 | 0.60669 | -0.01566 | ? | ? | 미제출 |
| V315 | 0.62235 | +0.00000 | ? | ? | V308 동일 |
| V316 | 0.61800 | -0.00435 | **0.650549** | **+0.0116** | ❌ LB 악화 |
| V317 | 0.61859 | -0.00376 | 0.63982 | +0.00089 | ❌ |
| V318 | 0.63327 | +0.01092 | 0.64428 | +0.00535 | ❌ |

### V319 — Meta C=500 (V312 동일 검증)
- OOF: 0.61448 | Δ vs V308: **-0.00787**
- V312와 완전히 동일 config (15 seeds, C=500)
- **제출 안 함** — V312와 동일하므로 V312 LB 검증 필요
- **교훈: C=500이 C=10보다 OOF에서 우위**

### V320 — Weighted Student Ensemble (No Meta Learner) (실패 ❌)
- OOF: 0.67566 | Δ vs V308: **+0.05331** (심각 악화!)
- Weight optimization이 단일 seed만 선택 (S1: seed3 100%, S2: seed6 99.88%)
- Stacking의 LR meta learner가 weighted avg보다 나은 평균화 역할
- **교훈: LR meta learner가 이미 optimal weight 찾음. weighted avg은 underfitting**

### V321 — Feature Bagging + Stacking (OOF 개선 ⭐)
- OOF: 0.60569 | Δ vs V308: **-0.01666** | Δ vs V312: **-0.00879**
- Predicted LB: 0.62527 (V308 0.63893 대비 **-0.014**)
- 각 seed마다 random feature subset (75%) 사용 → ensemble diversity 증가
- 모든 타겟 개선 (Q1: 0.639, Q2: 0.610, Q3: 0.612, S1: 0.561, S2: 0.604, S3: 0.601, S4: 0.613)
- Student gap 큼 (Q1: 0.15, S2: 0.15) → V316 같은 test overfitting 리스크
- **핵심 차이 V316 vs V321**: V316은 per-target cfg customization (실패), V321은 동일 cfg + feature bagging (성공 가능성)
- **제출 후보** — V312 LB 미검증, V321이 V312보다 OOF에서도 우위
- **리스크**: student gap이 V308보다 큼. 실제 LB에서 gap 유지 여부 중요

## 교훈 정리
- **OOF 개선 ≠ LB 개선** — V313은 OOF 0.595인데 LB 0.6467, V316은 OOF 0.618인데 LB 0.6505
- **per-target cfg customization = test overfitting** (V316)
- **gap balancing = OOF 희생** (V317/V318)
- **weighted ensemble (meta learner 없음) = underfitting** (V320)
- **feature bagging이 ensemble diversity 증가에 유효** (V321)
- **LR meta learner가 이미 optimal aggregation** — meta learner 제거하면 오히려 악화
- **V308이 현재까지 유일한 LB 검증 성공 모델**
- **V321이 OOF 기준으로 V312/V308 모두 초월** — 실제 LB 검증 필요
