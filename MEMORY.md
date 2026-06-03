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

## V325-V330 실험 결과 (2026-06-02 실행)

### V325 — Per-Subject Modeling (LOO CV) (실패 ❌)
- OOF: 0.68773 | Δ vs V321: **+0.08204** (심각 악화)
- 10개 subject별 모델 학습 → LOO CV로 예측
- **교훈: subject별 학습은 학습 데이터 부족으로 완전 붕괴. 다른 subject 데이터로는 individual patterns 학습 불가**

### V326 — Heavy Feature Engineering + V321 Stacking (성공 ⭐⭐)
- OOF: **0.59159** | Δ vs V321: **-0.01410**
- Predicted LB: 0.61059
- 추가 feature: interaction features (HR×pedo, Light×Screen, GPS×BLE), rolling window stats, per-subject z-scores
- 443 features total (141 base + 141 zscore + 151 per-subject z-score + 10 interaction)
- **핵심 발견: heavy feature engineering이 V321 baseline을 넘어섬!**
- 모든 타겟 개선: S1(-0.030), S2(-0.029), S3(-0.034), S4(-0.030), Q1(-0.023), Q2(-0.015), Q3(-0.033)
- **리스크**: Student-Meta gap 큼 (0.10~0.15) → V316/V314과 유사한 overfitting 패턴

### V327 — Double Stacking (실패)
- 10 configs × 15 seeds → L1 LR → L2 LR
- Student avg OOF ~1.2~1.4 (anomalous, likely data formatting issue)
- **교훈: double stacking은 구현 복잡하고 metrics unreliable**

### V328 — XGBoost + CatBoost + LGBM Ensemble (실패 ❌)
- XGBoost OOF: ~1.21 (predicting ~0.5 always — random)
- CatBoost OOF: ~1.20 (same issue on mixed raw+zscore features)
- Standardized features: LGBM 0.91 (from 0.62 → catastrophic), CatBoost 0.78
- **핵심 발견: LGBM is the ONLY tree model that works on this dataset**
- **교훈: CatBoost/XGB cannot handle mixed raw+zscore feature space. Feature standardization destroys LGBM signal too.**

### V329 — Target-Specific Feature Engineering (LEAKAGE ❌❌❌)
- Reported OOF: 0.001 (catastrophic leakage — perfect memorization)
- Pairwise target interactions (`S1_x_S2`, `S1_minus_S2`) used actual target values as features
- Inter-target CV predictions also leaked since derived from target values
- **핵심 교훈: target values CANNOT be used as features, even via CV. Massive leakage risk.**

### V330 — Meta Learner as GBM Tree (실패 ❌)
- LR Meta AVG OOF: 0.60543 (V321 baseline: 0.60569 — essentially identical)
- GBM Meta (best=deep) AVG OOF: 0.272 (catastrophic overfitting)
- GBM gap vs student: +0.39~+0.50 (LR gap: +0.05~+0.15)
- **교훈: GBM meta on 15 features + 450 samples = severe overfitting. LR remains optimal.**
- **Same pattern as V159: non-linear meta = overfitting with only 450 samples**

## V325-V330 종합 요약 테이블

| 실험 | OOF | Δ vs V321 | Δ vs V326 | Status | Notes |
|------|-----|-----------|-----------|--------|-------|
| V325 | 0.68773 | +0.08204 | +0.09614 | ❌ | Per-subject LOO崩溃 |
| V326 | 0.59159 | -0.01410 | baseline | ⭐ | Heavy feat engineering works |
| V327 | — | — | — | ❌ | Double stacking buggy |
| V328 | — | — | — | ❌ | XGBoost/CatBoost useless |
| V329 | 0.001* | -0.60463 | -0.59053 | ❌❌ | Massive leakage |
| V330 | 0.60543 | -0.00026 | +0.01384 | ❌ | GBM meta overfits |

*V329 leakage OOF (useless)

## V325-V334 실험 결과 (2026-06-02 실행)

### V328 — V326 Enhanced: More Per-Subject Features (성공 ⭐⭐⭐)
- OOF: **0.56298** | Δ vs V326: **-0.02861**
- Predicted LB: **0.58198**
- 추가 feature: per-subject rolling mean/std(3,5), min/max/median, ratio, deviation, acceleration
- 784 features total

### V329 — V328 + Cross-Subject Features (성공 ⭐⭐⭐⭐ BEST)
- OOF: **0.54365** | Δ vs V328: **-0.01933**
- Predicted LB: **0.56265**
- 추가 feature: cross-subject z-scores, quartiles, acceleration, day-of-week, entropy
- 2047 features total
- **현재 BEST 모델**

### V330 — V329 + Domain Cross-Interactions (실패)
- OOF: 0.55603 | Δ vs V329: **+0.01238** (악화)
- **교훈: domain cross-interactions이 noise로 작용**

### V331 — V329 + Top-100 Feature Selection (실패 ❌)
- OOF: **0.58429** | Δ vs V329: **+0.04064** (심각 악화)
- **교훈: top-100 feature removal이 signal 손실. 2047 features 중 noise가 아님.**
- Q-targets가 특히 나쁨 (Q1: 0.621, Q2: 0.614)

### V332 — V329 + 30 Seeds + Meta C=500 (OOF 개선 but student gap 큼 ⚠️)
- OOF: **0.51539** | Δ vs V329: **-0.02826** (개선)
- Predicted LB: 0.53439
- **학생 gap 문제**: Q1(0.129), Q2(0.163), S1(0.140), S2(0.152) — V313과 동일한 패턴
- Student avg OOF: V329와 거의 동일 (~0.65-0.69). C=500은 meta만 낮춤.
- **V313 교훈**: 이 패턴은 LB에서 OOF-LB gap이 클 것 (V313: OOF 0.595 → LB 0.647)

### V333 — V329 + Stronger Regularization (실패 ❌)
- OOF: **0.57818** | Δ vs V329: **+0.03453** (악화)
- Stronger regularization이 student OOF를 낮췄으나(meta OOF도 낮아져서) net negative
- **교훈: student OOF 개선 ≠ meta OOF 개선. student gap을 메우는 것이 아님.**

### V334 — V329 + 30 Seeds + C=500 Combined
- OOF: **0.51539** (V332와 동일)
- 같은 config이므로 같은 결과

## V325-V334 종합 요약 테이블

| 실험 | OOF | Δ vs V329 | Status |
|------|-----|-----------|--------|
| V326 | 0.59159 | +0.04794 | ⭐ |
| V328 | 0.56298 | +0.01933 | ⭐⭐⭐ |
| V329 | **0.54365** | **baseline** | ⭐⭐⭐⭐ BEST |
| V335 | 0.61709 | +0.07344 | ❌ target-grouped sharing no benefit |
| V336 | 0.58242 | +0.03877 | ❌ domain pairwise interactions noise |
| V330 | 0.55603 | +0.01238 | ❌ domain cross-interactions noise |
| V331 | 0.58429 | +0.04064 | ❌ Top-100 너무 aggressive, signal 손실 |
| V332 | 0.51539 | -0.02826 | ⚠️ 큰 student gap (Q1: 0.13, Q2: 0.16) |
| V333 | 0.57818 | +0.03453 | ❌ Regularization too strong, net negative |
| V334 | 0.51539 | -0.02826 | ⚠️ V332 동일 config |
| V335 | 0.61709 | +0.07344 | ❌ target-grouped student sharing no benefit |
| V336 | 0.58242 | +0.03877 | ❌ domain pairwise interactions noise |
| V337 | 0.86361 | +0.31996 | ❌ identical feature sets ensemble useless |
| V338 | 0.60532 | +0.06167 | ❌ C=500 fails with 30 seeds (student gap 0.30+) |
| V337 | 0.86361 | +0.31996 | ❌ V329+V308 identical predictions (same seed → same bag) |
| V338 | 0.60532 | +0.06167 | ❌ 30 seeds + C=500 worse than 15 seeds (student gap 0.30+) |

### V337 — V329 + V308 Cross-Validated Ensemble (실패 ❌❌)
- OOF: 0.86361 | Δ vs V329: **+0.31996** (치명적 실패)
- V329와 V308의 OOF가 동일 (Q1: 1.31700 = V308 1.31700)
- **원인**: feature bagging seed가 동일해서 두 pipeline이 완전히 같은 features 선택
- V308은 V329의 subset이므로, 같은 features → identical predictions
- **교훈: identical feature subsets을 쓰는 두 pipeline을 ensemble해도 의미 없음**
- **새로운 교훈: ensemble을 하려면 완전히 다른 feature set 필요**

### V338 — V329 + Multi-Config Ensemble (실패 ❌)
- OOF: 0.60532 | Δ vs V329: **+0.06167** (실패)
- Approach A (30 seeds + C=500)가 선택되었지만 student gap 큼 (Q1: 0.37)
- Approach B (4 configs × 7 seeds)는 더 나쁨
- **핵심 교훈: V329의 heavy feature set + C=500 조합은 student bottleneck 해결 못 함**
- **C=500은 15 seeds 환경에서만 작동. 30 seeds로는 overfitting**

### V335 — V329 + Target-Grouped Students (Q_pool + S_pool) (실패 ❌)
- OOF: 0.61709 | Δ vs V329: **+0.07344**
- Q-targets 공유, S-targets 공유 → 전혀 도움이 안 됨
- **교훈: target별 독립 학습이 이미 최적. target-group sharing은 overfitting/underfitting 유발**

### V336 — V329 + Deep Feature Interactions (도메인 간 pairwise) (실패 ❌)
- OOF: 0.58242 | Δ vs V329: **+0.03877**
- 도메인 간 pairwise interaction 추가 (pedo×hr, light×screen 등)
- V330과 같은 패턴 → interactions는 noise
- **교훈: feature interactions은 이미 V330에서 실패. 재탕 금지.**

### V331 — V329 + Top-100 Feature Selection (실패 ❌)
- OOF: 0.58429 | Δ vs V329: **+0.04064**
- 2047 features 중 top-100만 사용 → signal 대폭 손실
- **교훈: 2047 features는 noise보다 signal이 많음. top-K feature selection은 오히려 해로움.**

### V332 — V329 + 30 Seeds + Meta C=500 (OOF 개선 but student gap 큼 ⚠️)
- OOF: 0.51539 | Δ vs V329: **-0.02826**
- Predicted LB: 0.53439
- **학생 gap 문제**: Q1(0.129), Q2(0.163), S1(0.140), S2(0.152) — V313과 동일한 패턴
- Student avg OOF: V329와 거의 동일 (~0.65-0.69). C=500은 meta만 낮춤.
- **V313 교훈**: 이 패턴은 LB에서 OOF-LB gap이 클 것 (V313: OOF 0.595 → LB 0.647)
- **제출 안 함**: OOF는 좋으나 OOF-LB correlation 붕괴 리스크 큼

### V333 — V329 + Stronger Regularization (실패 ❌)
- OOF: 0.57818 | Δ vs V329: **+0.03453**
- Stronger regularization이 student OOF를 낮췄으나(meta OOF도 낮아져서) net negative
- **교훈: student OOF 개선 ≠ meta OOF 개선. student gap을 메우는 것이 아님.**

### V334 — V329 + 30 Seeds + C=500 Combined
- OOF: 0.51539 (V332와 동일)
- 같은 config이므로 같은 결과

### V337 — Two-Stage Stacking: Student Predictions as Features (실패 ❌)
- **OOF: 0.60195** | Δ vs V329: **+0.05830** (심각 악화)
- **Student avg OOF: 0.65905** | Δ vs V329: **+0.01207** (오히려 나빠짐)
- Approach: Stage 1에서 15 seeds V329 → OOF predictions → Stage 2에서 이 preds를 features로 추가 → LR meta
- Q-targets 특히 나쁨: Q2 OOF=0.61866 (student=0.75192, gap=+0.1333)
- **교훈: 두꺼운 stacking이 student bottleneck을 해결 못함. Stage 1 student 자체가 V329보다 안좋음.**

### V338 — Student Features + Aggressive Feature Bagging 50% (실패 ❌)
- **OOF: 0.58670** | Δ vs V329: **+0.04305** (악화)
- **Student avg OOF: 0.64290** | Δ vs V329: **-0.00408** (미미한 개선)
- Predicted LB: 0.60570
- Approach: V329 + cross-subject z-scores + feature bagging 50% + two-stage stacking
- S1: 0.53993 (V329 대비 -0.00372 개선) — 가장 나은 결과
- **교훈: feature bagging 50%가 V329 student OOF를 약간 낮췄으나(meta OOF는 높아짐) net negative**

### V339 — Student Features + Cross-Subject Z + Aggressive Bagging (실패 ❌)
- **OOF: 0.58689** | Δ vs V329: **+0.04324** (악화)
- **Student avg OOF: 0.64922** | Δ vs V329: **+0.00224** (동일 수준)
- Predicted LB: 0.60589
- V338과 거의 동일한 결과. cross-subject z-scores가 추가 benefit 없음.
- **교훈: V329의 feature set이 이미 optimal. 추가 feature engineering이 도움이 안됨.**

### V400 — Three-Stage Stacking (실패 ❌)
- **OOF: 0.59037** | Δ vs V329: **+0.04672** (악화)
- **Student avg OOF: 0.66994** | Δ vs V329: **+0.02296** (나빠짐)
- Predicted LB: 0.60937
- Approach: S1(base) → S2(S1 preds) → S3(S1+S2 preds) → LR meta
- **교훈: 3단계 stacking은 student OOF를 오히려 악화시킴. overfitting risk 큼.**

### V401 — Three-Stage Stacking + 50% Bagging (실패 ❌)
- **OOF: 0.59123** | Δ vs V329: **+0.04758** (악화)
- **Student avg OOF: 0.68149** | Δ vs V329: **+0.03451** (심각 나빠짐)
- Predicted LB: 0.61023
- **교훈: V400보다 더 나쁨. feature bagging 50%가 3-stage stacking에서 해로움.**

## V337-V339, V400-V401 종합 요약 테이블

| 실험 | OOF | Δ vs V329 | Student OOF | Status |
|------|-----|-----------|-------------|--------|
| V329 | **0.54365** | **baseline** | 0.64698 | ⭐⭐⭐⭐ BEST |
| V337 | 0.60195 | +0.05830 | 0.65905 | ❌ identical feat set ensemble |
| V338 | 0.58670 | +0.04305 | 0.64290 | ❌ bagging 50% net negative |
| V339 | 0.58689 | +0.04324 | 0.64922 | ❌ cross-subject z no benefit |
| V400 | 0.59037 | +0.04672 | 0.66994 | ❌ 3-stage overfitting |
| V401 | 0.59123 | +0.04758 | 0.68149 | ❌ 3-stage + 50% bagging worst |

## V325-V339, V400-V401 핵심 인사이트
1. **Per-subject modeling fails**: too few samples per subject (45 rows)
2. **Heavy feature engineering works**: per-subject features + cross-subject z-scores + quartiles + acceleration + dow (V329)
3. **LGBM is the only viable tree model**: XGBoost and CatBoost fail on mixed features
4. **Target values as features = leakage**: cannot use S1/S2 etc as features even via CV
5. **GBM meta overfits**: LR remains optimal meta-learner for 15-dimensional input, 450 samples
6. **C=500 helps OOF but not student avg**: V332/V334 student avg ~0.65-0.69 (same as V329 C=10)
7. **Student bottleneck persists**: student avg OOF 수렴 ~0.65-0.69. 근본적 feature engineering 필요.
8. **Top-100 feature removal backfires**: 2047 features contain mostly signal, not noise
9. **Stronger regularization helps student but hurts meta**: net negative
10. **V329 is the best confirmed model**: OOF=0.54365, Predicted LB=0.56265
11. **V332 student gap pattern matches V313**: OOF 0.515 → LB likely 0.62-0.65 (OOF-LB gap ~0.10+)
12. **LB 0.500 목표**: student OOF를 ~0.55까지 낮추는 것이 필요 — 새로운 feature engineering 방향이 필요
13. **V335 교훈: target-grouped sharing = no benefit** — target별 독립이 이미 최적
14. **V336 교훈: domain pairwise interactions = noise** — V330과 동일
15. **V337 교훈: stacking two similar pipelines = useless** — identical feature subsets → identical predictions
16. **V338 교훈: C=500은 15 seeds까지만 가능** — 30 seeds로 늘리면 student gap 0.30+로 폭증
17. **V338-V339 교훈**: feature bagging 50%가 student OOF를 약간 낮췄으나, meta OOF가 더 높아짐 → net negative
18. **V400-V401 교훈**: 3-stage stacking이 student OOF를 오히려 악화. stacking depth 증가 = overfitting
19. **V329가 여전히 BEST**: 모든 새로운 접근법이 OOF를 악화시킴 → V329 pipeline이 이미 local optimum
20. **Student avg OOF ~0.65에서 수렴**: student pipeline 개선이 bottleneck. new feature engineering 필요.
