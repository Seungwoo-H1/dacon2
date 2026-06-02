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

### V308 — Z-Score Enriched Stacking (OOF 테스트 완료 — 2026-06-02)
- OOF: 0.62235 | Δ vs V146: **-0.00934**
- 2026-06-02 테스트 예측 생성 완료
- **제출 파일**: `submission_v308_zscore_20260602_021028.csv`
- 구성: 15 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- 282 features (141 base + 141 zscore) + per-target top-K selection
- 모든 타겟 개선 (S2 제외: -0.005 ~ -0.022, S2: +0.012)
- Student OOF 안정성 우수 (S1: 0.59-0.60, S3: 0.62-0.63)
- Predicted LB: ~0.624 (V146 대비 -0.008 개선)

### V146 — Optimized Stacking (제출 완료 — 2026-05-30)
- OOF: 0.63169 | Δ vs V140: **-0.00941**
- 2026-05-30 수동 제출 완료
- Leaderboard 결과 기다리는 중
- **제출 파일**: `submission_v146_submit_20260530.csv`
- 구성: 5 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- V308이 OOF -0.00934로 V146 초월

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
- V146 meta output too conservative (|pred - 0.5| < 0.05 for all)
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

## ⭐ V160 — Current BEST (OOF 테스트 완료, 미제출)
- **AVG OOF: 0.62240** (vs V146: 0.63169, Δ=-0.00929)
- **All 7 targets improved** (no degradation)
- **Low overfitting risk**: same LR(C=10) meta as V146
- **Architecture**: 15 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- **Seed selection**: SEED=42, step=7 (42,49,56,...,140)
- **Expected LB**: ~0.620-0.625 (similar OOF-LB gap as V146)
- **Submission candidate**: YES
- **Status**: Waiting for user approval — V146 LB result 확인 필요

## V146 — Optimized Stacking (제출 완료 — 2026-05-30)
- OOF: 0.63169 | Δ vs V140: **-0.00941**
- 2026-05-30 수동 제출 완료
- Leaderboard 결과 기다리는 중
- **제출 파일**: 
- 구성: 5 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- V160 OOF에서 개선 확인되었으나 LB 제출 검증 필요

### V141 — Drift-Aware Stacking (테스트됨, LB=0.64031)
- OOF: 0.63678 → LB: 0.64031
- OOF-LB 갭: 0.00353 (V140 대비 8배 증가)
- fold drift weighting은 noise fitting으로 일반화 저하
- **V140→V146: V140 OOF-LB correlation 최상, V146이 OOF 개선**

### V142 — Stability-Weighted Stacking (OOF만 테스트)
- V142-A (feat sel only): OOF=0.63565
- V142-B (fold drift weights): OOF=0.63483
- OOF는 V140보다 낮지만, OOF-LB correlation 깨질 가능성 높음
- **제출 안 함** — OOF 개선이 LB로 직결되지 않음 (V142 교훈)

### V143 — Multi-Config Stacking (실패)
- 4 configs × 3 seeds = 12 students
- OOF=0.63873 (대시시) but student OOF 1.2~1.4 (성능 붕괴)
- 모든 config에 동일 features 사용 → prediction scale 붕괴
- V140의 config→target 매핑이 이미 최적
- **제출 안 함**

### V144 — Double-Blind Ensemble Stacking (실패)
- Pipeline A (V140 top-K) + Pipeline B (4× wider) → equal-weight ensemble
- OOF: 0.64987 | Δ: +0.00877 (악화)
- 4× widened features는 noise만 추가
- **제출 안 함**

### V145 — Heterogeneous Multi-Model Stacking (실패)
- 2 LGBM + 2 CatBoost + 2 XGBoost = 6 students
- OOF: 0.66032 | Δ: +0.01922 (심각 악화)
- CatBoost early stopping으로 17 iteration만 learning → OOF 1.3
- 다른 model family는 V140의 정교한 LGBM 설계를 망가뜨림
- **제출 안 함**

## 핵심 결과 비교표 (V140-V146)

| Version | Method | Seeds | Meta C | AVG OOF | Δ vs V140 | Status |
|---------|--------|-------|--------|---------|-----------|--------|
| V140 | LGBM stacking | 3 | 0.1 | 0.64110 | baseline | LB=0.64072 (submitted) |
| V141 | Drift-aware | 5 | 0.1 | 0.63678 | -0.00432 | LB=0.64031 (gap↑) |
| V142 | Stability-weighted | 5 | 0.1 | 0.63483 | -0.00627 | local (overfit risk) |
| V143 | Multi-config | 12 | 0.1 | 0.63873 | -0.00237 | failed |
| V144 | A+B ensemble | 6 | - | 0.64987 | +0.00877 | worse |
| V145 | Hetero (CB+XGB) | 6 | 0.1 | 0.66032 | +0.01922 | worse |
| **V146** | **Optimized** | **5** | **10** | **0.63169** | **-0.00941** | **submitted 5/30** |
| **V155** | **Multi-model** | **7** | **weighted** | **0.46447** | **-0.17669** | **NOT submitted (overfit)** |

## 핵심 인사이트
- **V140이 BEST (LB=0.64072, OOF=0.64116, 갭=0.00044)**
- V140의 핵심: proper CV stacking, OOF≈LB (stable generalization)
- **fold drift weighting은 noise fitting** (V142 실패 — OOF↓했지만 LB↑)
- **config→target 매핑이 이미 최적** — 깨면 성능 붕괴 (V143 실패)
- **V142 교훈: OOF↓ ≠ LB↓**, OOF-LB correlation 유지가 최우선
- **meta C=10이 C=0.1보다 OOF 개선** (V146, -0.00941). OOF-LB gap 확인 필요.
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
- 다음 실험: V160 제출 후보 (OOF 0.62240 vs V146 0.63169)

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
