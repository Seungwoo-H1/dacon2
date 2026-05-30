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

### V140 — Proper CV Stacking (BEST — 2026-05-18)
- **Leaderboard: 0.64072** | OOF: 0.64116
- V146 (2026-05-30 제출) 결과 확인 전

### V146 — Optimized Stacking (제출 완료 — 2026-05-30)
- OOF: 0.63169 | Δ vs V140: **-0.00941**
- 2026-05-30 수동 제출 완료
- Leaderboard 결과 기다리는 중
- **제출 파일**: `submission_v146_submit_20260530.csv`
- 구성: 3 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=0.1)
- OOF-LB 갭: 0.00044 (overfitting 거의 없음, generalization 매우 안정적)
- Submission: `submission_v140_stacking_20260518_021155.csv`

### V146 — Optimized Stacking (OOF 테스트, 미제출)
- OOF: 0.63169 | Δ vs V140: **-0.00941**
- 구성: 5 LGBM seeds × GroupKFold 5-fold → LR meta-learner (**C=10**, V140: 0.1)
- **핵심 발견**: V140의 C=0.1은 과소 규제. C=10이 OOF에서 유의미 개선.
- **OOF-LB gap 아직 확인 안 됨** — 3일 제출 제한으로 미제출
- OOF-LB gap < 0.002면 제출 후보
- Submission: `submission_v146_optimized_20260518_181206.csv`

## V140-V146 결과 정리

### V141 — Drift-Aware Stacking (테스트됨, LB=0.64031)
- OOF: 0.63678 → LB: 0.64031
- OOF-LB 갭: 0.00353 (V140 대비 8배 증가)
- fold drift weighting은 noise fitting으로 일반화 저하
- **V140이 현재 BEST (OOF-LB correlation 최상)**

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
| V140 | LGBM stacking | 3 | 0.1 | 0.64110 | baseline | ⭐ LB=0.64072 |
| V141 | Drift-aware | 5 | 0.1 | 0.63678 | -0.00432 | LB=0.64031 (gap↑) |
| V142 | Stability-weighted | 5 | 0.1 | 0.63483 | -0.00627 | local (overfit risk) |
| V143 | Multi-config | 12 | 0.1 | 0.63873 | -0.00237 | failed |
| V144 | A+B ensemble | 6 | - | 0.64987 | +0.00877 | worse |
| V145 | Hetero (CB+XGB) | 6 | 0.1 | 0.66032 | +0.01922 | worse |
| **V146** | **Optimized** | **5** | **10** | **0.63169** | **-0.00941** | **local only** |

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
