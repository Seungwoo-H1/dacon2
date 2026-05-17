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

### V127 — 3-way Ensemble (2026-05-11)
- **Leaderboard: 0.64763** | OOF: 0.53731
- 구성: V121 (pairwise) + V123 (transformed) + V115 (base)
- Weight: 0.35 / 0.25 / 0.40
- Temperature scaling T=0.86
- **아직 0.5 점수가 나옴** — V127이 현재 undisputed best
- Submission: `submissions/submission_v127_20260511_224000.csv`

## 핵심 결과 타임라인

### 베이스라인 (2026-05-01)
- **LGBM V10**: CV 0.6038

### Ensemble 실험 (2026-05-07)
- **V58** (LGBM+CatBoost+XGB stacking): avg CV 0.6253
- **V59** (multi-seed): V58과 동등
- **V60** (interactions): 역개선 (-0.0046)

### Leakage Clean (2026-05-07)
- **V61** (CatBoost + leakage-clean): avg CV 0.5830
- S4 전용 feature set (single LGBM, n_feat=25)

### V63~V72 (2026-05-08)
- Stacking, Ensemble, Calibration, interaction 실험 등 다수
- V127보다 낮은 성능 — V127이 undisputed best

### V259~V263 (2026-05-14)
- **V259 Isotonic Calibration**: Δ=-0.019 (OOF)
- **V260 Quantile+PSI**: 실제 LB=0.714592 (버림 — 역효과)
- **V262 2x2x2 Factorial**: Isotonic Δ=-0.073 (강력한 gain)
- **V263**: V127이 BEST 확인 (LB=0.64763, OOF=0.53731)
- V45a/V46: leakage 발견 (100%/99.6% accuracy)

### V62 Full Feature Pipeline (2026-05-17) — 확인 안 됨
- 142 features train AND test 동일하게 적용 (column-set 일치 검증 완료)
- 100-seed LGBM ensemble + JTD calibration + z-score personalization
- V127보다 더 높은 점수 나올 가능성 있지만 **아직 리더보드 확인 안 됨**
- Submission: `submissions/submission_v62_20260517_055940.csv`

## 핵심 아키텍처 패턴

### V127 (3-way Ensemble — BEST)
```
V121 (pairwise_interactions) + V123 (transformed) + V115 (base)
→ Weight: 0.35 / 0.25 / 0.40
→ Temperature scaling T=0.86
→ LB: 0.64763, OOF: 0.53731
```

### V53 (Baseline)
```
LGBM seed ensemble (50 seeds)
→ 개인별 z-score feature
→ 타겟별 top-K feature selection
→ Leakage column 제거
```

### V62 (Full 142-Feature — 아직 제출 전)
```
02_feature_engineering.py → train+test 동일 pipeline 적용
Train: 450 rows × 153 cols (142 features + 7 targets + meta)
Test:  250 rows × 146 cols (142 features + meta)
100-seed LGBM ensemble + JTD calibration + z-score per-subject
```

## 주요 인사이트
- **V127이 undisputed best** (LB=0.64763, OOF=0.53731)
- V45a leakage 발견 → pipeline에 leakage 가능성 상존
- Distribution correction (Quantile+PSI): 효과 없음
- Isotonic calibration: OOF에서는 강력하지만 LB에는 역효과
- **LB 0.50은 매우 어려움** (OOF <0.47 필요)
- Ensemble diversity 중요 (avg correlation 0.785 — 낮을수록 좋음)
- **V62는 142 features 풀 파이프라인이 train+test 모두에서 완벽하게 작동함을 확인** — V127 beating이 목표

## 파일 구조
- `/home/mwoo423/projects/dacon2/src/` — 모든 실험 스크립트
- `/home/mwoo423/projects/dacon2/submissions/` — 제출물 + 메타
- `/home/mwoo423/projects/dacon2/experiments/` — OOF, 로깅
- `/home/mwoo423/projects/dacon2/docs/` — 분석 문서
- `/home/mwoo423/.openclaw/workspace/data_processed/` — 전처리 데이터
- `/home/mwoo423/.openclaw/workspace/submissions/` — submissions (workspace 기준)

## 작업 원칙
1. 모든 실험은 메타 JSON 남기기
2. 신규 실험은 **V127 (LB=0.64763) 기준** 비교 후 제출
3. Leak 제거 필수: LEAK_S, LEAK_Q, NIGHTTIME_LEAK, SLEEP_DIRECT_LEAK
4. CV/OOF = 리더보드 점수 아님. 오버피팅 주의
5. Submission은 승우さんが 수동으로 DaCon에 제출 (API 403)
