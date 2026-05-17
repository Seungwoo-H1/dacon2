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

## 핵심 결과 타임라인

### 베이스라인 (2026-05-01)
- **LGBM V10**: CV 0.6038 — 장기간 최고 성능 기록

### Deep Feature Engineering (2026-05-06)
- **V53 final**: 리더보드 0.6535822621 (LGBM seed ensemble 50)
- **V53 swept**: n_feat 미세조정, avg CV +0.0081 개선
- 기준 제출물: `submissions/submission_v53_swept_20260507_151447.csv`

### Ensemble 실험 (2026-05-07)
- **V58** (LGBM+CatBoost+XGB stacking): avg CV 0.6253
- **V59** (multi-seed): V58과 동등
- **V60** (interactions): 역개선 (-0.0046)
- V58이 single architecture 중 최고

### Leakage Clean (2026-05-07)
- **V61** (CatBoost + leakage-clean): avg CV 0.5830
- sleep/wrist 직접 누수 feature 제거
- S4 전용 feature set (single LGBM, n_feat=25)

### V62 Full Feature Pipeline (2026-05-17)
- **Train**: 450 rows × 153 cols (142 features + targets)
- **Test**: 250 rows × 146 cols (142 features)
- Column-set 완벽 일치 (original features.parquet 검증 완료)
- 100-seed LGBM ensemble + JTD calibration + per-target n_feat 최적화
- **Leaderboard: 0.6540794708** → 기존 V53 (0.6535822621)보다 0.0005점 개선
- CV avg baseline: 0.5936
- Target별 n_feat: Q1=18, Q2=12, Q3=20, S1=22, S2=16, S3=8, S4=22

### 오늘 (2026-05-08) — V63~V72
- **V63**: Stacking + Temporal + Calibration (LGBM+XGB+Cat → LR meta)
- **V64**: CatBoost+LGBM+XGB Ensemble + Calibration
- **V65**: CatBoost + z-score + temporal + param sweep
- **V66**: CatBoost + leakage-clean + 50 seeds
- **V67**: CatBoost + leakage-clean + 30 seeds + conservative regularization
- **V70**: V61 + V69 ensemble (50/50)
- **V71**: V61 exact replication
- **V72**: V61 + interaction features

## 핵심 아키텍처 패턴

### V53 (Baseline)
```
LGBM seed ensemble (50 seeds)
→ 개인별 z-score feature
→ 타겟별 top-K feature selection
→ Leakage column 제거
```

### V58 (Stacking)
```
Level 0: LGBM + CatBoost + XGBoost → OOF average
Level 1: LogisticRegression(C=1.0) → stacked predictions
Target별: V53 swept config 적용
```

### V61 (Leakage Clean)
```
CatBoost on features_clean_v60
→ LEAK_S/LEAK_Q/NIGHTTIME_LEAK/SLEEP_DIRECT_LEAK 제거
→ S4: single LGBM (n_feat=25)
→ Q*: standard leakage-clean features
```

## 주요 파라미터
- GroupKFold n_splits=5
- n_jobs=1 (WSL2 제한)
- Data: train 450×282, test 250×?
- RAM 16GB + Swap 4GB

## 현재 기준점
- **Best CV**: V53 swept (GroupKFold 3-fold, avg 0.6806)
- **Best Leaderboard**: V62 (0.6540794708)
- **Previous LB Record**: V53 final (0.6535822621)
- **Best Architecture**: V62 — full 142-feature pipeline + 100-seed ensemble + JTD cal

## 파일 구조
- `/home/mwoo423/projects/dacon2/src/` — 모든 실험 스크립트
- `/home/mwoo423/projects/dacon2/submissions/` — 제출물 + 메타
- `/home/mwoo423/projects/dacon2/experiments/` — OOF, 로깅
- `/home/mwoo423/projects/dacon2/docs/` — 분석 문서
- `/home/mwoo423/projects/dacon2/data_processed/` — 전처리 데이터

## 작업 원칙
1. 모든 실험은 메타 JSON 남기기
2. 신규 실험은 V53 swept 기준 비교 후 제출
3. Leak 제거 필수: LEAK_S, LEAK_Q, NIGHTTIME_LEAK, SLEEP_DIRECT_LEAK
4. CV = 리더보드 점수 아님. 오버피팅 주의

## V259~V263 (2026-05-14)
### V259 Isotonic Calibration
- Isotonic regression → Δ=-0.019 (OOF 기준)
-最有效的 calibration method

### V260 Quantile+PSI
- 분석 OOF: 0.614019 (Δ=-0.098555)
- **실제 제출 LB: 0.714592** → 예상과 큰 차이, 버림
- Quantile normalization이 테스트 set에서 역효과

### V262 2x2x2 Factorial (Q × ISO × CLUST)
- Baseline: OOF=0.66141
- **Isotonic**: OOF=0.58819 (Δ=-0.073) ← Biggest gain
- Clustering: Δ=-0.007 (harmful), Quantile: Δ=-0.001
- Best: QFalse_ISOTrue_CLUSTFalse, 252 features
- PSI filter bug: psi.mean() → psi.sum()으로 수정 필요

### V263 Submission LB Analysis
- 246개 submission, 37개 OOF 파일 분석
- **V127이 여전히 BEST**: LB=0.64763, OOF=0.53731
- v53: LB=0.65358 (두 번째)
- v83: OOF=0.54575, estimated LB≈0.646 (가장 근접)
- v45a 100% accuracy → **leakage 발견!**
- v260 LB=0.714592로 확인 → 버림
- 문서: `docs/submission_lb_analysis.md`

## 주요 인사이트
- Distribution correction (Quantile+PSI)은 효과 없음
- Isotonic calibration이 가장 강력: Δ=-0.073
- V127이 260+ 실험 중 **undisputed BEST**
- V45a leakage 발견 → pipeline에 leakage 존재 가능성
- LB 0.50은 매우 어려움 (OOF <0.47 필요)
