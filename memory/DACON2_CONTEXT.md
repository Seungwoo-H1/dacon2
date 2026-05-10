# DACon2 ETRI 경진대회 — Context File

승우가 매일 새 세션에 들어올 때, 이 파일을 먼저 읽어서 대회 컨텍스트를 복원한다.

## 📌 대회 개요
- **대회명**: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
- **플랫폼**: Dacon (https://dacon.io/competitions/official/236690)
- **태그**: 알고리즘, 정형, 라이프로그, 분류
- **메트릭**: Average Log-Loss (낮을수록 좋음)
- **마감**: 06.26 리더보드/논문, 09.01 코드제출, 10.15 시상식

## 🎯 예측 타깃 (7개, 모두 이진 분류)
| 타깃 | 의미 |
|------|------|
| Q1 | 수면의 질 |
| Q2 | 취침 전 피로도 |
| Q3 | 스트레스 |
| S1 | 총 수면시간 |
| S2 | 수면효율 |
| S3 | 수면 지연시간 |
| S4 | 수면 중 각성시간 |

## 📊 데이터
- 라이프로그 12개 항목, 700일분
- 레이블 7개 지표, 450일분 (train.csv)
- 개인별 데이터 (personalized) 포함

## 🏆 현재 BEST 모델
- **V53 Swept**: LGBM seed ensemble (n_seeds=50) + per-target config
- **리더보드 점수**: **0.6535822621**
- **CV 점수**: **0.6813** (AVG CalCV, 50 seeds, no rolling)
- **피처**: 원본 141 + 개인화 zscore 141 = 282 cols (mean/std drop)
- **target별 cfg**: Q1=deep/19, Q2=deep/14, Q3=v48/11, S1=wide/21, S2=deep/19, S3=safety/23, S4=wide/20
- **제출 파일**: `submissions/submission_v53_swept_20260510_215247.csv` (재생성 완료)
- **생성 스크립트**: `src/gen_submission_v53_swept.py`

## 📈 모델 역사 & 실패 분석
| 버전 | 방식 | OOF/CV | 리더보드 | 상태 |
|---|---|---|---|---|
| V10 | LGBM cal | 0.6038 | — | 베이스라인 |
| V13 | 개인별 z-score | 0.6385 | Worse | ✗ overfit |
| V53 Swept | LGBM swept | 0.6813 (CV) | **0.65358** | ✅ **BEST** |
| V83 | KRR stacking | 0.6499 (OOF) | **0.838** | ✗ 과적합 |
| V94 | rolling + 50seeds | 0.6264 (OOF) | **0.76409** | ✗ rolling 실패 |

## 🔑 핵심 교훈
1. **Rolling features (3/7일 rolling mean+std)**: 오버피팅/분포 mismatch 유발 → **DISCARD**
2. **KRR stacking (V83)**: OOF↑ LB↓ — 모델 복잡도 증가가 항상 LB 개선 아님
3. **OOF과 LB 상관관계 없음**: V92/OOF 0.6264가 V94 LB 0.76409로 이어짐
4. **V53 Swept가 현재 최고**: rolling 없음, n_feat sweep, per-target config
5. **50 seeds가 안정적**

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py  ← 현재 BEST
│   └── v90-v94 optimization scripts
├── submissions/
├── experiments/
├── data_processed/
├── data_raw/
└── memory/DACON2_CONTEXT.md
```

## 🎯 다음 방향
- V53 Swept 유지 (현재 BEST)
- V53 Swept 리더보드 재제출 확인
