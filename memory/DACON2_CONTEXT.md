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

## 🏆 현재 최고 모델 (V53 Swept)
- **V53 Swept**: LGBM seed ensemble (n_seeds=50) + n_feat swept + per-target config
- **리더보드 점수**: **0.6535822621** ← **현재 BEST**
- **CV 점수 (재계산)**: **0.6813** (AVG CalCV, 50 seeds, no rolling)
- **피처**: 원본 141 + 개인화 zscore 141 = 282 cols (mean/std drop)
- **target별 cfg**: Q1=deep/19, Q2=deep/14, Q3=v48/11, S1=wide/21, S2=deep/19, S3=safety/23, S4=wide/20
- **제출 파일**: `submissions/v53_swept_*` 또는 `gen_submission_v53_swept.py`

## 📈 모델 역사 & 교훈
| 버전 | 방식 | OOF/CV | 리더보드 | 비고 |
|---|---|---|---|---|
| V10 | LGBM cal | 0.6038 | — | 베이스라인 |
| V13 | 개인별 z-score | 0.6385 | Worse | overfit |
| V53 | LGBM swept | 0.6813 (CV) | **0.65358** | **BEST** |
| V58 | CatBoost | 0.5904 | ? | OOF은 좋지만 LB 확인 안됨 |
| V62 | Ensemble | ? | ? | — |
| V83 | KRR stacking | 0.6499 (OOF) | 0.838 | **실패** — OOF↑ LB↓ (과적합) |
| V90 | V53 opt | timeout | — | — |
| V91 | rolling experiment | SIGKILL | — | — |
| V92 | rolling vs no | 0.6264 (AVG OOF) | — | rolling이 V92 내에서는 개선, BUT V53 대비 아님 |
| V93 | rolling + 30seeds | 0.6264 | — | — |
| V94 | rolling + 50seeds | 0.6264 | **0.76409** | **실패** — V53 대비 0.11 악화 |

## 🔑 핵심 교훈
1. **Rolling features (3/7일 rolling mean+std)**: OOF에서는 개선 보이나 리더보드에서 **악화**. temporal leakage 또는 distribution mismatch 원인 추정
2. **KRR stacking (V83)**: OOF은 좋아졌으나 LB가 0.838로 폭망 — **OOF과 LB 간 상관관계 없음**에 주의
3. **OOF 기준 최적화 위험**: V92/OOF 0.6264가 V53/OOF 0.6813보다 좋아 보이지만, V53의 0.6813은 V10의 0.6038과 **서로 다른 평가 방식** — V53 Swept의 CV 0.6813은 V92의 OOF과 직접 비교 불가
4. **Personalized z-score는 유효**: base 141 + zscore 141이 V53에서 잘 작동
5. **n_feat sweep 효과 확인**: V53에서 nf ±3 탐색으로 모든 타깃 개선
6. **50 seeds가 안정적**: 30 seeds보다 LB에서 더 안정적인 결과

## ⚠️ 주의사항
- LGBM 훈련 시 `n_jobs=1` (다른 스레드 충돌 방지)
- `verbose=-1`은 `lgb.train()`에 전달, Dataset에는 전달하지 마세요
- `**cfg` unpacking 시 monotone_constraints mismatch — `build_lgb_params()` 사용
- **Rolling features는 현재 실패** — 다시 사용하지 마세요
- OOF 개선을 무조건 LB 개선으로 보지 마세요

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py
│   ├── v90_v53_optimization.py
│   ├── v91_v53_optimization.py
│   ├── v92_v53_rolling.py
│   ├── v93_v53_submission.py
│   └── v94_v53_final.py
├── submissions/
├── experiments/
├── data_processed/
├── data_raw/
└── memory/DACON2_CONTEXT.md
```

## 🎯 다음 방향
- V53 Swept 유지 (현재 BEST)
- 새로운 피처 엔지니어링 접근법 탐색 (rolling 대신)
- CatBoost V58 OOF 0.5904 → 리더보드 검증 필요
- Ensemble (V53 + 다른 모델) 시도
