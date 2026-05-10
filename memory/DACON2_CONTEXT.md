# DACon2 ETRI 경진대회 — Context File

승우가 매일 새 세션에 들어올 때, 이 파일을 먼저 읽어서 대회 컨텍스트를 복원한다.

## 📌 대회 개요
- **대회명**: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
- **플랫폼**: Dacon (https://dacon.io/competitions/official/236690)
- **태그**: 알고리즘, 정형, 라이프로그, 분류
- **메트릭**: Average Log-Loss (낮을수록 좋음)
- **마감**: 06.26 리더보드/논문, 09.01 코드제출, 10.15 시상식
- **핵심 요구사항**: 리더보드 제출 + 논문 제출(ICTC 2026) + 재현성 검증

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
- 라이프로그 12개 항목 (activity, hr, light, ambience, wifi, gps 등), 700일분
- 레이블 7개 지표, 450일분 (train.csv)
- 개인별 데이터 (personalized) 포함

## 🏆 현재 최고 모델
- **V94**: LGBM seed ensemble (n_seeds=50) + rolling features + per-target config
- **AVG CalOOF**: 0.6264 (7 targets)
- **rolling features 효과가 모든 타깃에서 +0.008~0.019 CalOOF 개선**
- **S3 가장 큰 개선 (+0.0189), S1 두 번째 (+0.0181)**
- **target별 cfg**: Q1=deep/23, Q2=deep/23, Q3=v48/20, S1=wide/17, S2=deep/23, S3=safety/17, S4=wide/17
- **제출 파일**: `submissions/v94_submission.csv`
- **OOF vs Leaderboard**: V94는 OOF 기준, V53 리더보드 0.65358 대비 OOF 0.6264

## 📈 모델 역사
- **V10**: LGBM cal OOF 0.6038 (베이스라인)
- **V53**: LGBM swept 리더보드 0.65358 (baseline)
- **V53 Swept**: AVG CV 0.6500 (n_feat ±3 탐색)
- **V83**: KRR stacking 실패 (OOF 0.6499 vs LB 0.838) — 과적합
- **V90**: V53 opt (timeout)
- **V91**: rolling experiment (SIGKILL)
- **V92**: rolling features 확정 (AVG CalOOF 0.6264)
- **V93**: rolling + 30 seeds submission
- **V94**: rolling + 50 seeds 최종 submission

## 🔑 핵심 발견
1. **Rolling features(3일/7일 rolling mean+std)**가 V53 대비 CalOOF 0.008~0.019 개선
2. **시드 수 30→50의 개선 미미** (Δ<0.001) — 30 seeds면 충분하나 50으로 제출
3. **n_feat ±3 탐색은 V53 sweep 때 이미 검증됨**
4. **LGBM이 FT-Transformer(0.5847 AVG AUC)보다 안정적**
5. **S1은 DL(LGBM)이 LGBM보다 오히려 더 좋음** (+0.0655 AUC)

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/                    # 훈련/제출 스크립트
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py
│   ├── v90_v53_optimization.py
│   ├── v91_v53_optimization.py
│   ├── v92_v53_rolling.py
│   ├── v93_v53_submission.py
│   └── v94_v53_final.py
├── submissions/            # 제출물 및 메타
├── experiments/            # 실행 로그
├── data_processed/         # preprocessed data (parquet)
├── data_raw/               # 원본 데이터
└── *.md
```

## ⚠️ 주의사항
- LGBM 훈련 시 `n_jobs=1`로 설정해야 다른 스레드와 충돌 없음 (1600% CPU 사용 방지)
- `verbose=-1`은 Dataset에 전달하지 말고 `lgb.train()`에 전달
- `**cfg` unpacking 시 monotone_constraints mismatch 오류 발생 — `build_lgb_params()` 사용
- RAM 15GB, CPU 24코어, WSL2 환경
