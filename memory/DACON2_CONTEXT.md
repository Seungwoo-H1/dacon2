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
- **V53 swept**: LGBM seed ensemble (n_seeds=50)
- **리더보드 점수**: 0.6535822621
- **CV 점수**: ~0.6500
- **피처**: 원본 141 + 개인화 zscore 141 + 메타 12 = 294 cols
- **타깃별 cfg**: Q1=deep/19, Q2=deep/14, Q3=v48/5, S1=wide/21, S2=deep/19, S3=safety/21, S4=wide/20
- **제출 파일**: `submissions/submission_v53_swept_20260507_151447.csv`

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/                    # 훈련/제출 스크립트
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py
│   └── v53_cv_baseline.py
├── submissions/            # 제출물 및 메타
├── experiments/            # 실행 로그
├── data_raw/               # 원본 데이터
└── *.md                    # 계획/분석/레포트 문서들
```

## 🔜 다음 작업 우선순위
1. V53 swept 리더보드 실제 점수 검증
2. Seed 안정성 검증 (50→30/70)
3. cfg 파라미터 tuning (num_leaves, max_depth, lr, n_estimators)
4. 더 넓은 n_feat 탐색 (±5)
5. 논문 초안 작성 (마감 06.26)

## 📝 메모장
- 새로운 실험 결과, 발견사항, 결정사항은 매일 `memory/YYYY-MM-DD.md`에 기록
- 중요한 교훈은 이 파일에 업데이트
