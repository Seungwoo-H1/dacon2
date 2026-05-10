# DACon2 ETRI 경진대회 — Context File

승우가 매일 새 세션에 들어올 때, 이 파일을 먼저 읽어서 대회 컨텍스트를 복원한다.

## 📌 대회 개요
- **대회명**: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
- **플랫폼**: Dacon (https://dacon.io/competitions/official/236690)
- **메트릭**: Average Log-Loss (lower is better)

## 🏆 현재 모델 순위
| Rank | 버전 | 방식 | OOF CalOOF | 리더보드 | 상태 |
|---|---|---|---|---|---|
| 1 | **V97** | Temp Scaling + V53 Swept | **0.6354** | ⏳ 대기 | ✅ **submit 필요** |
| 2 | V53 Swept | Linear Cal + V53 Swept | 0.6813 | 0.65358 | ✅ BEST (기존) |
| 3 | V94 | Rolling + Linear | 0.6264 | 0.76409 | ✗ 실패 |
| 4 | V83 | KRR Stacking | 0.6499 | 0.838 | ✗ 실패 |

## 🎯 V97 (Temperature Scaling) — 다음 제출 대상
- **AVG CalOOF**: 0.6354 (V53 0.6813 대비 **-0.0459** 개선)
- **Temperature T**: Q1=3.246, Q2=1.917, Q3=1.733, S1=1.236, S2=2.260, S3=1.292, S4=1.602
- **가장 큰 개선**: S2 (-0.1243), Q1 (-0.1022), Q3 (-0.0412), Q2 (-0.0370)
- **S3만 미미히 악화** (+0.0070) — 전체 AVG로는 크게 개선
- **제출 파일**: `submissions/v97_submission.csv` (250 rows, 10 cols)
- **생성 스크립트**: `src/v97_submit.py`
- **📌 데콘 리더보드에 직접 업로드 필요**

## 📈 모델 역사 & 교훈
1. **Rolling features (3/7일 rolling)**: 오버피팅/분포 mismatch → **DISCARD**
2. **KRR stacking (V83)**: OOF↑ LB↓ — **DISCARD**
3. **Isotonic (V96)**: OOF에서는 0.6029 개선 but test 예측 시 폭주 → **OVERFIT**
4. **Temperature scaling (V97)**: OOF 0.6354, test 예측 안정적 → **CURRENT BEST**
5. **V53 Swept가 기본**: rolling 없음, n_feat swept, per-target config
6. **50 seeds가 안정적**

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py  ← V53 baseline
│   ├── v97_temperature_scaling.py   ← V97 experiment
│   ├── v97_submit.py                ← V97 submission generator
│   └── v90-v94 scripts
├── submissions/
│   ├── submission_v53_swept_*.csv   ← V53 BEST
│   └── v97_submission.csv           ← V97 (제출 대상)
├── experiments/
├── data_processed/
├── data_raw/
└── memory/DACON2_CONTEXT.md
```

## ⚠️ 주의사항
- API 업로드 실패 — **직접 데콘 브라우저에서 업로드 필요**
- LGBM: `n_jobs=1`, `verbose=-1` (Dataset에는 전달하지 않음)
- **Rolling features는 절대 사용하지 마세요**
