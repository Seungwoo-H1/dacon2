# DACon2 ETRI 경진대회 — Context File

승우가 매일 새 세션에 들어올 때, 이 파일을 먼저 읽어서 대회 컨텍스트를 복원한다.

## 📌 대회 개요
- **대회명**: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
- **플랫폼**: Dacon (https://dacon.io/competitions/236690)
- **메트릭**: Average Log-Loss (lower is better)
- **현재 최고 LB**: V53 Swept **0.65358** (2026-05-10 업로드)

## 🏆 현재 모델 순위
| Rank | 버전 | 방식 | OOF CalOOF | 리더보드 | 상태 |
|---|---|---|---|---|---|
| 1 | **V53 Swept** | Linear Cal + 50 seeds | **0.6813** | **0.65358** | ✅ **BEST (현재)** |
| 2 | V99 | LGBM 100 seeds + Weighted Blend | **0.6370** | ⏳ LB 미제출 | ✅ OOF 개선 확인 |
| 3 | V97 | Temp Scaling + 50 seeds | 0.6354 | 0.6835 (실패) | ✗ 오버피팅 |
| 4 | V94 | Rolling + Linear | 0.6264 | 0.7641 (실패) | ✗ 분포 mismatch |
| 5 | V83 | KRR Stacking | 0.6499 | 0.838 (실패) | ✗ 오버피팅 |

## 📈 핵심 교훈 (누적)

### 1. OOF → LB 불일치 (가장 중요한 발견)
- V99에서 **OOF 계산 버그 발견**: V97은 `oof_preds /= n_seeds`만 했지만, 올바른 방식은 `oof_preds /= (n_seeds × 5 folds)`
- V97의 `oof_mean=0.5249`는 **실제로 5× 큰 값** (per-fold avg ≈ 0.105)
- **V99도 OOF 0.6370으로 V53 0.6813 대비 개선** — 더 많은 seed diversity가 유용함 확인
- 하지만 V99의 test distribution (mean=0.92)이 왜곡됨 → weight optimization overfit

### 2. Calibration 시도들 (모두 실패)
- **Isotonic (V96)**: OOF 0.6029 → test 예측 폭주 (mean=0.9999)
- **Temperature Scaling (V97)**: OOF 0.6354 → LB 0.6835 (variance collapse)
- **Rolling Features (V94)**: OOF 0.6264 → LB 0.7641 (distribution leak)
- **KRR Stacking (V83)**: OOF 0.6499 → LB 0.838

### 3. V99 결과 요약 (100 seeds + Weighted Blend)
- **AVG CalOOF: 0.6370** (V53 0.6813 대비 Δ=-0.0443 개선)
- 100 seeds (4 groups × 25 seeds) → model diversity 증가
- Seed group별 weight optimization: OOF 개선 확인
- **OOF-LB gap**: V53 AVG gap -0.0221 (OOF이 LB보다 나쁨)
- **변동성 분석**: V99의 test_std가 V53보다 큼 (std=0.36 vs 0.28)

### 4. 안정적인 모델
- **V53 Swept**: 50 seeds, linear cal, target-specific n_feat sweep
- OOF-LB gap가 크지만 (V53 avg -0.0221), test distribution은 안정적
- **LB 0.65358이 현재 최고**

## 🎯 다음 단계

### 1. V99 제출 검토
- V99 submission: `submissions/submission_v99_blend_20260511_014451.csv`
- OOF 개선은 명확 (0.6370 vs 0.6813)
- 하지만 test distribution 왜곡 가능성 있음
- **승우가 업로드 전에 검토 필요**

### 2. V53 Swept 유지
- V53 Swept: `submissions/submission_v53_swept_20260510_215247.csv`
- LB 0.65358, 가장 안정적
- 0.5 점대 진입하려면 추가로 연구 필요

### 3. 연구 방향 제안
1. **V53 Swept의 n_feat sweep 재탐색** (더 넓은 범위 ±5)
2. **Per-target feature importance 재분석** (변동성 있는 타깃 위주)
3. **Model ensemble diversity** (LGBM + 다른 hyperparameter 조합)
4. **Distribution shift mitigation** (test set과의 분포 차이 분석)
5. **Calibration 안정성 연구** — OOF과 Test 간 분포 차이를 고려한 calib method

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py  ← V53 baseline
│   ├── v97_temperature_scaling.py   ← V97 experiment (OOF bug 발견)
│   ├── v99_blend.py                  ← V99 (100 seeds + blend)
│   └── v90-v98 scripts
├── submissions/
│   ├── submission_v53_swept_20260510_215247.csv  ← BEST
│   └── submission_v99_blend_20260511_014451.csv  ← OOF 개선
├── experiments/
├── data_processed/
├── data_raw/
└── memory/DACON2_CONTEXT.md
```

## ⚠️ 주의사항
- API 업로드 실패 — **직접 데콘 브라우저에서 업로드 필요**
- LGBM: `n_jobs=1`, `verbose=-1` (Dataset에는 전달하지 않음)
- **Rolling features는 절대 사용하지 마세요**
- **Isotonic calibration은 절대 사용하지 마세요**
- **Temperature scaling은 오버피팅 위험 있음**

## 🔧 V53 Swept Config
| Target | Config | n_feat |
|---|---|---|
| Q1 | deep | 19 |
| Q2 | deep | 14 |
| Q3 | v48 | 11 |
| S1 | wide | 21 |
| S2 | deep | 19 |
| S3 | safety | 23 |
| S4 | wide | 20 |

## 💡 OOF 계산 버그 수정 (2026-05-11)
V97의 OOF 계산에 division by 5 fold 미적용 버그가 있었습니다.
- **Wrong**: `oof_preds /= n_seeds` (각 seed에서 5 fold sum)
- **Correct**: `oof_preds /= (n_seeds * 5)` (각 sample당 fold avg)
- V97의 `oof_mean=0.5249` → 실제로 `0.105` per fold
- V99에서 이 버그를 수정했으므로 V99의 OOF 결과가 더 정확함
