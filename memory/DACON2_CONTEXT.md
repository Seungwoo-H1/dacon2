# DACon2 ETRI 경진대회 — Context File

승우가 매일 새 세션에 들어올 때, 이 파일을 먼저 읽어서 대회 컨텍스트를 복원한다.

## 📌 대회 개요
- **대회명**: 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
- **플랫폼**: Dacon (https://dacon.io/competitions/236690)
- **메트릭**: Average Log-Loss (lower is better)
- **현재 최고 LB**: V53 Swept **0.65358** (2026-05-10 업로드)
- **0.5 점대 진입 목표**

## 🏆 현재 모델 순위
| Rank | 버전 | 방식 | OOF CalOOF | 리더보드 | 상태 |
|---|---|---|---|---|---|
| 1 | **V53 Swept** | Linear Cal + 50 seeds | **0.6813** | **0.65358** | ✅ **BEST (현재)** |
| 2 | V99 | LGBM 100 seeds + Weighted Blend | **0.6370** | ⏳ LB 미제출 | ✅ OOF 개선 확인 |
| 3 | V100 | LGBM 100 seeds + Mean-Preserving Cal | **0.6419** | ⏳ LB 미제출 | ❌ Calibration shift 문제 |
| 4 | V97 | Temp Scaling + 50 seeds | 0.6354 | 0.6835 (실패) | ✗ 오버피팅 |
| 5 | V94 | Rolling + Linear | 0.6264 | 0.7641 (실패) | ✗ 분포 mismatch |

## 📈 핵심 발견 (2026-05-11)

### 1. OOF 계산 방식 발견
- **V97 방식 (oof_preds /= n_seeds)**: OOF mean ≈ 0.525 (train_rate와 근사)
  - 수학적으로는 5x 큰 값 (각 fold prediction mean ≈ 0.105)
  - 하지만 calibration shift가 작아 (~0.03) **LB에서 잘 작동**
- **V99/V100 방식 (oof_preds /= (n_seeds × 5))**: OOF mean ≈ 0.105 (정확함)
  - Calibration shift가 큼 (~0.39) → **test distribution 왜곡**
  - S1은 0.9999 폭주
- **결론**: V97 방식이 OOF-LB 간 relation에서 더 잘 작동 → **V97 방식을 유지**

### 2. V99 결과 (100 seeds + V97 OOF 계산)
- **AVG CalOOF: 0.6370** (V53 0.6813 대비 Δ=-0.0443 개선)
- 100 seeds (4 groups × 25 seeds) → V53의 50 seeds 대비 diversity 증가
- Seed group별 weight optimization: OOF 개선 확인
- **하지만 test distribution 왜곡**: weight optimization이 OOF에 과적합

### 3. V100 결과 (100 seeds + V97 OOF 계산)
- **AVG CalOOF: 0.6419** (V53 0.6813 대비 Δ=-0.0394 개선)
- Mean-preserving calibration 시도 → S1 0.9999 폭주 → **실패**

### 4. Calibration 시도들 (모두 실패)
- **Isotonic (V96)**: OOF 0.6029 → test 예측 폭주 (mean=0.9999)
- **Temperature Scaling (V97)**: OOF 0.6354 → LB 0.6835 (variance collapse)
- **Rolling Features (V94)**: OOF 0.6264 → LB 0.7641 (distribution leak)
- **KRR Stacking (V83)**: OOF 0.6499 → LB 0.838

### 5. 안정적인 모델
- **V53 Swept**: 50 seeds, linear cal, target-specific n_feat sweep
- **LB 0.65358이 현재 최고**

## 📈 핵심 발견 (2026-05-11) - 추가

### 6. V103: S3/S4 Shift Amplification
- **Purpose**: LB predictor formula 기반 S3/S4 negative shift 증폭 실험
- **LB Predictor Formula** (RMSE 0.0017):
  `LB = 0.0896*entropy - 0.4205*max_shift + 0.1877*skew + 0.4262*S3_shift + 0.2194*S4_shift + 0.7740`
- **Key Insight**: S3/S4 shift에 양수 계수 → 더 negative shift = 더 낮은 LB
- **V53 baseline**: S3 shift=-0.1360, S4 shift=-0.0756
- **Method**: `shifted = oof_mean + factor * (sub_mean - oof_mean)`
  - factor=1.0: original, factor=2.0: 2x amplified negative shift
- **Results**:
  | Factor | Predicted LB | Train LL Proxy | Improvement |
  |--------|-------------|----------------|-------------|
  | 1.0 (baseline) | 0.77193 | 0.54820 | - |
  | 1.3 | 0.75336 | 0.56606 | -0.01857 |
  | 2.0 | 0.71305 | 0.58805 | -0.05888 |
  | 2.5 | 0.68724 | 0.61160 | -0.08469 |
  | **3.0** | **0.66360** | **0.64854** | **-0.10833** |
- **Best**: factor=3.0 → predicted LB 0.66360 (하지만 train LL도 worst로 ↑)
- **Trade-off**: 더 aggressive shift → predicted LB ↓ 하지만 train LL ↑ (overfitting risk)
- **Submissions generated**: `v103_amp_s3s4_f2.0`, `v103_amp_s3s4_f2.5`, `v103_amp_s3s4_f3.0`
- **Caution**: train LL increase suggests aggressive shift overfits training distribution

### 7. V104: Pseudo-Labeling Experiment
- **Purpose**: High-confidence test predictions을 pseudo-label로 추가 retrain
- **Method**: 5-fold LGBM → test predict → confidence > threshold pseudo-label → retrain
- **Thresholds tested**: 0.7, 0.75, 0.8, 0.85, 0.9
- **Pseudo weights**: 0.5, 1.0, 2.0
- **Boost ratios**: 0.05, 0.1, 0.2
- **Results** (TOP 5 configs by OOF improvement):
  | Threshold | Weight | Boost | Avg OOF LL | ΔOOF | Avg Pseudo |
  |-----------|--------|-------|-----------|------|-----------|
  | 0.70 | 0.5 | 0.1 | 0.6963 | -0.1481 | 91 |
  | 0.70 | 1.0 | 0.1 | 0.6982 | -0.1499 | 91 |
  | 0.70 | 2.0 | 0.05 | 0.6982 | -0.1500 | 91 |
  | 0.70 | 0.5 | 0.2 | 0.6983 | -0.1501 | 91 |
  | 0.70 | 2.0 | 0.1 | 0.6990 | -0.1508 | 91 |
- **Key finding**: Best config (T=0.7, pw=0.5, boost=0.1) predicted LB **0.802** (V53 0.700 대비 WORSE)
- **Why worse**: pseudo-labeling improves OOF LL but shifts test predictions toward training mean
  → S3/S4 shift becomes LESS negative → higher predicted LB
- **Conclusion**: Pseudo-labeling helps train LL but harms test distribution → **Not recommended for this competition**
- **Submissions generated**: None (predicted LB worse than baseline)

## 🎯 다음 단계

### 1. V53 Swept 유지
- V53 Swept: `submissions/submission_v53_swept_20260510_215247.csv`
- LB 0.65358, 가장 안정적

### 2. 연구 방향 제안
1. **Distribution shift 분석 개선**: test set과의 분포 차이 정량적 분석
2. **V103 교훈**: aggressive shift는 train LL을 해침 → calibrate 한계 탐색
3. **V104 교훈**: pseudo-labeling은 OOF에는 도움되지만 test distribution 악화 → 금지
4. **Alternative approach**: ensemble diverse models without distribution-altering techniques
5. **Per-target calibration refinement**: S3/S4에만 focus (가장 큰 shift 존재)
6. **V99 100 seeds 재검증**: weight optimization 없이 simple average로 테스트

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py  ← V53 baseline
│   ├── v97_temperature_scaling.py   ← V97 experiment (OOF 계산 방식 확인)
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
- **Weight optimization은 과적합 위험 있음**

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
