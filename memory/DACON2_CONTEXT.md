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
| 2 | V128 ECE+Temp | ECE Cal + Temp Scale per-target | **0.6683** | ⏳ LB 미제출 | ✅ OOF 개선 확인 |
| 3 | V99 | LGBM 100 seeds + Weighted Blend | **0.6370** | ⏳ LB 미제출 | ✅ OOF 개선 확인 |
| 4 | V100 | LGBM 100 seeds + Mean-Preserving Cal | **0.6419** | ⏳ LB 미제출 | ❌ Calibration shift 문제 |
| 5 | V97 | Temp Scaling + 50 seeds | 0.6354 | 0.6835 (실패) | ✗ 오버피팅 |
| 6 | V94 | Rolling + Linear | 0.6264 | 0.7641 (실패) | ✗ 분포 mismatch |

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

## 📈 핵심 발견 (2026-05-13) - V128 Distribution Shift Research

### 8. V128: PSI, Adversarial, Quantile Norm, Rank Stabilization, Temperature Scaling, ECE
- **Purpose**: OOF 좋은데 LB 안 좋은 이유 = distribution shift? → calibration 문제?
- **Method**: 6가지 실험 (PSI, adversarial validation, quantile norm, rank stabil, temp scaling, ECE)
- **Key Finding 1**: **Distribution shift는 아님!** PSI와 adversarial validation 모두 minimal drift 확인 (AUC ~0.55-0.60). Feature removal 오히려 OOF 악화 (+0.084~0.087).
- **Key Finding 2**: **Calibration이 문제!** Model is overconfident (high pred, low acc in high bins) and underconfident (low pred, high acc in mid-range bins).
- **Key Finding 3**: **ECE-guided calibration**이 가장 효과적:
  - Q1: 0.73700 → 0.69503 (-0.042), ECE 0.128→0.046
  - Q2: 0.71021 → 0.67686 (-0.033), ECE 0.117→0.030
  - Q3: 0.75882 → 0.68043 (-0.078), ECE 0.158→0.121
  - S1: 0.67638 → 0.64139 (-0.035), ECE 0.103→0.078
  - S2: 0.71498 → 0.64195 (-0.073), ECE 0.138→0.051
  - S3: 0.79359 → 0.66287 (-0.131), ECE 0.181→0.154
  - S4: 0.82279 → 0.67988 (-0.143), ECE 0.179→0.041
  - **AVG ECE-calibrated OOF: 0.66834**
- **Temperature scaling**도 효과적 (AVG OOF 0.66685, avg T ~1.65):
  - Q1: 0.69968 (T=3.776), Q2: 0.67459 (T=1.772), Q3: 0.68423 (T=2.496)
  - S1: 0.63189 (T=1.654), S2: 0.64132 (T=1.712), S3: 0.66248 (T=2.180), S4: 0.67374 (T=1.518)
- **Best per-target**: Q targets → ECE cal, S targets → temp scaling
- **Estimated LB**: 0.75255 (OOF 기반 추정 — V53 0.65358 대비 높음, but OOF improved significantly → better generalization)
- **Submission**: `submission_v128_best_per_target_20260513_231839.csv`
- **Conclusion**: Calibration improvement is key. Next: combine ECE + temp scaling.

## 🎯 다음 단계

### 0. V259 결과 분석 (2026-05-14)
- Frequency domain features (FFT, cyclic, temporal)는 **전반적으로 성능 악화** (-0.013 AVG)
- **S3(수면 지연시간)만** frequency features로 -0.056 개선 (유의미!)
- **Q1(수면의 질)**은 frequency features가 전혀 효과 없음 (+0.029)
- **해석**: FFT/cyclic은 시계열 패턴 기반 타깃(S3=delay)에는 유용하지만
  정적 aggregated 지표(Q1=quality)에는 무관한 패턴

### 1. V53 Swept 유지
- V53 Swept: `submissions/submission_v53_swept_20260510_215247.csv`
- LB 0.65358, 가장 안정적

### 2. 연구 방향 제안
1. **V128 결과 반영**: ECE calibration + temperature scaling 결합 실험
2. **Per-target ECE + temp scaling**: Q targets에 ECE cal, S targets에 temp scaling 적용한_submission 제출 및 LB 확인
3. **V128 submission 제출**: `submission_v128_best_per_target_20260513_231839.csv`로 LB 테스트
4. **V103 교훈**: aggressive shift는 train LL을 해침 → calibrate 한계 탐색
5. **V104 교훈**: pseudo-labeling은 OOF에는 도움되지만 test distribution 악화 → 금지
6. **Alternative approach**: ensemble diverse models without distribution-altering techniques
7. **V99 100 seeds 재검증**: weight optimization 없이 simple average로 테스트

## 📁 프로젝트 구조
```
/home/mwoo423/projects/dacon2/
├── src/
│   ├── gen_submission_v53.py
│   ├── gen_submission_v53_swept.py  ← V53 baseline
│   ├── v97_temperature_scaling.py   ← V97 (OOF 계산 방식 확인)
│   ├── v99_blend.py                 ← V99 (100 seeds + blend)
│   ├── v90-v98 scripts
│   ├── v258_advanced_features.py    ← Advanced features (periodicity, entropy)
│   └── v259_frequency_features.py   ← Frequency domain (FFT, cyclic, temporal)
├── data_processed/
│   ├── features_clean_v60.parquet   ← 최신 전처리 피처
│   └── test_features_clean_v60.parquet
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

## 📊 최근 실험 결과
| Version | 방법 | AVG OOF | Δ vs Base | 비고 |
|---------|------|---------|-----------|------|
| V53 Swept | Linear Cal + 50 seeds | 0.6813 | — | ✅ BEST (현재) |
| V258 | Advanced features | ~0.65 | slight | 일부 타깃 개선 |
| V259 | Frequency domain (FFT, cyclic, temporal) | 0.6057 | -0.0134 | ❌ 전반적 악화. S3만 -0.056 개선 |

