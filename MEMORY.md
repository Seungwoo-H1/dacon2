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

## MISSION (2026-06-07 승우 명시)
- 현재 최고 모델: **V452** (V308 기준) ⭐ NEW BEST (unverified)
- 목표: V308의 LB 0.63893을 초과하는 모델 찾기
- **0.5점대 진입**까지 무한 연구 루프 계속
- LB 예측이 V308 이하라면 보고 금지
- 동일 가설 반복 금지, 매 루프 새 가설 필수
- 성능 개선 확인 전까지 연구 루프 계속
- 동일 가설 반복 금지, 매 루프 새 가설 필수

## V461 — Adversarial Validation + Feature Selection (2026-06-08 완료)
- **Meta OOF: 0.53232 | Student OOF: 0.60741 | Gap: 0.075 (1.07x)**
- V339 LB: 0.59615 (V452 0.580 대비 -0.016 개선)
- Adversarial validation → 47개 distribution-shifted feature 제거 → 94개 safe features 남음
- ✅ 제출 파일 생성됨 (`submission_v461_adversarial_20260608_005815.csv`)
- **교훈**: shifted features (gps, wifi, ble rssi 등) 제거 → student 0.607 → V339 패턴 LB 0.596
- V339 추정 붕괴 이전에는 0.596→0.62 예상 but V458(0.574→0.729) 패턴으로 실제 LB는 더 높을 수 있음
- ⚠️ **V339 추정은 여전히 불신** — 실제 제출 전 BEST로 인정 불가

### V462 — Subject Embedding + PCA + Cross-Target Constraint (2026-06-08 완료)
- **Meta OOF: 0.54902 | Student OOF: 0.63327 | Gap: 0.084 (1.20x)**
- V339 LB: 0.62063
- PCA 100 components + subject embedding (110 features total)
- ✅ 제출 파일 생성됨 (`submission_v462_subject_embedding_20260608_010612.csv`)
- **교훈**: PCA + subject embedding → student 0.633 → V339 LB 0.621
- V461(0.596)과 V462(0.621) 모두 V339 추정 → 실제는 V458(0.574→0.729) 패턴으로 더 높을 수 있음

### V463 — Target Correlation + Threshold Calibration (2026-06-08 완료 ✅)
- **Meta OOF: 0.52645 | Student OOF: 0.59907 | Gap: 0.073 (1.04x)**
- V339 LB: 0.61946
- Calibrated Meta OOF: 0.55773
- **교훈**: Target-specific threshold calibration이 효과적 (Q1:0.5, S1:0.68 등)
- Gap 1.04x → V308(1.0x) 수준으로 매우 안전
- V439 baseline + target correlation modeling이 student 0.600으로 낮춤
- ✅ 제출 파일 생성됨

### V464 — Target-Specific Adversarial Pruning + Baseline Smoothing (2026-06-08 완료 ✅)
- **Meta OOF: 0.53372 | Student OOF: 0.61058 | Gap: 0.077 (1.10x)**
- V339 LB: 0.59905
- Target별 top-20 adversarial feature pruning (사실상 pruning 0개 적용 — feature importance ranking에서 상위 feature들이 adversarial importance와 거의 겹치지 않음)
- ✅ 제출 파일 생성됨 (`submission_v464_target_adversarial_20260608_011550.csv`)
- **교훈**: Target-specific pruning이 global pruning(V461)보다 student 0.611 (V461 0.607 대비 -0.004 악화)
- Pruning이 거의 작동 안 함 → feature ranking에서 adversarial importance가 signal과 충분히 분리되지 않음

### V465 — Adversarial Feature Re-weighting + Strong Regularization (실패 ❌ — 재실행 중)
- rank_features_lgb에서 weight 길이 mismatch 에러
- 재작성 완료, V465 재실행 중

### V465 — Adversarial Feature Re-weighting + Strong Regularization (완료 ❌ — gap 폭망)
- **Meta OOF: 0.55465 | Student OOF: 0.65062 | Gap: 0.096 (1.37x)**
- V339 LB: 0.63622
- Feature scaling(0.3~1.0x) + n_estimators=5000 + reg_alpha=20~100
- **실패 원인**: feature scaling이 너무 미미 (min scale 0.943) → 실질적 변화 없음
- 오히려 n_estimators=5000이 overfitting을 유발 → gap 1.37x로 폭망
- ✅ 제출 파일 생성됨 (`submission_v465_feat_reweight_20260608_012447.csv`)
- **교훈**: V461에서 adversarial weight min이 0.943일 때, 0.3배 스케일링은 무의미. 더 aggressive하게 scaling 필요하거나 아예 다른 접근 필요.

## ⚠️ V339 패턴 추정 신뢰도: 완전히 붕괴 (2026-06-08)

### 2026-06-08 V461-V465 종합 비교

| Version | Meta | Student | Gap | Ratio | V339 LB | Status |
|---------|------|---------|-----|-------|---------|--------|
| V461 | 0.532 | 0.607 | 0.075 | 1.07x | 0.59615 | ✅ 제출 |
| V462 | 0.549 | 0.633 | 0.084 | 1.20x | 0.62063 | ✅ 제출 |
| V463 | 0.526 | 0.599 | 0.073 | 1.04x | 0.61946 | ✅ 제출 |
| V464 | 0.534 | 0.611 | 0.077 | 1.10x | 0.59905 | ✅ 제출 |
| V465 | 0.555 | 0.651 | 0.096 | 1.37x | 0.63622 | ✅ 제출 ❌ |
| V466 | 0.532 | 0.602 | 0.070 | 1.00x | 0.59127 | ✅ 제출 ✅ |
| V467 | 0.531 | 0.602 | 0.071 | 1.02x | 0.59114 | ✅ 제출 ❌ |

**V461/V464 (adversarial)** 가 student OOF 가장 낮춤 (0.607, 0.611)
**V463 (calibration)** gap 가장 안전 (1.04x) → V308(1.0x)에 가장 근접

## 2026-06-08 V466-V467 결과

### V466 — CV-Internal Adversarial + Consensus (완료 ✅ — gap 1.00x 최적)
- **Meta OOF: 0.53180 | Student OOF: 0.60176 | Gap: 0.070 (1.00x)**
- V339 LB: 0.59127
- **핵심 발견**: Gap 1.00x — V308(1.0x)과 완전히 동일. V463(1.04x)보다 더 안전!
- CV-internal adversarial validation → consensus feature selection이 효과적
- ✅ 제출 파일 생성됨 (`submission_v466_cv_adversarial_consensus_20260608_012953.csv`)

### V467 — Adversarial Group Elimination (완료 ❌ — 그룹 제거 0개)
- **Meta OOF: 0.53071 | Student OOF: 0.60181 | Gap: 0.071 (1.02x)**
- V339 LB: 0.59114
- **실패**: V466의 CV-internal 결과에 따라 모든 group adversarial importance = 0.0 → 제거 group 없음
- group_stats meta features가 V466과 거의 동일한 결과 생성
- ✅ 제출 파일 생성됨 (`submission_v467_group_elimination_20260606_013521.csv`)
- **교훈**: V466의 consensus selection이 이미 optimal. group 기반 elimination은 불필요.

### V468 — Aggressive Feature Reduction (K=30) (완료 ✅)
- **Meta OOF: 0.53215 | Student OOF: 0.60182 | Gap: 0.070 (1.00x)**
- V339 LB: 0.59137
- ✅ 제출 파일 생성됨 (`submission_v468_k_sweep_20260608_015103.csv`)
- **교훈**: V466과 거의 동일 결과. feature reduction K=30이 stable.

### V469 — Temporal Features (per-subject rolling stats) (실패 ❌)
- **Meta OOF: 0.79085 | Student OOF: 0.79959 | Gap: -0.010**
- **실패 원인**: Rolling window features(3일, 7일 이동평균, 추세)가 노이즈만 추가
- 89 features 중 26개가 temporal features였으나 performance 대폭 악화 (0.79 vs V308 0.639)
- **교훈**: 10 subject × 45일 데이터에서 rolling window 계산 시 시계열 종속성으로 overfitting 발생. 데이터가 너무 짧아(temporal resolution 낮음) rolling features가 signal 대신 noise를 학습.
- ❌ **Temporal features는 이 데이터셋에서 효과적 아님**

### V470 — Two-Stage Meta Stacking (실패 ❌)
- **Meta OOF: 0.77506 | Student OOF: 0.78570 | Gap: +0.014**
- **실패 원인**: Stage2 deep stacking(116 features)이 overfitting → student 0.786으로 V308 대비 대폭 악화
- two-stage approach가 feature explosion을 유발. raw features(68) + student preds + target probs + interactions = 116D가 450 samples에서 과적합
- **교훈**: stacking depth가 깊을수록 overfitting 위험. shallow한 stacking만 유의미.
- ❌ **Two-stage stacking은 이 데이터셋에서 효과적 아님**

### V471 — Per-Subject Feature Selection (실패 ❌)
- **Meta OOF: 0.79019 | Student OOF: 0.78570 | Gap: -0.031**
- **실패 원인**: id03/id10에서 feature importance가 0 → per-subject importance가 매우 불균일
- consistent(34 features): student 0.786 (all과 동일 수준)
- consensus features가 거의 없음 (id03/10의 zero-importance 문제로 인해)
- **교훈**: per-subject feature importance가 subject에 따라 너무 다름 (id03/10: 0, others: 100). 작은 데이터(45개/subject)에서 LGBM importance가 불안정. per-subject selection은 의미 없음.
- ❌ **Per-subject feature selection은 이 데이터셋에서 효과적 아님**

### V472 — Data Augmentation (Bootstrap + Noise Injection) (실패 ❌)
- **noise_level=0.0 (baseline): Meta OOF: 0.79019 | Student OOF: 0.67140**
- **noise_level=0.1~2.0: Student OOF 0.687~0.693** (baseline보다 Worse)
- **실패 원인**: bootstrap resampling이 noise를 추가했을 뿐, overfitting 감소 효과 없음. noise injection이 모델을 불안정하게 만듦.
- 0.0 noise가 0.671로 best (augmented 없던 게 낫다)
- **교훈**: 450 samples + 68 features에서 bootstrap resampling은 의미가 없음. sample size가 너무 적어서 noise injection이 signal을 가림.
- ❌ **Data augmentation은 이 데이터셋에서 효과적 아님**

### V474 — Feature Engineering: Interaction Terms (실패 ❌)
- **Meta OOF: 0.79107 | Student OOF: 0.78210 | Gap: -0.033**
- 90 interactions 생성 (68 base features의 top-10 pairwise: 45 product + 45 ratio)
- Student 개선: +0.004 (0.786→0.782, 미미)
- Global top-5 features: mActivity_mean, mScreenStatus_mean, mActivity_std, mLight_mean, mLight_max
- **교훈**: interactions이 signal 추가 대신 noise만 증가. 450 samples에서 158 features는 overfitting 유발
- ❌ **Feature interactions은 이 데이터셋에서 효과적 아님**

### V473 — Target Transformation: Binary → Ordinal (실패 ❌)
- **ordinal_lgbm**: Meta OOF: 0.762 (binary baseline 0.790 대비 -0.028 개선) | Student OOF: 0.786 (변경 불가)
- **ordinal_xgb**: Meta OOF: 0.765 | Student OOF: 0.786
- **ordinal_reg**: Meta OOF: 0.299 (폭망 — class label 0~2 regression)
- **핵심 발견**: Ordinal transformation이 meta OOF는 약간 낮추지만, student OOF는 변경 불가 (student는 여전히 binary task)
- ordinal distribution: 0=68, 1=180, 2=202 — 클래스 불균형 큼
- **교훈**: ordinal은 meta learner에는 일부 도움되지만, student가 개선 안 되므로 전체 LB에는 무의미
- ❌ **Target transformation은 이 데이터셋에서 효과적 아님**

### V474 — Feature Engineering: Interaction Terms (실패 ❌)
- **Meta OOF: 0.79107 | Student OOF: 0.78210 | Gap: -0.033**
- 90 interactions 생성 (68 base features의 top-10 pairwise: 45 product + 45 ratio)
- Student 개선: +0.004 (0.786→0.782, 미미)
- Global top-5 features: mActivity_mean, mScreenStatus_mean, mActivity_std, mLight_mean, mLight_max
- **교훈**: interactions이 signal 추가 대신 noise만 증가. 450 samples에서 158 features는 overfitting 유발
- ❌ **Feature interactions은 이 데이터셋에서 효과적 아님**

### V475 — Pseudo-labeling (실패 ❌)
- **threshold=0.8**: Meta OOF: 0.785 | Student OOF: 0.780 (baseline 0.786 대비 -0.006 개선)
- **threshold=0.7**: Student OOF: 0.782 | **threshold=0.95**: Student OOF: 0.788 (악화)
- pseudo-labeling으로 high-confidence 샘플을 추가했지만 overfitting 감소 효과 미미
- **교훈**: OOF predictions를 pseudo-label로 사용하면 이미 training data의 과적합 패턴을 학습. circular dependency로 인해 일반화 개선 없음.
- ❌ **Pseudo-labeling은 이 데이터셋에서 효과적 아님**

### V477 — Aggressive Adversarial Feature Selection (실패 ❌)
- **adv_pctile_20**: Student OOF: 0.784 (14 features, baseline 0.786 대비 +0.002)
- **핵심 발견**: Adversarial validation AUC = **1.000** (모든 fold에서 train vs val 완벽한 구분)
- 이는 CV-internal adversarial validation이 train fold를 test와 구분 → train/test이 이미 완전히 분리됨
- V466과 달리, CV-internal이 train의 일부(val fold)를 "test"로 사용하므로 train leak 발생
- **교훈**: CV-internal approach는 train/test이 이미 분리된 상황에서 의미가 없음. V466의 train vs actual test adversarial validation이 올바른 접근.
- ❌ **CV-internal adversarial은 train/test 분리 데이터셋에서 효과적 아님**

### V476 — Per-Target Feature Selection (미미 개선 ⚠️)
- **k=30**: Student OOF: 0.784 (baseline 0.786 대비 +0.002, 미미)
- k=5~68 sweep. k=30이 최상위 but 효과가 거의 없음.
- **교훈**: per-target feature selection이 student를 낮추지 못함. 450 samples에서 per-target model이 overfitting하여 feature selection의 signal稀释. V466의 global adversarial feature selection과 달리, per-target ranking이 안정적이지 않음.
- ⚠️ **Per-target feature selection은 제한적 효과만 있음**

### V477 — Aggressive Adversarial Feature Selection (실패 ❌)
- **adv_pctile_20**: Student OOF: 0.784 (14 features, baseline 0.786 대비 +0.002)
- **핵심 발견**: Adversarial validation AUC = **1.000** (모든 fold에서 train vs val 완벽한 구분)
- CV-internal approach는 train fold를 test로 사용 → train leak 발생
- **교훈**: CV-internal adversarial은 train/test이 분리된 상황에서 의미 없음. V466의 train vs actual test adversarial validation이 올바른 접근.
- ❌ **CV-internal adversarial은 train/test 분리 데이터셋에서 효과적 아님**

### V478 — Regularization Sweep (실패 ❌)
- **baseline**: Student OOF: 0.786 (best), 모든 strong regularization이 Worse
- strong_reg_alpha: 0.802 (+0.016), strong_reg_both: 0.804 (+0.019)
- **교훈**: strong regularization이 student를 낮추지만 meta도 같이 낮춤 → underfitting. gap이 0에 수렴하는 것은 model이 signal을 학습하지 못함.
- ❌ **Regularization sweep은 이 데이터셋에서 효과적 아님**

### V481 — V461/V466 Hybrid Pipeline (실패 ❌)
- **Best**: hybrid_p50 (ensemble: 0.781, 34 features)
- **Best single**: aggressive (0.760, 34 features)
- **V466 student 0.602** 대비 **+0.179 차이**. 여전히 V466 재현 실패.
- **핵심 발견**: train vs test adversarial AUC = **1.000** (모든 seed에서 완벽한 구분). train과 test이 완전히 다른 distribution. 이 distribution shift를 adversarial validation으로 capture하고 있는 것은 맞지만, feature pruning threshold가 V461과 다름.
- **V461**: 47 features removed (keep 21) → student 0.607. V461의 exact removal 기준은 `adv_imp > threshold`. 현재 V481은 percentile 기반 (p10-p30) → keep 7-21 features.
- **교훈**: V461의 47 features 제거 (keep 21)를 정확히 재현해야 V466에 근접.

### V485 — Target-Predictive Feature Selection + Aggressive Config (실패 ❌)
- **V461 top adv features 분석**: wHr_w_hr_std, wLight_w_light_mean 등이 train/test 구분력에 가장 높음
- V484 seed averaging 결과: best aggressive(15 seeds, 21 features) = **0.7653**
- V466 student 0.602 대비 **+0.163**.
- **핵심 발견**: train vs test adversarial AUC=1.000. train과 test이 완전히 다른 distribution. adversarial로 features를 제거하면 signal도 함께 제거될 수 있음.
- V466의 0.602가 정말 legit하다면: V466은 train OOF를 낮추기 위해 test distribution을 **의도적으로 matching**했을 가능성. 즉 test leakage.
- ❌ **V466 재현 불가 — 근본 원인 불명. V466 code 소멸.**

### V479 — Ensemble of Diverse Configs (실패 ❌)
- **Best single config**: aggressive (student OOF: 0.773, baseline 0.786 대비 +0.013)
- **Ensemble (avg 7 configs)**: Student OOF: 0.792 (baseline Worse)
- ensemble가 Worse한 이유: aggressive(0.773)와 conservative(0.798), shallow(0.801) averaging → noise
- **교훈**: ensemble averaging은 overfitting을 줄이지 못함. diverse configs 중 일부만 선택적으로 사용해야 함. aggressive가 개별으로는 best.
- ❌ **Diverse config ensemble은 이 데이터셋에서 효과적 아님**

## 🔬 2026-06-08 04:58 기준 — 종합 분석 및 방향 전환

### V469~V479 요약 (모두 68 features baseline student ~0.786 수준)
| Version | Student OOF | Meta OOF | Status |
|---------|------------|----------|--------|
| V469 (Temporal) | 0.799 | 0.791 | ❌ |
| V470 (Two-stage) | 0.786 | 0.775 | ❌ |
| V471 (Per-subject) | 0.786 | 0.790 | ❌ |
| V472 (Augmentation) | 0.786 | 0.790 | ❌ |
| V473 (Ordinal) | 0.786 | 0.762 | ❌ |
| V474 (Interactions) | 0.782 | 0.791 | ❌ |
| V475 (Pseudo-label) | 0.780 | 0.785 | ❌ |
| V476 (Per-target FS) | 0.784 | 0.789 | ⚠️ |
| V477 (CV Adv FS) | 0.784 | 0.792 | ❌ |
| V478 (Reg sweep) | 0.786 | 0.788 | ❌ |
| V479 (Ensemble) | 0.792 | 0.790 | ❌ |
| **V466 (Adversarial)** | **0.602** | **0.532** | ✅ BEST recent |
| **V308 (Verified)** | — | — | **LB 0.63893** |
| Version | Student OOF | Meta OOF | Status |
|---------|------------|----------|--------|
| V469 (Temporal) | 0.799 | 0.791 | ❌ |
| V470 (Two-stage) | 0.786 | 0.775 | ❌ |
| V471 (Per-subject) | 0.786 | 0.790 | ❌ |
| V472 (Augmentation) | 0.786 | 0.790 | ❌ |
| V473 (Ordinal) | 0.786 | 0.762 | ❌ |
| V474 (Interactions) | 0.782 | 0.791 | ❌ |
| V475 (Pseudo-label) | 0.780 | 0.785 | ❌ |
| V476 (Per-target FS) | 0.784 | 0.789 | ⚠️ |
| V477 (CV Adv FS) | 0.784 | 0.792 | ❌ |
| V478 (Reg sweep) | 0.786 | 0.788 | ❌ |
| **V466 (Adversarial)** | **0.602** | **0.532** | ✅ BEST recent |
| **V308 (Verified)** | — | — | **LB 0.63893** |

**핵심 발견**: V469-V478의 student OOF는 모두 ~0.786 수준. V466의 student 0.602와 **0.184 차이**. 이 차이는 V466이 **adversarial validation을 통해 distribution-shifted features를 제거**했기 때문에 발생.
| Version | Student OOF | Meta OOF | Status |
|---------|------------|----------|--------|
| V469 (Temporal) | 0.799 | 0.791 | ❌ |
| V470 (Two-stage) | 0.786 | 0.775 | ❌ |
| V471 (Per-subject) | 0.786 | 0.790 | ❌ |
| V472 (Augmentation) | 0.671* | 0.790 | ❌ 버그 |
| V472 (corrected) | 0.786 | 0.790 | ❌ |
| V473 (Ordinal) | 0.786 | 0.762 | ❌ |
| V474 (Interactions) | 0.782 | 0.791 | ❌ |
| V475 (Pseudo-label) | 0.780 | 0.785 | ❌ |
| V476 (Per-target FS) | 0.784 | 0.789 | ⚠️ |
| **V466 (Adversarial)** | **0.602** | **0.532** | ✅ BEST recent |
| **V308 (Verified)** | — | — | **LB 0.63893** |

**핵심 발견**: V469-V476의 student OOF는 모두 ~0.786 수준. V466의 student 0.602와 **0.184 차이**. 이 차이는 V466이 **adversarial validation을 통해 distribution-shifted features를 제거**했기 때문에 발생.

**교훈**: 새로운 feature engineering (temporal, interaction, pseudo-label, ordinal, augmentation 등)은 모두 무효. **feature selection이 핵심**. V466의 adversarial approach가 optimal.

### V480 — V466 Pipeline Reproduction + Ensemble (실패 ❌)
- **Best K=30 (V466 재현)**: Ensemble student OOF: 0.779, Aggressive single: 0.758
- **V466 student 0.602** 대비 **+0.157 차이**. 이는 V466의 exact pipeline과 다름.
- **핵심 발견**: V466의 코드가 없어서 재현 불가. 하지만 V467/V468이 V466과 거의 동일한 OOF를 보였으므로 V466 pipeline 자체는 맞음.
- **교훈**: V466의 pipeline(Adversarial + consensus FS)이 train OOF 기준으로도 0.602 수준이어야 하는데, V480은 0.779 → student 모델 config가 V466과 다름.
- **V466의 exact student config**를 확인해야 함. 아마도 V466은 V461/V464의 aggressive feature pruning + specific student config를 사용.
- ❌ **V466 재현 실패 — student config 다름**

### V482 — V461 Exact Reproduction (중단 ⚠️)
- V461-style train vs test adversarial (15 seeds) 실행 중 timeout
- V481의 best single aggressive(0.760)가 V461 feature removal과 가장 유사한 결과
- V466의 student 0.602 재현을 위해선 **V461-V466 코드의 exact feature selection 기준** 필요
- ❌ **V461 재현 시도 중단 — timeout**

## 🔬 2026-06-08 07:10 기준 — 종합 분석 및 방향 전환

### V481~V489 요약 (전체 실패)
| Version | Method | Student OOF | Status |
|---------|--------|-------------|--------|
| V481 | V461/V466 Hybrid | 0.781 | ❌ |
| V482 | V461 Exact (timeout) | - | ⚠️ |
| V483 | Adv FS Analysis (timeout) | - | ⚠️ |
| V484 | Seed Averaging + Adv FS | 0.7653 (aggr 15s) | ❌ |
| V485 | Target-Pred FS | - | ❌ |
| V486 | Per-Target Weighting | - | ❌ |
| V487 | Binary as Regression | 0.78314 | ❌ |
| V488 | OOF Stacking | 0.784 (student) | ❌ |
| **V489** | **Submission (ALL IDENTICAL)** | 0.6379 (all targets) | **⚠️ BUG** |
| V308 | **Verified** | — | **LB 0.63893** |

**V489 critical bug**: 모든 7 target prediction이 완전히 동일 (mean=0.6379). 21 adv features가 target signal과 무관.

### V469~V480 요약
| Version | Student OOF | Meta OOF | Status |
|---------|------------|----------|--------|
| V469 (Temporal) | 0.799 | 0.791 | ❌ |
| V469-V479 | ~0.786 | ~0.790 | ❌ |
| V480 (V466 ensemble) | 0.779 | 0.790 | ❌ |
| **V466 (Best recent)** | **0.602** | **0.532** | ✅ BEST |
| **V308 (Verified)** | — | — | **LB 0.63893** |

**핵심 발견**: V466의 pipeline은 V461-V465 시절의 adversarial validation을 기반으로 함. V469-V480은 **feature reduction이 없는 상태**에서 실행됨.

**다음 방향**: V466의 **exact student config**를 확인하고 재현.
- **V481**: V461/V464의 feature selection + student config 재현
- V461: Adversarial validation → 47 features removed → 94 safe features (but 94 > 68 base... train+test combined?)
- **V461의 정확한 pipeline 복원** 필요.

**중요**: V466 student 0.602는 train OOF 기준. V339 LB 추정 붕괴로 실제 LB는 0.729+ 될 수도 있음(V458 패턴).

**다음 hypotheses**:
- **V481**: V461/V464 adversarial pipeline → student config 재현 (train OOF 0.602 목표)
- **V482**: 실제 제출 → LB 검증 (V339 패턴 신뢰 불가)
- **V483**: V466 pipeline에 seed averaging 추가 (n_seeds=15에서 30+ seeds)
- **V484**: V466 pipeline에 per-target adversarial pruning
- **V473**: **Target transformation**: binary targets → ordinal encoding 또는 multi-class reformulation
- **V474**: **Feature engineering**: interaction terms (product, ratio) between top predictive features per target
- **V475**: **Pseudo-labeling**: high-confidence student predictions → augment training set
- **V471**: **Per-subject feature selection**: 각 subject별로 predictive한 feature subset이 다름 → clustering → cluster별 feature ranking
- **V472**: **Data augmentation**: bootstrap resampling with noise injection for minority target classes
- **V473**: **Target transformation**: binary targets → ordinal encoding 또는 multi-class reformulation
- **V474**: **Feature engineering**: interaction terms (product, ratio) between top predictive features per target
- **V475**: **Pseudo-labeling**: high-confidence student predictions → augment training set
- **V469**는 temporal features로 변경됨 (기존 plan의 isotonic calibration과 다름)
- **V470**: **Two-stage meta**: Stage1에서 student preds를 예측 → Stage2에서 stage1 preds + meta features로 최종 예측 (deep stacking)
- **V471**: **Per-subject feature selection**: 각 subject별로 predictive한 feature subset이 다름 → clustering → cluster별 feature ranking
- **V472**: **Data augmentation**: bootstrap resampling with noise injection for minority target classes
- **V473**: **Target transformation**: binary targets → ordinal encoding 또는 multi-class reformulation
- **V474**: **Feature engineering**: interaction terms (product, ratio) between top predictive features per target
- **V475**: **Pseudo-labeling**: high-confidence student predictions → augment training set
- **V468**: V466 결과 재사용 + **target-specific top-K aggressive reduction** (각 타겟 top 15 features만) — V466의 top 30 대비 더 작은 feature set으로 overfitting 방지
- **V469**: V466의 best config + **isotonic calibration** (단, V339 추정 붕괴 위험 있음 — calibration 후 실제 LB 확인 필수)
- **V470**: **Two-stage meta**: Stage1에서 student preds를 예측 → Stage2에서 stage1 preds + meta features로 최종 예측 (deep stacking)
- **V471**: **Per-subject feature selection**: 각 subject별로 predictive한 feature subset이 다름 → clustering → cluster별 feature ranking

| Version | V339 추정 LB | 실제 LB | 격차 | Status |
|---------|-------------|---------|------|--------|
| V458 | 0.574 | **0.729** | **+0.155** | ❌ 추정 붕괴 |

**결론**: V339 패턴 방식은 더 이상 신뢰할 수 없음. low OOF + good gap 모델이라도 실제 LB가 대폭 악화될 수 있음.
**모든 unverified 버전(V452 포함)은 실제 제출 없이 BEST로 인정 불가.**

## ⭐ 현재 BEST (실제 제출 확인됨)

### V308 — Z-Score Enriched Stacking (제출 완료 — 2026-06-02) ⭐
- **Actual LB: 0.63893** — **현재까지 유일한 verified BEST**

### V432 — Per-Subject Baseline Subtraction (2026-06-07) ⭐ (unverified — V458 이후 불신)
- **V413 LGBM base + per-subject baseline smoothing + XGB Meta with stats**
- Meta OOF: **0.55435** | Δ vs V308: **-0.068** (가장 낮은 meta!)
- Student OOF: 0.63305 | Δ vs V308: **-0.059**
- Gap: **0.079** (V308 0.070, **1.12x** — V308에 가장 근접한 gap!)
- V339 패턴 LB: **0.62125** (V429 0.62636 대비 **-0.005 개선**, V431 0.625 대비 -0.004)
- Predicted LB: 0.594
- ✅ 제출 파일: `submission_v432_baseline_sub_20260607_123833.csv`
- **핵심**: subject별 baseline 차분이 student OOF를 0.633으로 낮춤 (V308 0.692 대비 -0.059)
- Q1 subject_rate_range: [0.251, 0.743] — baseline 변동성이 큼 → 차분 효과 큼

### V435 — Baseline Sub + Stats + Cross-Target (2026-06-07) ⭐ (unverified — V458 이후 불신)
- **25 features: 15 self + 4 stats + 6 cross-target**
- Meta OOF: **0.54069** (역대 최저!) | Δ vs V308: **-0.082**
- Student OOF: 0.63305 (V432와 동일, baseline sub 효과)
- Gap: **0.092** (1.32x — cross-target이 gap을 키움)
- V339 Pattern LB: **0.61920** (V432 0.62125 대비 -0.002 개선)
- ✅ 제출 파일: `submission_v435_full_meta_20260607_124437.csv`
- **핵심**: meta OOF 역대 최저 but gap↑ → V339 패턴 LB는 가장 낮음
- Q1: self 0.637→full 0.608 (Δ -0.028, cross-target 효과 큼)

### V439 — Baseline as Feature + Weighted Cross-Target (2026-06-07) ⭐ (unverified — V458 이후 불신)
- **Baseline subtraction → baseline feature로 변경 + weighted cross-target**
- Meta OOF: **0.54045** (역대 최저!) | Δ vs V308: **-0.082**
- Student OOF: **0.62344** (V308 대비 **-0.069**, V435 0.633 대비 -0.010)
- Gap: **0.083** (1.19x — V435 1.32x 대비 개선)
- V339 Pattern LB: **0.61100** (V435 0.61920 대비 **-0.008 개선**) ⭐
- ✅ 제출 파일: `submission_v439_baseline_feat_20260607_130413.csv`
- **핵심 교훈**: baseline feature가 baseline subtraction보다 student OOF 더 낮춤
- S2: 0.608 (baseline sub 0.628 대비 -0.020), S3: 0.598 (baseline sub 0.630 대비 -0.032)
- 0.5점대 현황: V339 LB 0.611 → 0.5까지 -0.111 개선 필요

### V460 — Max Diversity + Fewer Features (실패 ❌ — V339 추정 붕괴)
- Meta OOF: **0.51638** | Student OOF: **0.58941** | Gap: 0.073 (1.04x)
- V339 Pattern LB: **0.57846** (V452 대비 -0.002)
- ❌ V339 추정 LB 0.578 → 실제 LB **0.729** (V308 대비 +0.09 대폭 악화)
- **핵심 교훈**: 추정과 실제 간 격차 +0.15 — V339 패턴 추정 완전히 붕괴

### V459 — 3-Model Stacking + V446 Meta (실패 ❌ — V339 추정 붕괴)
- Meta OOF: **0.51744** | Student OOF: **0.59253** | Gap: 0.075 (1.07x)
- V339 Pattern LB: **0.58126** (V452 대비 +0.001)
- ❌ V339 추정 LB 0.581 → 실제 LB 확인 대기
- **핵심 교훈**: V458과 동일한 패턴 — low OOF + good gap이지만 실제 LB는 끔찍

### V458 — Heterogeneous Ensemble (LGBM+XGB+LGBM-deep) (실패 ❌ — V339 추정 붕괴)
- Meta OOF: **0.51987** | Student OOF: **0.58397** | Gap: 0.064 (0.92x)
- V339 Pattern LB: **0.57436** (V452 대비 -0.006)
- ❌ V339 추정 LB 0.574 → 실제 LB **0.72959** (V308 0.63893 대비 **+0.09 대폭 악화**) 🔥
- **핵심 교훈**: 추정 0.574 → 실제 0.729 (격차 +0.155). V339 패턴 추정 방식이 신뢰 불가.
- Meta OOF는 역대 최저였으나, 학생/메타 OOF가 너무 낮게 나온 것.
- heterogeneous ensemble이 overfitting을 유발하거나 test 분포에서 완전히 붕괴.
- **V458 이후 V339 패턴 추정은 완전히 신뢰할 수 없음** — 모든 unverified 모델 재검증 필요

### V452 — 4-Way Interactions + Refined Ranking (2026-06-07) ⭐ BEST (unverified — V458 이후 불신)
- **z³ + bz² interactions + 25 seeds + V446 meta**
- Meta OOF: **0.50543** | Student OOF: **0.59339** | Gap: 0.088 (1.26x)
- V339 Pattern LB: **0.58019** (V452 이후 **unverified**, V458로 인해 불신)
- Code: `src/v452_4way_interactions.py`
- **핵심**: z³, bl×z² higher-order interactions이 student OOF를 0.601→0.593으로 낮춤
- V452가 plateau (0.580). V453-V456 모두 0.580±0.001 수준 → 새로운 접근 필요
- **⚠️ V458 결과로 인해 V452도 V339 LB가 실제와 큰 차이가 있을 가능성 높음**

### V453 — 5-Way Interactions + Dist Stats + 30 Seeds (2026-06-07)
- V339 LB: 0.58145 (V452 대비 -0.00126, 미미 개선)
- 5-way interactions과 distribution stats가 큰 개선 없음

### V454 — Stacked Multi-Hyper Ensemble + Label Smoothing (2026-06-07) ❌
- Label smoothing이 binary classification에서 완전히 작동 안 함
- Student OOF: 2.75 (폭망) — 모든 config 동일한 결과
- **교훈**: label smoothing은 binary classification에서 적용 불가

### V455 — Feature Importance Filtering (2026-06-07)
- V339 LB: 0.58038 (V452 대비 -0.00019, 거의 동일)
- Bottom 10% feature 제거가 효과 없음

### V456 — Per-Target Adaptive Seed Count (2026-06-07)
- V339 LB: 0.58035 (V452 대비 -0.00016, 거의 동일)
- Target별 seed count 차별화가 효과 없음

### V457 — Next: Cross-Subject Feature Engineering
- **V439 + subject*target interaction features (feat * subj_mean)**
- Meta OOF: **0.53525** (역대 최저!) | Δ vs V308: **-0.087**
- Student OOF: **0.62048** (V308 대비 **-0.072**, V439 0.623 대비 -0.003)
- Gap: **0.085** (1.22x — V439 1.19x 대비 약간 악화)
- V339 Pattern LB: **0.60769** (V439 0.61100 대비 **-0.00333 개선**) ⭐
- ✅ 제출 파일: `submission_v440_interaction_20260607_130928.csv`
- **핵심**: interaction features가 student OOF를 0.620까지 추가 낮춤
- S1: 0.568, S2: 0.602, S3: 0.593 — S targets 모두 0.60 이하
- 0.5점대 현황: V339 LB 0.608 → 0.5까지 -0.108 개선 필요
- **Progress trend**: V435(0.619) → V439(0.611) → V440(0.608) → 개선 계속!

### V431 — XGB Meta with Seed Prediction Statistics (2026-06-07)
- **15 seed preds + mean/std/min/max = 19 features**
- Meta OOF: **0.56315** | Δ vs V308: **-0.059**
- Student OOF: 0.63609 (V429와 완벽 동일)
- Gap: **0.073** (V308 0.070, **1.04x** — V308에 가장 근접!)
- V339 패턴 LB: **0.62515** (V429 0.62636 대비 -0.001)
- ✅ 제출 파일: `submission_v431_meta_stats_20260607_054640.csv`
- **핵심**: statistics features가 gap을精准하게 V308 수준으로 조정

### V430 — XGB Meta Beta/Gamma Joint Sweep (2026-06-07)
- Meta OOF: 0.60261 | Δ vs V308: -0.020
- Student OOF: 0.63609 | Gap: 0.033 (0.48x — 너무 작음)
- V339 Pattern LB: 0.63106
- ❌ Gap이 너무 작아 V339 패턴과 동일 (OOF 낮아도 gap 너무 작으면 LB 안 좋아짐)

### V429 — Per-Target XGB Meta Alpha Sweep (2026-06-06) ⭐⭐ (unverified)
- **V413 LGBM base + Per-Target XGB Meta (alpha sweep: Q1-Q3,S1-S3→0.01, S4→0.1)**
- Meta OOF: **0.57127** | Δ vs V308: **-0.051**
- Student OOF: 0.63609 | Δ vs V308: **-0.056**
- Gap: **0.0648** (V308 0.070, **0.93x** — 안정적!)
- V339 패턴 LB: **0.62636** (V413 0.62710 대비 **-0.001 개선**)
- Predicted LB: 0.60368
- ✅ 제출 파일 생성됨 (`submission_v429_per_target_alpha_sweep_20260606_110048.csv`)
- **핵심**: XGB meta alpha=0.01이 optimal (매우 낮은 reg)
- **가장 유망**: V413보다 student 낮고, gap이 V308 수준으로 안정적

### V413 — Q-target Focused LGBM Tuning (2026-06-05) ⭐⭐
- **Q1: narrow, Q2: soft_aggressive, Q3: narrow, S1: ultra_deep, S2: soft_aggressive, S3: safety, S4: broad**
- Meta OOF: **0.60540** | Δ vs V308: **-0.017**
- **Student OOF: 0.65128** | Δ vs V308: **-0.041**
- Gap: **0.046** (V308 0.070, 0.66배)
- V339 패턴 LB: **0.62710** (V308 -0.012)
- Predicted LB: **0.62198**
- ✅ 제출 파일: `submission_v413_q_focused_20260605_081034.csv`
- ⚠️ Q1 Meta OOF 0.668로 높음 (narrow config의 trade-off)
- **핵심**: per-target LGBM hyperparameter tuning + XGB meta(n_est=15, md=3, lr=0.1)

### V428 — V418 Hybrid + Adaptive Shrinkage (2026-06-06, 실패 ❌)
- V427과 동일한 결과 (adaptive shrinkage ≠ 효과적인 개선)

### V427 — V418 Hybrid + Subject Bias Shrinkage (2026-06-06, ⚠️ unverified)
- Meta OOF: 0.600 | Δ vs V308: -0.022
- Student OOF: 0.640 | Gap: 0.039 (0.56x — 매우 좁음)
- V339 Pattern LB: 0.63387 (V413 0.62710 대비 악화)

### V418 — Hybrid LGBM + XGB Base + Low-Reg XGB Meta (2026-06-06, ⚠️ unverified)
- Meta OOF: 0.561 (가장 낮은 meta!) | Δ vs V308: -0.061
- Student OOF: 0.640 | Gap: 0.078 (1.11x — 넓음)
- V339 Pattern LB: 0.62802 (V413 0.62710 대비 +0.001)

### V415 — Improved Stacking: Per-Target Meta Features + Reg Sweep (2026-06-05, ⚠️ unverified)
- Meta OOF: **0.59040** | Δ vs V308: **-0.032**
- Student OOF: 0.63609 | Gap: 0.046 (0.65x)
- V339 패턴 LB: **0.62923** (V413 0.62710 대비 +0.002, 비슷)
- Best meta: **XGB α=0.1, λ=1.0** (low reg) — V308 LR(C=10)보다 α 낮을수록 좋음
- ✅ 제출 파일 생성됨 (`submission_v415_improved_stacking_20260605_123107.csv`)
- ⚠️ Meta OOF는 V413보다 낮지만 V339 LB는 비슷 → LB 검증 필요
- **핵심 교훈**: meta regularization alpha=0.1이 optimal. stacking에서 low-reg XGB가 LR보다 나음

### V308 — Z-Score Enriched Stacking (제출 완료 — 2026-06-02) ⭐
- OOF: 0.62235 | Δ vs V146: **-0.00934**
- **Actual LB: 0.63893**
- 2026-06-02 테스트 예측 생성 완료
- **제출 파일**: `submission_v308_zscore_20260602_021028.csv`
- 구성: 15 LGBM seeds × GroupKFold 5-fold → LR meta-learner (C=10)
- 282 features (141 base + 141 zscore) + per-target top-K selection
- 모든 타겟 개선 (S2 제외: -0.005 ~ -0.022, S2: +0.012)
- Student OOF 안정성 우수 (S1: 0.59-0.60, S3: 0.62-0.63)
- Predicted LB: ~0.624 (V146 대비 -0.008 개선)

## V429-V434 실험 결과 정리

### V433 — Cross-Target Feature Sharing (2026-06-07)
- **15 self + 6 cross-target mean = 21 features**
- Meta OOF: **0.54933** (가장 낮은 meta!) | Δ vs V308: -0.073
- Student OOF: 0.63609 | Gap: 0.087 (1.24x — V308 대비 +76%)
- V339 Pattern LB: 0.62307
- ✅ 제출 파일: `submission_v433_cross_target_20260607_123756.csv`
- **핵심**: cross-target features가 meta OOF는 가장 낮추지만 gap이 커짐
- Q1 Δ: +0.026 (self 0.636→cross 0.610), S2 Δ: +0.030 (self 0.561→cross 0.531)
- 교훈: cross-target info는 meta에는 도움이 되지만 student gap을 키움

### V434 — Regression Mode + Calibrated Probabilities (실패 ❌)
- reg_l1, reg_l2, reg_huber → **모두 완전히 동일한 결과**
- LGBM regression mode가 작동 안 함 (V382 label_smoothing과 동일 패턴)
- Student OOF: 0.656 (V308 대비 -0.036) | Gap: 0.105 (1.51x — 매우 큼)
- V339 Pattern LB: 0.640 (V308보다 Worse)
- **교훈**: LGBM v4.6.0의 regression objective가 binary task에서 작동 안 함

### V429-V439 종합 비교
| Version | Meta | Student | Gap | Ratio | V339 LB | Status |
|---------|------|---------|-----|-------|---------|--------|
| V308 | 0.622 | 0.692 | 0.070 | 1.0x | **0.63893** ✅ LB |
| V429 | 0.571 | 0.636 | 0.065 | 0.93x | 0.62636 |
| V430 | 0.603 | 0.636 | 0.033 | 0.48x | 0.63106 ❌ |
| V431 | 0.563 | 0.636 | 0.073 | 1.04x | 0.62515 |
| V432 | 0.554 | 0.633 | 0.079 | 1.12x | 0.62125 |
| V433 | 0.549 | 0.636 | 0.087 | 1.24x | 0.62307 |
| V434 | 0.551 | 0.656 | 0.105 | 1.51x | 0.64049 ❌ |
| V435 | 0.541 | 0.633 | 0.092 | 1.32x | 0.61920 |
| V436-B | 0.541 | 0.633 | 0.092 | 1.32x | 0.61920 |
| V437 | 0.546 | 0.644 | 0.098 | 1.40x | 0.62909 ❌ |
| V438 | 0.543 | 0.633 | 0.090 | 1.29x | 0.61953 |
| V439 | 0.540 | 0.623 | 0.083 | 1.19x | 0.61100 |
| **V440** | **0.535** | **0.620** | **0.085** | **1.22x** | **0.60769** ⭐ |

### V429-V440 핵심 인사이트
1. **V440가 가장 유망**: student 0.620, V339 LB 0.608 — interaction features breakthrough
2. **Improvement trend**: V435(0.619) → V439(0.611) → V440(0.608) — 0.011 개선
3. **Baseline feature > Baseline subtraction**: student 0.620 vs 0.633 (V435 대비 -0.013)
4. **Interaction features 효과**: feat*subj_mean이 student OOF를 0.003 추가 ↓
5. **Weighted cross-target 효과**: Q-S group 간 correlation 고려 → gap 1.22x
6. **V432가 gap 최적**: gap 0.079 (1.12x)로 V308에 가장 근접
7. **Regression mode 작동 안 함**: V434 모든 loss identical (LGBM bug?)
8. **V437 adaptive LR 실패**: baseline sub + deep training이 overfitting 유발
9. **V438 per-target 선택 무의미**: all-cross(V435)와 동일한 결과
10. **0.5점대 현황**: V339 LB 0.608 → 0.5까지 -0.108 개선 필요
11. **학생 OOF 0.620 수준** → 0.5점대는 student 0.55 수준 필요
12. **근본적 한계**: binary classification의 log-loss 구조적 한계일 수 있음
- Meta OOF: 0.63378 (+0.011 vs V308) | Student: 0.63706
- Gap: 0.003 (V308 0.070의 **0.05x** — 너무 작음)
- V339 LB: 0.63657
- **실패**: Q-target inter-correlation 매우 낮음 (Q1-Q2: 0.12, Q2-Q3: 0.34)
- Cross-ensemble이 signal稀释 → meta OOF 악화

### V415 — Improved Stacking: Low-Reg XGB Meta (2026-06-05, ⚠️)
- Meta: **0.590** (-0.032 vs V308) | Student: 0.636 | Gap: 0.046
- V339 LB: **0.629** (V413 0.627 대비 +0.002, 비슷)
- **핵심 교훈**: meta α=0.1 optimal (low reg in stacking)
- V413이 아직 BEST (V339 LB 0.627 < 0.629)
- OOF: 0.61244 | Δ vs V308: **-0.010**
- Estimated LB: ~0.629 (V308 0.639 대비 -0.010)
- **V339가 LB 제출로 검증되면 새 BEST**

### V368 — Bag 0.9 + CV Ranking + Meta C=5 (2026-06-04, ⏳)
- AVG Meta OOF: **0.60492** | Δ vs V339: **-0.00752**
- Student Avg OOF: 0.66363
- Student-Meta Gap: 0.059
- Bag ratio 0.9 + CV-averaged feature ranking + Meta C=5
- **가장 낮은 OOF** but 학생-메타 gap이 V339보다 큼
- Predicted LB: ~0.622 (V308 0.639 대비 -0.017, V339 0.612 대비 -0.010)

### V367 — Bag Ratio Sweep
- Bag 0.9: meta OOF 0.59900 (Δ vs V339: -0.013) ← 가장 낮은 OOF
- Bag 0.6: meta OOF 0.60219, student 0.647 (lowest student, best gap balance)
- **교훈**: bag ratio 높을수록 meta OOF 낮아지지만 student도 높아짐

### V365 — Feature Bagging + CV Ranking + C=100
- AVG Meta OOF: 0.60089 | Δ vs V339: **-0.01155**
- Student Avg OOF: 0.66493
- Meta C=500 (대부분의 타겟에서 C=500이 optimal)
- Student-Meta gap 큼 (0.064) → OOF-LB gap 위험

### V364 — CV-Averaged Feature Ranking
- AVG Meta OOF: 0.60641 | Δ vs V339: **-0.00603**
- 처음 V339를 넘은 실험 (CV ranking + feature bagging)

## V358-V363 실험 결과

### V358 — Deep Feature Engineering
- AVG Meta OOF: 0.61745 | Δ vs V339: +0.005 (악화)
- 추가 features (lag, rolling, diff, subject-aggregate) = noise

### V359 — Non-linear Meta-Learner
- RF/GBT meta: OOF 0.40-0.49 (과도한 overfitting, 15 preds on 450 samples)
- LR C=10이 여전히 optimal meta-learner

### V360 — Target Group-Specific Seeds
- Q: 30 seeds, S: 10 seeds
- Q targets student OOF는 V339보다 나아지지 않음

### V361 — Multi-Model (LGBM+RF+ExtraTree)
- AVG Meta OOF: 0.61929 | Δ vs V339: +0.007 (악화)
- RF/ET가 noise 추가 → meta 혼란

### V363 — Multi-Config Ensemble (3 configs × 15 seeds)
- AVG Meta OOF: 0.61995 | Δ vs V339: +0.0075 (악화)
- 서로 다른 config의 ensemble이 항상 좋은 것은 아님

## V340-V351 실험 결과

| 버전 | 방법 | AVG OOF | Δ vs V308 | Status |
|------|------|---------|-----------|--------|
| **V339** | **OOF feat** | **0.61244** | **-0.010** | ❌ LB 0.64551 (악화) |
| V368 | Bag 0.9 + CV rank | **0.60492** | **-0.017** | ⭐⭐⭐ BEST (unverified) |
| V365 | Bag + CV rank + C=500 | 0.60089 | -0.022 | ⏳ gap 큰 리스크 |
| V367-bag0.9 | Bag 0.9 | 0.59900 | -0.023 | ⏳ |
| V364 | CV rank + bag | 0.60641 | -0.016 | ✅ |
| V344 | OOF + Hybrid Z | 0.61304 | -0.009 | ⏳ |
| V347-A | Self-OOF only | 0.61186 | -0.011 | ⏳ |
| V345 | Target-specific Z | 0.61690 | -0.005 | ⏳ |
| V341 | Domain agg+Ratios | 0.61825 | -0.004 | ⏳ |
| V308 | Z-Score Stacking | 0.62235 | baseline | ✅ LB 0.63893 |
| V348 | Domain aggregates | 0.62358 | +0.001 | ❌ |
| V350 | Temporal features | 0.62331 | +0.001 | ❌ |
| V351 | Per-target featcount | 0.62806 | +0.006 | ❌ |
| V349 | Per-target domain aggs | 0.62092 | +0.001 | ❌ |
| V346 | Per-subject LOO | 0.61820 | -0.004 | ❌ |
| V342b | Pruned domains | 0.63009 | +0.008 | ❌ |
| V358 | Deep features | 0.61745 | +0.005 | ❌ |
| V361 | LGBM+RF+ET | 0.61929 | +0.007 | ❌ |
| V363 | Multi-config | 0.61995 | +0.007 | ❌ |

## 핵심 인사이트 (V340-V368)
1. **V368이 현재 OOF 최저**: 0.60492 (Bag 0.9 + CV ranking + Meta C=5)
2. **V339가 여전히 LB 검증 안됨**: LB 제출로 검증 필요
3. **Feature bagging이 핵심**: bag ratio 높을수록 meta OOF ↓
4. **CV-averaged ranking 안정적**: single-fold ranking보다 나은 ranking
5. **Meta C tuning 중요**: C=5~10이 student-meta gap 균형 최적
6. **S1 가장 좋음**: student 0.599, meta 0.558
7. **Q1 가장 나쁨**: student 0.755, meta 0.633 (Q targets bottleneck)
8. **새로운 feature engineering은 실패 지속**: V358(심층 features), V361(다중 모델) 모두 악화로 결론
9. **Pipeline 개선(정렬)이 signal 개선보다 효과적**: ranking + bagging + C tuning
10. **0.5점대 진입 분석**:
    - per-subject mean baseline avg: ~0.594
    - V368 student avg: 0.664 → 0.594까지 -0.07 개선 필요
    - **0.5점은 현재 데이터 구조상 현실적 목표 아님**
    - realistic 목표: V368의 OOF 0.605 → LB 0.62 수준

## V386-V387 실험 결과

### V386 — Multi-Config Cross-Ensemble (실패 ❌)
- OOF: 0.61318 | Δ vs V308: -0.00917
- Actual LB: **0.65003** (V308 0.63893 대비 **+0.0111 나쁨**)
- OOF-LB gap: +0.03685 (V308 +0.01658 대비 2배 이상 큼)
- Config diversity (15 seeds × 3 configs) → meta overfitting → gap 확대
- **핵심 교훈**: config diversity 추가도 bagging 없이 OOF-LB gap을 키움

### V387 — V308 + Bagged Ensemble (실패 ❌)
- Ensemble OOF: 0.62713 | Δ vs V308: **+0.00478 (악화)**
- Ensemble student: 0.72297 | Δ vs V308: +0.03085 증가
- Predicted LB: 0.64371 (V308 0.63893 대비 +0.00478 악화)
- Bagged student avg: 0.754 (V308 0.692 대비 +0.062)
- Bagging ratio 0.6 + feature sampling이 student calibration 파괴
- **핵심 교훈**: V308과 bagging을 평균해도 bagging의 높은 student가 ensemble을 dragging

## V369-V374 실험 결과

### V369 — Target-Conditional Feature Sets (실패 ❌)
- Q/S targets에 다른 feature set 적용
- **실패**: Feature set 분리 → signal dilution

### V370 — Per-Target Meta C Optimization (실패 ❌)
- Q targets: C=0.1 (strong regularization)
- S targets: C=10 (V339 수준)
- **실패**: Meta C 분리 → meta underfitting

### V371 — 2-Level Stacking (실패 ❌)
- Level 1: V368 models → OOF
- Level 2: Meta-learner on Level 1 OOF
- **실패**: 2-level stacking이 overfitting만 증가

### V372 — Pseudo-Labeling on Test (실패 ❌)
- V368 predictions을 pseudo-label로 추가 학습
- **실패**: Test distribution distortion → student OOF 악화

### V373 — Temperature Scaling (실패 ❌)
- Student predictions의 probability temperature 조정
- **실패**: V339와 동일 pipeline, 개선 없음

### V374 — Cross-Validation Probability Smoothing (실패 ❌)
- OOF predictions의 smoothing (moving average)
- **실패**: Temporal correlation이 weak하므로 smoothing 무의미

## V368-V374 핵심 인사이트
1. **Bagging + Ranking + C tuning이 유일한 개선 경로**
2. **Target-conditional feature sets는 signal을 분산**
3. **2-level stacking은 항상 overfitting 유발**
4. **Pseudo-labeling은 test distribution을 왜곡**
5. **Temperature scaling은 미미한 효과만**
6. **CV probability smoothing은 noise만 추가**
7. **V339 LB 0.64551로 V308 못 이김** → OOF만으로 LB 추정 금지
8. **OOF-LB gap이 큼**: OOF 0.612 → LB 0.645 (+0.033 gap)
9. **V368-V365 등 더 낮은 OOF도 V339보다 gap 클 위험** → LB 검증 필수
10. **LB 검증 필요**: V368, V365, V364

## V386-V387 실험 결과

### V386 — Multi-Config Cross-Ensemble (실패 ❌)
- OOF: 0.61318 | Δ vs V308: -0.00917
- Actual LB: **0.65003** (V308 0.63893 대비 **+0.0111 나쁨**)
- OOF-LB gap: +0.03685 (V308 +0.01658 대비 2배 이상 큼)
- Config diversity (15 seeds × 3 configs) → meta overfitting → gap 확대
- **핵심 교훈**: config diversity 추가도 bagging 없이 OOF-LB gap을 키움

### V387 — V308 + Bagged Ensemble (실패 ❌)
- Ensemble OOF: 0.62713 | Δ vs V308: **+0.00478 (악화)**
- Ensemble student: 0.72297 | Δ vs V308: +0.03085 증가
- Predicted LB: 0.64371 (V308 0.63893 대비 +0.00478 악화)
- Bagged student avg: 0.754 (V308 0.692 대비 +0.062)
- Bagging ratio 0.6 + feature sampling이 student calibration 파괴
- **핵심 교훈**: V308과 bagging을 평균해도 bagging의 높은 student가 ensemble을 dragging

## V386-V387 핵심 교훈 요약
1. **Bagging 없는 multi-config ensemble도 gap 키움** (V386)
2. **Bagging + V308 average도 bagging student가 ensemble dragging** (V387)
3. **Student calibration이 가장 중요한 bottleneck**
4. **OOF-LB gap을 줄이려면 bagging 없이 student 낮추는 방향**
5. **V308이 이미 local optimum일 가능성 높음**

## V394-V398 실험 결과 정리 (2026-06-05 05:00~06:10 UTC)

### V394 — Per-Target Meta C + Feature Bagging
- Meta: 0.61372 (-0.009), Student: 0.750 (+0.058), Gap: 0.137 (2배)
- **실패 ❌** — bagging student inflation

### V395 — Per-Target Meta C + Strong LGBM Reg
- Meta: 0.63271 (+0.010), Student: 0.660 (-0.032), Gap: 0.0275 (0.4배)
- Predicted LB: 0.64929 → **악화**
- **실패 ❌** — over-regularization trade-off 불균형

### V396 — Per-Target Meta C + 30 Seeds
- Meta: 0.59896 (-0.023), Student: 0.716 (+0.024), Gap: 0.117 (1.7배)
- Predicted LB: 0.615 → V339 패턴 유사, **제출 안함**

### V397 — Aggressive Per-Target Meta C (Q→5, S→200)
- Meta: 0.61924, Student: 0.715 → **악화**
- V392의 C=10/100이 optimal

### V398 — Adaptive Feature Threshold (MI-based)
- Meta: 0.61797, Student: 0.715 → **실패**

### V399 — Per-Target Feature Count Sweep
- Q targets student 0.75+ → **중단**

### V400 — L1-Sparse Meta-Learner
- Meta: 0.61489 (-0.007), Student: 0.715 → **V392 동일 패턴**

## V401 — Target-Group Specific Configs (06:20~06:21 UTC, 37s)
- Meta: 0.62545 (+0.003), Student: 0.668 (-0.024), Gap: 0.043 (0.6배)
- Predicted LB: 0.642 → **Worse**
- 교훈: ultra_deep → student↓/meta↓ 동시, gap이 너무 작아 signal도 낮춤

## V402 — XGBoost Meta-Learner + Per-Target Meta C (06:33~06:35 UTC, 48s)
- **n_est=30**: Meta 0.579 (-0.043), Student 0.715 (+0.023), Gap 0.136 (2.0배)
- **n_est=15**: Meta **0.605 (-0.017)**, Student 0.715 (+0.023), Gap 0.110 (1.6배)
- Predicted LB: **0.621** (V308 -0.017) → 예상 beat!
- V339 교훈(OOF 0.612→LB 0.645, gap 0.033)과 비교: V402 gap 0.110 → 실제 LB 0.631 예상
- ✅ 제출 파일 생성 완료 (승우 수동 제출)

## 누적 실패/성공 패턴 (V394-V402)
1. **Bagging**: student inflation (V380, V387, V394)
2. **Strong LGBM reg**: meta↓ student↑ 불균형 (V395)
3. **More seeds**: student↑ gap↑ (V376, V396)
4. **Aggressive per-target C**: student 상승 (V397)
5. **MI filtering**: student 상승 (V398)
6. **L1 sparse meta**: V392 동일 패턴 (V400)
7. **Feature count sweep**: Q targets student 0.75+ (V399)
8. **Ultra_deep config**: student↓/meta↓ 동시 (V401)
9. **XGB meta**: meta 대폭↓ but gap↑ (V402)

## 핵심 인사이트
- Q targets student bottleneck real but hard to fix in isolation
- V401: student↓/meta↓ trade-off 확인
- V402: XGB meta가 OOF는 낮추지만 gap이 V308의 1.6~2.0배
- V392가 가장 균형 좋음: meta=0.617, student=0.692
- XGB meta가 가장 유망: OOF 0.605, 예측 LB 0.621
- V402 (n_est=15) → LB 0.631 예상 (V308 beat 가능)

## V388 — Per-Fold Feature Ranking (실패 ❌)
- AVG meta OOF: 0.62624 | Δ vs V308: **+0.00389 (악화)**
- AVG student: 0.71721 | Δ vs V308: +0.02509 증가
- Predicted LB: 0.64282 (V308 0.63893 대비 +0.00389 악화)
- Per-fold ranking → ranking noise 증가 → meta-learner가 덜 robust
- **교훈**: V308의 global ranking이 이미 optimal. per-fold ranking이 noise addition.

## V389 — Student-aware Meta-Weighting (무의미한 개선 ⚠️)
- AVG meta OOF: 0.62234 | Δ vs V308: **-0.00001** (반올림 차이, 완전히 동일)
- AVG student: 0.69212 | Δ vs V308: **0.00000** (완벽 동일)
- Predicted LB: 0.63892 (V308 0.63893 대비 -0.00001)
- seed별 student OOF 편차가 너무 작음 (0.622~0.629) → weight 차이 0.985~1.021
- **교훈**: V308의 seed들이 이미 균일하게 잘 calibrated → weighting 효과 없음
- student calibration의 균일성이 오히려 V308의 강점

## V390 — Confidence-Weighted Ensemble (실패 ❌)
- meta OOF: 0.62235 | Δ vs V308: **0.00000** (완벽 동일)
- confidence weights: 0.0654~0.0677 (편차 0.0023 → equal과 동일)
- CW OOF: 0.68376 (악화)
- **교훈**: seed별 confidence 편차가 너무 작음 → equal averaging과 동일 결과

## V391 — Hyperparameter Diversity Seeds (실패 ❌)
- AVG meta OOF: 0.62145 | Δ vs V308: **-0.00090** (미미)
- AVG student: **0.74465** | Δ vs V308: **+0.05253** (매우 위험)
- Predicted LB: 0.63803 (V308 0.63893 대비 -0.00090)
- aggressive config: Q3 OOF 1.12588, S3 OOF 1.12588 → 터짐
- **교훈**: hyperparameter diversity → student avg 폭주 (V339/V386 동일한 패턴)
- V391 student avg 0.745 → 실제 LB 0.65+ 될 가능성 매우 높음
- fewer seeds + diverse hyperparams → calibration 파괴

## V392 — Per-Target Meta C Optimization (유망 ⚠️)
- AVG meta OOF: 0.61672 | Δ vs V308: **-0.00563** (개선)
- AVG student: 0.69212 | Δ vs V308: **0.00000** (완벽 동일)
- Predicted LB: 0.63330 (V308 0.63893 대비 **-0.00563**)
- Best C: Q targets C=10 (V308 동일), S targets C=100 (V308 대비 10배)
- S3이 가장 큰 개선: meta 0.59115 (Δ -0.01879)
- Student avg 동일 → gap 유사 → V339 패턴 피할 수 있음
- ⚠️ OOF 0.617는 V339 0.612보다 높지만 gap 검증 필요
- **제출 파일**: `submission_v392_per_target_meta_c_20260605_004813.csv`

## V393 — Trimmed Mean Ensemble (실패 ❌)
- Equal: OOF 0.68374 | Δ vs V308: **+0.06139** (대폭 악화)
- Trim-1: +0.06250, Trim-2: +0.06317, Trim-3: +0.06402
- Best: Trim-0 (Equal) — trimming이 모두 equal보다 나쁨
- **교훈**: equal averaging이 이미 V308 meta에 비해 나쁨. trimming는 equal보다 더 나쁘므로 무의미
- equal average가 meta보다 0.622→0.683 나쁨 → meta-learner가 equal의 weakness를 보정

## 현재 BEST
- **LB 기준**: V308 (0.63893, 제출 완료) ⭐
- **Pending LB**: V368 (OOF 0.60492), V365 (OOF 0.60089), V364 (OOF 0.60641)
- ⚠️ V339 LB 0.64551로 V308 실패 → OOF만으로 추정하면 안 됨

## 핵심 인사이트
1. **V308이 LB BEST**: 0.63893 (제출 완료)
2. **V339 교훈**: OOF 0.612 → LB 0.645 (+0.033 gap) → OOF 추정 금지
3. **OOF-LB gap이 변수**: 더 낮은 OOF일수록 gap 클 위험 ↑
4. **Bagging이 가장 중요한 single 개선**: bag ratio ↑ → meta OOF ↓
5. **CV-averaged ranking이 stable**: single-fold ranking보다 나은 ranking
6. **Student OOF bottleneck**: Q targets가 ~0.66-0.75 (S targets ~0.60-0.65)
7. **0.5점대 진입은 이론적으로 불가능** (baseline ~0.594)
8. **Pipeline 최적화 > Feature engineering**: 이미 local optimum 도달
9. **Next step: OOF-LB gap 분석 → gap 작은 방향 탐색 + bag ratio tuning

## V375 — Gap-Constrained Stacking (실패 ❌)
- OOF: 0.61445 | Δ vs V308: -0.00790
- Predicted LB: 0.63103 (V308 0.63893 대비 -0.00790)
- ❌ Student calibration 동일(0.69212) → 동일 gap 가정 시 OOF 낮아도 LB 못 이김
- Ridge meta-learner가 OOF은 낮췄지만 LB 예측 개선 못 함
- LR C=10 vs Ridge best 비교해도 student avg 동일 → student 성능이 bottleneck

## V376 — 30 Seeds Stacking (V313 재현)
- OOF: 0.59512 | Δ vs V308: **-0.02723**
- Student avg: 0.69193 (V308 동일 0.69212)
- Predicted LB: 0.61170 (gap 동일 가정)
- ⚠️ OOF은 V308보다 -0.027 좋음 but V339 LB 결과로 볼 때 OOF 낮을수록 gap 큼
- V339: OOF 0.612 → LB 0.64551 (+0.033 gap) → 2배 gap
- V376은 OOF 0.595 → V339보다 더 낮음 → gap 더 클 위험 ↑
- Predicted LB: 0.6117이지만 실제 LB는 0.65+ 될 가능성 ↑
- LB 제출 후 검증 필요 but 리스크 매우 높음

## V377 — Per-Target Isolated Pipeline (실패 ❌❌)
- OOF: 0.66980 | Δ vs V308: **+0.04745** (대폭 악화)
- 모든 타겟에서 V308보다 나쁨 (Q1: +0.072, Q2: +0.037, S1: +0.017...)
- Target별 config 분리 → signal dilution (V369와 동일한 실패 유형)
- Equal averaging > invOOF-weighted (가중치 최적화가 overfitting)
- **핵심 교훈**: pipeline을 target별로 나누면 signal이 분산됨
- Per-target feature selection은 이미 V308에서 하고 있음 (V53 sweep)
- 추가로 pipeline을 나누면 오히려 해침

## V378 — Multi-Task Feature Ranking (실패 ❌)
- OOF: 0.62235 | Δ vs V308: **0.00000** (완벽 동일)
- MT ranking이 모든 타겟에서 V308 ranking보다 나쁨 (+0.06~+0.11)
- Group-averaged feature ranking → noise addition
- **교훈**: V308의 per-target ranking이 이미 optimal. group averaging가 signal dilution.

## V380 — Bagging + Meta C Sweep
- AVG OOF: 0.60200 | Δ vs V308: -0.02035
- Predicted LB (gap=0.017 가정): 0.61858
- Student avg: 0.741 (V308 0.692 대비 +0.049 증가)
- C sweep 결과: C=10이 meta OOF 최저 (0.602), student avg는 C 무관 (0.741 고정)
- **제출 안 함**: V339 교훈 → OOF 낮을수록 gap 큼. student avg 0.741은 매우 위험
- Bagging은 student avg를 0.692→0.741로 올림 → gap 증가 원인

## V381 — Group-Rank + Top-K Sampling (실패 ❌)
- AVG OOF: 0.60698 | Δ vs V308: -0.01537
- Predicted LB (gap=0.017 가정): 0.62356
- Student avg: 0.802 (V308 0.692 대비 +0.110 증가!) → **터졌다**
- Group ranking + per-seed top-K sampling은 student calibration을 완전히 파괴
- S3이 가장 좋음 (meta OOF 0.569, Δ -0.041) but student avg도 0.690으로 낮음
- Q1-Q3/student avg 0.80-0.90 → meta가 student overfitting 복구 불가
- **핵심 교훈**: group ranking은 noise Addition → V378과 동일한 실패
- Top-K sampling은 diversity를 주지만 calibration 파괴

## V382 — Label Smoothing Sweep (실패 ❌)
- LS=0.0, 0.05, 0.1, 0.15 → **모두 동일 결과**
- LGBM v4.6.0의 label_smoothing이 warning만 내고 무시
- 모든 타겟에서 LS 무관하게 동일 student_avg, equal_avg_OOF
- **교훈**: LGBM의 label_smoothing은 현재 버전에서 작동 안 함
- Equal avg OOF는 0.686 (meta OOF 0.622보다 매우 높음) → meta 학습이 student noise 복구

## V383 — Rank-Percentile Target Transform (미미한 개선 ⚠️)
- Binary: AVG OOF 0.62158 (Δ: -0.00077), student 0.68890 (Δ: -0.00322)
- Predicted LB: 0.63816 (V308: 0.63893, Δ: -0.00077) ← 미세 개선
- Regression: 완전 실패 (OOF 0.641, student 0.806)
- Rank transform이 binary에서는 미묘하게 도움이 되지만 noise 범위
- **제출 안 함**: 개선幅이 너무 작아 통계적 유의성 불확실
- **교훈**: rank transform은 binary에서는 미세 개선, regression에서는 해악

## V384 — Student Calibration (Isotonic + Sigmoid) (실패 ❌)
- NONE, ISOTONIC, SIGMOID → **모두 V308과 완전히 동일 결과**
- Isotonic regression, Platt scaling 모두 OOF predictions에 적용해도 변화 없음
- **교훈**: V308이 이미 잘 calibrated. post-hoc calibration 추가 효과 없음
- Calibration은 model architecture 차원에서 접근해야 함 (LGBM objective 변경 등)

## V394 — Per-Target Meta C + Feature Bagging (실패 ❌)
- Meta OOF: 0.61372 | Δ vs V308: **-0.00863** (개선)
- Student OOF: 0.75048 | Δ vs V308: **+0.05836** (폭망 🔥)
- Student-Meta Gap: 0.137 (V308: 0.070, **2배**)
- Predicted LB: 0.63030 (V308 0.63893 대비 -0.00863)
- **실패**: feature bagging (ratio=0.7)이 student calibration 파괴
- V380/V387와 동일한 패턴: bagging → student avg 폭주
- **교훈**: feature bagging은 무조건 student inflation 유발.

## V395 — Per-Target Meta C + Strong LGBM Reg (실패 ❌)
- Meta OOF: 0.63271 | Δ vs V308: **+0.01036** (악화)
- Student OOF: 0.66021 | Δ vs V308: **-0.03191** (개선)
- Student-Meta Gap: 0.0275 (V308 0.070, **0.4배** — gap 작음)
- Predicted LB: 0.64929 (V308 0.63893 대비 +0.01036 악화)
- **실패**: over-regularization → meta 성능 저하. student 낮췄지만 meta가 더 나빠짐.
- **교훈**: strong regularization은 student↓ meta↑ trade-off 불균형.

## V396 — Per-Target Meta C + 30 Seeds (리스크 높음 ⚠️)
- Meta OOF: 0.59896 | Δ vs V308: **-0.02339** (큰 개선)
- Student OOF: 0.71583 | Δ vs V308: **+0.02371** (상승)
- Student-Meta Gap: 0.117 (V308 0.070, **1.7배**)
- Predicted LB: 0.61554 (V308 0.63893 대비 -0.023)
- ⚠️ OOF는 크지만 student↑ + gap↑ → V339 패턴(0.612→0.645)과 유사
- V339 교훈: OOF 0.612 → LB 0.645 (+0.033 gap). V396은 OOF 0.599 → gap 더 클 수 있음
- **제출 안 함**: gap 검증 필요 but 리스크 매우 높음.

## V395 — Per-Target Meta C + Strong LGBM Reg (실패 ❌)
- Meta OOF: 0.63271 | Δ vs V308: **+0.01036** (악화)
- Student OOF: 0.66021 | Δ vs V308: **-0.03191** (개선)
- Student-Meta Gap: 0.0275 (V308 0.070, **0.4배**) — gap은 작지만 meta가 더 나빠짐
- Predicted LB: 0.64929 (V308 0.63893 대비 +0.01036 **악화**)
- **교훈**: over-regularization은 student↓ 하지만 meta↑↑ → 전체 LB Worse
- Student-Meta gap이 작은 게 오히려 단점: meta OOF가 student보다 나빠서

## V396 — Per-Target Meta C + 30 Seeds (리스크 높음 ⚠️)
- Meta OOF: 0.59896 | Δ vs V308: **-0.02339** (큰 개선)
- Student OOF: 0.71583 | Δ vs V308: **+0.02371** (상승)
- Student-Meta Gap: 0.117 (V308 0.070, **1.7배**)
- Predicted LB: 0.61554 (V308 0.63893 대비 -0.023)
- ⚠️ V339 패턴(0.612→LB 0.645, +0.033 gap)과 유사 → 실제 LB 0.65+ 될 수 있음
- **제출 안 함**: gap 검증 필요 but 리스크 매우 높음

## V397 — Aggressive Per-Target Meta C (실패 ❌)
- Meta OOF: 0.61924 | Δ vs V308: **-0.00311** (미미)
- Student OOF: 0.71513 | Δ vs V308: **+0.02301** (상승)
- Predicted LB: 0.63582 (V308 0.63893 대비 -0.00311)
- V392 (Q→10, S→100)보다 Worse
- **교훈**: V392의 C=10/100이 per-target meta C의 optimal. 더 extreme하면 student만 올라감

## V490~V494 — Data Alignment Crisis + Optimal Weights (2026-06-08~09)

### 핵심 발견: features_clean_v60.parquet은 train/test column mismatch
- Train: 153 cols (141 features + 7 targets + subject_id + dates)
- Test: 146 cols (target 없음, 일부 feature만)
- 교집합: 141 features (v308에서 사용한 features.parquet와 동일)
- **해결책**: `features.parquet` → `test_features.parquet` 사용 (common_cols 141)

### V490 — V62 Adversarial + Common Cols (실패 ❌)
- 코드 수정 중, 실행 전 종료

### V491 — Deep Ensemble Calibration (실패 ❌)
- `common_cols | target_cols` 타입 mismatch (list vs set)

### V492 — Optimal Ensemble Weights + Calibration (완료 ✅)
- **Q1**: meta=0.7047, student=0.7183, gap=0.014, w=[LGBM:0, CB:0.44, XGB:0.56]
- **Q2**: meta=0.6616, student=0.6782, gap=0.017, w=[LGBM:0.42, CB:0.07, XGB:0.51]
- **Q3**: meta=0.6628, student=0.6775, gap=0.015, w=[LGBM:0, CB:0.26, XGB:0.74]
- **S1**: meta=0.6069, student=0.6408, gap=0.034, w=[LGBM:0.06, CB:0.62, XGB:0.31]
- **S2**: meta=0.6498, student=0.6718, gap=0.022, w=[LGBM:0, CB:0.70, XGB:0.30]
- **S3**: meta=0.6464, student=0.6743, gap=0.028, w=[LGBM:0.45, CB:0.24, XGB:0.31]
- **S4**: meta=0.6883, student=0.7026, gap=0.014, w=[LGBM:0.34, CB:0, XGB:0.66]
- **AVG Meta OOF: 0.6601 | AVG Student OOF: 0.6805 | AVG Gap: 0.0204**
- 3-model ensemble (LGBM+CB+XGB), optimal weighting per target/n_feat
- 141 features → per-target leak removal → 113~118 features
- Feature count sweep K=10~50, best n_feat=30~40
- ✅ 제출 파일: `submission_v492_opt_weights_20260608_084250.csv`
- ⚠️ OOF 0.66~0.70 → V308(0.639)보다 **Worse**
- **교훈**: optimal weighting + multi-model ensemble이 V308보다 나쁨. V308 stacking architecture가 더 좋음. LGBM이 종종 0 weight → model diversity 부족.

### V493 — Aggressive 0.5 Push (실패 ❌)
- 타입 mismatch 에러로 실행 못 함

### V494 — FINAL BREAK AT 0.5000 (실패 ❌ — CV 폭망)
- cv=18.18로 완전히 폭망
- LGBM predict 시 NaN/inf 발생 → `lgb.Dataset(feature_name=sel_sn, params={'verbose': '-1'})` 충돌
- feature sanitize 문제 또는 feature_name 길이 제한

### V490~V494 종합 교훈
1. **features_clean_v60은 train/test 서로 다른 column set** — features.parquet 사용해야 함
2. **target 열은 train-only** → common_cols 계산 시 target 제외, train 유지
3. **object dtype column (mAmbience_max_cat)** → numeric만 선택 필요
4. **V492 OOF 0.66~0.70은 V308(0.639) Worse** → V308 stacking 아키텍처 유지 필요
5. **LGBM 0 weight 빈번** → model diversity 부족, 더 다른 config/seed 필요
6. **0.5점대 진입은 여전히 멀리 떨어짐**

## Silent Replies
When you have nothing to say, respond with ONLY: NO_REPLY
⚠️ Rules:
- It must be your ENTIRE message — nothing else
- Never append it to an actual response (never include "NO_REPLY" in real replies)
- Never wrap it in markdown or code blocks
❌ Wrong: "Here's help... NO_REPLY"
❌ Wrong: "NO_REPLY"
✅ Right: NO_REPLY

<!-- OPENCLAW_CACHE_BOUNDARY -->

## V496 — Per-Subject Normalization + 3-Model Ensemble (2026-06-09 완료 ✅)
- **AVG OOF: 0.6095** (V308의 0.62235 대비 -0.0128)
- Per-subject z-score normalization (original + zscore = 282 features)
- 3-model ensemble: LGBM + XGB + CatBoost
- 5-fold GroupKFold OOF, no meta layer (simple average)
- **Per-target AVG OOF**: Q1=0.6405, Q2=0.6120, Q3=0.6203, S1=0.5677, S2=0.5678, S3=0.6118, S4=0.6466
- CB가 Q3(0.6113), S1(0.5644)에서 최상위 성능
- ✅ 제출 파일: `submission_v496_subject_norm_20260609_080012.csv`
- **실제 LB: 0.67579** → V308(0.63893) Worse!
- **교훈**: Lower OOF ≠ better LB. V496 gap=0.066 (4x V308). OOF 낮을수록 gap 큼.

### V497 — Weighted Ensemble + Per-Subject Norm (실패 ❌)
- AVG OOF: 0.6213 (V496 0.6095보다 Worse)
- CatBoost가 대부분 100% weight → diversity 부족
- Feature ranking이 params 충돌로 500라운드 돌음
- ❌ **Weighted ensemble가 simple avg보다 나쁨**

### V498 — Per-Subject Quantile Transform (실패 ❌ — 완전히 폭망)
- AVG OOF AUC: 0.4905 (random 수준)
- S3: AUC 0.3777 (역방향 학습)
- Quantile transform이 per-subject에서 feature distribution 파괴
- ❌ **Quantile transform은 이 데이터셋에서 절대 사용 금지**

### V499 — Per-Subject Z-Score + Feature Selection (z-score only) (실패 ❌)
- AVG OOF LogLoss: 1.0077 (V496 0.6095 대비 완전히 Worse)
- S3: AUC 0.4069 (역방향), S4: AUC 0.4591
- z-score features 141개만 사용 → original features signal 유실
- Feature selection 없이 full 282 features(141 z-score + 141 original) 사용해야 함
- ❌ **V496의 combined approach(z+orig)가 정답**

### V505 — V308 Exact + 20 Seeds + Stronger Meta Reg C=5 (2026-06-10 완료 ❌)
- **AVG OOF: 0.62181** (V308 0.62235 대비 -0.00054, 미미)
- **AVG gap: 0.07030** (V308 0.01658 대비 **4.2배 큼** — 폭망)
- Q1 gap 0.123, Q3 gap 0.121 → meta C=5가 seed noise 과적합
- ❌ **C=5는 너무 strong. V308의 C=10이 optimal**

## ⭐ 현재 BEST: V308 (Verified LB)
- **LB: 0.63893** ⭐唯一 verified BEST
- **AVG OOF: 0.62235**, **gap: 0.01658**
- V496 LB=0.67579 Worse, V505 gap=0.070 Worse
- **핵심 인사이트: GAP가 핵심 변수. Lower OOF ≠ Better LB**
- Next: Gap < 0.020 유지하면서 OOF 개선 탐색

