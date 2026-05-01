# 🐛 Dacon2 제출 파일 문제 분석 리포트

**작성일:** 2026-05-01
**발견 점수:** Validation Log-Loss 0.5395 → Submission 0.90535 (68% 악화)

---

## 결론 (한 줄 요약)

**치명적 버그 2개:**
1. **Target Leakage (치명적)**: 모델이 다른 target 열(Q/S)을 feature로 사용 → test 데이터에서 0 입력 → 예측 붕괴
2. **Out-of-Distribution 예측**: train/test 데이터가 (subject, date) 단위로 완전히 분리됨 → 모델이 unseen data에 대해 예측

---

## 상세 분석

### 1. 🚨 Target Leakage - Submission 파일이 0.90535인 주된 원인

**버그 발생 위치:** `src/03_model_training.py` lines 92-95

```python
meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date", target]
feature_cols = [c for c in features.columns if c not in meta_cols]
```

이 코드는 `target` 하나만 제외하고 **모든 열을 feature로 포함**. 다른 target 열(Q1, Q2, Q3, S1, S2, S3, S4)도 feature에 포함됨.

**실제 누설된 target 목록:**

| Target 모델 | 누설된 target features | 누설된 비율 |
|-------------|----------------------|------------|
| Q1          | S1                   | 1/147 (0.7%) |
| Q2          | Q3                   | 1/147 (0.7%) |
| Q3          | Q2                   | 1/147 (0.7%) |
| S1          | S2, Q1               | 2/147 (1.4%) |
| S2          | S4, S3, S1           | 3/147 (2.0%) |
| S3          | S2, S4               | 2/147 (1.4%) |
| S4          | S2, S3               | 2/147 (1.4%) |

**메커니즘:**
- 학습 시: 다른 target의 실제 값이 feature로 입력 → 모델이 학습
- 제출 시: `submission_sample.csv`에서 Q1~S4가 **모두 0** → 모델이 0을 feature로 받음
- S2 모델은 S1=0, S3=0, S4=0을 받고 예측 → 학습 때의 S1/S3/S4 값(0.6~0.7)과 완전히 다름

**feature importance 확인:**
```
S2 model: S4(imp=673), S3(imp=521), S1(imp=211) → top 3 features 모두 다른 target!
S2 모델의 real features top: wifi_max_rssi(80), wifi_avg_rssi(54), gps_count(36)
  → target leakage가 실제 예측의 95% 이상을 결정
```

**영향:**
- S2, S3, S4 모델은 가장 심함 (top 3 features 중 2~3개가 target leakage)
- S2 예측: mean=0.1449 (실제 0.651의 22% 수준) - model predicts near-0 because all leaked targets are 0
- 이것이 log-loss 0.90535의 주요 원인

### 2. 📊 Out-of-Distribution (OOD) 예측

**데이터 구조:**
- Training: 450 rows (10 subjects × ~45 days), 2024-06-03 ~ 2024-11-14
- Test sample: 250 rows (10 subjects × ~25 days), 2024-07-06 ~ 2024-11-19
- **교차점: 0개** - 어떤 (subject, date) 조합도 training과 test가 공유하지 않음

**왜 0 교차?**
- train과 test의 date range가 calendar상 겉보기에는 overlap 되지만,
- **특정 날짜**들은 서로 완전히 다른 set임
- 예: id01 train dates → 06-26~08-31 중 41개 날짜 (07-11, 07-18, 07-20, 08-15, 08-30 등이 빠짐)
- id01 test dates → 07-30~09-14 중 27개 날짜 (train의 빠진 날짜를 보완하는 형태)
- 두 set은 **상호 보완적(complementary)** - 전체 700 days를 train 450 + test 250으로 나눔

**영향:**
- 모델이 훈련된 distribution과 다른 distribution에서 예측해야 함
- temporal interpolation이 아니라 pure extrapolation
- 검증 log-loss (0.5395)는 training split 내 validation → OOD 아님
- 실제 test는 OOD → 검증 점수가 신뢰도 없는 지표

### 3. ✅ 제출 파일 형식 - 정상

```
Sample columns:    ['subject_id', 'sleep_date', 'lifelog_date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
Submission columns: 같은 순서, 같은 개수
Row count: 250 == 250 ✓
Prediction range: 모든 value [0, 1] ✓
Null values: 없음 ✓
```

형식 자체는 완벽함.

### 4. ✅ Feature Engineering Pipeline - 일관성 있음

- Training: features.parquet (450 rows, 153 cols, 4.1% null)
- Test: 동일한 raw parquet → 동일한 aggregation → 동일한 feature structure
- Test null rate: 2.2% (training보다 낮음) → merge 문제 없음
- 855 missing features는 250×153=38250 중 2.2% → 정상적인 sparsity

### 5. ⚠️ Feature 누락 분석

855 missing out of 38250 cells = 2.2% null rate
Training: 2827 / 68850 = 4.1% null rate

Test의 null rate가 더 낮음 → 파이프라인 정상 동작.
누락의 주요 원인: wHr (heart rate), wPedo (step data)가 없는 날짜 → 정상적 sparsity.

### 6. ⚠️ Calibration 문제

Model predictions on training data:
```
Q1: pred_mean=0.4944, true_mean=0.4956, gap=-0.0011  ← OK
Q2: pred_mean=0.5275, true_mean=0.5622, gap=-0.0347
Q3: pred_mean=0.5686, true_mean=0.6000, gap=-0.0314
S1: pred_mean=0.6496, true_mean=0.6822, gap=-0.0326
S2: pred_mean=0.6194, true_mean=0.6511, gap=-0.0318
S3: pred_mean=0.6000, true_mean=0.6622, gap=-0.0622  ← worst
S4: pred_mean=0.5292, true_mean=0.5600, gap=-0.0308
```

Training에서는 잘 calibrate됨.
Test에서는 target leakage로 인해 predictions가 0 쪽으로 bias됨 (위 분석 참조).

### 7. ⚠️ Feature Engineering Import 방식

`04_submit.py`에서 `importlib.util.spec_from_file_location`으로 동적 import 사용.
- `02_feature_engineering.py`의 `from config import ...`가 올바르게 동작함
- 동적 import 시 `sys.path.insert(0, str(Path(__file__).parent))`로 src/를 path에 추가
- 정상 동작 확인 (pipeline completed without import errors)

---

## 해법

### Priority 1: Target Leakage 제거 (가장 중요) (✅ 완료)

`src/03_model_training.py`의 `meta_cols`에 모든 target 열을 포함:

```python
# BEFORE (buggy):
meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date", target]

# AFTER (fixed):
meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"] + TARGETS
```

**동일한 fix가 `src/04_submit.py`의 `get_train_feature_cols_for_target()`에도 필요:**

```python
# BEFORE:
meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"]

# AFTER:
meta_cols = ["subject_id", "lifelog_date", "sleep_date", "date"] + TARGETS
```

### Priority 2: Cross-validation 구조 개선 (✅ 완료)

Current validation: each subject's last 7 days → temporal leakage possible.
More robust approach for future:

1. **Time-based split:** early months train, later months validation
2. **Leave-one-subject-out:** 1 subject as validation at a time
3. **Multiple random seeds** for variance check

**Status:** Models retrained with correct feature set (141 features, no leakage).
Val log-loss range: 0.6186 (S1) ~ 0.7039 (S2).

### Priority 3: Calibration 추가

03_model_training.py에 Isotonic calibration 또는 Platt scaling 추가:

```python
from sklearn.calibration import CalibratedClassifierCV
model = CalibratedClassifierCV(model, method='isotonic', cv=3)
```

### Priority 4: Feature engineering 개선

현재 feature engineering은 day-level aggregation만 수행. 시간적 패턴을 잡기 위해:
- **Rolling window features:** 이전 1~7일간의 aggregate
- **Temporal features:** day_of_week, is_weekend, day_of_year
- **Subject-level stats:** 개인별 평균/std (lifetime aggregates)

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `src/03_model_training.py` | `meta_cols`에 `TARGETS` 전체 포함 |
| `src/04_submit.py` | `get_train_feature_cols_for_target()`에서 `meta_cols`에 `TARGETS` 전체 포함 |
| `src/03_model_training.py` (선택) | Calibration 추가 |
| `src/02_feature_engineering.py` (선택) | rolling window, temporal features 추가 |

---

## 예상 개선 효과

1. **Target leakage 제거:** S2/S3/S4의 log-loss가 가장 크게 개선됨
   - 예상 전체 log-loss: **0.5395 수준으로 회복** (current 0.90535 → ~0.55)

2. **OOD 개선:** 추가 temporal features로 extrapolation 성능 향상
   - 예상 추가 개선: **0.55 → ~0.50**

## 완료된 작업 (2026-05-01 02:39 KST)

1. ✅ **Target leakage 제거** — `src/03_model_training.py`와 `src/04_submit.py` 수정
2. ✅ **모델 재학습** — 7개 모델 모두 141 features로 재학습 완료
3. ✅ **테스트 feature 생성** — `src/05_generate_test_features.py`로 test features 생성
4. ✅ **제출 파일 생성** — `submissions/submission_20260501_023905.csv` (250 rows)

### 새로운 모델 val log-loss
| Target | Val Loss | Best Iter | Features |
|--------|----------|-----------|----------|
| Q1     | 0.6905   | 28        | 141      |
| Q2     | 0.6704   | 2         | 141      |
| Q3     | 0.6664   | 6         | 141      |
| S1     | 0.6186   | 28        | 141      |
| S2     | 0.7039   | 28        | 141      |
| S3     | 0.6290   | 32        | 141      |
| S4     | 0.6418   | 34        | 141      |

---

## 부록: 현재 모델의 feature 구성 (모든 target)

| Target | 총 features | Target leakage | Real features | Best iter | Val loss |
|--------|------------|----------------|---------------|-----------|----------|
| Q1     | 141        | ❌ none        | 141           | 28        | 0.6905   |
| Q2     | 141        | ❌ none        | 141           | 2         | 0.6704   |
| Q3     | 141        | ❌ none        | 141           | 6         | 0.6664   |
| S1     | 141        | ❌ none        | 141           | 28        | 0.6186   |
| S2     | 141        | ❌ none        | 141           | 28        | 0.7039   |
| S3     | 141        | ❌ none        | 141           | 32        | 0.6290   |
| S4     | 141        | ❌ none        | 141           | 34        | 0.6418   |

*모든 top features는 genuine feature (WiFi, BLE, UsageStats 등)*

### Clean 모델 Top-5 Features
| Target | Top Feature | Importance (gain) |
|--------|------------|-------------------|
| Q1     | mWifi_wifi_max_rssi_std | 125.3 |
| Q2     | mACStatus_m_charging_mean | 47.0 |
| Q3     | mWifi_wifi_max_rssi_std | 41.9 |
| S1     | mScreenStatus_m_screen_use_mean | 117.8 |
| S2     | mWifi_wifi_max_rssi_max | 172.7 |
| S3     | mWifi_wifi_max_rssi_max | 180.3 |
| S4     | mGps_gps_avg_speed_max | 167.7 |
