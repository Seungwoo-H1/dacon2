# Dacon2 V53 파이프라인 감사 결과

**작성일:** 2026-05-07  
**감사 대상:** V53 Deep Feature Engineering 파이프라인  
**검토 범위:** 데이터 leakage, feature 품질, 모델 진단, train/test 분포

---

## 1. 결론 (Executive Summary)

### 주요 발견사항 (중요도 순)

| # | 항목 | 심각도 | 설명 |
|---|------|--------|------|
| 1 | **Temporal Z-Score Leakage** | 🔴 Critical | 10개 subject 전원이 train/test date 범위 중첩. z-score stats가 오염된 상태에서 계산됨 |
| 2 | **Feature Selection Leakage** | 🟡 High | Global ranking 후 CV OOF → 전역 ranking이 validation data 정보 간접 누수 |
| 3 | **LEAK_S/Q 제거 시 성능 급감** | 🟡 High | S1/S2 타깃에서 LEAK_S 제거 시 CV loss +0.016~+0.019. CV-LB gap의 핵심 원인 |
| 4 | **S4 Calibration Over-Quantization** | 🟡 Medium | S4 예측값이 12개 값으로만 분포. coarse mean-matching calibration |
| 5 | **Train/Test 분포 불일치** | 🟡 Medium | wHr 관련 7개 feature에서 KS stat > 0.14. 도메인 시프트 존재 |

**CV-LB Gap (~0.10)의 주요 원인:**
1. `LEAK_S`/`LEAK_Q` feature 제거 (S1/S2 타깃에 결정적 영향)
2. Z-score personalization의 temporal contamination
3. Feature selection leakage (global ranking → overoptimistic CV)
4. S4 quantization 오류

---

## 2. Leakage 분석

### 2.1 Temporal Z-Score Leakage (🔴 Critical)

**발견:** 모든 10개 subject(train id01~id10)에서 train date range와 test date range가 중첩됨.

| Subject | Train Period | Test Period | Overlap |
|---------|-------------|-------------|---------|
| id01 | 2024-06-27..09-01 | 2024-07-31..09-15 | train 중 13일, test 중 14일 |
| id02 | 2024-07-18..09-28 | 2024-08-26..10-16 | train 중 16일, test 중 16일 |
| id03 | 2024-07-18..09-13 | 2024-08-17..10-10 | train 중 11일, test 중 11일 |
| id04 | 2024-08-01..10-27 | 2024-09-10..10-30 | train 중 23일, test 중 24일 |
| id05 | 2024-08-29..11-15 | 2024-09-29..11-20 | train 중 18일, test 중 16일 |
| id06 | 2024-06-04..08-19 | 2024-07-07..08-25 | train 중 18일, test 중 18일 |
| id07 | 2024-06-10..08-14 | 2024-07-14..09-02 | train 중 16일, test 중 15일 |
| id08 | 2024-06-26..09-17 | 2024-08-01..09-20 | train 중 24일, test 중 17일 |
| id09 | 2024-07-02..09-04 | 2024-08-06..09-22 | train 중 13일, test 중 14일 |
| id10 | 2024-07-07..09-15 | 2024-08-05..09-27 | train 중 11일, test 중 11일 |

**영향도:**
- test rows 중 **156/250 (62.4%)** 가 train date range 내에 존재
- `zscore_stats.json`은 ALL train data로 per-subject mean/std 계산
- 동일한 subject의 test data에 train data 기반 z-score 적용 → **temporal contamination**

**해결 방안:**
```python
# Current (leaky):
# zscore_stats computed from ALL train data per subject

# Fix: Compute z-score stats using ONLY past data (before test start)
for subject_id, test_start_date in test_start_dates.items():
    mask = (f['subject_id'] == subject_id) & (f['sleep_date'] < test_start_date)
    stats = f[mask].groupby(...).agg(['mean', 'std'])
```

### 2.2 Feature Selection Leakage (🟡 High)

**발견:** V53 파이프라인에서 feature ranking을 **전체 train 데이터**에 대해 수행 → 선택된 top-N features로 **모든 fold**에서 training/evaluation.

**문제:**
- ranking에 validation data의 정보가 간접적으로 포함 (CV fold의 pattern이 ranking에 반영됨)
- 올바른 방법: 각 fold별로 train data만으로 ranking → fold별 선택 feature로 evaluation

**영향:** CV score가 과최적화 (overoptimistic). 실제 LB에서는 ranking 순서가 다를 수 있어 성능 하락.

### 2.3 LEAK_S/Q Leakage 검증

**실험 결과:** LEAK_S 제거 시 S1/S2 타깃 CV loss가 **악화**됨.
- S1: loss +0.0161 증가
- S2: loss +0.0193 증가
- LEAK_Q: Q 타깃에 미미한 영향

**해석:** LEAK_S 컬럼이 예측에 유용한 signal을 포함 → LEAK_S 제거가 CV-LB gap의 **주요 원인** 중 하나.

**LEAK_S 컬럼 목록 (S1/S2 타깃용):**
```
wLight_w_light_mean, wLight_w_light_std, wLight_w_light_min,
wLight_w_light_max, wLight_w_light_count,
wHr_hr_mean, wHr_hr_std, wHr_hr_min, wHr_hr_max,
wHr_hr_median, wHr_hr_count,
wPedo_pedo_step_mean, wPedo_pedo_step_sum,
wPedo_pedo_step_frequency_mean, wPedo_pedo_step_frequency_sum,
wPedo_pedo_running_step_mean, wPedo_pedo_running_step_sum,
wPedo_pedo_walking_step_mean, wPedo_pedo_walking_step_sum,
wPedo_pedo_distance_mean, wPedo_pedo_distance_sum,
wPedo_pedo_speed_mean, wPedo_pedo_speed_sum,
wPedo_pedo_burned_calories_mean, wPedo_pedo_burned_calories_sum
```

### 2.4 Train/Test Row Overlap (🟢 Clean)

- train 450 rows ↔ test 250 rows → **0 duplicate rows**
- 104개 날짜가 train과 test에서 공통으로 존재하지만, **동일 subject 내에서 중첩 없음** (temporal separation 유지)

---

## 3. Feature Quality 분석

### 3.1 Feature Stats

| 지표 | 값 |
|------|-----|
| 총 feature 수 (features.parquet) | 142 (수치형) |
| 총 feature 수 (features_v57.parquet) | 324 |
| LEAK_S 컬럼 (v57) | **0** (제거 완료) |
| Constant feature | 12 |
| NaN 있는 feature | 102 (72.0%) |
| Outlier (>3σ) 있는 feature | 68 (48.2%) |

### 3.2 High Missing Rate Features

| Feature | Missing Rate | 출처 |
|---------|-------------|------|
| wHr_hr_std | 24.7% | Heart Rate |
| wHr_hr_mean | 13.1% | Heart Rate |
| wHr_hr_count | 13.1% | Heart Rate |
| wPedo_pedo_* (모두) | 10.2% | Pedometer |
| mBle_ble_*_std | 8.2% | BLE |

**주목:** wHr 데이터는 12:00-21:00만 수집 (야간 데이터 없음). Pedometer도 15.7% nighttime data.

### 3.3 High Correlation Groups (|r| > 0.9, 56 pairs)

**핵심 다중공선성 그룹:**

1. **wPedo step/distance/speed (r ≈ 1.0):**
   - `wPedo_pedo_step_mean ↔ wPedo_pedo_step_frequency_mean` (r=1.000)
   - `wPedo_pedo_distance_mean ↔ wPedo_pedo_speed_mean` (r=0.996)
   - `sum` series도 동일 패턴

2. **BLE device count (r ≈ 1.0):**
   - `mBle_ble_count_std ↔ mBle_ble_device_count_std` (r=1.000)
   - `mBle_ble_count_max ↔ mBle_ble_device_count_max` (r=0.954)

3. **Activity/Screen/AC count (r ≈ 0.96-0.98):**
   - `mACStatus_m_charging_count ↔ mScreenStatus_m_screen_use_count` (r=0.988)
   - `mACStatus_m_charging_count ↔ mActivity_m_activity_count` (r=0.984)

4. **Ambience categories (r ≈ 0.92-0.96):**
   - `mAmbience_ambience_vehicle_sum ↔ mAmbience_ambience_car_sum` (r=0.916)

5. **WiFi RSSI groups:**
   - `wifi_max_rssi` mean/std/min/max 가 모두 상호 상관
   - `wifi_avg_rssi` mean/std/max 도 상관

### 3.4 Global Top 20 Features (Q1 타깃 기준)

```
1.  mUsageStats_usage_major_ratio_mean
2.  mWifi_wifi_max_rssi_std
3.  mWifi_wifi_max_rssi_mean
4.  mBle_ble_device_count_mean
5.  mWifi_wifi_max_rssi_min
6.  mUsageStats_usage_total_time_mean
7.  mBle_ble_count_std
8.  mACStatus_hour_night
9.  mBle_ble_count_mean
10. mBle_ble_count_max
11. mScreenStatus_hour_morning
12. mWifi_wifi_avg_rssi_max
13. mActivity_m_activity_mean
14. mBle_ble_rssi_std_std
15. mAmbience_ambience_truck_sum
16. mScreenStatus_m_screen_use_mean
17. mBle_ble_max_rssi_max
18. mBle_ble_device_count_max
19. mWifi_wifi_strong_ratio_max
20. mScreenStatus_m_screen_use_std
```

**특징:** BLE/WiFi RSSI 관련 feature가 압도적 다수. UsageStats, ScreenStatus도 다수 포함.

---

## 4. Model Diagnostics

### 4.1 Class Imbalance

| Target | Mean | Class 0 | Class 1 | Imbalance Ratio (0/1) |
|--------|------|---------|---------|----------------------|
| Q1 | 0.496 | 227 | 223 | 1.02 |
| Q2 | 0.562 | 197 | 253 | 0.78 |
| Q3 | 0.600 | 180 | 270 | 0.67 |
| S1 | 0.682 | 143 | 307 | 0.47 |
| S2 | 0.651 | 157 | 293 | 0.54 |
| S3 | 0.662 | 152 | 298 | 0.51 |
| S4 | 0.560 | 198 | 252 | 0.79 |

**분석:** S1-S3가 가장 심한 imbalance (class 0이 class 1의 약 1/2). `scale_pos_weight` 파라미터로 어느 정도 보정됨.

### 4.2 OOF Prediction Distribution

| Target | Range | Mean | Std | Unique (4dp) | <0.05 | >0.95 |
|--------|-------|------|-----|-------------|-------|-------|
| Q1 | 0.0001-0.9999 | 0.496 | 0.209 | 12 | 5 | 6 |
| Q2 | 0.0001-0.9999 | 0.562 | 0.227 | 13 | 15 | 11 |
| Q3 | 0.0001-0.9999 | 0.600 | 0.204 | 13 | 3 | 10 |
| S1 | 0.0001-0.9999 | 0.682 | 0.230 | 13 | 3 | 26 |
| S2 | 0.0001-0.9999 | 0.651 | 0.222 | 15 | 3 | 23 |
| S3 | 0.0001-0.9999 | 0.662 | 0.223 | 12 | 8 | 60 |
| S4 | **0.1176-0.9999** | 0.560 | 0.232 | **12** | 0 | 2 |

### 4.3 S4 예측 문제 (🟡 Medium)

**S4만 다른 패턴:**
- **min prediction = 0.1176** (다른 타깃은 0.0001). 낮은 예측값이 없음
- **오직 12개 unique 값**으로만 분포 (others는 12-15개)
- 예측값 분포: `[0.1176, 0.2, 0.225, 0.316, 0.5, 0.526, 0.575, 0.741, 0.783, 0.884, 0.929, 0.9999]`

**원인:** coarse mean-matching calibration이 fold/subject 평균을 기반으로 예측을 분할하여 quantization 유발. 10 subjects × 5 folds의 작은 샘플에서 과도한 quantization 발생.

**해결 방안:**
1. Isotonic regression 적용 (discrete quantization 완화)
2. 더 많은 fold 사용 (Leave-One-Subject-Out 권장)
3. S4 별도 모델로 training (class imbalance가 다른 구조)

### 4.4 CV Stability

GroupKFold 5-fold (2 subjects/fold)로 Q1 확인 시 fold 간 loss variance가 큼. 10 subjects only → fold 간 성능 차이가 큼 → CV estimate의 신뢰도 낮음.

---

## 5. Train/Test Distribution 분석

### 5.1 Kolmogorov-Smirnov Test (Feature Distribution Shift)

**7개 feature에서 유의미한 분포 차이 (p < 0.01, KS stat > 0.1):**

| Feature | KS Stat | p-value | 해석 |
|---------|---------|---------|------|
| wHr_hr_count | **0.955** | 4.7e-155 | 극단적 차이 |
| wHr_hr_std | **0.914** | 2.7e-127 | 극단적 차이 |
| wHr_hr_mean | 0.242 | 3.0e-08 | 상당한 차이 |
| mGps_gps_count_std | 0.202 | 7.1e-06 | 상당한 차이 |
| mGps_gps_count_mean | 0.192 | 2.5e-05 | 상당한 차이 |
| mWifi_wifi_avg_rssi_max | 0.141 | 3.3e-03 | 중간 차이 |
| wPedo_pedo_burned_calories_mean | 0.135 | 6.7e-03 | 중간 차이 |

**주목:** wHr (Heart Rate) features에서 KS stat이 0.9+로 극단적. train에서는 59 rows가 NaN (13.1%)이고 test에서는 NaN이 0. train/test 데이터 수집 조건이 다를 가능성.

### 5.2 Missing Value Pattern

| Feature | Train NaN | Test NaN | 차이 |
|---------|-----------|----------|------|
| wLight_* (5개) | 36 | 0 | ⚠️ Test에 없음 |
| wPedo_pedo_step_* (6개) | 46 | 1 | ⚠️ Test에 거의 없음 |
| wHr_hr_* (6개) | 59-111 | 0 | ⚠️ Test에 없음 |

**해석:** train에서는 wearable sensor data의 누락이 많지만, test에서는 누락이 거의 없음 → test 데이터의 품질이 train보다 좋을 가능성 (또는 전처리 차이).

---

## 6. Z-Score Personalization 분석

### 6.1 구조

```python
# zscore_stats.json: 4860 entries
# 구조: {feature_name: {'mean': {subject_id: value}, 'std': {subject_id: value}}}
# 예: mACStatus_m_charging_mean: {'mean': {'id01': 0.184, ...}, 'std': {'id01': 0.077, ...}}
```

- **per-feature, per-subject mean/std** → 올바른 personalization 접근
- 10개 subject × 486 features = 4860 entries

### 6.2 Leakage Risk Assessment

**✅ 긍정적:**
- train/test 계산 분리 (`add_personalization`에서 train과 test 별도 계산)
- per-subject 기반 → 개인별 정규화

**⚠️ 부정적:**
- 모든 10개 subject가 train과 test에 공통 → z-score stats computation에 test period data 포함
- 62.4%의 test rows가 train date range 내에 존재

**Impact:** z-score mean/std가 test period data로 오염 → 개인별 패턴이 과대평가됨 → CV score inflation

---

## 7. CV-LB Gap 원인 분석

### 7.1 Gap 요약

- **V53 LB:** 0.65358
- **V53 CV (OOF):** ~0.5479 (calibrated)
- **Gap:** ~0.1056

### 7.2 Gap 원인 기여도 (추정)

| 원인 | 기여도 | 설명 |
|------|--------|------|
| LEAK_S/Q 제거 | **중대** | S1/S2에서 loss +0.016~+0.019. 전체 가중치로 0.04+ 기여 가능 |
| Z-score temporal contamination | **중대** | 62.4% test rows 오염 → z-score가 테스트 시나리오와 다름 |
| Feature selection leakage | **보통** | Global ranking → CV overoptimistic |
| S4 quantization | **소규모** | S4의 12개 unique value → 예측 불확실성 증가 |
| Train/test distribution shift | **소규모** | wHr, GPS 등 7개 feature의 분포 차이 |
| Small subject count (N=10) | **중요** | CV의 높은 variance → 신뢰도 낮은 estimate |

### 7.3 V57 전환 권고

`features_v57.parquet`는 **LEAK_S 컬럼이 0개**로 이미 클린. V57 파이프라인으로의 전환을 검토:
- CV score는 다소 하락할 수 있음 (LEAK_S 제거 효과)
- 하지만 LB score는 더 안정적일 가능성 (leakage-free)
- Z-score leakage도 동일하므로 추가 수정 필요

---

## 8. 권장 사항

### 우선순위 1 (필수)

1. **Z-score personalization temporal fix**
   - 각 subject의 test 시작일 이전에 train data만 사용하여 z-score stats 계산
   - 코드: `data_processed/zscore_stats.json` 재생성

2. **Feature selection을 per-fold로 변경**
   - Global ranking 대신 각 fold별로 train data ranking → fold별 feature 선택

### 우선순위 2 (권장)

3. **S4 calibration 개선**
   - Isotonic regression 적용
   - 또는 S4 별도 모델로 training

4. **GroupCrossValidation 사용**
   - GroupKFold 대신 GroupLeaveOneOut (10 subjects → LOO = 10-fold)
   - 더 안정된 CV estimate

### 우선순위 3 (개선)

5. **High-correlation feature 제거**
   - wPedo step/mean/distance/speed: mean만 유지
   - BLE count/device_count: count만 유지
   - WiFi RSSI max/avg: max만 유지

6. **Missing value 처리 개선**
   - train/test 누락 패턴 차이 확인 (wearable sensor 전처리 일관성)
   - 결측치 패턴 feature 추가 고려

7. **External data 통합 검토**
   - `external_data.parquet`에 holiday/month/season feature가 있으나 미사용
   - leakage 없는 calendar data는 유용할 수 있음

### 우선순위 4 (전략적)

8. **V57 파이프라인으로의 마이그레이션**
   - 이미 LEAK_S 0개
   - 더 clean한 baseline 제공
   - 단, z-score leakage는 동일하므로 별도 수정 필요

---

## 부록 A: LEAK_S/Q 컬럼 정의

### LEAK_S (Sleep targets S1-S4)
```
wLight_w_light_mean, wLight_w_light_std, wLight_w_light_min,
wLight_w_light_max, wLight_w_light_count,
wHr_hr_mean, wHr_hr_std, wHr_hr_min, wHr_hr_max,
wHr_hr_median, wHr_hr_count,
wPedo_pedo_step_mean, wPedo_pedo_step_sum,
wPedo_pedo_step_frequency_mean, wPedo_pedo_step_frequency_sum,
wPedo_pedo_running_step_mean, wPedo_pedo_running_step_sum,
wPedo_pedo_walking_step_mean, wPedo_pedo_walking_step_sum,
wPedo_pedo_distance_mean, wPedo_pedo_distance_sum,
wPedo_pedo_speed_mean, wPedo_pedo_speed_sum,
wPedo_pedo_burned_calories_mean, wPedo_pedo_burned_calories_sum
```

### LEAK_Q (Quality targets Q1-Q3)
```
wHr_hr_mean, wHr_hr_std, wHr_hr_min, wHr_hr_max,
wHr_hr_median, wHr_hr_count
```

---

## 부록 B: 주요 파일 경로

| 파일 | 설명 |
|------|------|
| `src/gen_submission_v53.py` | 제출용 파이프라인 (z-score + LEAK 제거) |
| `src/v53_cv_baseline.py` | CV 평가용 baseline |
| `src/53_v53_deep_feature_engineering.py` | Deep feature engineering |
| `data_processed/features.parquet` | Train features (450 rows, 142 features) |
| `data_processed/test_features.parquet` | Test features (250 rows, 142 features) |
| `data_processed/features_v57.parquet` | Clean features (653 rows, 324 features, 0 LEAK_S) |
| `data_processed/zscore_stats.json` | Per-subject z-score stats (4860 entries) |
| `data_processed/oof_v53.csv` | V53 OOF predictions |
| `data_processed/external_data.parquet` | Calendar/holiday data (미사용) |
| `src/oof_v43.csv` | V43 OOF (reference: CV 0.55688) |

---

*감사 종료*
