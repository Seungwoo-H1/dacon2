# DAcon2 v8 코드 심층 분석 보고서

## 1. Feature Leakage 검증

### 🔴 CRITICAL: 수면 중 라이프로그 데이터 누수 (S1-S4 타겟)

**발견사항:**
- `wLight`, `wHr`, `wPedo` 의 wrist device 데이터가 **24시간 전일 단위**로 집계되어 sleep quality 타겟(S1-S4)의 입력 피처로 사용됨
- Nighttime(22:00-06:00) wLight 평균: 17.3 lux (매우 낮음, 어두운 환경)
- Daytime wLight 평균: 263.9 lux
- Nighttime wHr 평균 낮음 (수면 중 심박수)
- Nighttime wPedo step: 258,212 / 2,582,121 (10%만 밤에 발생)
- 즉, **"밤에 얼마나 어두웠는지", "밤에 얼마나 적게 움직였는지"** 가 피처에 직접 포함되어 수면 품질 예측에 활용됨

**근거:**
```python
# 02_feature_engineering.py: aggregate_numeric() - 시간 제한 없음
def aggregate_numeric(df, col, agg_cols):
    # 모든 시간대 데이터를 무조건 aggregation → sleep-time 데이터 포함
    grouped = df.groupby(["subject_id", "date"])[col].agg(["mean", "std", "min", "max", "count"])

# wrist device aggregation:
# wLight_w_light_mean  ← night 포함 24h mean
# wHr_hr_mean          ← night 포함 24h mean  
# wPedo_pedo_step_sum  ← night 포함 24h total
```

**위험도:** 🔴 CRITICAL — sleep quality 타겟에 wrist device의 수면 중 데이터를 직접 사용 = 모델이 정답을 "눈으로 보는" 것과 동일

### 🟡 MEDIUM: Phone 기반 수면 간접 데이터

- `mACStatus_m_charging_mean`: 밤에 충전 중 비율 → 수면 위치 추론 가능
- `mScreenStatus_m_screen_use_mean`: 밤에 화면 사용 비율 → 낮음
- `mACStatus_hour_night` / `mScreenStatus_hour_night`: 시간대별 비율
- 상관계수: mACStatus_hour_evening ↔ mScreenStatus_hour_evening: r=0.9912 (고민공)

### 🟢 LOW: 기타 잠재적 누수

- WiFi BLE device count: 위치 기반 추론 (약한 누수)
- Ambience "Inside, small room": 실내 체류 패턴 (간접적)

---

## 2. Feature Quality 분석

### 결측률 분석 (450행 × 142개 피처)

| 결측률 | 수 | 주요 피처 |
|--------|-----|-----------|
| 0% (100%) | 40개 | Numeric aggregation 전체, ambience, time-of-day |
| 3-5% | ~50개 | mUsageStats(10), mWifi(11-22), GPS(25), BLE(34) |
| ~8% | 3개 | wHr_hr_mean(59), wPedo(46), wLight(36) |
| ~25% | 1개 | wHr_hr_std(111) |

### 🔴 Constant / Near-constant Features (13개) — 즉시 제거 대상

```
mACStatus_m_charging_min          → 항상 0 (상수)
mACStatus_m_charging_max          → {0, 1} (2값만)
mLight_m_light_min                → {0, 9} (2값만)
mScreenStatus_m_screen_use_min    → 항상 0
mScreenStatus_m_screen_use_max    → {0, 1}
wPedo_pedo_running_step_mean      → 항상 0 (누적 합 0)
wPedo_pedo_running_step_sum       → 항상 0
wPedo_pedo_walking_step_mean      → 항상 0
wPedo_pedo_walking_step_sum       → 항상 0
mGps_gps_has_speed_mean/std/max/min → {0, 1} (4개 모두 상수)
mUsageStats_usage_major_ratio_min → 항상 0
mUsageStats_usage_game_ratio_mean → {0, ~0.005} (거의 0)
mUsageStats_usage_game_ratio_std  → {0, ~0.034} (거의 0)
mUsageStats_usage_game_ratio_max  → {0, 0.25}
mUsageStats_usage_game_ratio_min  → 항상 0
```

### 🟡 높은 다공선성 (Multicollinearity) — 28개 쌍

```python
# r=1.0000 (완전 다중공선성)
wPedo_pedo_step_mean       ↔ wPedo_pedo_step_frequency_mean
wPedo_pedo_step_sum        ↔ wPedo_pedo_step_frequency_sum
wPedo_pedo_distance_mean   ↔ wPedo_pedo_speed_mean
mBle_ble_count_mean        ↔ mBle_ble_device_count_mean
mWifi_wifi_count_mean      ↔ mWifi_wifi_bssid_count_mean
mGps_gps_avg_speed_min     ↔ mGps_gps_max_speed_min
mUsageStats_usage_game_ratio_* (3개 완전 중복)

# r>0.99 (거의 중복)
wPedo_pedo_step_mean ↔ wPedo_pedo_distance_mean (r=0.9971)
wPedo_pedo_step_mean ↔ wPedo_pedo_speed_mean (r=0.9971)
```

### 🟡 wHr_hr_mean 이상치

- range: [2.0, 240.0] — 정상 HR은 40-180
- hr_mean < 10인 row: 62개 (13.8%)
- hr_mean = 2는 데이터 수집 오류 가능성 (wrist 장치의 심박수 측정 실패 시 2로 기록되는 듯)

---

## 3. Time Leakage 가능성

### 수면 중 라이프로그 데이터 누수 (상세)

**메커니즘:**
```
lifelog_date (2024-06-26) → sleep_date (2024-06-27)
  │
  ├─ mACStatus: 2024-06-26 전체 시간대 (00:00-23:59) aggregation
  ├─ mActivity: 2024-06-26 전체 aggregation
  ├─ wLight:    2024-06-26 24시간 aggregation ← 수면 중 포함!
  ├─ wHr:       2024-06-26 24시간 aggregation ← 수면 중 포함!
  ├─ wPedo:     2024-06-26 24시간 aggregation ← 수면 중 포함!
  └─ ...
  
  → sleep_quality (S1-S4)를 예측하는 데 사용
```

**구체적 문제점:**

1. **wLight (wrist)**: 수면 중 암흑 환경(lux ≈ 0)이 `wLight_w_light_mean`에 반영 → 이 값이 낮을수록 수면 품질 좋음으로 연결
2. **wHr (wrist)**: 수면 중 심박수 저하가 `wHr_hr_mean`에 반영 → 낮을수록 좋은 수면
3. **wPedo (wrist)**: 수면 중 활동 최소화 → `wPedo_pedo_step_sum`이 낮을수록 좋음
4. **mACStatus (phone)**: 밤에 충전 중 → 충전 패턴으로 수면 시간 추론 가능

**해결 방향:**
- 수면 대상(S1-S4)에는 수면 중 시간대(22:00-06:00 또는 00:00-07:00)의 wrist data만 사용
- 또는 daytime(06:00-22:00)만 사용 (전혀 다른 정보)
- Q1-Q3 (daytime activity)에는 nighttime data를 제외하는 것이 타당

---

## 4. Personalization 부재

**문제:** 모든 피처가 전역(day-level) aggregation. 개인별 baseline deviation, z-score 없음.

**근거:**
```
142개 피처 중 subject별 정규화 피처: 0개
```

**추가해야 할 개인화 피처:**

```python
# 1) Personal baseline deviation
for each subject s:
    s_baseline_mean = mean(df[df.subject_id == s][feature])
    s_baseline_std = std(df[df.subject_id == s][feature])
    new_feature = (feature - s_baseline_mean) / s_baseline_std  # z-score
    
# 2) Personal rate of change (day-over-day)
df[f'{feature}_delta'] = df.groupby('subject_id')[feature].diff()

# 3) Personal percentile rank
df[f'{feature}_pctl'] = df.groupby('subject_id')[feature].rank(pct=True)

# 4) Personal variance ratio
#    그날의 분산 / 전체 기간 분산
```

**장점:**
- id01은 평소와 비교한 "어제보다 덜 움직였다"가 의미 있음
- id10은 평소보다 "약간 더 움직였다"가 의미 있을 수 있음
- 전역 평균과 비교하는 것은 개인차가 큰lifelog 데이터에서 noise

---

## 5. Aggregation Window 문제

**발견사항:** `config.py`에 `AGG_WINDOWS = [1, 3, 6, 12, 24]`가 정의되어 있으나 `02_feature_engineering.py`에서 **단 한 번도 사용되지 않음**.

**현재 상태:** 100% day-level aggregation (전 일괄 aggregation)

**시간 윈도우의 의미:**
- 1h window: 직전 1시간의 순간적 활동 (예: 직전 1시간 동안 얼마나 움직였나)
- 24h window: 하루 전체 (현재와 동일)

**수면 타겟에 대한 시간 윈도우 의미:**
- S1-S4 (수면 품질): 수면 중(22:00-07:00) 데이터만 의미 있음
  - → 24h aggregation은 leakage
  - → 6h/8h aggregation (수면 시간대 포함)만 의미 있음
- Q1-Q3 (daytime activity): 1h/3h/6h window가 의미 있음
  - → 직전 활동 패턴이 다음날 활동에 영향

**개선 방향:**

```python
# 수면 타겟용: nighttime aggregation (22:00-06:00)
night_mask = (df['hour'] >= 22) | (df['hour'] < 6)
night_features = df[night_mask].groupby(['subject_id', 'date']).agg(...)

# daytime aggregation: daytime only (06:00-22:00)
day_mask = ~night_mask
day_features = df[day_mask].groupby(['subject_id', 'date']).agg(...)

# 또는 window별: 직전 N시간
# df[df['hour'] <= target_hour + window].groupby(...)
```

---

## 6. Missing Pattern 분석

### Feature 생성이 실패한 case

| 원인 | 수 | 예시 |
|------|-----|------|
| Raw 데이터 없음 (wrist device 미착용) | 36-59개/day | wLight(36), wHr(59), wPedo(46) |
| GPS 데이터 없음 | 25개/day | mGps (실내에서 GPS 없음) |
| WiFi/BLE 데이터 없음 | 10-22개/day | mWifi, mBle |
| UsageStats 데이터 없음 | 10개/day | mUsageStats |

**분석:**
- `wHr_hr_count` range: [1.0, 61.0] → 하루에 고작 1~61개 측정 (수초마다 1개)
- `wLight_count` range: [30, 146] → 하루에 30~146개 (10~50분 간격)
- **심박수 데이터가 매우 희박** → hr_mean, hr_std 신뢰도 낮음
- GPS 미수신은 실내에서 당연 → `gps_count` 자체가 실내/야외 지표로 사용 가능

---

## 7. Model Training 분석

### CV Logic 문제

```python
# 03_model_training.py
def create_cv_splits(df, n_splits=5, val_days=7):
    # 각 subject의 마지막 7일만 validation
    # 나머지 subjects는 training에 포함
    # → 각 subject는 단 한 번만 validation에 참여
    # → proper cross-validation 아님 (subject가 고정됨)
```

**문제점:**
1. 각 subject의 마지막 7일만 validation → 10개 subject가 있으면 10회 split 중 각 subject 1회만 validation
2. `n_splits=5`지만 실제 fold는 subject 수(10)에 의해 결정 → 5-fold와 다름
3. Time-series split이 아닌 "random subject-based" split

**올바른 CV:**
```python
# GroupKFold로 subject 단위로 fold 분할
# 또는 temporal split: 최근 N일만 validation
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=features['subject_id']):
    ...
```

### Hyperparameter 문제

```python
# config.py의 기본값
LGBM_PARAMS = {
    "num_leaves": 63,        # 너무 큼 (과적합 위험)
    "max_depth": -1,         # 무제한 → 과적합
    "learning_rate": 0.05,   # 표준
    "n_estimators": 1000,    # early_stopping=50로 조절
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,        # 너무 약한 regularization
    "reg_lambda": 1.0,
    "min_child_samples": 20, # 적절
}
```

v5에서는 `num_leaves=10-25, max_depth=3-5, reg_alpha=0.5-5.0` 등 훨씬 더 conservative한 hyperparameter를 사용.

### Missing value 처리

```python
# 03_model_training.py
train_df_san = train_df[feature_cols].fillna(0).rename(columns=rename_map)
```

- 단순 `fillna(0)`: 결측이 "0"인 것과 "데이터 없음"을 구분 못 함
- `wHr_hr_std`는 25% 결측 → 0으로 채우면 심박수 변동성을 "없음"으로 해석
- Better: `fillna(median)`, 또는 dedicated `missing_indicator` 추가

---

## 8. 전체 요약 및 권장 개선사항

### Priority 1: Leakage 해결 (가장 중요)

```python
# 수면 타겟(S1-S4)용: nighttime-only features
# phone data: 00:00-07:00만
phone_night = df[(df['hour'] >= 0) & (df['hour'] < 7)]
sleep_features = phone_night.groupby(['subject_id', 'date']).agg(...)

# 또는: wrist device nighttime data만 (22:00-06:00)
wrist_night = df[(df['hour'] >= 22) | (df['hour'] < 6)]
sleep_features = wrist_night.groupby(['subject_id', 'date']).agg(...)

# 일일 활동 타겟(Q1-Q3)용: daytime-only features  
# 22:00-06:00 제외
daytime = df[~((df['hour'] >= 22) | (df['hour'] < 6))]
day_features = daytime.groupby(['subject_id', 'date']).agg(...)
```

### Priority 2: Constant/고다공선성 피처 제거 (13개 상수 + 28개 쌍)

```python
# 제거 대상:
constants = [
    'mACStatus_m_charging_min',
    'mLight_m_light_min',
    'mScreenStatus_m_screen_use_min',
    'wPedo_pedo_running_step_mean',
    'wPedo_pedo_running_step_sum',
    'wPedo_pedo_walking_step_mean',
    'wPedo_pedo_walking_step_sum',
    'mGps_gps_has_speed_mean',
    'mGps_gps_has_speed_std',
    'mGps_gps_has_speed_max',
    'mGps_gps_has_speed_min',
    'mUsageStats_usage_major_ratio_min',
    'mUsageStats_usage_game_ratio_min',
]

# 다중공선성 제거 (상관 > 0.99인 것 중 하나만 유지)
# 예: wPedo_pedo_step_mean과 wPedo_pedo_distance_mean → 둘 중 하나만
#     mBle_ble_count_mean과 mBle_ble_device_count_mean → 둘 중 하나만
#     mWifi_wifi_count_mean과 mWifi_wifi_bssid_count_mean → 둘 중 하나만
```

### Priority 3: Personalization 추가

```python
# 개인별 baseline deviation
for feature in all_features:
    for subject in subjects:
        subj_data = df[df.subject_id == subject][feature]
        overall_mean = df[feature].mean()
        overall_std = df[feature].std()
        df.loc[df.subject_id == subject, f'{feature}_personal_zscore'] = (
            subj_data - subj_data.mean()
        ) / subj_data.std()
```

### Priority 4: CV 개선

```python
# GroupKFold 사용
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=features['subject_id']):
    ...
```

### Priority 5: Missing value 처리 개선

```python
# Median imputation + missing indicator
for col in numeric_cols:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())
```

### Priority 6: wHr 이상치 처리

```python
# hr_mean < 20인 값은 이상치로 처리
df = df[(df['wHr_hr_mean'] >= 20) | df['wHr_hr_count'].isnull()]
# 또는 winsorize
from scipy.stats import mstats
df['wHr_hr_mean'] = mstats.winsorize(df['wHr_hr_mean'], limits=[0.01, 0.99])
```

---

## 부록: 현재 데이터 구조

```
Shape: 450 rows × 153 columns
  - Meta: subject_id, lifelog_date, sleep_date, date
  - Targets: Q1, Q2, Q3, S1, S2, S3, S4
  - Features: 142 columns (141 numeric + 1 categorical)
  
Subjects: 10 (id01-id10), all overlap between train and submission
Time range: 2024-06-03 ~ 2024-11-14 (41-57 days per subject)
```
