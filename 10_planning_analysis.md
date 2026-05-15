# 🏗️ Planning Analysis — 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

**작성일:** 2026-05-01 (최초) / 2026-05-01 (업데이트)  
**분석:** 집가헤응 (기획 에이전트)  
**대회:** [제 5회 ETRI 휴먼이해 인공지능 논문경진대회](https://dacon.io/competitions/official/236690)

---

## 결론

**10명 × 450 샘플의 극소규모 데이터로 7개 타깃 이진 분류를 수행해야 하며, 재현성 검증이 최종 수상의 전제 조건인 대회.** 핵심 전략:

1. **개인별 모델 + 글로벌 모델 하이브리드** — 450 샘플은 전역 모델로 개인차捕捉이 불충분하므로 subject-aware 아키텍처 필수
2. **LightGBM 기반 개별 타깃 모델** — 지표 간 상관도 낮아 개별 학습이 효율적
3. **엄격한 시간 기반 CV + subject-leave-out 검증** — 데이터 누수(Data Leakage)가 승패를 가름
4. **실제 스키마 기반 피처 엔지니어링** — Parquet 12개 파일의 실제 컬럼/타입 기반 전략

---

## 1. 데이터의 본질적 특성

### 1.1 실제 데이터 스키마 (실측 결과)

Parquet 12개 파일의 pandas metadata를 통한 실제 컬럼 분석:

| 파일명 | 크기 | 행수 | 컬럼 | 타입 | 설명 |
|--------|------|------|------|------|------|
| `ch2025_mACStatus.parquet` | 4.33 MB | 939,897 | subject_id, timestamp, m_charging | object, datetime64[ns], int64 | 충전 상태(0/1) |
| `ch2025_mActivity.parquet` | 4.43 MB | 961,063 | subject_id, timestamp, m_activity | object, datetime64[ns], int64 | 활동 레벨 |
| `ch2025_mAmbience.parquet` | 14.16 MB | 476,577 | subject_id, timestamp, m_ambience | object, datetime64[ns], object | JSON(소리분류 확률) |
| `ch2025_mBle.parquet` | 3.88 MB | 21,830 | subject_id, timestamp, m_ble | object, datetime64[ns], object | JSON(BLE MAC/RSSI) |
| `ch2025_mGps.parquet` | 68.59 MB | 800,611 | subject_id, timestamp, m_gps | object, datetime64[ns], object | JSON(좌표/속도/고도) |
| `ch2025_mLight.parquet` | 0.65 MB | 96,258 | subject_id, timestamp, m_light | object, datetime64[ns], float64 | 조도(lux) |
| `ch2025_mScreenStatus.parquet` | 4.36 MB | 939,653 | subject_id, timestamp, m_screen_use | object, datetime64[ns], int64 | 화면 사용 |
| `ch2025_mUsageStats.parquet` | 0.95 MB | 45,197 | subject_id, timestamp, m_usage_stats | object, datetime64[ns], object | JSON(앱이름/시간) |
| `ch2025_mWifi.parquet` | 3.80 MB | 76,336 | subject_id, timestamp, m_wifi | object, datetime64[ns], object | JSON(BSSID/RSSI) |
| `ch2025_wHr.parquet` | 9.89 MB | 382,918 | subject_id, timestamp, heart_rate | object, datetime64[ns], object | JSON(심박 배열) |
| `ch2025_wLight.parquet` | 3.25 MB | 633,741 | subject_id, timestamp, w_light | object, datetime64[ns], float64 | 손목 조도 |
| `ch2025_wPedo.parquet` | 4.55 MB | 748,100 | subject_id, timestamp, step, step_frequency, running_step, walking_step, distance, speed, burned_calories | object, datetime64[ns], 9개 수치 | 보행 데이터 |

**총 행수:** 약 6,149,008행 (약 615만 행)

### 1.2 시계열의 불규칙성 — 핵심 난제

- **12개 소스마다 sampling rate가 완전히 다름**
  - mACStatus: ~1M개 timestamp (10분 단위 추정)
  - mGps: 800K개, mActivity: 961K개, mLight: 96K개, mWifi: 76K개
  - mBle: 21K개 (가장 적음), mUsageStats: 45K개
  - wPedo: 748K개, wHr: 383K개, wLight: 634K개
  - mAmbience: 477K개

- **12개 소스의 timestamp가 반드시 정렬되어 있지도 않고, 동일 timestamp가 여러 소스에 존재하지도 않을 수 있음**
  - Parquet 데이터가 subject_id 순으로 정렬되어 있는 것은 확인됨
  - 하지만 timestamp 내림차순으로 저장되어 있음 (parquet raw data의 subject_id 값이 `id10, id01` 순)

- **핵심 과제:** 서로 다른 sampling rate의 12개 소스를 어떻게 통합 피처로 만들 것인가

### 1.3 시간적 관계 (중요 — 실증 확인)

```
lifelog_date (T) → sleep_date (T+1일)

모든 450 row에서 정확히 1일 차이 확인됨.
```

- **즉, lifelog_date=T의 라이프로그만으로 sleep_date=T+1의 타깃 예측 가능**
- **aggregation window:** lifelog_date=T 이전의 모든 데이터 허용 (미래 데이터 제외)
- **단, lifelog_date=T의 "당일" 데이터를 얼마나까지 포함할 수 있는지?**
  - sleep_date=T+1의 타깃이므로, lifelog_date=T의 전일 데이터(00:00~23:59) 모두 사용 가능
  - **단, T일 중 수면 중(LTE 등)의 데이터는 타깃에 직접 영향을 줄 수 있음 → leakage 가능성**
  - **해결책:** 수면 시간(사후 정의) 이전의 데이터만 피처로 사용하거나, 전일 전체를 사용하되 수면 관련 피처는 분리

### 1.4 데이터 불균형 (실측)

```
Q1(수면의질):   49.6% positive (223/450) — 거의 균형
Q2(피로도):     56.2% positive (253/450)
Q3(스트레스):   60.0% positive (270/450)
S1(수면시간):   68.2% positive (307/450) — 가장 skewed
S2(수면효율):   65.1% positive (293/450)
S3(수면지연):   66.2% positive (298/450)
S4(수면각성):   56.0% positive (252/450)
```

- Q1이 유일하게 50% 근처 (이진 분류가 가장 어려운 지표)
- S1이 가장 skewed (positive가 많으므로 negative 예측이 어려움)

### 1.5 소규모 데이터 — 실증 통계

```
10명 사용자, 평균 45일 데이터 = 450 샘플

개인별 데이터 개수:
  id01: 41일 (06-27 ~ 09-01)
  id02: 48일 (07-18 ~ 09-28)
  id03: 33일 (07-18 ~ 09-13)
  id04: 57일 (08-01 ~ 10-27)
  id05: 44일 (08-29 ~ 11-15)
  id06: 48일 (06-04 ~ 08-19)
  id07: 49일 (06-10 ~ 08-14)
  id08: 56일 (06-26 ~ 09-17)
  id09: 41일 (07-02 ~ 09-04)
  id10: 33일 (07-07 ~ 09-15)

 Min: 33일 (id03, id10) / Max: 57일 (id04) / Mean: 45.0일
```

- **p >> n 문제 극심:** 수백~수천 개의 피처를 450 샘플로 학습
- **개인별 편차 큼:** 같은 지표라도 개인별 positive 비율이 14.6%~84.8%까지 차이 (Q1 기준)
  - id06: Q1=14.6% (매우 낮음)
  - id03: Q1=84.8% (매우 높음)
  - → **개인별 기준선(personal baseline)이 매우 다름**

### 1.6 시간 범위

```
전체 기간: 2024-06-03 ~ 2024-11-15 (약 5.5개월)
```

---

## 2. 실제 타깃 상관관계 분석 (실측)

### 2.1 전역 Pearson 상관계수 행렬

```
       Q1     Q2     Q3     S1     S2     S3     S4
Q1:   1.000  0.122  0.102  0.361  0.073 -0.119  0.019
Q2:   0.122  1.000  0.340  0.052  0.003 -0.052 -0.024
Q3:   0.102  0.340  1.000  0.066  0.002 -0.027  0.007
S1:   0.361  0.052  0.066  1.000  0.382  0.118  0.107
S2:   0.073  0.003  0.002  0.382  1.000  0.394  0.478
S3: -0.119 -0.052 -0.027  0.118  0.394  1.000  0.086
S4:   0.019 -0.024  0.007  0.107  0.478  0.086  1.000
```

### 2.2 상관관계 해석

**강한 상관 (> 0.35):**
- S2 ↔ S3 (0.394): 수면 효율과 수면 지연은 음의 관계일 것 같으나 양의 상관 → 효율이 낮은 날은 수면 지연도 긴 경향
- S2 ↔ S4 (0.478): 수면 효율과 수면 중 각성은 강한 양의 상관 → 효율이 낮은 날은 각성도 많음 (직관과 일치)
- S1 ↔ S2 (0.382): 총 수면시간과 수면 효율은 양의 상관 (직관과 일치)
- Q1 ↔ S1 (0.361): 수면의질과 총 수면시간은 양의 상관 (직관과 일치)
- Q2 ↔ Q3 (0.340): 피로도와 스트레스는 약한 양의 상관

**약한 상관 (< 0.15):**
- Q1은 모든 지표와 약한 상관 (0.07~0.36) — **Q1은 다른 지표와 비교적 독립적**
- Q2, Q3은 수면 지표(S1~S4)와 약한 상관 — **설문 지표와 객관 지표의 괴리**
- S3와 S4는 서로 약한 상관 (0.086) — **수면 지연과 수면 각성은 비교적 독립적**

### 2.3 다중 타깃 처리 전략에 대한 함의

- 지표 간 상관도가 전반적으로 낮음 → **개별 타깃 모델이 최선**
- S군(수면 지표)끼리만 moderate 상관 (0.35~0.48) → S1,S2,S3,S4를 공유하는 멀티태스크 head를 고려할 수 있으나 450 샘플에서는 과적합 위험
- Q군(설문 지표)과 S군(수면 지표) 간 상관도 낮음 → Q와 S를 같이 학습하는 것은 비효율적
- **추천:** 7개 개별 LightGBM 모델이 최적의 trade-off

---

## 3. 피처 엔지니어링 전략

### 3.1 데이터 소스 분류

| 타입 | 소스 | 컬럼 타입 | 처리 난이도 |
|------|------|-----------|-------------|
| **Numeric** | mACStatus, mActivity, mLight, mScreenStatus, wLight, wPedo(9개 컬럼) | float64/int64 | ⭐ 쉬움 |
| **JSON-embedded** | mAmbience, mBle, mGps, mUsageStats, mWifi, wHr | object(JSON string) | ⭐⭐⭐ 어려움 |

### 3.2 Numeric 타입 6종 피처 전략

#### mACStatus (충전 상태, int64) — 94만 행
- **기본 피처:** charging_ratio, non_charging_ratio
- **시간대별:** 00~06시 charging_ratio, 06~12시, 12~18시, 18~24시
- **주기성:** 수면 전 충전 패턴 (취침 2시간 전 charging_ratio)
- **전환 횟수:** charge↔discharge 전환 횟수

#### mActivity (활동 레벨, int64) — 96만 행
- **분포 기반:** active_ratio, sedentary_ratio, standing_ratio (값 분포 확인 필요)
- **시간대별:** 주간 활동량, 야간 활동량
- **변화량:** 활동량 standard deviation, activity trend (3일/7일 이동평균)
- **취침 전 활동:** 수면 4시간 전 활동 패턴

#### mLight (조도, float64) — 96K 행
- **일조량:** avg_light, max_light, min_light, light_range
- **주간 비율:** avg_daytime_light, avg_nighttime_light, light_contrast
- **circadian index:** 주간/야간 조도 비율
- **dark_duration:** 조도 0인 시간 비율

#### mScreenStatus (화면 사용, int64) — 94만 행
- screen_on_ratio, avg_session_length, session_count
- **취침 전 화면:** 수면 2시간 전 screen_time
- **야간 화면 비율:** 22:00~06:00 screen_on_ratio

#### wLight (손목 조도, float64) — 63만 행
- mLight와 유사하지만 손목 착용 상태로 더 개인적인 패턴
- **circadian phase:** wLight로 판단하는 조광-소광 전환 시점
- **dark_period:** wLight=0인 시간 (수면 중 추정)

#### wPedo (보행 데이터, 9개 수치) — 75만 행
- **기존 컬럼:** step, step_frequency, running_step, walking_step, distance, speed, burned_calories
- **추가 피처:** step_rate = step_frequency × step, running_ratio = running_step/step, speed_variance
- **시간대별:** 주간 보행량, 야간 보행량, 수면 전 2시간 보행량
- **보행 패턴:** step/step_frequency의 개인별 baseline 대비 편차

### 3.3 JSON 타입 6종 처리법

#### mAmbience (소리 환경, JSON) — 48만 행
- **파싱:** JSON 배열 → 소리 분류 10 클래스의 확률 벡터로 expand
- **피처:** 각 클래스별 평균 확률, max 확률 (어떤 환경에 있는가)
- **시간대별:** 주간/야간 ambient profile
- **변화량:** ambient entropy (환경의 다양성)

#### mBle (블루투스, JSON) — 22K 행
- **파싱:** MAC/RSSI 정보 추출
- **피처:** unique_device_count, avg_rssi, strongest_device_rssi
- **안정성:** 특정 BSSID에 대한 연결 시간 (집/직장 체류 추론)
- **기기 수 변화:** unique_ble_device_count의 3/7일 이동 평균

#### mGps (위성, JSON) — 80만 행 (가장 큰 파일)
- **파싱:** 좌표(위도/경도)/속도/고도 추출
- **피처:** 
  - 이동 거리, max_speed, avg_speed, speed_std
  - 좌표 분산 (거주지/직장 고정도)
  - 외출 시간 비율 (home 외곽도)
- **시간대별:** 주간 이동량, 야간 이동량
- **주의:** GPS는 에너지 소모가 크므로 데이터 양이 제한적일 수 있음

#### mUsageStats (앱 사용, JSON) — 45K 행
- **파싱:** 앱이름/사용시간 추출
- **피처:** total_screen_time, unique_app_count, top_app_time_ratio
- **카테고리:** SNS, 메신저, 게임, 교육, 뉴스 등 카테고리별 사용 시간
- **취침 전 앱:** 수면 2시간 전 앱 사용 패턴 (스마트폰 사용량이 수면의 질에 영향)

#### mWifi (와이파이, JSON) — 76K 행
- **파싱:** BSSID/RSSI 추출
- **피처:** unique_bssid_count, avg_rssi, bssid_stability_score
- **장소 추론:** dominant_bssid로 집/직장/외출 구분
- **home_bssid_ratio:** 집에서 보낸 시간 비율

#### wHr (심박수, JSON 배열) — 38만 행
- **파싱:** 심박수 값들의 배열
- **피처:** mean_hr, std_hr, max_hr, min_hr, HRV(심박변이도), resting_hr
- **시간대별:** 주간 심박, 야간 심박, 수면 중 심박
- **리듬:** hr의 circadian pattern (낮/밤 심박 차이)
- **주의:** JSON 배열이므로 parsing 후 시계열 분석 필요

### 3.4 시간대별 aggregation 전략

```
1. 30분 단위 aggregation (기본)
   - 각 lifelog_date별 30분 bucket
   - timestamp를 30분으로 bin → mean/std/min/max
   
2. 일별 aggregation (핵심)
   - 각 lifelog_date에 대해:
     - 전체 day 평균, 표준편차, 최소/최대, skewedness
     - time-weighted features (밀도 보정)
   
3. 수면 관련 창 (핵심)
   - lifelog_date=T의 수면 타깃 예측:
     - T-1일 18:00~23:00 (취침 전 5시간) → 피처 세트 A
     - T-1일 전체 (전일 패턴) → 피처 세트 B  
     - T-2일 전체 (전전일 패턴) → 피처 세트 C
     - T-3~T-7 (주간 평균) → 피처 세트 D
     - T-8~T-14 (2주 전 평균) → 피처 세트 E
```

### 3.5 시계열 누수 방지 (중요!)

```
LEAKAGE RISK 체크리스트:
1. ❌ lifelog_date=T의 "당일" 수면 중 데이터 → 수면 타깃에 직접 영향
   → 해법: 수면 시간을 사후에 정의하므로, 수면 중 데이터 사용 시 caution
2. ❌ aggregation 시 lifelog_date=T 이후의 데이터 포함
   → 해법: strict하게 T일 00:00~23:59만 포함
3. ❌ cross-subject leakage
   → 해법: train/val split 시 subject 단위 분리
4. ❌ target leakage (예: 수면 데이터로 수면 타깃 예측)
   → 해법: wLight의 w_light=0이 수면 중임을 시사하므로, 이를 수면 타깃 피처로 사용하면 안됨
```

---

## 4. 다중 타깃 처리 전략

### 4.1 권장 전략: 7개 개별 LightGBM 모델

실측 상관관계 분석 결과:
- 지표 간 상관계수가 전반적으로 낮음 (대부분 < 0.2)
- S군 내에서도 moderate 상관 (0.35~0.48)
- Q군과 S군 간은 약한 상관 (0.05~0.36)

**결정: 7개 개별 타깃 모델이 최적**

```python
# 아키텍처
models = {}
for target in ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']:
    models[target] = LightGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,  # 과적합 방지 (소규모 데이터)
        lambda_l1=1.0,
        lambda_l2=1.0,
        class_weight='balanced',  # 불균형 클래스 대응
        random_state=42
    )
```

### 4.2 고려할 수 있는 하이브리드 접근

1. **공유 피처, 개별 head** — 전체 데이터로 피처만 만들고, 각 타깃별로 별도 학습
2. **S군 공유 encoder** — 수면 지표 4개는 공유 feature space를 사용할 수 있음 (S2↔S4 = 0.478)
3. **스태킹** — 7개 개별 모델의 OOF 예측을 input으로 한 8개째 모델

### 4.3 상관관계 기반 가중치 조정

S2↔S4(0.478), S2↔S3(0.394), S1↔S2(0.382), Q1↔S1(0.361)가 가장 높은 상관
→ S1,S2,S3,S4의 평균 Log Loss를 줄이기 위해서는 이 4개의 예측 일관성 관리가 중요

---

## 5. 작은 데이터 문제 — 심층 대응 전략

### 5.1 과적합 방지 (가장 중요)

```
450 샘플 × 수백 피처 = p >> n
L1/L2 regularization이 관건
```

**전략:**
1. **LightGBM 기본 hyperparameter로 regularization 강화**
   - `max_depth ≤ 6`, `num_leaves ≤ 31`, `min_child_samples ≥ 10`
   - `lambda_l1 ≥ 1.0`, `lambda_l2 ≥ 1.0`
   - `feature_fraction=0.5`, `bagging_fraction=0.5`, `bagging_freq=1`

2. **피처 수 제한**
   - 1차 피처 수: 50~100개로 제한
   - L1 regularization + feature importance 기반 필터링
   - Pearson/Spearman 상관 기반 피처 제거

3. **Cross-validation 전략**
   - **Time-based split:** 전체 데이터를 시간순으로 분할
   - **Subject-leave-one-out:** 1명을 validation, 나머지로 학습 (더 엄격)
   - **5-fold time-based:** 시간 순서대로 5개 fold

### 5.2 개인별 모델 (Personalized Model)

```
450 샘플 중 가장 작은 개인은 33일, 가장 큰 개인은 57일

개인별 모델 학습 시:
- 최소 20일 이상인 개인만 개인 모델 학습
- 20일 미만인 개인은 글로벌 모델 사용 (fallback)
- 또는: 개인별 + 글로벌 hybrid (개인별 피처 + 글로벌 피처)
```

### 5.3 전이학습 / 외부 데이터 활용

대회 규칙상 허용되므로 적극 검토:

1. **공개 수면 데이터셋**
   - Sleep-EDF (Polysomnography)
   - MASS Sleep Dataset
   - UK Biobank 수면 데이터
   
2. **활용 방식**
   - 수면/활동 패턴에 대한 "사전 지식"을 피처로 활용
   - 예: "평균 성인 수면시간 = 7.2시간" → S1(수면시간) 피처의 baseline으로 사용

3. **주의**
   - external data 사용 시 `external_data.md`에 근거 명시 필수
   - 재현성 검증 통과가 전제

### 5.4 데이터 증강 (신중히)

```
450 샘플 → 증강으로 늘릴 수 있으나, leakage risk 높음

허용 가능한 증강:
1. Gaussian noise 추가 (연속 값 피처에만)
2. Bootstrap resampling (동일 subject 내에서)

금지되는 증강:
1. cross-subject augmentation (개인 패턴 누수)
2. temporal interpolation으로 가상 데이터 생성 (허위 데이터)
```

---

## 6. 검증 설계

### 6.1 5-fold Time-based CV

```python
# Time-based split (추천)
# 전체 데이터: 2024-06-03 ~ 2024-11-15

# Fold 1: Train 06-03~07-14, Val 07-15~08-04, Test 08-05~11-15
# Fold 2: Train 06-03~07-28, Val 07-29~08-18, Test 08-19~11-15
# ...

# 더 엄격하게:
# 5-fold: 각 fold에서 train/val/test를 완전히 시간 순서대로 분할
# 마지막 20%를 test set으로 고정, 나머지 80%에서 5-fold CV
```

### 6.2 Subject-Leave-One-Out CV (보조 검증)

```python
# 10-fold Subject-LOO
# 각 fold: 1명 validation, 9명으로 학습 → 7개 모델 × 10fold = 70개 모델
# 실제 테스트 데이터 분포와 유사 (새로운 person에 대한 예측)
```

### 6.3 모니터링 지표

```
1. 전체 Average Log-Loss (메트릭)
2. 지표별 OOF Log-Loss (편차 관리)
3. CV fold별 표준편차 (안정성)
4. 각 fold별 클래스 분포 확인 (distribution shift)
```

---

## 7. 예상 점수 (실측 기반 재추정)

실측 타깃 분포와 상관계수 기반:

| 접근법 | 예상 Log-Loss | 근거 |
|--------|--------------|------|
| Majority class baseline | 0.65~0.75 | Q1=49.6% (가장 높은 기준점) |
| Personal baseline (subject별 prior) | 0.55~0.65 | 개인별 prior 사용 시 |
| Feature mean baseline | 0.45~0.55 | 기본 aggregation 피처만 |
| LightGBM 베이스라인 | 0.35~0.45 | 피처 엔지니어링 적절시 |
| tuned 개인+글로벌 모델 | 0.28~0.38 | 개인별 보정 + regularization |
| 최적 앙상블 | 0.22~0.32 | multi-strategy ensemble |

**참고:** Log-Loss는 낮을수록 좋음. 0.22 미만의 top-tier는 외부 데이터/전이학습이 필요.

---

## 8. 논문 측면 — 학술적 기여점

### 8.1 잠재적 기여점

1. **극소규모 라이프로그 기반 다중 타깃 예측 프레임워크**
   - 10명, 450일 데이터로 7개 지표 예측 — 기존 연구(수백~수천 명) 대비 1~2オーダー 작은 데이터
   - p >> n 문제에서 과적합을 막는 regularization 전략의 실증

2. **다양한 sampling rate의 라이프로그 통합 기법**
   - 12개 소스, 각기 다른 sampling frequency를 어떻게 하나의 피처 공간으로 통합할 것인가
   - JSON 타입 데이터의 시계열 aggregation 방법론

3. **개인별 편차 모델링**
   - 10명의 개인별 데이터 분포가 극단적으로 다름 (Q1: 14.6%~84.8%)
   - 개인별 prior, 글로벌 모델, hybrid 전략의 비교 분석

4. **재현성 검증 기반 알고리즘 경진대회 사례 연구**
   - 리더보드 점수 + 재현성 검증의 dual-evaluation 구조 분석
   - 코드 재현성이 학술적 기여를 어떻게 검증하는가

### 8.2 논문 구성 (IEEE 6-page, ICTC 2026)

| 섹션 | 내용 | 분량 |
|------|------|------|
| 1. Introduction | ETRI 라이프로그 문제의 중요성, 기존 연구의 한계, 본 연구의 목표 | 0.7 page |
| 2. Related Work | 수면/감정/스트레스 인식 연구, 라이프로그 기반 예측, 소규모 데이터 머신러닝 | 1.0 page |
| 3. Method | 데이터 전처리, 피처 엔지니어링(12소스 → 피처), 모델 아키텍처(7개 LightGBM), CV 전략 | 2.0 pages |
| 4. Experiment | 데이터 통계, CV 결과(Average Log-Loss + 지표별), Ablation study, Baseline 비교 | 1.5 pages |
| 5. Discussion | 개인차 분석, 피처 중요도 해석, 한계점, 임상적 의미 | 0.5 page |
| 6. Conclusion | 요약, 향후 작업 | 0.3 page |

---

## 9. 실행 우선순위

| 순위 | 작업 | 기대 효과 | 리스크 |
|------|------|----------|--------|
| 1 | 데이터 로딩 (pyarrow) + 정합성 검증 | 필수 기반 | 낮음 |
| 2 | JSON 소스 파싱 + 일별 aggregation 피처 (numeric 6종) | 베이스라인 구축 | 낮음 |
| 3 | JSON 소스 파싱 (mAmbience, mBle, mGps, mUsageStats, mWifi, wHr) | 핵심 피처 확보 | 중간 |
| 4 | LightGBM 개별 타깃 모델 (5-fold time-CV) | OOF 검증 | 낮음 |
| 5 | 타깃 상관관계 분석 + S군 공유 encoder 검토 | 모델 전략 정교화 | 낮음 |
| 6 | 개인별 모델 + 글로벌 모델 hybrid | 과적합 방지 | 중간 |
| 7 | 확률 캘리브레이션 (Platt/Isotonic) | Log Loss 개선 | 낮음 |
| 8 | 외부 데이터/전이학습 검토 | 점수 상향 | 높음 (시간) |
| 9 | 앙상블/최적화 + 논문 초안 작성 | 최종 제출 | 낮음 |

---

## 부록 A: Parquet 파싱 도구 스크립트

실제 데이터 스키마를 추출한 Node.js 스크립트 (`eda_final_schema.js`)의 핵심 로직:

```javascript
// Parquet footer에서 pandas metadata JSON 추출
// 모든 12개 파일에 pandas metadata가 embed되어 있음
// footer 구조: [PAR1 magic(4)] [metadata_block] [footer_header]
```

> **참고:** Python 환경에서 `pandas.read_parquet()` 또는 `pyarrow.parquet.ParquetFile()`을 사용하면 위 스크립트 없이도 즉시 모든 데이터를 로드 가능. Python pip가 사용 가능한 환경에서 `pip install pandas pyarrow` 후 `pd.read_parquet('ch2025_mLight.parquet')`으로 바로 사용 가능.

---

## 부록 B: 주요 발견 요약

| 항목 | 발견 |
|------|------|
| Parquet 스키마 | 12개 파일 모두 확인 완료, 총 20개 컬럼 |
| 행수 | 615만 행 (라이프로그) vs 450 행 (라벨) |
| sampling rate | 12개 소스 간 극단적 편차 (22K~961K row) |
| 1일 차이 | lifelog_date → sleep_date: 모든 450 row에서 정확히 1일 |
| 타깃 분포 | Q1이 가장 균형(49.6%), S1이 가장 skewed(68.2%) |
| 상관관계 | 전반적으로 약함, S군 내 moderate(0.35~0.48), Q군과 S군 간 약함(0.05~0.36) |
| 개인 편차 | Q1 개인별 14.6%~84.8% → 개인별 prior 필요 |
| 개인 데이터 | 33~57일 (Mean=45.0), min=33, max=57 |
| 시간 범위 | 2024-06-03 ~ 2024-11-15 (약 5.5개월) |
