# dacon2 베이스라인 파이프라인 설계

**작성일:** 2026-05-01  
**대회:** 제 5회 ETRI 휴먼이해 인공지능 논문경진대회 (dacon2)

---

## 1. 프로젝트 구조

```
dacon2/
├── data_raw/
│   ├── ch2025_data_items/          ← 12개 parquet 파일
│   ├── ch2026_metrics_train.csv    ← 학습 라벨 (450 row)
│   ├── ch2026_submission_sample.csv ← 제출 형식 (250 row)
│   └── ch2026_metrics_description.pdf
├── src/
│   ├── config.py                   ← 설정 (경로, 하이퍼파라미터, 타겟)
│   ├── 01_load_data.py             ← 12개 parquet + 라벨 로딩
│   ├── 02_feature_engineering.py   ← 특징 공학 (JSON 파싱 + aggregation)
│   ├── 03_model_training.py        ← LightGBM 학습 + CV
│   └── 04_submit.py                ← 제출 파일 생성 + 검증
├── data_processed/                 ← 파이프라인 중간 산출물
├── models/                         ← 학습된 모델 파일
├── submissions/                    ← 생성된 제출 파일
└── 11_development_design.md        ← 이 파일
```

---

## 2. 데이터 이해

### 2.1 라벨 (training CSV)

| 항목 | 값 |
|------|-----|
| 행 수 | 450 |
| 대상 | 10명 (id01~id10) × 평균 45일 |
| 타겟 | Q1(수면의질), Q2(피로도), Q3(스트레스), S1(수면시간), S2(수면효율), S3(수면지연), S4(수면각성) |
| positive 비율 | Q1=0.496, Q2=0.562, Q3=0.600, S1=0.682, S2=0.651, S3=0.662, S4=0.560 |
| 시간관계 | lifelog_date(당일) → sleep_date(다음날, +1일 차이) |

**핵심 규칙:** `sleep_date - lifelog_date = 1일` (모든 450 row에서 동일)

### 2.2 라이프로그 12개 파일

| 파일 | 행 수 | 주요 열 | 데이터 타입 |
|------|-------|---------|-------------|
| mACStatus | 940K | m_charging (0/1) | 정수 (시간당 평균 약 250 row) |
| mActivity | 961K | m_activity (레벨) | 정수 (시간당 평균 약 257 row) |
| mAmbience | 477K | m_ambience (JSON) | object — 10개 카테고리 × 확률 |
| mBle | 22K | m_ble (JSON) | object — MAC/RSSI/장비유형 |
| mGps | 801K | m_gps (JSON) | object — lat/lon/alt/speed |
| mLight | 96K | m_light (lux) | float64 |
| mScreenStatus | 940K | m_screen_use (0/1) | 정수 |
| mUsageStats | 45K | m_usage_stats (JSON) | object — app_name/total_time |
| mWifi | 76K | m_wifi (JSON) | object — BSSID/RSSI |
| wHr | 383K | heart_rate (배열) | object — 1~40개 심박수 값 |
| wLight | 634K | w_light (조도) | float64 |
| wPedo | 748K | step/breakdown | 정수 (7열: step, freq, run, walk, dist, speed, cal) |

### 2.3 테스트 데이터

| 항목 | 값 |
|------|-----|
| 행 수 | 250 |
| 대상 | 동일 10명 (subject당 19~32일) |
| 날짜 범위 | 2024-07-06 ~ 2024-11-19 |
| 제출 형식 | `subject_id, sleep_date, lifelog_date, Q1..Q4, S1..S4` (모든 0으로 초기화) |

---

## 3. 파이프라인 설계

### 3.1 `01_load_data.py` — 데이터 로딩

**역할:** 12개 parquet + 라벨 CSV를 로드하고 공통 키(date)를 생성

**핵심 로직:**
```
1. parquet_dir에서 PARQUET_FILES 매핑 따라 12개 파일 로딩
2. timestamp → date + hour + minute 분리 (병합용 키)
3. 라벨 CSV: sleep_date, lifelog_date 파싱
4. 반환: dict[str, DataFrame] (12개 parquet) + DataFrame (라벨)
```

**고려사항:**
- parquet 파일이 크지 않으므로 전체 메모리 로딩 가능
- JSON 열(mAmbience 등)은 raw 그대로 유지 (02에서 파싱)

### 3.2 `02_feature_engineering.py` — 특징 공학

**역할:** 라이프로그를 day-level 피처로 aggregation + 라벨 병합

#### 3.2.1 Numeric 열 aggregation (source별)

```
mACStatus:
  - charging_rate: charging==1 인 비율
  - charging_duration_sum: total charging time

mActivity:
  - activity_mean/std/min/max: 활동 레벨 통계
  - activity_distribution: 각 레벨(0=stationary,1=standing,2=walking,3=running,4=unknown) 비율

mLight:
  - light_mean/std/min/max: 조도 통계

mScreenStatus:
  - screen_on_ratio: screen_use==1 인 비율
  - screen_duration_sum: total screen time

wLight:
  - wlight_mean/std/min/max: 손목 조도

wPedo:
  - step/step_frequency/running_step/walking_step/distance/speed/burned_calories
  - 각 열의 mean, sum
```

#### 3.2.2 JSON 열 파싱 (source별)

| JSON 열 | 파싱 방식 | 추출 피처 |
|---------|-----------|-----------|
| mAmbience | category × prob → day별 확률 평균 | Speech, Music, Vehicle 등 10개 category별 avg prob + top5_sum |
| mBle | list of dict → list aggregation | count, device_count, avg_rssi, max_rssi, min_rssi, rssi_std |
| mGps | list of dict → list aggregation | avg_speed, max_speed, alt_range, has_speed |
| mUsageStats | list of dict → list aggregation | app_count, total_time, major_ratio, game_ratio |
| mWifi | list of dict → list aggregation | bssid_count, avg_rssi, max_rssi, strong_ratio(-60dBm 초과) |
| wHr | 배열 → 단일 값 | mean, std, min, max, median HR |

#### 3.2.3 시간대별 피처

```
mACStatus, mScreenStatus:
  - time_bin: night(0-6), morning(6-12), afternoon(12-18), evening(18-24)
  - hour_ratio: 각 time_bin에서의 사용 비율
```

#### 3.2.4 병합

```
labels (subject_id, lifelog_date, sleep_date, Q1..S4)
  ↓ merge on date=lifelog_date
features (date + day-level aggregated features)
  ↓
merged_df (450 rows, ~50+ features)
```

### 3.3 `03_model_training.py` — 모델 학습

#### 3.3.1 모델 아키텍처

```
7개 개별 LightGBM Binary Classifier
  ├─ Q1: 수면의질
  ├─ Q2: 피로도
  ├─ Q3: 스트레스
  ├─ S1: 수면시간
  ├─ S2: 수면효율
  ├─ S3: 수면지연
  └─ S4: 수면각성
```

#### 3.3.2 교차 검증 전략 (시간 기반)

```python
# 각 subject의 마지막 7일을 validation으로 사용
# → 시간 누수(time leakage) 완전 방지

for subject in subjects:
    sorted_dates = sort_by_date(subject_days)
    train = sorted_dates[:-val_days]   # 과거 days
    val   = sorted_dates[-val_days:]    # 최근 days (7일)
```

**대안:** GroupKFold (group=subject_id)도 가능하지만, 동일 subject 내 time leakage 위험 있음. **시간 기반 split을 우선.**

#### 3.3.3 LightGBM 하이퍼파라미터

```python
# 기본
num_leaves=63, max_depth=-1, learning_rate=0.05, n_estimators=1000
subsample=0.8, colsample_bytree=0.8
min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0
early_stopping_round=50

# scale_pos_weight: 타겟별 class ratio 동적 조정
# 예: Q1(0.496) → spw≈1.0, S1(0.682) → spw≈0.47
```

#### 3.3.4 검증 메트릭

```
Validation Log-Loss (각 타겟별)
+ Feature Importance (gain 기준)
+ Best Iteration
```

### 3.4 `04_submit.py` — 제출 파일 생성

**검증 체크리스트:**
1. ✅ 컬럼명/순서: `subject_id,sleep_date,lifelog_date,Q1,Q2,Q3,S1,S2,S3,S4`
2. ✅ 행 수: 250 row
3. ✅ 확률값 범위: [0, 1]
4. ✅ 누락값 없음: NaN 없음

---

## 4. 특징 공학 상세 설계

### 4.1 피처 카테고리

```
1. 직접 수치: mLight, wLight, mActivity, wPedo(7열)
2. Binary 비율: mACStatus(charging_rate), mScreenStatus(screen_ratio)
3. JSON 파싱: mAmbience(10cat), mBle(6통계), mGps(5통계),
              mUsageStats(4통계), mWifi(5통계), wHr(5통계)
4. 시간대: night/morning/afternoon/evening 비율
```

### 4.2 JSON 파싱 패턴 (실제 데이터 기반)

```python
# mAmbience: numpy.ndarray of numpy.ndarray
#   [[category_name, probability], ...]  (10개 카테고리)
# 예: [["Music", "0.309"], ["Vehicle", "0.082"], ...]

# mBle: numpy.ndarray of dict
#   [{"address": "xx:xx:xx...", "device_class": "0", "rssi": -82}, ...]

# mGps: numpy.ndarray of dict
#   [{"altitude": 110.6, "latitude": 0.208, "longitude": 0.170, "speed": 0.0}, ...]

# mUsageStats: numpy.ndarray of dict
#   [{"app_name": "NAVER", "total_time": 549}, ...]

# mWifi: numpy.ndarray of dict
#   [{"bssid": "xx:xx:xx...", "rssi": -78}, ...]

# wHr: numpy.ndarray (1차원 배열)
#   [134, 134, 135, 133, ...]  # 1~40개 심박수 값
```

### 4.3 시계열 누수 방지 전략

```
⚠️ 주의: aggregation 시 미래 데이터를 포함하면 안 됨

올바른 접근:
  - lifelog_date 기준 aggregation (당일 데이터만)
  - 예측: lifelog_date의 데이터 → sleep_date의 라벨
  - train/val split: 각 subject의 과거 → 최근 시간 순

잘못된 접근 (피해야 할 것):
  - sleep_date까지 포함하여 aggregation
  - subject 전체를 random shuffle
  - sliding window에서 미래 데이터 포함
```

---

## 5. 구현 순서

```
1. config.py 생성           ← 모든 경로/파라미터 중앙화
2. 01_load_data.py 작성      ← 데이터 로딩 테스트
3. 02_feature_engineering.py ← 특징 공학 (가장 많은 코드)
4. 03_model_training.py      ← LightGBM 학습
5. 04_submit.py              ← 제출 파일 생성
```

### 실행 순서

```bash
cd dacon2/src
python 01_load_data.py          # 로딩 테스트
python 02_feature_engineering.py  # 특징 생성 → data_processed/features.parquet
python 03_model_training.py     # 모델 학습 → models/lgbm_*.json
python 04_submit.py             # 제출 생성 → submissions/submission_*.csv
```

---

## 6. 고려사항 및 향후 개선

### 6.1 현재 설계의 한계
- **단일 day aggregation:** time-series 패턴(주간 패턴, 추세) 미반영
- **JSON 피처 단순 통계:** top category, trend 등 고차원 패턴 미활용
- **LightGBM만:** XGBoost, CatBoost, neural network 등 비교 없음

### 6.2 향후 개선 방향
1. **Weekly pattern features:** 요일별, 주별 패턴 추출
2. **Inter-subject features:** 동일 subject 간 시계열 관계 모델링
3. **Multi-task learning:** 7개 타겟 joint learning
4. **Feature selection:** feature importance 기반 선택
5. **Ensemble:** LightGBM + XGBoost + CatBoost stacking
6. **Prob calibration:** Platt scaling / isotonic regression

### 6.3 논문 작성 시 주목할 점
- ETRI 대회 특성상 **논문 + 재현성 검증**이 핵심
- 코드 및 모델 설명서 제출 필요 (논문 채택 시)
- 데이터 전처리 과정, 특징 공학 방법, 모델 아키텍처를 명확히 문서화

---

## 7. 파일별 책임 요약

| 파일 | 책임 | 주요 함수 |
|------|------|-----------|
| `config.py` | 설정 관리 | `DATA_DIR`, `TARGETS`, `LGBM_PARAMS` |
| `01_load_data.py` | 데이터 로딩 | `load_all_parquet()`, `load_labels()` |
| `02_feature_engineering.py` | 특징 공학 | `create_day_features()`, `save_features()` |
| `03_model_training.py` | 모델 학습 | `train_all_targets()`, `save_models()` |
| `04_submit.py` | 제출 생성 | `create_submission()`, `verify_submission_format()` |
