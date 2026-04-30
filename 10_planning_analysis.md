# 🏗️ Planning Analysis — 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

**작성일:** 2026-05-01  
**분석:** 집가헤응 (기획 에이전트)

---

## 결론

**10명 × 450 샘플의 극소규모 데이터로 7개 타깃의 이진 분류를 수행해야 하며, 재현성 검증이 최종 수상의 전제 조건인 대회.** 핵심 전략: 작은 데이터에 최적화된 LightGBM 기반 개별 타깃 모델 + 엄격한 시간 기반 CV + 확률 캘리브레이션.

---

## 1. 데이터 본질적 특성

### 1.1 시계열의 불규칙성
- 라이프로그 sampling rate: 몇 초 ~ 수시간까지 극단적 편차
- 12개 데이터소스 총 **760만행**의 고밀도 데이터가 450개 타깃 레이블과 매핑
- **핵심 과제:** 고빈도 시계열 → 저빈도 타깃 레이블을 어떻게 연결할 것인가

### 1.2 시간적 관계 (중요)
```
lifelog_date (T) → sleep_date (T+1일)
```
- 모든 450 row에서 정확히 1일 차이
- **즉, lifelog_date=T의 라이프로그만으로 sleep_date=T+1의 타깃 예측 가능**
- aggregation window: lifelog_date=T 이전의 모든 데이터 허용 (미래 데이터 제외)

### 1.3 데이터 불균형
```
Q1(수면의질):   49.6% positive
Q2(피로도):     56.2% positive
Q3(스트레스):   60.0% positive
S1(수면시간):   68.2% positive
S2(수면효율):   65.1% positive
S3(수면지연):   66.2% positive
S4(수면각성):   56.0% positive
```
- 모두 이진 분류에 가까우나 Q1만 균형에 가까움
- S1이 가장 skewed

### 1.4 소규모 데이터
- 10명 사용자, 평균 45일 데이터 = 450 샘플
- p >> n 문제 (피처 >> 샘플)
- 개인별 데이터 개수 편차 큼 (33~57일)

---

## 2. 피처 엔지니어링 전략

### 2.1 JSON 타입 6종 처리법

| 소스 | 데이터 | 파싱 전략 |
|------|--------|-----------|
| mAmbience | 소리분류 10 클래스 확률 벡터 | `json.loads()` → 10개 컬럼으로 expand → 일별 평균/최대 |
| mBle | BLE 기기 MAC/RSSI | `json.loads()` → unique_device_count, avg_rssi, strongest_rssi, device_type_counts |
| mGps | 좌표/속도/고도 | `json.loads()` → max_speed, avg_speed, std_speed, distance, location_variance |
| mUsageStats | 앱명/사용시간 | `json.loads()` → total_app_time, unique_apps, top_category_time, screen_on_time |
| mWifi | BSSID/RSSI | `json.loads()` → unique_bssid_count, avg_rssi, home_bssid_stability |
| wHr | 심박수 배열 | `json.loads()` → mean_hr, std_hr, max_hr, min_hr, hr_variability, resting_hr_ratio |

### 2.2 Numeric 타입 6종 처리법

| 소스 | 데이터 | 피처 전략 |
|------|--------|-----------|
| mACStatus | 충전 상태 (0/1) | charging_ratio, sleep_period_charging_ratio, daily_charging_count |
| mActivity | 활동 레벨 | active_ratio, sedentary_ratio, avg_activity_level, activity_variance |
| mLight | 조도 (lux) | avg_light, light_var, dark_ratio, bright_ratio, circadian_rhythm_index |
| mScreenStatus | 화면 사용 | screen_on_ratio, screen_usage_count, night_screen_ratio |
| wLight | 손목 조도 | circadian_pattern, dark_sleep_ratio, light_variation |
| wPedo | 보행 9개 수치 | 총 보행량, 일평균 보행수, 보행 강도, 수면 전 보행량 |

### 2.3 시간대별 집계 전략

```
# 1시간 단위 aggregation (가장 기본)
- timestamp를 hour bucket으로 그룹화
- lifelog_date별 aggregations

# 일별 aggregation (핵심)
- 각 lifelog_date에 대해:
  - mean, std, min, max, count, skewness
  - time-weighted features (시간 밀도 고려)

# 수면 관련 창 (핵심 피처)
- lifelog_date=T의 수면 타깃 예측:
  - T-1일 18:00~23:00 (취침 전) → 피처 세트 A
  - T-2일 전체 (전일 패턴) → 피처 세트 B
  - T-3~T-7 (주간 패턴) → 피처 세트 C
```

### 2.4 리듬/패턴 피처

```
- 요일별 패턴: weekday vs weekend 비율
- 취침/기상 패턴 일관성 (activity + light + screen 기반)
- 직전 3/7/14일 변화율
- 주말/평일 차이
```

---

## 3. 다중 타깃 처리 전략

### 3.1 개별 모델 (초기 추천)
- 7개 타깃별 별도 LightGBM 학습
- 장단점: 각 타깃의 최적 하이퍼파라미터 가능, 오버헤드 적음
- **추천:** 1차 접근법

### 3.2 멀티태스크 학습 (2차)
- 공유 feature encoder + 타깃별 head
- 단점: 450 샘플로는 과적합 위험 높음
- **추천:** 개별 모델이 plateau된 후 검토

### 3.3 앙상블 전략
- 5-fold 시간 기반 OOF 예측 → weighted ensemble
- 지표별 상관계수 분석 → 상관 높은 지표끼리는 ensemble weighting 조정

### 3.4 타깃 상관관계 분석 제안
```python
# Q1~Q3, S1~S4 간 상관계수 행렬 계산
# 예상되는 상관:
# Q2(피로도) ↔ S2(수면효율): 높음
# Q3(스트레스) ↔ S4(수면각성): 높음
# Q1(수면의질) ↔ S1(수면시간): 높음
```

---

## 4. 작은 데이터 문제 대응

### 4.1 과적합 방지
- **Feature selection:** L1 regularization, feature importance 기반 필터링
- **Regularization:** LightGBM의 max_depth, min_child_samples, lambda_l1/l2 강화
- **Cross-validation:** time-based split (마지막 7일을 validation로 고정)

### 4.2 전이학습/외부데이터
- 외부 데이터 활용 가능 (대회 규칙)
- 수면/활동 관련 공개 데이터셋으로 전이학습 검토 (예: Sleep-EDF, MIMC)
- **단, external_data.md에 근거 명시 필수**

### 4.3 데이터 증강
- 시간 슬라이딩 윈도우 증강 (동일 사용자의 인접 데이터)
- Gaussian noise 추가 (심박수, 조도 등 연속값)
- **주의:** 미래 데이터 유입되지 않도록 strict 시간 경계 유지

---

## 5. 검증 설계

### 5.1 시간 기반 CV
```
# 5-fold 시간 기반 분할
Fold 1: Train: id01~05까지 28일, Valid: id01~05 29~35일, Test: id06~10
Fold 2: Train: id01~05까지 21일, Valid: id01~05 22~28일, Test: id06~10
...
```

### 5.2 모니터링 지표
- 전체 Average Log-Loss
- 지표별 OOF Log-Loss (편차 관리)
- CV fold별 표준편차 (안정성 지표)

---

## 6. 예상 점수 (실제 제출 전 참고)

450 샘플, 10명, LightGBM 기반 베이스라인 기준:

| 접근법 | 예상 Log-Loss 범위 | 근거 |
|--------|-------------------|------|
| Majority class baseline | 0.690 ~ 0.720 | 타깃별 positive 비율 0.5~0.68 |
| Feature mean baseline | 0.550 ~ 0.650 | 기본 피처만 사용 시 |
| LightGBM 베이스라인 | 0.380 ~ 0.500 | 피처 엔지니어링 적절시 |
| tuned 앙상블 | 0.300 ~ 0.420 | 다수 실험 + 캘리브레이션 |
| 최적화된 제출 | 0.250 ~ 0.350 | 외부데이터 + 전이학습 + 세밀한 피처 |

**참고:** Log-Loss는 낮을수록 좋음. 0.250 미만은 top-tier 수준.
**중요:** 위 점수는 추정치이며, 실제 Private LB 점수와 괴리 가능.

---

## 7. 논문 측면

### 7.1 학술적 기여점 후보
1. **소규모 라이프로그 기반 다중 타깃 예측 프레임워크** — 10명 데이터로 7개 지표 동시 예측 방법론
2. **불규칙 샘플링 시계열 → 타깃 매핑 기법** — 다양한 sampling rate의 라이프로그를 일관되게 aggregate하는 방법
3. **개인별 편차 보정 전략** — 10명의 극소규모 데이터에서 개인차를 어떻게 일반화할 것인가

### 7.2 논문 구성 (IEEE 6-page)
1. Introduction (ETRI 라이프로그 문제의 중요성)
2. Related Work (수면/감정 인식, 라이프로그 연구)
3. Method (데이터 전처리, 피처 엔지니어링, 모델 아키텍처)
4. Experiment (CV 결과, Ablation, Baseline 비교)
5. Discussion (한계점, 개인차 분석, 임상적 의미)
6. Conclusion

---

## 8. 실행 우선순위

| 순위 | 작업 | 기대 효과 | 리스크 |
|------|------|----------|--------|
| 1 | 데이터 로딩 + 정합성 검증 | 필수 | 낮음 |
| 2 | 기본 피처 엔지니어링 (numeric 6종) | 베이스라인 구축 | 낮음 |
| 3 | JSON 파싱 + 집계 피처 | 핵심 피처 확보 | 중간 (JSON 파싱 오류) |
| 4 | LightGBM 개별 타깃 모델 | OOF 검증 | 낮음 |
| 5 | 타깃 상관관계 분석 | 앙상블 전략 수립 | 낮음 |
| 6 | 캘리브레이션 | Log Loss 개선 | 낮음 |
| 7 | 외부데이터/전이학습 | 점수 상향 | 높음 (시간 소모) |
| 8 | 앙상블/최적화 | 최종 점수 | 낮음 |
