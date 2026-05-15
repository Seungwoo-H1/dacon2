# Dacon2 Modeling Pipeline — Security/Quality/Performance Audit

**작성일:** 2026-05-01  
**분석 대상:** 전체 modeling pipeline (`src/01_` ~ `05_`, `config.py`)

---

## 1. Data Leakage Audit

### 🔴 CRITICAL: Sleep-period time leakage (S1-S4)

**발견:** `lifelog_date` 기반 aggregation window (00:00–23:59)가 `sleep_date`의 수면 기간(당일 밤 ~次日 아침)과 **22:00–23:59 구간에서 중첩**됩니다.

```
lifelog_date=2024-06-26 → aggregation: 2024-06-26 00:00 ~ 23:59
sleep_date=2024-06-27  → 수면 기간: 2024-06-26 22:00 ~ 2024-06-27 07:00
중첩: 2024-06-26 22:00 ~ 23:59 ← 이 시간이 S1-S4 예측에 사용됨
```

**누출 가능성 있는 features:**

| Feature Group | Example | Risk Level |
|---|---|---|
| `mScreenStatus_hour_night` | 밤 화면 사용 비율 (23:00 포함) | 🔴 High |
| `mACStatus_hour_night` | 밤 충전 비율 (23:00 포함) | 🟡 Medium |
| `wLight_w_light_mean` | 야간 조명 (00:00-06:00 포함 가능) | 🟡 Medium |
| `mActivity_*_mean` | 야간 활동량 | 🟡 Medium |
| `mLight_*` | 조도 (야간 실내등 누출 가능) | 🟡 Medium |

> **해결:** `06_improved_modeling.py` 작성 시, aggregation을 **06:00–21:59 (daytime)** 만으로 제한하거나, time-window를 lifelog_date 18:00까지로 자름.

### 🟢 OK: Target 간 leakage 없음

- 모든 파일(`03_`, `04_`, `05_`)에서 `TARGETS` 전체를 feature 제외 (`meta_cols`)
- S1-S4가 feature로 포함되지 않음 확인

### 🟢 OK: Sleep-date 직접 데이터 미포함

- features는 `lifelog_date`에서 파생, `sleep_date` 직접 참조 없음
- sleep_date = lifelog_date + 1일 (검증 완료)

---

## 2. Code Quality Audit

### 🔴 CRITICAL: CV split inefficiency (O(n²))

**파일:** `03_model_training.py` line 45–63

```python
for idx, row in df.iterrows():  # 450회 반복
    subject_rows = df[df['subject_id'] == sid].sort_values('lifelog_date')  # 매번 정렬!
```

- 450 sample × 10 subjects × 매번 sort → **O(n²)** 연산
- **해결:** pre-compute `val_cutoff` per subject → vectorized assign

### 🟡 MEDIUM: `_parse_json_column` dead code

**파일:** `01_load_data.py` line 34–40

- `_parse_json_column()` 함수가 **호출되지 않음** (전체 코드에서 grep 결과: 정의만 존재)
- JSON 컬럼 파싱은 `02_feature_engineering.py`에서 별도 처리 (의도적)
- **해결:** dead code 제거 또는 docstring에 의도 명시

### 🟡 MEDIUM: JSON parser code duplication

**파일:** `02_feature_engineering.py`

- `parse_ambience()` 함수와 `_amb_group()` 그룹 함수가 **동일 로직을 2번 구현**
- `parse_usage_stats()` 카테고리 매핑도 `parse_ambience` 스타일과 일관성 없음
- **해결:** 단일 함수로 통합

### 🟡 MEDIUM: `04_submit.py`의 `load_feature_columns()` inconsistency

**파일:** `04_submit.py` line 95–104

```python
# 이 함수는 meta_cols에 TARGETS를 포함하지 않음!
feat_cols = [c for c in df.columns if c not in meta_cols + TARGETS]
# TARGETS를 추가했지만 meta_cols에 lifelog_date 등 누락 가능
```

- `get_train_feature_cols_for_target(target)`는 TARGETS 전체를 제외 (올바름)
- `load_feature_columns()`는 호출되지 않는 것으로 보이나, consistency 유지 필요

### 🟡 MEDIUM: wHr parsing — `iterrows()` 성능

**파일:** `02_feature_engineering.py` line 206–217

- `wHr` 382,918 rows에 `iterrows()` → **매우 느림**
- **해결:** `apply()` + `np.array()` 또는 `pd.Series.explode()` 활용

---

## 3. Performance Bottlenecks

### 🔴 CRITICAL: `iterrows()` on 382k rows

```python
# 02_feature_engineering.py
for _, row in df.iterrows():  # wHr: 382,918 iterations
    hr_vals = [float(v) for v in row[json_col] if v is not None]
```

- 권장: `df[json_col].apply(lambda x: ...)` or vectorized parsing
- 예상 시간: iterrows() → 수 분, apply() → 수 초

### 🟡 MEDIUM: 반복 JSON aggregation

각 JSON column(5개) × stat columns(4~8개) = **20+개 groupby 연산**이 별도 DataFrame으로 생성 → merge
- **해결:** `agg({'col1': 'mean', 'col2': 'std'})` 한 번에 처리

### 🟡 MEDIUM: 중복 merge + drop_duplicates

```python
def _merge_feat(left, right):
    merged = left.merge(right, on=["subject_id", "date"], how="outer")
    return merged.drop_duplicates(["subject_id", "date"])
```

- 4 단계 × 매번 outer merge + dedup = cumulative overhead
- **해결:** 모든 feature를 별도 DataFrame에 쌓은 후 **1회 merge**

### 🟢 OK: 450 samples × 141 features × 7 targets × 20 CV runs

- 전체 연산량: ~558,000 training iterations
- LightGBM으로 1회당 수 초 → 전체 30분~1시간 예상 (실제 05_v5_robust.py처럼 parallelization 필요)

---

## 4. Reproducibility

### 🟡 MEDIUM: Seed 고정 (partial)

| 파일 | `random_state` 설정 | `np.random.seed` |
|---|---|---|
| `03_model_training.py` | ✅ LightGBM `random_state=RANDOM_SEED` | ❌ |
| `03_model_training_improved.py` | ✅ `random_state=RANDOM_SEED` | ❌ |
| `03_model_training_v2.py` | ✅ `random_state=RANDOM_SEED` | ❌ |
| `05_v5_robust.py` | ✅ `RANDOM_SEED = 42` | ❌ |
| `config.py` | ✅ `RANDOM_SEED = 42` | — |

**문제:** `np.random.seed()` 호출 없음 → sklearn의 내부 랜덤(예: `scale_pos_weight`의 float division, DataFrame sampling)이 비결정적일 수 있음

**권장:**
```python
np.random.seed(42)
import random
random.seed(42)
```
파이프라인 시작 시 전역 시드 고정

### 🔴 CRITICAL: 환경 버전 관리 없음

- `requirements.txt`, `pyproject.toml`, `environment.yml` **모두 미존재**
- LightGBM, XGBoost, CatBoost, scikit-learn, pandas 버전이 실행 환경에 따라 달라질 수 있음
- 논문의 재현성 검증에 치명적

**해결:** `pip freeze > requirements.txt` 실행

### 🟢 OK: 데이터 경로

- `config.py`에 `PROJECT_ROOT`, `DATA_RAW`, `DATA_DIR` 등 모든 경로 중앙화
- 절대 경로 의존 없음 ✅

---

## 5. Metric Alignment

### ✅ Average Log-Loss 계산 — 일치

```python
# sklearn log_loss with binary classification
log_loss(y, p, labels=[0, 1])
```

- 검증: manual 계산 `(1/N) Σ[-y·log(p) - (1-y)·log(1-p)]` = sklearn 결과 ✓
- imbalanced binary에서도 동일 (sklearn의 기본 label_weight가 binary에 대해 unweighted average와 동일)
- Dacon의 Average Log-Loss (per-target log_loss 평균)와 호환 ✅

### 🟡 MEDIUM: Probability clipping

| 파일 | clipping 여부 | 범위 |
|---|---|---|
| `04_submit.py` | ❌ 없음 | raw predict |
| `04_submit_clean.py` | ❌ 없음 | raw predict |
| `04_submit_final.py` | ✅ 있음 | clip |
| `04_submit_improved.py` | ✅ 있음 | clip |
| `05_v5_robust.py` | ✅ 있음 | `np.clip(0.0001, 0.9999)` |

- LightGBM `predict()` for binary는 sigmoid → 이론상 (0, 1)
- 하지만 extreme case에서 `log_loss` 발산 위험
- **해결:** submission 시 `np.clip(pred, 1e-15, 1-1e-15)` 필수

---

## 6. Summary & Priority Fixes

| Priority | Issue | Severity | Files Affected |
|---|---|---|---|
| **P0** | Sleep-period time leakage (22:00-23:59) | 🔴 Critical | `02_feature_engineering.py` aggregation window |
| **P0** | 환경 버전 관리 부재 | 🔴 Critical | 프로젝트 루트 |
| **P1** | CV split O(n²) inefficiency | 🟡 High | `03_model_training.py` |
| **P1** | wHr parsing iterrows() | 🟡 High | `02_feature_engineering.py` |
| **P1** | Global seed 미설정 | 🟡 Medium | 전체 training files |
| **P2** | Probability clipping 누락 | 🟡 Medium | `04_submit.py` (old) |
| **P2** | JSON parser code duplication | 🟡 Medium | `02_feature_engineering.py` |
| **P2** | Dead code `_parse_json_column` | 🟢 Low | `01_load_data.py` |
| **P3** | Merge inefficiency | 🟢 Low | `02_feature_engineering.py` |

---

## 7. Recommendations for `06_improved_modeling.py`

1. **Time-window 제한:** aggregation을 `lifelog_date 06:00~18:00` (daytime only)으로 제한
   - 수면 기간(22:00~07:00)의 sensor data를 feature에서 제거
   - `mScreenStatus_hour_night`, `mACStatus_hour_night` 제외
2. **CV split vectorization:** pre-compute subject별 cutoff 후 `apply`
3. **Global seed:** `np.random.seed(42)` + `random.seed(42)` 추가
4. **Clip predictions:** `np.clip(pred, 1e-15, 1-1e-15)`
5. **wHr vectorization:** `iterrows()` → `apply()` + `np.mean()`
6. **requirements.txt** 생성 (pip freeze)
7. **JSON aggregation:** 20+ groupby → `agg()` 한 번에
