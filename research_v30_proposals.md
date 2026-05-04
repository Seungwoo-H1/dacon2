# Dacon2 V30 — 연구 제안서

**작성일:** 2026-05-03  
**현재 최고:** V10 cal OOF = 0.6038  
**V25 진행 중:** Cal OOF = 0.5778 (승우 확인)

---

## 0. GPU 환경 분석

### 현재 GPU 상태
- **RTX 4060 Laptop**, VRAM 8GB, CUDA 12.4
- Driver: 581.83 (WSL2)
- **GPU는 idle** (V25가 CPU로만 돌고 있음)

### GPU 가능 여부 (Framework별)

| Framework | GPU 지원 | WSL2에서의 동작 | 권장 여부 |
|-----------|---------|----------------|----------|
| **XGBoost 3.2.0** | `hist` + `predictor='gpu_predictor'` | **가능** (CUDA runtime 직접 사용) | ✅ 추천 |
| LightGBM 4.3.0 | `device='gpu'` (OpenCL) | **불가** (WSL2 OpenCL 불가) | ❌ |
| CatBoost 1.2.10 | `task_type='GPU'` | 가능 (CUDA 사용) | ✅ 대안 |
| PyTorch 2.6.0+cu124 | Native | 가능 | DL 모델용 |

### GPU 사용 권장 전략
1. **XGBoost `hist` + `gpu_predictor`**가 WSL2에서 가장 확실함
2. V30에서 XGBoost GPU를 기본 tree 모델로 사용
3. LGBM도 계속 사용하되 `n_jobs` 증가로 CPU 병렬화 강화

---

## 1. 피처 엔지니어링 개선 (High Impact)

### 1.1 ✅ 이미 V25에서 일부 적용: Rolling Window
V25는 `rolling(window=[3,7])` mean/std을 적용 중. 하지만:
- **Window 부족**: V25는 3,7만. 1,2,5,10,14,21도 고려
- **std만**: rolling min/max, median, skewness도 추가 가능

### 1.2 외부 데이터 통합 (Weather + Calendar)

**feature_study_results.md**에서 가장 강력한 신호가 확인됨:
- `week_of_year`과 S3 상관관계: **0.31** (가장 강한 single predictor!)
- 월별 패턴: S3가 6월 87.9% → 11월 28.6% (3배 차이)

#### 1.2.1 기상 데이터 (Daejeon KMA / Meteostat)
```
- 기온 (max/min/avg), 습도, 강수량, 일조량
- Daejeon 기상청 또는 Meteostat API (무료)
- 예상 효과: -0.02 ~ -0.05 log-loss
```

#### 1.2.2 휴일/계절 이벤트
```
- 한국 공휴일 (holidays.KR)
- 등하교 시즌, 시험기간
- 음력 명절 (설날, 추석)
- 예상 효과: -0.01 ~ -0.03 log-loss
```

#### 1.2.3 시간/일조량 (역산 가능)
```
- 일출/일몰 시간 (sunrise-sunset calculator)
- 일사량 추정 (위도/위성 데이터 없이 역산)
- 예상 효과: -0.01 ~ -0.02 log-loss
```

### 1.3 리듬/정규성 피처

```
- activity_regularity: 주별 활동 패턴 consistency (std of daily activity)
- circadian_consistency: 하루 중 활동 시간의 표준편차
- sleep_rhythm: wLight/wHr 패턴의 주별 일관성
```

---

## 2. 모델링 개선 (Medium Impact)

### 2.1 ✅ XGBoost GPU 병렬 사용

**문제:** 현재 V10, V25는 LGBM만 사용. XGBoost를 병렬로 사용하면:
- 서로 다른 tree 구조로 다양한 예측 → ensemble 효과
- GPU로 속도와 메모리 효율성 개선

```python
params_xgb = {
    'tree_method': 'hist',
    'predictor': 'gpu_predictor',  # GPU 사용
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.03,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
}
```

### 2.2 Ensemble 전략 고도화

현재 V25: 20 seeds ensemble (동일 모델)
→ **다양한 모델 ensemble**으로 전환:

| Ensemble | Weight | 설명 |
|----------|--------|------|
| LGBM 20 seeds | 40% | 기존 LGBM |
| XGBoost GPU 20 seeds | 40% | 새로운 XGB GPU |
| CatBoost GPU 10 seeds | 20% | 추가 다양성 |

### 2.3 Calibrator 개선

현재: `simple_mean_match` (단순 mean-matching)
→ **isotonic regression** 시도 (V9에서 실패한 이유 재확인 후):
- V9가 실패한 이유: training rate와 prediction mean 차이가 너무 커서
- 해결책: **stratified calibration** (per-subject 또는 per-week)

### 2.4 Per-Subject Models

450 샘플이지만 10명의 subject가 있음. 각 subject당 약 45 샘플.
- subject별 모델은 데이터 부족으로 과대적합 위험
- **hybrid 접근**: global model + subject-level bias term

---

## 3. 검증 방법론 개선

### 3.1 시간 기반 검증 (Time-Series CV)

현재: GroupKFold (무작위 split)
→ **시간 순서 고려**:
```python
# 더 현실적인 검증: 과거로 테스트, 미래로 검증
for train_end in pd.date_range('2024-06-01', '2024-10-01', freq='M'):
    train = df[df['lifelog_date'] < train_end]
    val = df[(df['lifelog_date'] >= train_end) & (df['lifelog_date'] < train_end + 30)]
```

### 3.2 Per-Subject Holdout

현재: GroupKFold는 subject 간 분할.
→ **완전 분리 검증**: 8명 학습, 2명 검증 (repeat 5-fold)

### 3.3 Bootstrap Confidence Interval

```python
# 1000 bootstrap resampling으로 prediction confidence 계산
# outlier detection 및 prediction quality scoring
```

---

## 4. 실험 로드맵 (V30)

### Phase 1: Quick Win (1-2일)
1. ✅ **XGBoost GPU ensemble** 추가 (`hist` + `gpu_predictor`)
2. ✅ **rolling window 확대** (1,2,3,5,7,10,14,21)
3. ✅ **temporal cyclical feature** 추가 (이미 V25에 일부 있으나 강화)
4. **기대 효과:** -0.01 ~ -0.03 log-loss

### Phase 2: External Data (2-3일)
1. ☐ **Meteostat weather data** 다운로드 (Daejeon)
2. ☐ **Korean holidays** integration
3. ☐ **Daylight hours** calculation
4. **기대 효과:** -0.02 ~ -0.05 log-loss

### Phase 3: Advanced Modeling (3-5일)
1. ☐ **CatBoost GPU** ensemble 추가
2. ☐ **Per-subject bias term** hybrid 모델
3. ☐ **Stratified calibration** 재시도
4. **기대 효과:** -0.01 ~ -0.03 log-loss

### 총 예상 효과: -0.04 ~ -0.11 log-loss
→ V10 (0.6038) → 0.56 ~ 0.57 가능

---

## 5. GPU 설정 가이드

### XGBoost GPU (WSL2 호환)
```python
params = {
    'tree_method': 'hist',
    'predictor': 'gpu_predictor',  # ← 핵심!
    'gpu_id': 0,
    ...
}
```

### LGBM GPU (불가 확인)
```
Error: No OpenCL device found
```
WSL2의 OpenCL 미지원으로 불가. 대체:
- CPU `n_jobs` 증가 (현재 -1, 더 많은 코어 사용)
- 또는 Docker Desktop GPU 사용 (복잡함)

### CatBoost GPU (대안)
```python
params = {
    'task_type': 'GPU',
    'devices': '0:1',  # 1개 GPU 사용
    'loss_function': 'Logloss',
    ...
}
```

---

## 6. 현재 V25와 비교

| 항목 | V25 (진행중) | V30 (제안) |
|------|-------------|-----------|
| Rolling window | 3, 7 | 1,2,3,5,7,10,14,21 |
| Personalization | ✅ | ✅ |
| Seasonal | ✅ (doy_sin/cos, is_weekend, month) | ✅ + holiday, weather |
| External data | ❌ | ✅ (weather, holidays) |
| Models | LGBM only | LGBM + XGB GPU + CatBoost GPU |
| Seeds | 20 | 20 × 3 모델 = 60 |
| Calibration | simple_mean_match | simple + stratified isotonic |
| GPU | ❌ (CPU only) | ✅ (XGB, CatBoost) |
