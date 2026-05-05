# V45 연구 — 점수 개선을 위한 방향 분석

## 📊 현재 상황

- **V10**: Avg Cal 0.6038 (현재 최선)
- **V43 재현**: Avg Cal 0.6059 (Δ: +0.0021)
- **평가 지표**: Binary LogLoss (낮을수록 좋음)

## 🔍 V10의 한계 분석

### 1. 피처 수: 10~20개만 선택
- V10은 top-10 또는 top-20만 사용 → **정보 손실**
- features.parquet에 148개 numeric column 있음
- sch_csm(1위): 770개 피처 사용 → **피처 부족 가능성 높음**

### 2. aggregation window 부족
- 02_feature_engineering.py의 `AGG_WINDOWS = [1, 3, 6, 12, 24]`가 config에 있지만,
  features.parquet에는 window별 피처가 **전부 없음**
- aggregation window가 다 적용된 parquet가 있는지 확인 필요

### 3. rolling/expanding/EMA feature 부족
- sch_csm의 피처 엔지니어링에 포함:
  - **시계열**: diff, pct_change, EMA, expanding mean
  - **패턴**: zero-crossing
  - **변동성**: IQR, MAD
- features.parquet에는 **전혀 없음**

### 4. date/period feature 부족
- **날짜 기반**: dayofweek, month, hour, is_weekend, date_diff
- features.parquet에 `date` column은 있지만, 실제 날짜 기반 피처는 없음

### 5. multi-window aggregation 누락
- 02_feature_engineering.py에 정의되어 있지만 features.parquet에 반영 안됨

### 6. rolling feature 누락
- window별 aggregation이 아닌, 시계열 rolling window 기반 feature

### 7. 모델: LGBM only
- sch_csm(1위): LightGBM + CatBoost ensemble (0.3:0.7)
- wanniboy(2위): LSTM-based
- 단머스(3위): LLM + TabPFN + LightGBM + XGBoost ensemble

### 8. per-subject 패턴 누락
- 각 subject의 **시간적 추세**(rolling mean/std over days) 반영 안됨
- day-level이 아닌 **event-level** aggregation 활용 안됨

## 💡 V45 개선 방향

### Priority 1: Rolling/Expanding Feature 추가
```
- activity_7d_rolling_mean/std (최근 7일 평균)
- usage_expanding_mean (누적 평균, 추세 반영)
- day_diff features (전날 대비 변화량)
- ema (지수이동평균)
```

### Priority 2: Date/Period Feature 추가
```
- dayofweek, is_weekend, is_holiday
- month, season
- date_diff_from_first (관측 시작일로부터 차이)
```

### Priority 3: Rolling Statistics (event-level)
```
- 각 subject별 rolling mean/std over time (1h, 3h, 6h windows)
- diff (전 이벤트 대비 변화)
- pct_change
- zero_crossing_rate
```

### Priority 4: CatBoost Ensemble
```
- sch_csm이 CatBoost로 큰 성능 향상 확인
- CatBoost categorical feature 잘 처리
- LightGBM + CatBoost ensemble
```

### Priority 5: Feature Count 증가
```
- top-20 → top-50, top-100으로 확대
- regularization 강화로 overfitting 방지
```

### Priority 6: External Data / Domain Knowledge
```
- 수면 관련 외부 지식 활용:
  - 취침 시간 패턴 (hour distribution)
  - 주말 vs 평일 차이
  - 활동량 vs 수면 효율 상관관계
- 계절성, 요일 효과
```

## 🧪 실험 계획

### V45a: Rolling + Date Feature
- features.parquet에 rolling(expanding), date features 추가
- 동일 LGBM pipeline, feature count 50
- 예상: Avg Cal 0.59~0.60

### V45b: V45a + 더 많은 feature (top-50)
- feature selection top-50
- expected: Avg Cal 0.585~0.595

### V45c: V45a + CatBoost Ensemble
- LGBM + CatBoost ensemble
- expected: Avg Cal 0.58~0.59

### V45d: Multi-window aggregation
- 02_feature_engineering.py의 window별 aggregation 적용
- expected: Avg Cal 0.57~0.58

### V45e: Full feature set + CatBoost + LGBM
- 모든 feature + ensemble
- expected: Avg Cal 0.56~0.58

## ⚠️ 주의사항

1. **메모리**: 16GB RAM + 4GB swap. rolling feature는 메모리 주의
2. **데이터 양**: 450 samples, 10 subjects. 과대적합 주의
3. **leakage**: rolling window가 미래 데이터를 포함하지 않도록 주의
4. **시간**: rolling feature 생성은 OOM 없이 해야 함
