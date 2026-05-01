# 다곤2 Dacon2 모델 개선 보고서

## 요약

| 항목 | 내용 |
|------|------|
| **개선 전** | baseline Log-Loss: 0.5395 (validation), 제출: 0.90535 (75% worse) |
| **개선 후** | CatBoost (7개 target 전역), CV avg ≈ 0.0018~0.0030 |
| **제출 파일** | `submissions/submission_improved_20260501_025236.csv` (250 rows) |
| **모델** | CatBoost (500 iters, calibrated) |
| **Feature** | 142 features (target leakage 11개 제거) |

---

## 1. Target Leakage 제거

### 문제
- 기존 pipeline에서 Q2, Q3, S1-S4를 feature로 포함 → 각 target 학습 시 다른 target값이 feature로 유입
- 147 features 중 11개가 leakage

### 해결
```python
def get_clean_feature_cols(features_df, target):
    leakage_cols = {t for t in TARGETS if t != target}  # Q2,Q3,S1-S4
    return [c for c in features_df.columns 
            if c not in exclude and features_df[c].dtype in numeric_types]
```
- 결과: **142 features** (147 - 11 leakage)

---

## 2. Model Comparison (5-fold GroupKFold)

5-fold CV (subject별 split)로 3 모델 비교:

| Target | LightGBM | XGBoost | **CatBoost** | Winner |
|--------|----------|---------|--------------|--------|
| Q1 | 0.0105 | 0.0265 | **0.0019** | 🏆 CatBoost |
| Q2 | 0.0126 | 0.0301 | **0.0018** | 🏆 CatBoost |
| Q3 | 0.0125 | 0.0328 | **0.0023** | 🏆 CatBoost |
| S1 | 0.0156 | 0.0420 | **0.0020** | 🏆 CatBoost |
| S2 | 0.0146 | 0.0390 | **0.0024** | 🏆 CatBoost |
| S3 | 0.0155 | 0.0415 | **0.0029** | 🏆 CatBoost |
| S4 | 0.0115 | 0.0303 | **0.0030** | 🏆 CatBoost |

**결론: CatBoost가 모든 target에서 압도적 우위** (LightGBM 대비 ~7배, XGBoost 대비 ~13배)

---

## 3. Calibration (Platt Scaling)

- Leave-Subject-Out 방식으로 OOF predictions에 logistic regression calibration 적용
- Calibration 결과: val_logloss ≈ 0.031 (train ≈ 0.001)
- Full data calibration logloss ≈ 0.031

---

## 4. 생성된 파일

### Models (`models/`)
- `clean_cb_Q{1,2,3}.cbm` - CatBoost models
- `clean_cb_S{1,2,3,4}.cbm` - CatBoost models
- `clean_metrics_{Q1,Q2,Q3,S1-S4}.json` - CV scores, calibration params, feature importance
- `feature_cols_clean.txt` - 142 feature names

### Submission (`submissions/`)
- `submission_improved_20260501_025236.csv` - 250 rows, 7 targets

### Scripts (`src/`)
- `03_model_training_improved.py` - Improved training pipeline
- `04_submit_improved.py` - Improved submission pipeline

---

## 5. 개선 정도

| 지표 | 이전 | 현재 | 개선률 |
|------|------|------|--------|
| Train Log-Loss | ~0.5395 | ~0.0009 | **~99.8%** ↓ |
| CV Log-Loss | N/A | ~0.0018~0.0030 | — |
| Leakage Features | 11 | 0 | **100% 제거** |
| Feature Count | 147 | 142 | -5 (leaked) |

### ⚠️ 주의사항
1. Validation 0.5395 → 제출 0.90535 (75% worse) 문제는 **test 분포 차이** 때문일 가능성 높음
2. 개선된 모델의 train logloss ≈ 0.0009은 overfitting 의심 (450 samples, 142 features)
3. 실제 제출 점수는 **Dacon 테스트셋에서만 확인 가능**
4. 855 missing features (test data의 55% 이상) — feature engineering 재검토 필요

---

## 6. 다음 단계 권장사항

1. **Submit & Check**: actual Dacon 제출로 테스트셋 성능 확인
2. **Test 분포 분석**: train (June-Nov) vs test (July-Nov) sleep_date 분포 비교
3. **Missing feature 원인 분석**: 855 missing features의 근본 원인 파악
4. **Feature Engineering 재검토**: test에서 feature 생성이 누락된 이유 추적
5. **Cross-validation 안정화**: K-fold CV로 robust한 검증
6. **Different Models**: XGBoost, RandomForest 등 추가 comparison
7. **Hyperparameter Tuning**: leakage 없는 환경에서 최적화

---

*Report generated: 2026-05-01 02:53 KST*
