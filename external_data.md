# 외부 데이터 사용 근거 및 내역

## 대회 규칙상 허용 여부
- `01_rules_and_constraints.md` 명시: "외부 데이터 사용 가능"
- 사전학습 모델 사용 가능 (공식 공개 가중치 + 라이선스 조건 충족 시)

## 사용 외부 데이터셋

### 1. Sleep Health and Lifestyle Dataset (Kaggle)
- **소스:** https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
- **크기:** 400 rows × 13 columns
- **타입:** synthetic dataset (fictive persons)
- **열:** Person ID, Gender, Age, Occupation, Sleep Duration, Quality of Sleep (1-10), Physical Activity Level, Stress Level (1-10), BMI Category, Blood Pressure, Heart Rate, Daily Steps, Sleep Disorder
- **용도:** 수면 품질/스트레스/심박수/활동량 간 피처-타깃 상관관계 패턴 추출 → LGBM regularization guidance
- **라이선스:** Kaggle 공개 데이터셋 (저작자 명시 필요)

### 2. WESAD (Wearable Stress and Affect Detection) - UCI
- **소스:** https://archive-beta.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
- **크기:** 15 subjects, wrist + chest wearable sensor data
- **센서:** Blood Volume Pulse, ECG, Electrodermal Activity, EMG, Respiration, Body Temperature, 3-axis Acceleration
- **용도:** 웨어러블 생체신호(심박, EDA, 가속도) 패턴 → 라이프로그 피처 엔지니어링 방향성 추출
- **라이선스:** Academic use, CC BY-NC 4.0

### 3. Sleep-EDF (PhysioNet)
- **소스:** https://physionet.org/content/sleep-edf/1.0.0/
- **크기:** 80+ recordings, PSG (EEG, EOG, EMG, airflow, temperature)
- **용도:** 수면 단계 분류 패턴 → 수면 지표(S1-S4) 피처 설계 참고
- **라이선스:** CC BY 4.0

## 활용 방식
1. **피처 중요도 Prior:** 외부 데이터에서 계산한 피처-타깃 상관을 LGBM의 `feature_fraction`, `lambda_l1`, `lambda_l2` regularization guidance로 사용
2. **피처 엔지니어링 방향성:** WESAD의 심박/가속도 패턴 → wHr, wPedo 피처 설계 개선
3. **정규화 하이퍼파라미터 튜닝 guidance:** 외부 데이터에서 발견된 과적합 패턴으로 regularization strength 조정
4. **직접 학습에는 사용되지 않음:** 외부 데이터는 탐색/가이드 목적으로만 사용, 최종 모델 학습에는 dacon2 내부 데이터만 사용

## 재현성
- 모든 외부 데이터셋은 공개적으로 접근 가능
- 각 데이터셋의 원본 링크 및 라이선스 명시
- 외부 데이터 기반 피처 중요도 계산 스크립트는 `src/05_external_data_analysis.py`에 저장
