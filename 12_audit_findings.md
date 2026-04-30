# 🔍 Audit Findings — 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

> 작성일: 2026-05-01
> 대상: `dacon2/` 프로젝트 전체
> 근거 파일: `00_overview.md`, `01_rules_and_constraints.md`, `02_metric_formula.md`, `03_submission_spec.md`, `04_data_inventory.md`, `05_strategy_plan.md`

---

## Executive Summary

| Severity | Count | Items |
|----------|-------|-------|
| 🔴 Critical | 4 | 데이터 누수 구조적 리스크, 재현성 검증 실패 = 탈락, 제출 마감 병목, 확률 무한대 Loss |
| 🟡 High | 5 | 과적합(p >> n), 개인별 편차, Public shake-up, CV 안정성 부족, 1일 3회 제출 제한 |
| 🟠 Medium | 4 | 논문/코드 제출의 순차적 의존성, 데이터 소스 불균형, 개인정보 윤리, 제출 파일 검증 |
| 🟢 Low | 3 | 코드 확장자 제한, 팀 구성 제약, 실행 환경 격리 |

**핵심 결론:** 이 대회는 "리더보드 점수 ≠ 수상"이 핵심 운영 규칙. 재현성 검증 통과가 최종 수상의 절대적 전제 조건. 모든 전략은 재현성을 최우선으로 설계해야 함.

---

## 1. 데이터 누수 (Data Leakage) — 🔴 Critical

### 1.1 sampling_rate 불규칙성 → 시간 유출

**리스크:**
- 라이프로그 데이터의 sampling rate이 몇 초~수시간까지 극단적 편차 존재
- `lifelog_date` 단위의 aggregation window 설계 시, **해당 날짜의 미래 시간대 데이터가 자연스럽게 유입**될 수 있음
- 수면/감정/스트레스 지표는 특정 수면 주기(취침~기상)에 연관되므로, aggregation이 취침 시작 이후의 라이프로그를 포함하면 **미래 정보가 레이블에 누수**됨

**구체적 시나리오:**
```
예: 수면 지표 S1(총 수면시간)이 2025-03-15의 수면 패턴을 예측한다고 가정

❌ bad aggregation:
  - lifelog_date = '2025-03-15'인 모든 라이프로그 포함
  - 2025-03-15 22:00~06:00 (취침 중)의 activity/screen 데이터가 S1 예측에 사용됨
  → 수면 중 데이터가 수면 지표 예측에 직접 사용됨 = 명백한 누수

✅ correct approach:
  - 수면 지표 예측에는 수면 시작 시간 이전의 data만 사용
  - strict time boundary: 수면 시작时刻 이전 N일까지만 aggregation
```

**mitigation 전략:**
1. 각 사용자별 수면/설문 시점(날짜)을 명시적으로 파악하여, 해당 시점 **이전까지만**의 라이프로그를 feature window로 제한
2. aggregation 함수(`mean`, `std`, `max`, `min`)를 계산할 때 `timestamp < target_date` 조건을 엄격히 enforcing
3. Time-based CV split: 훈련 세트의 모든 data가 검증 세트의 타깃보다 과거인지 검증 로직 삽입
4. `check_leakage()` 유틸리티 함수를 CI 파이프라인에 통합 — 각 feature가 target보다 미래 data를 포함하는지 스캔

**검증 테스트:**
```python
# 누수 체크: 각 feature의 max_timestamp < target_date 인지 확인
for user_id in users:
    for target_date in target_dates:
        feature_max_ts = train_data[
            (train_data['user_id'] == user_id) & 
            (train_data['feature_name'] == fname)
        ]['timestamp'].max()
        assert feature_max_ts < target_date, f"Leakage detected in {fname}"
```

### 1.2 사용자 간 공유 데이터 포인트

**리스크:**
- 10명이라는 극히 소규모 데이터셋에서 사용자 간 데이터 교차 가능
- 일부 라이프로그 데이터 포인트가 여러 사용자의 device에서 기록되어 overlap 발생 가능
- CV split 시 같은 물리적 이벤트가 훈련/검증 세트에 동시에 분할되면 모델이 "이벤트 패턴"을 암기하고 평가에서 고점수

**mitigation:**
- 모든 데이터 포인트에 고유 `user_id` + `timestamp` composite key 설정
- CV split 시 `user_id` 단위 분할 또는 user 내 시간 기반 분할로 엄격 분리
- `user_id` leakage: 같은 사용자의 훈련/검증 data에 동일한 이벤트가 있는지 검증

### 1.3 lifelog_date → sleep_date 1일 차이

**리스크:**
- `lifelog_date`가 `sleep_date`보다 항상 1일 앞서 있음
- 이 1일 차이는 intentional일 수 있지만, **그 1일 이전의 모든 데이터 사용 가능 여부**가 명확하지 않음
- 예: sleep_date = 2025-03-16 인 경우, lifelog_date = 2025-03-15 의 data만 사용 가능한가? 아니면 2025-03-14까지도 포함 가능한가?

**mitigation:**
- EDA 단계에서 `lifelog_date`~`sleep_date` 간격 분포를 정밀 분석
- 규칙 문서의 명시적 지침이 없으면 `sleep_date - N일` (N=1,3,7,14) 등 여러 window로 실험
- 가장 보수적 접근: `sleep_date - 1`까지만 사용 → 안전측 but information 부족 가능성

---

## 2. 과적합 리스크 (Overfitting) — 🔴 High

### 2.1 p >> n 문제

**리스크:**
- 학습 샘플: 450일 분 (10명 × 약 45일 평균)
- 피처 수: 12개 데이터 소스에서 수백~수천 피처 추출 가능
- **p(피처) >> n(샘플) 구조** → 모델이 noise를 signal로 학습할 확률 극대화

**mitigation 전략:**

| 레이어 | 접근법 | 세부 |
|--------|--------|------|
| 피처 설계 단계 | 물리 기반 피처 | 무작위 피처 생성 금지. 수면/감정/스트레스 이론에 근거한 명시적 피처만 |
| 선택적 필터링 | Univariate screening | 타깃과의 상관계수 기반 pre-filtering (p < 0.15 수준) |
| 모델 내재적 | L1/L2 regularization | LightGBM: `lambda_l1`, `lambda_l2` 큰 값으로 시작. CatBoost: `l2_leaf_reg` + `bagging_temperature=0` (무 bagging) |
| 앙상블 | OOF stacking | 과적합 모델이 앙상블에서 상쇄되도록 |
| 검증 | Time-series CV | 단순 k-fold 금지. Time-based split 필수 |

**특수 권장:**
- 12개 데이터 소스별 feature importances를 먼저 확인. contribution이 낮은 소스는 아예 feature extraction 건너뛰고 모델에서 제외
- PCA/feature hashing보다는 **도메인 기반 피처 선택**이 우선 — 라이프로그는 해석 가능성이 평가의 일부

### 2.2 타깃 불균형

**리스크:**
- 이진 분류 7개 타깃 중 일부는 class distribution에 extreme bias 가능
- Log Loss는 예측 확률이 극단적일수록 무한대 발산 → 불균형 클래스에서 catastrophic loss 발생 가능

**mitigation:**
- 모든 타깃에 대해 `class_distribution` 분석을 EDA 단계에서 필수 수행
- `scale_pos_weight` (LightGBM) / `class_weight` (scikit-learn)로 불균형 보정
- 확률 캘리브레이션(Platt scaling 또는 Isotonic regression)을 반드시 적용

---

## 3. 개인별 편차 (Inter-User Variability) — 🟡 High

**리스크:**
- 10명 × 33~57일 (평균 ~45일)로 데이터 편차가 큼
- 일부 사용자는 장기간 데이터, 일부는 단기간 → 모델이 "장기 사용자 패턴"에 편향될 가능성
- 개인별 baseline 패턴 vs 집단 공통 패턴: 어떤 것이 더 일반화 가능한지 명확하지 않음

**mitigation 전략:**

```
Layer 1: Personalization
  - 각 사용자별 mean/std로 표준화 (z-score per user)
  - 사용자별 baseline(평균) 대비 deviation 기반 피처

Layer 2: Cross-user features
  - 동일 시점의 집단 평균/분위수 기반 피처 추가
  - "해당 사용자의 값이 집단에서 얼마나 outlier인지"를 encoding

Layer 3: User embedding
  - user_id를 categorical feature 또는 embedding으로 사용
  - user별 data 개수를 feature로 추가 (data quality indicator)
```

**전략적 고려사항:**
- 10명은 충분히 작으므로 user별 모델 학습도 고려 가능
- 하지만 **user별 모델 = 10개 모델** → 코드 재현성 검증에서 10개 모델을 어떻게 패키징할지 명시 필요
- 우선: cross-user single model → performance 불충분하면 user-specific model로 단계적 접근

---

## 4. 평가 지표 관련 리스크 — 🔴 Critical

### 4.1 Log Loss의 극단값 민감도

**리스크:**
- Log Loss: `- (y*log(p) + (1-y)*log(1-p))`
- 예측 확률 `p = 0` 또는 `p = 1`일 때 Loss → ∞
- 리더보드에서 단 1개의 극단 예측이 전체 Average Log-Loss에 치명적 영향

**mitigation (엄격 필수):**
```python
# Inference 시 확률 clipping
p = np.clip(model.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
```
- 모든 타깃에 대해 inference 시 위 clip 적용
- validation set에서 극단 확률 발생 빈도 모니터링

### 4.2 Public/Private 괴리 (shake-up)

**리스크:**
- Public Score: 테스트의 사전 샘플링 44%
- Private Score: 테스트 100%
- Public leaderboard에서 상위권 → Private에서 추락하는 shake-up 현상
- Public 44% 데이터가 전체를 대표하지 않을 수 있음

**mitigation:**
- Public 점수追逐 금지. CV score의 **안정성**을 우선
- CV fold별 점수 표준편차가 낮은 모델 선택 (variability monitoring)
- Public LB에 occasional submit 하되, 주요 실험은 Private LB가 공개된 후에도 재현 가능한 설정으로
- 1일 3회 제한을 고려하여 submit 전 OOF score로 모든 검증 완료

---

## 5. 제출/재현성 리스크 — 🔴 Critical

### 5.1 재현성 검증 = 최종 탈락 조건

**리스크:**
- 09.01 코드 제출 마감 → 09.30 재현성 검증
- **재현성 검증 실패 = 수상 자격 상실**. 리더보드 점수가 아무리 높아도 의미 없음
- 제출 코드 조건:
  - `/data` 경로 포함
  - `.R`, `.rmd`, `.py`, `.ipynb` 만 허용
  - UTF-8 인코딩
  - 의존성 포함, 오류 없이 실행 가능
  - OS/라이브러리 버전 명시
  - 사전학습 모델 사용 시 출처/링크 명시
  - **Private 리더보드 스코어 복원 가능**

**mitigation 전략:**

| 항목 | 실행 계획 |
|------|-----------|
| 가상환경 | `environment.yml` (conda) + `requirements.txt` (pip) 동시 생성 |
| 버전 관리 | `pip freeze > requirements.txt` + `conda list > env_detailed.txt` |
| 실행 스크립트 | `run_all.sh` 또는 `Makefile`: 데이터 전처리 → 학습 → 검증 → 제출 생성의 전체 파이프라인 |
| 테스트 | 제출 코드 자체를 Docker 컨테이너에서 처음부터 실행하는 CI 파이프라인 구축 |
| 시드 고정 | `numpy.random.seed()`, `torch.manual_seed()`, `lightgbm_seed`, `python_hash_seed` 등 **모든** 시드 기록 |
| 데이터 경로 | 절대 경로 대신 `/data/` 상대 경로 사용. `os.path.join('/data', 'filename')` 패턴 |

**재현성 체크리스트 (상세):**
```yaml
reproducibility:
  seed_all:
    - python_hash_seed=0
    - numpy.random.seed(42)
    - torch.manual_seed(42)
    - random.seed(42)
    - lightgbm_params: seed=42
    - catboost_params: random_seed=42
  environment:
    - requirements.txt
    - environment.yml
    - OS: "Ubuntu 22.04 / Python 3.11"
    - cuda_version: "12.x (if GPU)"
  data_path:
    - root: "/data"
    - train: "/data/ch2026_metrics_train.csv"
    - items: "/data/ch2025_data_items/*.parquet"
  pipeline:
    - 01_data_load.py
    - 02_feature_engineering.py
    - 03_model_training.py
    - 04_evaluation.py
    - 05_submission.py
  validation:
    - run_all.sh: 전체 파이프라인 자동 실행
    - private_lb_reproduce_score: 제출 파일로 재추출 시 동일 점수
```

### 5.2 제출 마감 병목

**리스크:**
- 리더보드 및 논문 제출 동시 마감: 06.26
- 논문 제출 시스템(EDAS)과 리더보드 제출이 별도 프로세스 → 동시 제출 기술적 실패 가능성
- 코드 제출(09.01)은 논문 채택 결과 이후이므로, 논문 채택 실패 시 코드 준비가 무의미할 수 있음

**mitigation:**
- 06.20 이전: 리더보드 최종 제출 + 논문 초안 완료
- 06.22: EDAS 논문 제출 테스트 (가상의 PDF로 경로 및 형식 검증)
- 06.24: 최종 리더보드 submit
- 06.25: 최종 논문 submit
- 코드 제출은 논문 채택 여부와 무관하게 항상 준비된 상태로 유지

### 5.3 1일 3회 제출 제한

**리스크:**
- 1일 3회는 매우 제한적. 실험 결과 확인 → 수정 → 재제출 사이클이 느림
- Public shake-up 전에 3회를 모두 쓰는 것 방지 필요

**mitigation:**
- OOF(Local CV) score로 모든 실험을 먼저 검증 → OOF score가 개선되었을 때만 submit
- Submit 로그 파일 유지 (timestamp, OOF score, submitted LB score, 변경 사항)
- 1일 3회 제한은 **새로운 실험 제출**용, 이미 검증된 설정의 미세 조정에는 활용

---

## 6. 개인정보/윤리 리스크 — 🟠 Medium

**리스크:**
- 라이프로그 (활동량, 심박수, 위치, 화면 사용, WiFi 연결 정보 등)는 **매우 민감한 개인정보**
- 논문에서 개인정보 처리 방침 서술 필요:
  - IRB(연구윤리위원회) 승인 여부
  - 동의 절차 서술
  - 데이터 익명화/가명화 방법
  - 데이터 저장/처리 보안 조치

**mitigation:**
- 논문 초안 작성 시 별도 "Ethics Statement" 섹션 준비
- ETRI 연구진에게 IRB 관련 질의
- 논문에서 특정 사용자 식별 가능한 정보(위치, 시간 패턴 등)를 aggregation 수준으로만 제시

---

## 7. 데이터 소스 불균형 — 🟠 Medium

**리스크:**
- 12개 데이터 소스 중 일부는 저빈도采样, 일부는 고빈도
- 일부 사용자는 특정 소스 데이터가 결측일 수 있음 (device 문제, 배터리 소진 등)
- 소스별 data quality에 따라 feature reliability가 다름

**mitigation:**
- EDA 단계에서 소스별: (1) 총 row 수 (2) 결측률 (3) sampling 간격 분포 분석
- 결측률이 높은 소스는 feature extraction 건너뛰거나, 결측 패턴 자체를 feature로 활용
- 소스별 중요도 분석: `feature_importance` 기반 불필요 소스 pruning

---

## 8. 제출 파일 검증 리스크 — 🟠 Medium

**리스크:**
- `ch2026_submission_sample.csv` 형식과 완전히 일치해야 함
- 컬럼명, 순서, 데이터 형식(소수점 자릿수 등)의 미세 불일치도 제출 실패 원인
- 7개 타깃 = 7개 예측 컬럼. sample과 완전 비교 검증 필요

**mitigation:**
```python
# 제출 파일 검증 스크립트
import pandas as pd

sample = pd.read_csv('/data/ch2026_submission_sample.csv')
submission = pd.read_csv('submission.csv')

assert list(submission.columns) == list(sample.columns), "Column mismatch!"
assert submission.shape[0] == sample.shape[0], "Row count mismatch!"
assert submission['id'].equals(sample['id']), "ID mismatch!"

# 확률 범위 검증
for col in ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']:
    assert submission[col].between(0, 1).all(), f"{col} out of [0,1]"
```

---

## 9. 코드 확장자 제한 — 🟢 Low

**리스크:**
- 허용 확장자: `.R`, `.rmd`, `.py`, `.ipynb` 만
- Python으로 개발 시 `ipynb` → `.py` 변환 시 실행 순서/셀 구조 정보가 손실될 수 있음

**mitigation:**
- 핵심 파이프라인은 `.py` 파일로 유지
- `nbconvert`로 ipynb → py 변환 시 `--no-input` 옵션으로 코드만 추출
- `.R` 사용 가능성 검토: R도 성능적으로 경쟁력 있음(LightGBM R 패키지 존재)

---

## 10. 팀 구성 제약 — 🟢 Low

**리스크:**
- 팀 최대 4명. 지도교수는 저자만 포함, 팀원 아님
- 팀명에 소속/신분 식별 정보 포함 금지
- 중복 등록 불가

**mitigation:**
- 팀 구성 확정 시 멤버 4명 내외로 제한
- 팀명: 익명성 확보 (예: "Team Alpha" → "팀명_익명" 형식)

---

## 제출 체크리스트

### Phase 1: 데이터 수신 전 (현재 ~ 데이터 도착 시)

- [ ] `data_raw/` 폴더 구조 정리
- [ ] 데이터 다운로드 대기 (데이콘 로그인 필요)
- [ ] 환경 설정: Python 3.11 가상환경 생성
- [ ] 코드 구조 디렉토리 생성: `src/`, `notebooks/`, `scripts/`, `output/`
- [ ] 재현성 파운데이션: 시드 고정 유틸리티 함수 구현

### Phase 2: 데이터 수신 즉시 (Day 0)

- [ ] 파일 무결성 검증 (크기, 행수, 컬럼)
- [ ] 타임스탬프 정렬 및 키 정합성 확인
- [ ] 타깃 분포 분석 (7개 지표 class imbalance)
- [ ] 소스별 data quality 분석 (12개 라이프로그 소스)
- [ ] 누수 체크: `lifelog_date` vs 타깃 시점 관계 정밀 분석
- [ ] 사용자별 data 개수 분포 확인
- [ ] EDA 리포트 생성

### Phase 3: 베이스라인 (Day 1-3)

- [ ] Time-based CV split 구현 (누수 차단 확인 포함)
- [ ] 베이스라인 피처: 소스별 기본 통계 (mean, std, min, max)
- [ ] LightGBM 단일 모델 OOF 평가
- [ ] 확률 캘리브레이션 적용
- [ ] OOF Log-Loss 기반라인 점수 확보

### Phase 4: 실험 (Day 4-14)

- [ ] 피처 확장 (시간 윈도우, 주기성, 변화율)
- [ ] 모델 비교 (LightGBM / CatBoost / XGBoost)
- [ ] 멀티타깃 전략 (7개 개별 vs joint model)
- [ ] 앙상블 (OOF stacking / blending)
- [ ] CV 안정성 모니터링 (fold별 점수 편차)
- [ ] Public LB occasional submit (매주 최대 1회)

### Phase 5: 제출 준비 (06.20 이전)

- [ ] 최종 모델 고정
- [ ] 리더보드 최종 제출 (06.24 이전)
- [ ] 논문 초안 작성 완료 (06.20 이전)
- [ ] 코드 패키징: `src/`, `environment.yml`, `requirements.txt`, `run_all.sh`
- [ ] 재현성 검증 테스트: Docker에서 full pipeline 실행
- [ ] 제출 파일 검증 스크립트 통과 확인

### Phase 6: 최종 제출 (06.26)

- [ ] 리더보드 제출
- [ ] EDAS 논문 제출
- [ ] 제출 로그 파일 기록

### Phase 7: 코드 제출 (09.01)

- [ ] 코드 + 모델 설명서 제출
- [ ] Private LB 재현 테스트
- [ ] 코드 검증 대응 문서 준비 (09.30)

---

## Risk Summary Matrix

| # | 리스크 | Severity | 영향 범위 | mitigation 상태 |
|---|--------|----------|-----------|----------------|
| 1 | sampling_rate 불규칙 → 시간 누수 | 🔴 Critical | 모델 성능 + 재현성 | mitigation 설계 완료 |
| 2 | 사용자 간 데이터 overlap | 🔴 Critical | CV 신뢰도 | mitigation 설계 완료 |
| 3 | lifelog_date → sleep_date 1일차 불명확 | 🔴 Critical | feature window 설계 | EDA에서 우선 검증 |
| 4 | Log Loss 극단값 → 무한대 | 🔴 Critical | Public/Private 점수 | 확률 clipping 필수 |
| 5 | 재현성 검증 실패 = 탈락 | 🔴 Critical | 대회 참여 전체 | 재현성 파이프라인 구축 |
| 6 | p >> n 과적합 | 🟡 High | 일반화 성능 | regularization + 피처 선택 |
| 7 | 개인별 편차 | 🟡 High | 교차 검증 안정성 | per-user normalization |
| 8 | Public shake-up | 🟡 High | 리더보드 전략 | CV 안정성 우선 |
| 9 | CV 안정성 부족 | 🟡 High | Private 점수 예측 | fold별 편차 모니터링 |
| 10 | 1일 3회 제출 제한 | 🟡 High | 실험 사이클 속도 | OOF 우선 검증 |
| 11 | 논문/코드 제출 병목 | 🟠 Medium | 최종 수상 | 일찍 완료 |
| 12 | 데이터 소스 불균형 | 🟠 Medium | feature quality | EDA 품질 분석 |
| 13 | 개인정보 윤리 | 🟠 Medium | 논문 채택 | Ethics Statement 준비 |
| 14 | 제출 파일 검증 | 🟠 Medium | 제출 실패 | 검증 스크립트 자동화 |
| 15 | 코드 확장자 제한 | 🟢 Low | 코드 제출 | .py 유지 |
| 16 | 팀 구성 제약 | 🟢 Low | 참여 자격 | 4명 이내로 제한 |

---

## 권장 우선순위

1. **🔴 최우선:** 데이터 수신 시 누수 체크 → 엄격한 time-based CV split 구현
2. **🔴 다음:** 재현성 파이프라인 구축 (Docker CI 포함). 리더보드 점수보다 중요도 우선
3. **🟡 그 다음:** p >> n 대응 (정규화 + 피처 선택) + 개인별 편차 완화
4. **🟠 여유 시:** 논문 Ethics Statement + 제출 파일 자동 검증

> **핵심 원칙:** 이 대회는 "리더보드 점수 > 재현성"이 절대 아니다. 재현성이 통과하지 못하면 점수가 0점이다. 모든 의사결정은 재현성 가능성을 최우선으로 한다.
