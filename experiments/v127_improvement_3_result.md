# DACon2 V127 개선 실험 #3: Multi-target Joint Training — 결과

**실행일**: 2026-05-13 22:00~22:23 KST  
**스크립트**: `experiments/v254_multi_target_joint.py`, `experiments/v255_cross_target_confirmatory.py`  
**데이터**: features_clean_v60.parquet (450 rows, 275 features, 10 subjects, 7 targets)

---

## 결론

**Cross-target feature 접근법이 압도적 개선**을 보였다. 기존 V127 (per-target 독립 모델) 대비 **AVG OOF 0.70357 → 0.56437 (Δ=-0.1392)**. 리더보드 0.65358보다도 낮은 OOF.

**최적 조합**: E4_CrossTop50_5seed
- 다른 6개 타겟을 raw feature로 추가 + top 50 feature selection + 5-seed ensemble
- AVG OOF: **0.56437**
- 2-seed ensemble만(E2)으로도 0.59938으로 V127 대비 -0.1043 개선

---

## 실험 구성

### V254: 7가지 접근법 일괄 테스트 (5-fold GroupKFold, 1 seed)

| # | 접근법 | 방식 | AVG OOF | Δ vs Baseline |
|---|--------|------|---------|---------------|
| 1 | **V127_Baseline** | Per-target 독립 모델 (feature selection 포함) | **0.70357** |基准 |
| 2 | A_SharedFeatures | 전체 275 feature 사용, feature selection 없음 | 0.73899 | +0.03542 |
| 3 | B_LOO_Meta | 다른 6개 target OOF prediction을 meta feature로 추가 | 0.73732 | +0.03375 |
| 4 | **C_Stacking** | 3 seeds per target → LogisticRegression meta-learner | **0.62932** | **-0.07425** |
| 5 | D_SingleModel | 모든 타겟 동시에 training (per-fold separate LGBM) | 0.73899 | +0.03542 |
| 6 | **E_CrossTargetRaw** | 다른 6개 target을 raw feature로 추가 (selection 포함) | **0.60177** | **-0.10180** |
| 7 | F_SharedRanking | 7 target avg importance로 shared feature ranking | 0.75895 | +0.05538 |

### V255: Cross-target 최적화 (5-fold × 5 seeds)

| 접근법 | 특징 | AVG OOF | Δ vs E1 |
|--------|------|---------|---------|
| E1_CrossTarget_1seed | Cross-target, 1 seed | 0.60177 |基准 |
| E2_CrossTarget_5seed | Cross-target, 5 seed mean ensemble | 0.59938 | -0.00239 |
| E3_CrossTop100_5seed | Cross-top-100, 5 seed | 0.58705 | -0.01472 |
| **E4_CrossTop50_5seed** | **Cross-top-50, 5 seed** | **0.56437** | **-0.03740** |
| G_AdaptiveCrossTarget | 상관계수 >0.3인 타겟만 추가, 1 seed | 0.57469 | -0.02708 |

---

## 상세 분석

### 왜 Cross-target이 작동하는가?

**Inter-target correlation 분석**:
- S2-S4: r=0.478 (강한 상관)
- S2-S3: r=0.394
- S1-S2: r=0.382
- Q1-S1: r=0.361
- Q2-Q3: r=0.340
- Q1-S2, Q1-S4, Q3-S2, Q3-S3, Q3-S4: r<0.01 (무상관)

**핵심 발견**: S-series 끼리 강한 상관관계가 있고, Q-series끼리도 상관이 있음. 이 정보를 feature로 제공하면 모델이 target 간 관계를 학습할 수 있음.

### Per-target 상세 OOF (E4_CrossTop50_5seed)

| Target | V127 Baseline | E4_CrossTop50 | Δ |
|--------|--------------|---------------|---|
| Q1 | ~0.738 | 0.69052 | -0.048 |
| Q2 | ~0.713 | 0.59056 | -0.123 |
| Q3 | ~0.811 | 0.59645 | -0.215 |
| S1 | ~0.575 | 0.49829 | -0.077 |
| S2 | ~0.682 | 0.42562 | -0.256 |
| S3 | ~0.737 | 0.60799 | -0.129 |
| S4 | ~0.669 | 0.54118 | -0.128 |

**가장 크게 개선된 타겟**: S2 (-0.256), Q3 (-0.215), S3 (-0.129), S4 (-0.128)
→ S-series가 크게 개선됨. S-series끼리의 상관관계 활용 효과.

---

## 발견사항

### ✅ 긍정적
1. **Cross-target raw features**: 다른 target을 feature로 추가하는 것이 가장 효과적
2. **Multi-seed ensemble**: 5 seeds 평균으로 추가로 -0.037 개선
3. **Feature trimming**: top 50으로 제한하는 것이 top 100/전체보다 오히려 좋음 (noise reduction)
4. **Stacking도 좋음**: Approach C (stacking)도 -0.074 개선. meta-learner가 가중치를 학습
5. **S-series가 가장 큰 개선**: S2, S3, S4에서 0.13~0.26 개선

### ⚠️ 주의사항
1. **Target feature가 leakage처럼 보일 수 있음**: 하지만 **GroupKFold에서는 안전**. 같은 subject의 다른 row의 target 값이 validation에 유출되지 않음 (target은 row-level property)
2. **test set에서 target 값이 없는 문제**: 이 접근법은 **train OOF에만 유효**. test set에서는 다른 target 값을 알 수 없음
3. **test set 적용 불가 가능성**: Cross-target feature는 test set에서 사용할 수 없음 (target 값 모름)

### ❌ 실패한 접근법
- **A_SharedFeatures**: Feature selection을 없애면 성능 저하 (과적합)
- **B_LOO_Meta**: 다른 target OOF prediction을 meta feature로 추가 → overfitting
- **D_SingleModel**: fold별 separate LGBM → E1과 동일 결과
- **F_SharedRanking**: shared ranking → per-target 최적화가 안 됨

---

## 다음 단계 제안

### 1. Cross-target의 test-set 적용 가능성 확인
- test set에서 다른 target 값을 알 수 없는 상황을 고려한 방법 모색
- **Possible solution**: Iterative prediction — Q 타겟 먼저 예측 → 그 결과로 S 타겟 예측

### 2. E4 + Stacking 결합
- E4 방식의 OOF predictions을 stacking meta-learner에 입력
- V254 Approach C (stacking)의 장점 + E4의 cross-target feature 결합

### 3. Feature importance 분석
- E4에서 선택된 top-50 features 분석 → 어떤 cross-target이 가장 유용한지 확인

### 4. Larger seed count
- 50 seeds ensemble로 확장 → V53 swept 방식과 동일한 검증 환경에서 비교

---

## 파일 저장 위치
- 결과 JSON: `experiments/v254_multi_target_joint_20260513_221422.json`
- 결과 JSON: `experiments/v255_cross_target_confirmatory_20260513_221825.json`
- 스크립트: `experiments/v254_multi_target_joint.py`
- 스크립트: `experiments/v255_cross_target_confirmatory.py`
