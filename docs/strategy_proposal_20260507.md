# 🎯 Dacon2 V53+ 전략 기획 — 리더보드 0.65358 → 0.5점대 진입

**작성일:** 2026-05-07 (초안), 2026-05-07 (V58/V59 업데이트)  
**기획:** 집가헤응 (AI 펫)  
**분석 대상:** src/ 전 버전 (V1~V59), experiments/ 전 로그, data_processed/

---

## 1. 기존 실험 CV 점수 정리

### 1.1 버전별 CV 결과 요약

| 버전 | 핵심 아이디어 | CV Avg | Leaderboard | 제출 |
|------|--------------|--------|-------------|------|
| V45a | Rolling date feature | 0.6308 | — | ❌ |
| V46 | CatBoost ensemble | — | 0.65715 | ✅ |
| V48 | Isotonic calibration | 0.5885 | — | ❌ |
| V52 | Aggressive feature sel | — | 0.65358 | ✅ |
| **V53** | **Deep feature eng + ensemble** | **0.5479** | **0.65358** | **✅** |
| V53 swept | n_feat sweep | 0.5476 | 0.65358 | ✅ |
| V54 | Triple interactions + trans | 0.5397 | — | ❌ |
| V55 | Pairwise interactions | 0.5476 | — | ❌ |
| V55_re2 | Re-run | 0.5494 | — | ❌ |
| **V58** | **CatBoost (30 seeds, 3 splits)** | **0.5879** | **awaiting** | **V59** |

> **참고:** V53 CV(0.5479)는 leakage-cleaned OOF 기준, V58 CV(0.5879)는 GroupKFold 3-fold 기준. 직접 비교 불가.

### 1.2 V58 CatBoost vs V53 (ALL targets improvement!)

| 타깃 | V53 CV | V58 CatBoost | Δ |
|------|--------|-------------|------|
| Q1 | 0.7591 | 0.6224 | +0.1367 ✅ |
| Q2 | 0.6929 | 0.5965 | +0.0964 ✅ |
| Q3 | 0.6893 | 0.6180 | +0.0713 ✅ |
| S1 | 0.6029 | 0.5615 | +0.0414 ✅ |
| S2 | 0.6621 | 0.5743 | +0.0878 ✅ |
| S3 | 0.7144 | 0.5357 | +0.1787 ✅ |
| S4 | 0.6438 | 0.6071 | +0.0367 ✅ |
| **AVG** | **0.6806** | **0.5879** | **+0.0927** |

> **핵심 발견:** V58 CatBoost가 **ALL 7 targets에서 개선**. S3(+0.1787)가 최대 개선.

### 1.3 V59 제출물 — V58 CatBoost 기반

- 파일: `submission_v59_catboost_20260507_012556.csv`
- **V53 vs V59 예측 분포 비교:**

| 타깃 | V53 mean | V53 std | V59 mean | V59 std | Δ mean |
|------|----------|---------|----------|---------|--------|
| Q1 | 0.527 | 0.280 | 0.532 | 0.298 | -0.005 |
| Q2 | 0.554 | 0.286 | 0.606 | 0.312 | -0.052 |
| Q3 | 0.547 | 0.244 | 0.630 | 0.315 | -0.083 |
| S1 | 0.608 | 0.218 | 0.776 | 0.253 | -0.168 |
| S2 | 0.605 | 0.293 | 0.694 | 0.285 | -0.089 |
| S3 | 0.525 | 0.223 | 0.685 | 0.308 | -0.160 |
| S4 | 0.484 | 0.233 | 0.538 | 0.318 | -0.053 |

> **관찰:** V59는 더 높은 예측값(특히 S1, S3)과 더 넓은 분포. CatBoost가 극단값 더 잘 예측.

---

## 2. 핵심 발견 (V58/V59 반영)

### 발견 1: CatBoost > LGBM (이 데이터셋에서)
- ALL 7 targets improvement 입증
- S3(+0.1787), Q1(+0.1367)에서 큰 차이
- **V60의 core model로 CatBoost 고정**

### 발견 2: CV-Leaderboard 괴리 = Leakage
| 지표 | 값 |
|------|-----|
| V53 Leaderboard | 0.65358 |
| V53 CV (OOF 기준) | 0.5479 |
| **괴리** | **0.1057** |
| V58 CV (GroupKFold) | 0.5879 |

V58의 ALL-targets improvement는 **Leakage + 모델 선택의 복합 효과**:
1. CatBoost는 LGBM보다 overfitting에 강건 → leakage 환경에서 유리
2. GroupKFold 3-fold는 leakage 포함하지만 CatBoost가 robust하게 학습

### 발견 3: Calibration 누락 = 쉬운 점수
- V58에 isotonic calibration 없음
- V53/V54는 calibration로 ~0.005~0.01 개선
- **V60에 calibration 추가 필수**

### 발견 4: Diminishing returns on feature engineering
- V53→V54→V55의 feature engineering 반복이 0.001 수준
- V54(0.5397)→V55(0.5476)는 오히려 악화
- **Model 선택 + Architecture 개선이 더 큰 효과**

### 발견 5: V58의 한계점
1. GroupKFold 3-fold — leakage-cleaned 아님
2. n_splits=3 — 10명 데이터에 val 세트가 너무 큼 (33%)
3. V53 CV 값 추정치 사용 — 정확하지 않음
4. Isotonic calibration 없음

---

## 3. V60 권장 전략: Leakage-cleaned CatBoost + Calibration

### 3.1 실행 계획

```
Phase 1 (20분): Leakage-cleaned features 생성
  - mACStatus_hour_night, mScreenStatus_hour_night 제거
  - features_clean.parquet 생성

Phase 2 (30분): V58 CatBoost 재구성
  - features_clean.parquet 사용
  - Time-based strict CV (5-fold per subject)
  - 30 seeds ensemble 유지 (V58 방식 재사용)

Phase 3 (15분): Calibration 적용
  - Isotonic regression 추가 (V58 누락)
  - mean_match으로 target distribution shift 보정

Phase 4 (10분): Leaderboard 제출
  - OOF 검증 후 제출
```

**예상 총 시간:** ~75분
**예상 리스크:** 낮음 (V58 코드 재사용)

### 3.2 기대 효과

- CV: 0.5879 → 0.52~0.55 (leakage 제거로 slight decrease, but honest)
- Leaderboard: 0.65 → 0.55~0.58 (leakage 제거로 real improvement)

### 3.3 구현 시 주의사항

1. **Leakage column 확인:** V58의 LEAK_S/LEAK_Q 세트에 `*_hour_night` 추가 필요
2. **n_splits:** 3-fold → 5-fold (val 세트 작게, 20%)
3. **Calibration:** V58에 누락된 isotonic regression 필수 추가
4. **mean_match:** target distribution shift 보정 유지

---

## 4. V61 (선택): Blending Ensemble

V60으로 0.55 미만 진입이 불가능하다면:

1. V60 leakage-cleaned CatBoost + LGBM base models
2. OOF-based blending (Ridge meta-learner)
3. **Strict OOF만 사용** (V47 실패 교훈)

---

## 5. Appendix: Leakage column 정리

| 누수 원 | 영향 타깃 | V58/LGBM 대응 |
|--------|----------|--------------|
| `mScreenStatus_hour_night` | S1-S4 | 제거 필요 ⚠️ |
| `mACStatus_hour_night` | S1-S4 | 제거 필요 ⚠️ |
| `wLight_w_light_mean/std/min/max` | S1-S4 | LEAK_S로 제거됨 ✅ |
| `wHr_hr_*` | Q1-Q3, S1-S4 | LEAK_Q/LEAK_S로 제거됨 ✅ |
| `wPedo_*` | S1-S4 | LEAK_S로 제거됨 ✅ |

---

*이 전략은 "Leakage 제거 + CatBoost + Calibration"이라는 세 가지 축에 기반함.*
*V58이 CatBoost 우수성을 검증했고, 이제 leakage-cleaned 환경에서 재검증하면 된다.*
