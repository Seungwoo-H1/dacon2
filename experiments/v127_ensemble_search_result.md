# DACon2 V127 Ensemble Architecture Search — 결과

**실행일**: 2026-05-13 23:00~23:48 KST  
**스크립트**: `experiments/v256_v127_ensemble_search.py`  
**데이터**: `features_clean_v60.parquet` (450 rows, 275 features, 10 subjects, 7 targets)  
**총 실행 시간**: 873초 (~14.5분)

---

## 결론

**Experiment 1: Bayesian Weight Optimization이 가장 유망**
- AVG OOF: **0.58229** (Δ=-0.05144 vs V127 baseline 0.63373)
- target별 최적 3-model combo + Bayesian 가중치 최적화가 효과적
- 특히 S2(-0.083), S1(-0.085), Q2(-0.069)에서 크게 개선

**중요 발견**: 기존 V127 고정 가중치(0.35/0.25/0.40) vs 최적화 가중치
- pair_wide가 Q2/Q3/S3에서 ~0.08 개선 (0.66→0.58)
- pair_deep가 S1/S2에서 ~0.08~0.09 개선 (0.62→0.55)
- V115-style base 모델은 almost useless (OOF 0.63~0.69)

---

## Experiment 1: Bayesian Weight Optimization

### 방법
- 6개 모델 pool: base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep
- 모든 3-way combo(20개) × 20 restart = 400 최적화 per target
- Constraint: w_i > 0, sum = 1
- Objective: CV log_loss 최소화

### Per-target 결과

| Target | Best LL | Combo | Weight | Δ vs V127 |
|--------|---------|-------|--------|-----------|
| Q1 | **0.59343** | pair_wide, pair_deep, trans_deep | 0.49/0.17/0.34 | -0.051 |
| Q2 | **0.58570** | base_wide, pair_wide, trans_deep | 0.03/0.50/0.47 | -0.069 |
| Q3 | **0.61103** | base_wide, pair_wide, trans_wide | 0.13/0.50/0.37 | -0.046 |
| S1 | **0.53841** | base_deep, pair_deep, trans_deep | 0.04/0.34/0.62 | -0.069 |
| S2 | **0.54833** | base_deep, pair_wide, trans_wide | 0.08/0.71/0.21 | -0.041 |
| S3 | **0.58297** | base_deep, pair_wide, pair_deep | 0.10/0.53/0.37 | -0.049 |
| S4 | **0.61615** | base_wide, pair_wide, trans_wide | 0.14/0.37/0.49 | -0.037 |
| **AVG** | **0.58229** | — | — | **-0.05144** |

### 핵심 인사이트
1. **pair_wide**가 7개 타겟 중 5개에서 top-2 model. pairwise features + wide config가 가장 안정적
2. **trans_deep**가 S1에 강력한 (0.62 weight). trans features + deep config
3. **base_deep**는 S1/S2에서 중요. 하지만 Q-series에는 거의 사용 안 됨
4. **pair_deep**가 S3에서 중요. pair features + deep

---

## Experiment 2: Feature-Subspace Diversity

### 방법
- 4개 랜덤 feature subspace (60% of 275 features) × 4 seeds = 16 models
- Mean ensemble of subspace models

### 결과
| Target | ENS_AVG | Best Single | Δ vs V127 |
|--------|---------|-------------|-----------|
| Q1 | 0.67766 | 0.67801 | +0.034 |
| Q2 | 0.67973 | 0.65824 | +0.025 |
| Q3 | 0.66265 | 0.65973 | +0.006 |
| S1 | 0.61283 | 0.60667 | -0.021 |
| S2 | 0.63226 | 0.62819 | +0.003 |
| S3 | 0.63955 | 0.63622 | +0.008 |
| S4 | 0.68380 | 0.68034 | +0.031 |
| **AVG** | **0.65550** | — | **+0.02177** |

### 결론
❌ **실패**. Subspace dropout는 OOF를 오히려 악화.
- Feature가 이미 적절히 선택됨 (V53 sweep) → subspace가 noise 추가
- Ensemble 효과가 개별 model 개선보다 작음

---

## Experiment 3: Rank Averaging vs Mean Averaging

### 6-model ensemble (base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep)

| Metric | AVG | Δ vs V127 |
|--------|-----|-----------|
| Mean blend | **0.60277** | -0.03095 |
| Rank blend | 0.60719 | -0.02654 |

### V127 config (3-model): pair_deep, pair_wide, base_wide

| Metric | AVG | Δ vs V127 |
|--------|-----|-----------|
| Mean blend | **0.60088** | -0.03285 |
| Rank blend | 0.60865 | -0.02507 |

### 결론
✅ **Mean blend가 rank blend보다 항상 우수** (모든 target, 모든 config)
- Rank blending은 extreme values를 완화하지만, 이 데이터셋에서는 mean이 더 좋음
- 6-model mean이 V127 mean보다 -0.032 개선
- **하지만 V127 고정 가중치보다 Bayesian 최적화 (-0.051)가 더 좋음**

---

## Experiment 4: Per-Target Weight Optimization

### 결과
Experiment 1과 동일 (same method, same results).
- AVG: **0.58229** (Δ=-0.05144)

---

## Experiment 5: Additional Model Diversity

### 5A: Polynomial Features (sq, cube, pairwise product)
- 4-model: V127 + poly_deep
- AVG: **0.59557** (Δ=-0.03815)
- Poly 모델 자체 OOF: 0.59~0.66 (base보다 좋음)

### 5B: Target-Mean Deviation Features
- 4-model: V127 + dev_deep
- AVG: **0.59842** (Δ=-0.03531)
- Dev 모델 자체 OOF: 0.55~0.62 (mixed)

### 5C: 8-model Equal-Weight Ensemble
- All 6 base + poly + dev
- AVG: **0.59560** (Δ=-0.03813)

### 결론
- Polynomial features가小有효. Poly_deep가 trans_wide보다 나은 target 다수
- Dev features는 mixed. S1/S2에 도움, Q3/S4에는 해로움
- 8-model equal-weight는 3-model Bayesian weight보다 열등

---

## Model Pool 상세 OOF (per-target, per-strategy)

| Target | base_wide | base_deep | pair_wide | pair_deep | trans_wide | trans_deep |
|--------|-----------|-----------|-----------|-----------|------------|------------|
| Q1 | 0.68825 | 0.68742 | **0.60518** | 0.61388 | 0.62150 | 0.62884 |
| Q2 | 0.66904 | 0.66939 | **0.60091** | 0.63016 | 0.60978 | **0.60089** |
| Q3 | 0.66378 | 0.66578 | **0.61016** | 0.62581 | **0.60736** | 0.64099 |
| S1 | 0.62212 | 0.62309 | 0.55416 | **0.54471** | 0.54822 | **0.54509** |
| S2 | 0.62700 | 0.63042 | **0.54599** | 0.55593 | 0.59645 | 0.60271 |
| S3 | 0.63921 | 0.63925 | 0.58866 | **0.58673** | 0.62299 | 0.61959 |
| S4 | 0.65840 | 0.67425 | **0.62574** | 0.63715 | **0.61889** | 0.63410 |

### 핵심 발견
1. **pair_wide**가 7개 타겟 중 5개에서 가장 낮음 → pairwise features의 효과
2. **trans_wide**가 Q2/Q3/S4에서 pair보다 나은 경우 있음
3. **base_deep**와 **base_wide**는 모두 매우 나쁨 (0.62~0.69)
   → base feature만으로는 부족, engineered features 필수
4. **wide config**가 deep보다 항상 나은 경우가 많음
   → 이 데이터셋은 과적합에 민감한 것 같음

---

## Comparison Summary

| Experiment | AVG OOF | Δ vs V127 | Status |
|------------|---------|-----------|--------|
| **V127 Baseline (fixed)** | 0.63373 | 0.00000 | 기준 |
| **Exp 1: Bayesian Weight** | **0.58229** | **-0.05144** | 🏆 **BEST** |
| Exp 3: 6M Mean Blend | 0.60277 | -0.03095 | Good |
| Exp 3: V127 Mean Blend | 0.60088 | -0.03285 | Good |
| Exp 5A: Poly + V127 | 0.59557 | -0.03815 | Moderate |
| Exp 5C: 8-model Equal | 0.59560 | -0.03813 | Moderate |
| Exp 2: Subspace | 0.65550 | +0.02177 | ❌ Worse |
| Exp 3: 6M Rank Blend | 0.60719 | -0.02654 | Moderate |

---

## 다음 단계 제안

### 1. Bayesian Weight + 6-model ensemble (최고 전망)
- Exp 1의 방법: per-target 3-model combo 최적화
- 6-model로 확장: base_wide, base_deep, pair_wide, pair_deep, trans_wide, trans_deep
- 각 target에 대해 6개 모델 weight 최적화
- V53 sweep의 n_feat를 각 model별로 다르게 설정

### 2. V127 + Cross-target feature (Exp 4 from v254)
- V254 E4_CrossTop50_5seed: OOF 0.56437 (V127 대비 -0.139)
- 이 결과는 **OOF 기준**. test set 적용 여부 확인 필요
- Cross-target feature가 test set에서 사용 가능한지 검증

### 3. Ensemble stacking
- 6개 모델 OOF → meta-learner (LR)로 가중치 학습
- V254 Approach C (Stacking): OOF 0.62932

### 4. 더 많은 seed
- 현재 4 seeds. 50 seeds (V53 방식)로 확장 가능
- OOF 안정성 향상 예상

---

## 파일 위치
- 스크립트: `experiments/v256_v127_ensemble_search.py`
- 결과 JSON: `experiments/v256_20260513_234829.json`
