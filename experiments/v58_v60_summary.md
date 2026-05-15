# V58–V60 Experiment Summary

## 실행 환경
- GroupKFold n_splits=3
- n_jobs=1 (WSL2)
- Data: train 450 rows × 282 feats (141 base + 141 zscore), test 250 rows

## CV 결과 비교

| Target | V53 Swept | V58 Stack | V59 Multi | V60 Inter | **Best** |
|--------|-----------|-----------|-----------|-----------|----------|
| Q1     | 0.7591    | 0.6469    | 0.6454    | 0.6575    | V59      |
| Q2     | 0.6929    | 0.6310    | 0.6319    | 0.6546    | V58      |
| Q3     | 0.6893    | 0.6337    | 0.6365    | 0.6325    | V60      |
| S1     | 0.6029    | 0.5653    | 0.5667    | 0.5702    | V58      |
| S2     | 0.6621    | 0.6249    | 0.6266    | 0.6282    | V58      |
| S3     | 0.7144    | 0.6223    | 0.6260    | 0.6182    | V60      |
| S4     | 0.6438    | 0.6532    | 0.6432    | 0.6485    | V59      |
| **AVG**| **0.6806**| **0.6253**| **0.6252**| **0.6299**|          |

## V53 → V58: +0.0553 (8.1% 개선)

### 핵심 발견
1. **Three-model stacking (LGBM+CatBoost+XGBoost)** 이 기존 single-model ensemble 대비 명확한 우위
2. **Stacking meta-learner** 가 각 모델의 보완적 패턴을 효과적으로 통합
3. **S4** 는 stacking보다 single-model(LGBM)이 약간 더 나은 유일한 타겟

### V58 아키텍처
```
Level 0: LGBM(1 seed) + CatBoost(1 seed) + XGBoost(1 seed) → OOF
Level 1: LogisticRegression(C=1.0) → stacked predictions
Target별: V53 swept config(=cfg + n_feat) 적용, 타겟별 leak 제거
```

## V58 → V59: +0.0002 (미미한 변화)

### 다중 시드 효과
- 5 seeds per model → OOF averaging
- S4에 dedicated feature set(n_feat=28) 적용
- **결과:** 개별 모델 loss가 높아짐 (accumulation bug 영향 의심)
- Stacked result는 V58와 동등 수준

## V58 → V60: -0.0046 (역개선)

### 실패한 개선 시도
1. **Feature interactions** (top-3 cross-products): 오히려 과적합
2. **Platt calibration**: V58보다 낮은 성능
3. **Wider feature sets** (n_feat+3/+8): noise 증가

## 최종 결론: V58이 최상의 single architecture

### 제출 파일
- **submission_v58_ensemble_20260507_011530.csv**
- **meta_v58_ensemble_20260507_011530.json**

### 스크립트
- `src/v58_ensemble_tri.py` — V58 (세 모델 스택잉)

## 다음 단계 제안

1. **Leaderboard 제출 확인** — V58 submission으로 실제 점수 측정
2. **V58 seed ensemble** — 5~10 seeds로 추평균 (V59의 accumulation bug 없이)
3. **Stacking C tuning** — C=0.3~2.0 sweep
4. **Per-target model selection** — S4만 single(LGBM), 나머지는 stacking
5. **Temporal features** — rolling mean/std 추가 (V58에서 시도하지 않음)

## 코드 구조

```
src/
├── v58_ensemble_tri.py     ← V58 (추천: 세 모델 스택잉)
├── v59_multiseed.py        ← V59 (다중 시드, V58과 거의 동일)
└── v60_interactions.py     ← V60 (feature interaction, 역개선)

experiments/
├── oof_v58.csv             ← V58 OOF predictions
├── oof_v59.csv             ← V59 OOF predictions
├── oof_v60.csv             ← V60 OOF predictions
├── test_preds_v58.csv      ← V58 test predictions
└── test_preds_v59.csv      ← V59 test predictions

submissions/
├── submission_v58_ensemble_20260507_011530.csv  ← V58 제출
├── submission_v59_multiseed_20260507_014700.csv ← V59 제출
├── submission_v60_interactions_20260507_021425.csv ← V60 제출
├── meta_v58_ensemble_20260507_011530.json      ← V58 메타
├── meta_v59_multiseed_20260507_014700.json      ← V59 메타
└── meta_v60_interactions_20260507_021425.json   ← V60 메타
```
