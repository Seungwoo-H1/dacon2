# ETRI Dacon2 V13/V14 Experiment Report

## Decision
V10 (0.6038) beat하는 버전만 제출.

## Results

### V11 (New Features) - FAILED: 0.6290
- 날짜/인터랙션 피처 15개 추가 → +0.025 worse
- 기존 142 피처가 이미 최적

### V12 (CatBoost) - FAILED: 0.6105
- LGBM(0.625) > CatBoost(0.611)

### V13 (Better Calibration) - FAILED: OOM
- `lgb.Dataset` + `reference=tr_ds` 조합 메모리 누수
- 50 models × 5 folds = 250 Dataset 객체가 메모리에 누적 → SIGKILL

### V14 (Feature Subset Optimization) - FAILED: OOM
- V13 과 동일 OOM 문제

### V10 (Baseline) - CONFIRMED BEST: 0.6038
- Proven pipeline: personalization(z-score) + leakage fix + LGBM 20-seed ensemble
- Mean-match calibration (isotonic 없음)

## Conclusion
**V10이 최종 best version (0.6038)**

## Next Steps
1. V10 제출
2. Deep Learning 등 근본적 재도전
3. 다른 방향 시도
