# V13/V14 Experiment Report

## Decision
승우의 요청대로 **V13 (better calibration)** 과 **V14 (feature subset optimization)** 을 병렬로 실행.
V10 (0.6038) 을 beat 하는 버전만 제출.

## Results

### V11 (New Features) - FAILED: 0.6290
- 날짜/인터랙션 피처 15개 추가 → +0.025 worse
- 기존 142 피처가 이미 최적

### V12 (CatBoost) - FAILED: 0.6105
- LGBM(0.625) > CatBoost(0.611)
- CatBoost이 오히려 더 나쁨

### V13 (Better Calibration) - FAILED
- **원인 1**: 메모리 부족 (OOM → SIGKILL)
  - `lgb.Dataset` + `reference=tr_ds` 조합이 메모리 누수 유발
  - 50 models × 5 folds = 250 Dataset 객체가 메모리에 누적
- **원인 2**: `cal_per_subject_match`의 subject별 loop가 느림
- **원인 3**: numpy loop-based personalization이 비효율적
- **해결 방안**: V10 코드 fork 방식으로 재작성 필요 (proven pipeline)

### V14 (Feature Subset Optimization) - FAILED
- **원인**: V13 과 동일 OOM 문제
- LGBM importance / MI ranking / LASSO 비교 실행 불가

### V10 (Baseline) - CONFIRMED BEST: 0.6038
- Proven pipeline: personalization(z-score) + leakage fix + LGBM 20-seed ensemble
- Mean-match calibration (isotonic 없음)

## Conclusion

**V10이 최종 best version (0.6038)**.

V11/V12/V13/V14 모두 실패. 더 근본적인 접근이 필요하면:
1. Deep Learning (TabNet/FT-Transformer)
2. Data augmentation (sleep/wake state masking)
3. Multi-task learning (7 targets joint training)

## Files
- `src/07_v10_robust.py` — V10 (proven, best)
- `src/03_v13_calibration.py` — V13 (failed, OOM)
- `src/04_v14_feature_selection.py` — V14 (failed, OOM)
- `src/13_v12_catboost.py` — V12 (failed)
- `submissions/meta_v10_20260501_170715.json` — V10 meta

## Next Steps
승우의 의사 확인:
1. V10 제출
2. Deep Learning 등 근본적 재도전
3. 다른 방향 시도
