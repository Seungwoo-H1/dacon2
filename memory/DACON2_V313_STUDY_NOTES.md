# V313 Study Notes

## V308 Baseline
- OOF: 0.62235 | LB: 0.63893
- 15 seeds + z-score + stacking + C=10
- OOF-LB gap: 0.01658

## V312 Status
- OOF: 0.61448 | LB: **미검증**
- Meta C=500 (all targets)
- Expected LB: 0.631~0.640 (gap 동일 가정)
- **V312가 V308 LB를 넘으면 연구 종료**

## V313 Hypothesis Priority

### H1: More Seeds (15→30)
- V160 발견: seeds 증가 → ensemble diversity ↑ → OOF ↓
- V312 기준 30 seeds → student model 140개
- Cost: 2x but proven effective
- Risk: Low (same architecture, only seed count)
- Expected OOF: 0.610-0.612 (Δ -0.004~0.008 vs V312)

### H2: Pseudo-Labeling with V312
- V158/V161은 V146 기준에서 실패 (meta too conservative)
- V312는 stronger predictions → pseudo-labeling 더 가능
- Threshold: 0.30-0.40 (V158의 0.55보다 낮춤)
- Risk: Medium (V161 실패 이력)

### H3: Data Augmentation
- Class imbalance handling
- SMOTE / ADASYN for minority class
- Risk: Medium

## Action Plan
1. Wait for V312 LB result
2. If V312 LB > V308 LB → research complete
3. If not → V313 H1 (More Seeds)
