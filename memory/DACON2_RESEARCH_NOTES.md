# V313 Research Notes

## Current Best: V312
- OOF: 0.61448
- Meta C=500 (V308: C=10)
- 15 seeds + z-score + stacking

## V312 Analysis
- C=500이 모든 타겟에서 optimal
- S3가 -0.03621로 가장 큰 개선
- Q3가 -0.02025로 두 번째 개선
- V308과 동일 아키텍처(C만 변경) → low-risk

## V312 OOF-LB Gap Prediction
- V308: OOF 0.62235 → LB 0.63893 (gap 0.01658)
- V312: OOF 0.61448 → 예상 LB 0.631 (gap 0.01658 유지 가정)
- C=500은 meta regularization이 약해지므로 gap이 커질 수 있음 (+0.005~0.01)
- 예상 LB range: 0.631~0.640

## Next Hypothesis Candidates

### H1: Pseudo-Labeling with V312's Stronger OOF
- V158/V161이 V146 기준에서 실패한 건 V146 meta output이 너무 conservative
- V312는 OOF가 더 낮으므로 더 넓은 prediction distribution
- threshold를 낮춰 (0.30~0.40) pseudo-labeling 시도
- Risk: Medium (V161 실패 이력 있음)

### H2: More Seeds (15→30)
- V160 발견: seeds 증가가 ensemble diversity 개선
- V312는 아직 15 seeds
- 30 seeds면 student model 140개 → meta에 더 풍부한 input
- Cost: 2x but OOF 개선 가능성 높음
- Risk: Low

### H3: Interaction Features
- V156에서 group features가 noise였지만, target-domain interaction features는?
- 예: (sleep features) × (activity features)
- Feature count 통제 필요 (과적합 risk)
- Risk: Medium-High

### H4: Target Correlation Stacking
- 7 targets가 correlated → joint learning 가능
- Multi-task learning 또는 sequential stacking
- Risk: High (architectural change)

### H5: Data Augmentation / Synthetic Samples
- SMOTE 또는 similar for minority class
- V308이 binary로 분류하므로 class imbalance handling
- Risk: Medium (feature distribution 변화)

## Priority Order
1. H2: More Seeds (15→30) — low-risk, proven by V160
2. H1: Pseudo-Labeling (V312 기준) — medium-risk but potentially high reward
3. H5: Data Augmentation — medium-risk
4. H3: Interaction Features — medium-high risk
5. H4: Target Correlation — high risk, save for later
