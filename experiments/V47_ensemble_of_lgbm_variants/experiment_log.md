# V47: Ensemble of LGBM Variants (Multi-Config Stacking)

## Hypothesis
Training 6 diverse LGBM configurations per target and combining predictions via meta-learner stacking should improve over any single model by leveraging diversity.

## Method
- 6 configs per target (varying depth, LR, regularization)
- 5 seeds each = 30 models per target
- Two strategies: (A) simple average, (B) LogisticRegression meta-learner
- 20 features per target (top LGBM importance)
- GroupKFold CV, 5 folds

## Results
- Q1: B Cal=0.6334
- Q2: B Cal=0.5996
- Q3: B Cal=0.6126
- S1: B Cal=0.5745
- S2: B Cal=0.6004
- S3: B Cal=0.6234
- S4: B Cal=0.6213

**V47 Avg Cal: 0.6093**

## Conclusion
Meta-learner overfits with 450 samples. V48 (isotonic calibration) is much better.
