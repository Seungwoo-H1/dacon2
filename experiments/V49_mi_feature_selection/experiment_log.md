# V49: MI-Based Feature Selection

## Hypothesis
Mutual Information captures non-linear feature-target relationships independent of the model. MI-based feature selection + correlation pruning might find better feature subsets than LGBM importance.

## Method
- 4 ranking strategies: LGBM importance, MI only, rank fusion, score fusion
- Each tested at n=5,10,15,20 features
- Correlation pruning (|corr|>0.95)
- n_neighbors=10 for MI computation
- GroupKFold CV, 5 folds

## Results

| Target | LGBM best | MI best | Rank Fusion best | Score Fusion best |
|--------|-----------|---------|-----------------|-------------------|
| Q1 | 0.6496 (n=15) | 0.6890 (n=20) | 0.6437 (n=5) | 0.6644 (n=15) |
| Q2 | **0.6153 (n=10)** | 0.6631 (n=20) | 0.6172 (n=15) | 0.6249 (n=10) |
| Q3 | **0.6052 (n=10)** | 0.6675 (n=20) | 0.6545 (n=5) | 0.6383 (n=10) |

- LGBM∩MI overlap (top20): Q1=4, Q2=4, Q3=2
- V49 Avg Cal: 0.6201 (vs V10 0.6038 → worse)

## Conclusion
MI feature selection is worse than LGBM importance for this dataset. MI and LGBM select very different features (low overlap), but MI-selected features don't help. Feature selection alone doesn't close the gap.
