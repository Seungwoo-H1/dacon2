# DACon2 Autonomous Research — Master Experiment Queue

## Current Best: V127 Ensemble
- **LB: 0.65358** (actual)
- **OOF: 0.53731**
- **Structure:** 0.35×V121(pairwise+rank) + 0.25×V123(pairwise) + 0.40×V115(base+zscore)
- **Per-target configs:** Q1=deep/19, Q2=deep/14, Q3=v48/11, S1=wide/21, S2=deep/19, S3=safety/23, S4=wide/20
- **Pipeline:** features.parquet → z-score personalization → per-target ranking → top-N → 4 seeds × GroupKFold(5) → mean-match cal → blend

## Completed Experiments
| Ver | OOF | LB | Result |
|-----|-----|-----|--------|
| V127 | 0.53731 | 0.64763 | ⭐ BEST |
| V102 | ? | 0.61998 | LB=0.77, rejected |
| V128 | ? | ? | Group-wise LOO target encoding |
| V245 | 0.68-0.77 | - | MLP, rejected (overfit) |
| V246 | - | est 0.648 | TZ+V127 |
| V251 | 0.692 | - | External merge (worse) |
| V252 | 0.537 | - | Calendar FE (Δ=-0.003) |
| V253 | - | - | V09 personalized migration |
| V254 | 0.601-0.759 | - | Multi-target joint |
| V255 | ? | ? | Cross-target confirmatory |
| V09 | Δ=-0.023 | - | External proxy features |

## Active/Queued Experiments

### A. Group-wise LOO Target Encoding (V128)
**Hypothesis:** Group-wise leave-one-out target encoding per user behavioral pattern could capture non-linear user-target relationships without leakage.
**Status:** In progress (subagent)

### B. Diverse Feature Pipeline Ensemble
**Hypothesis:** Completely different feature engineering pipelines (statistical, temporal, cross-feature, rolling aggregates) trained separately and ensemble'd could find orthogonal signal.
**Status:** In progress (subagent)

### C. Multi-Target Joint Training (V254+)
**Hypothesis:** Leveraging inter-target correlations through cross-target features (leakage-free via OOF pseudo-targets) can improve prediction.
**Status:** In progress (subagent) — cross-target 5-seed ensemble found: OOF 0.564

### D. Advanced External Proxy Features (V09→V253→V256)
**Hypothesis:** Better external proxy features (circadian rhythm, entropy, routine regularity) can add signal not captured by current features.
**Status:** Queue — needs re-engineering

### E. Ensemble Architecture Search
**Hypothesis:** Bayesian weight optimization + feature-subspace diversity + rank averaging could squeeze 0.01-0.02 more from V127 ensemble.
**Status:** Queue

### F. Distribution Shift / Calibration
**Hypothesis:** Train/test PSI drift correction + quantile normalization + rank stabilization could improve LB generalization.
**Status:** Queue

### G. Advanced Feature Discovery
**Hypothesis:** FFT/cyclic decomposition of time series, frequency-domain features, clustering embeddings, anomaly scores could reveal hidden patterns.
**Status:** Queue

## Research Principles
1. No leakage — strict OOF evaluation
2. Hypothesis-driven — every experiment must have a "why"
3. Best model never overwritten
4. Record everything — version, features, seeds, OOF, LB estimate
5. LB generalization > OOF chasing
