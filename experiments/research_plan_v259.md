# Autonomous Research Plan — V259+

## Goal
Improve LB generalization beyond V127 (LB=0.64763)

## Research Pillars

### A. Distribution Analysis & Calibration
- PSI drift (train/test)
- Adversarial validation (train vs test)
- Fold-level drift analysis
- Target distribution calibration
- Quantile normalization
- Rank stabilization
- Temperature scaling (per-target)
- Local calibration

### B. Advanced Feature Discovery
- Feature interactions (auto-generated)
- Nonlinear transforms (log, sqrt, box-cox, rank)
- Target-conditioned statistics
- Frequency-domain (FFT, cyclic decomposition)
- Clustering embeddings
- Anomaly scores
- Reconstruction error (autoencoder)

### C. Ensemble & Model Architecture
- V127 true 3-model ensemble reimplementation
- Bayesian weight optimization
- Feature-subspace diversity
- Rank averaging ensemble

### D. External Data Integration
- Circadian rhythm features
- Entropy / routine regularity
- Mobility priors
- Sleep/activity heuristics
- Public temporal behavior distributions

## Research Principles
1. No leakage — strict OOF evaluation
2. Hypothesis-driven
3. Best model never overwritten
4. Record everything
5. LB generalization > OOF chasing
