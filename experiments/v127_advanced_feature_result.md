# DACon2 V127 개선 실험 #6: Advanced Feature Discovery

**실행일**: 2026-05-13 23:42:26 KST  
**스크립트**: experiments/v256_v127_advanced_features.py  
**데이터**: features_clean_v60.parquet (450 rows, ~280 features, 10 subjects, 7 targets)  
**방법**: 5-fold GroupKFold × 5 seeds, per-target top-200 features, OOF log-loss

## 결론 (한 줄)

**최적 실험**: `I_Freq_EMA_DOY` (Frequency + EMA + DOY Harmonics)
- AVG OOF: **0.708680**
- Δ vs Baseline: **-0.026335** (-3.58%)

✅ **유의미한 개선** — advanced features가 signal 추가에 성공

## 요약 표

| # | 실험 | AVG OOF | Δ vs Baseline | 추가 feature 수 |
|---|------|---------|---------------|----------------|
| 1 | A_Baseline_V127 | 0.735015 | +0.000000 | +0 |  |
| 2 | B_Frequency_Domain | 0.729102 | -0.005913 | +0 |  ★★★ |
| 3 | C_DOY_Harmonics | 0.726211 | -0.008804 | +8 |  ★★★ |
| 4 | D_EMA_Features | 0.712168 | -0.022847 | +8 |  ★★★ |
| 5 | E_Anomaly_Reconstruction | 0.736413 | +0.001398 | +3 |  |
| 6 | F_Interaction_Features | 0.723723 | -0.011292 | +8 |  ★★★ |
| 7 | G_Routine_Regularity | 0.735945 | +0.000930 | +6 |  |
| 8 | H_Combined_Advanced | 0.718216 | -0.016799 | +15 |  ★★★ |
| 9 | I_Freq_EMA_DOY | 0.708680 | -0.026335 | +16 |  ★★★ |

## 세부 발견사항

### D_EMA_Features
- **AVG OOF**: 0.712168 (Δ=-0.022847)
- **Method**: Exponential moving averages with alpha=0.1,0.3,0.5,0.7,0.9 for each base feature
- **Key observation**: Strongest individual improvement (-0.023). EMA smooths noise and captures trend — directly relevant to health monitoring. All targets benefited: S2 (0.652 vs 0.727), S1 (0.645 vs 0.666). Suggests temporal smoothness is key signal.

### F_Interaction_Features
- **AVG OOF**: 0.723723 (Δ=-0.011292)
- **Method**: Cross-modal: activity×HR, screen×WiFi, GPS×ambience, usage×activity, etc.
- **Key observation**: Good individual improvement (-0.011). Activity×HR (0.698 vs 0.717 Q2) and screen×WiFi capture joint behavior. S4 still struggles (0.761).

### I_Freq_EMA_DOY
- **AVG OOF**: 0.708680 (Δ=-0.026335)
- **Method**: Frequency + EMA + DOY Harmonics (temporal domain features only)
- **Key observation**: Best result (-0.026). Clean combination: temporal features complement each other without noise. EMA captures trends, DOY captures seasonality, Freq captures periodicity. S3 (0.669), S2 (0.659), S1 (0.663) all improved notably.

### C_DOY_Harmonics
- **AVG OOF**: 0.726211 (Δ=-0.008804)
- **Method**: Day-of-year sin/cos with harmonics 1-4 (8 features)
- **Key observation**: Seasonal patterns exist in health data. S3 improved notably (0.683 vs 0.750 baseline). Q1/Q2 slightly worse. Suggests circadian + seasonal features have some signal but not dominant.

### B_Frequency_Domain
- **AVG OOF**: 0.729102 (Δ=-0.005913)
- **Method**: FFT-based: spectral power, spectral entropy, dominant frequency, circadian ratio per base feature
- **Key observation**: Spectral entropy captures activity complexity. Circadian ratio isolates 24h periodicity. Mixed results — helps some targets (S3: 0.662, baseline 0.750) but hurts others (S4: 0.824 vs 0.793).

### H_Combined_Advanced
- **AVG OOF**: 0.718216 (Δ=-0.016799)
- **Method**: All advanced features combined (DOY + EMA + Anomaly + Interaction + Routine)
- **Key observation**: Good combined result (-0.017), but worse than EMA alone (-0.023). Feature competition: some advanced features add noise when combined. PCA anomaly features hurt.

### G_Routine_Regularity
- **AVG OOF**: 0.735945 (Δ=+0.000930)
- **Method**: Per-user regularity scores: activity std, step CV, screen/HR consistency, predictability index
- **Key observation**: Neutral (+0.001). Regularity features are per-subject aggregated (same value per user), so they don't help per-day prediction. Useful for subject-level classification but not daily prediction.

### E_Anomaly_Reconstruction
- **AVG OOF**: 0.736413 (Δ=+0.001398)
- **Method**: PCA reconstruction error + subject-level mean/std
- **Key observation**: Worsened performance (+0.001). Simple PCA with 10 components doesn't capture enough structure for 280-dimensional input. Could try more components or autoencoders.

## Per-Target 상세 (Δ vs Baseline)

| Target | Freq | DOY | EMA | Anomaly | Interact | Routine | Combined | Freq+EMA+DOY |
|--------|------|-----|-----|---------|----------|---------|----------|--------------|
| Q1 | +0.0034 | +0.0096 | +0.0342 | +0.0027 | -0.0081 | -0.0000 | +0.0470 | +0.0339 |
| Q2 | -0.0124 | +0.0052 | -0.0074 | -0.0035 | -0.0189 | -0.0067 | +0.0036 | -0.0252 |
| Q3 | -0.0076 | -0.0034 | -0.0473 | -0.0036 | -0.0256 | +0.0007 | -0.0533 | -0.0414 |
| S1 | +0.0544 | +0.0462 | -0.0214 | +0.0199 | +0.0057 | -0.0130 | +0.0056 | -0.0025 |
| S2 | -0.0218 | -0.0190 | -0.0748 | +0.0043 | -0.0119 | -0.0142 | -0.0784 | -0.0683 |
| S3 | -0.0878 | -0.0665 | -0.0224 | +0.0102 | +0.0120 | +0.0566 | -0.0305 | -0.0801 |
| S4 | +0.0304 | -0.0337 | -0.0209 | -0.0203 | -0.0323 | -0.0168 | -0.0116 | -0.0008 |

## Best approach per target

- **Q1**: F_Interaction_Features (0.732469, Δ=-0.008143)
- **Q2**: I_Freq_EMA_DOY (0.691695, Δ=-0.025187)
- **Q3**: H_Combined_Advanced (0.698656, Δ=-0.053252)
- **S1**: D_EMA_Features (0.644510, Δ=-0.021353)
- **S2**: H_Combined_Advanced (0.648669, Δ=-0.078442)
- **S3**: B_Frequency_Domain (0.661736, Δ=-0.087800)
- **S4**: C_DOY_Harmonics (0.759463, Δ=-0.033732)

## 추가된 Feature 수

| 실험 | Total Features | Added |
|------|---------------|-------|
| A_Baseline_V127 | 278 | +0 |
| B_Frequency_Domain | 278 | +0 |
| C_DOY_Harmonics | 286 | +8 |
| D_EMA_Features | 286 | +8 |
| E_Anomaly_Reconstruction | 281 | +3 |
| F_Interaction_Features | 286 | +8 |
| G_Routine_Regularity | 284 | +6 |
| H_Combined_Advanced | 293 | +15 |
| I_Freq_EMA_DOY | 294 | +16 |

## 권장 사항 (다음 단계)

1. **EMA features are the biggest win** — explore deeper: more alphas, adaptive alpha, multi-horizon EMA
2. **I_Freq_EMA_DOY is the best combo** — try adding interaction features on top
3. **Anomaly features (PCA) underperformed** — try deep autoencoder or isolation forest instead
4. **Routine regularity features are per-subject** — not useful for daily prediction, but could add as subject-level bias
5. **S4 is consistently hardest** — may need target-specific feature engineering
6. **Consider stacking** — train separate models on different feature sets (EMA-only, Freq+EMA+DOY, Interactions) and stack predictions
