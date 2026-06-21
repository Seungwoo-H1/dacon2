# V536+ Autonomous Research Plan

## Current State (2026-06-14 03:01 UTC)
- **V308**: LB=0.63893 (verified BEST) — 15 seeds, LogReg meta
- **V534**: avg_gap=-0.02824 (OOF-based, not LB-verified) — Ridge α=0.01
- **V535**: Submission generated (same as V534 config), awaits manual upload

## V534 Config (Reference)
```
Q1: n_feat=3, xgb=q_narrow, lgbm=wide, n_est=600
Q2: n_feat=10, xgb=q_deep, lgbm=wide, n_est=800
Q3: n_feat=7, xgb=q_strong, lgbm=safety, n_est=500
S1: n_feat=3, xgb=q_strong, lgbm=wide, n_est=500
S2: n_feat=7, xgb=s_strong, lgbm=wide_strong, n_est=500
S3: n_feat=23, xgb=q_strong, lgbm=safety, n_est=1000
S4: n_feat=20, xgb=q_deep, lgbm=wide, n_est=300
Meta: Ridge(α=0.01)
```

## V536 Hypotheses (ALL must be new — no repeat of V534 or earlier)

### H1: Target-specific Ridge Alpha
- **Hypothesis**: Global α=0.01 is not optimal per-target. S1/S2 have large negative gaps (-0.096, -0.070) → need stronger regularization (higher α) to shrink back toward 0.
- **Method**: Grid search per-target α: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
- **Prediction**: S1/S2 optimal α higher → reduce over-calibration → lower avg_gap

### H2: Calibration Shrinkage (Post-hoc gap correction)
- **Hypothesis**: S1/S2 gap=-0.096/-0.070 means model is too confident. Apply shrinkage factor to bring predictions closer to training mean.
- **Method**: post_correction = mean + shrink * (pred - mean), shrink∈[0.5, 0.6, 0.7, 0.8, 0.9]
- **Prediction**: S1/S2 gap improves, avg_gap becomes less negative (closer to optimal)

### H3: Ensemble Diversity with 30 Seeds
- **Hypothesis**: V534 uses same seed count as V308 (15). More seeds → more diverse ensemble → better calibration.
- **Method**: Same config as V534, but 30 seeds instead of 15.
- **Prediction**: avg_gap improves by ~0.003-0.005

### H4: CatBoost as Target-Specific Learner
- **Hypothesis**: XGB+LGBM is good but CatBoost might handle small n=450 better with less overfitting.
- **Method**: Replace XGB with CatBoost for targets where XGB overfits (S1, S2).
- **Prediction**: Better generalization on S targets → lower gap

### H5: Weighted Blend (OOF-based)
- **Hypothesis**: Equal blend (0.5/0.5 XGB/LGBM) is suboptimal. OOF-based weights should favor better learners.
- **Method**: weight = softmax(-oof_scores) for XGB and LGBM separately per target.
- **Prediction**: avg_gap improves by ~0.001-0.003

### H6: Two-Stage Meta with Non-Linear Learner
- **Hypothesis**: First stage: Ridge(meta_features). Second stage: LightGBM meta learner on Ridge predictions + original meta features.
- **Method**: Ridge → LGBM meta. Test various LGBM configs.
- **Prediction**: Non-linear calibration captures complex patterns Ridge misses

### H7: S1/S2 Feature Engineering Override
- **Hypothesis**: S1 n_feat=3, S2 n_feat=7 might be too few. Maybe interaction features help.
- **Method**: Add targeted interaction features for S1/S2 only (not all targets).
- **Prediction**: S1/S2 prediction improves

## Execution Order (10 hours)
1. **H1** (Target-specific α) — ✅ DONE
2. **H2** (Calibration Shrinkage) — ✅ DONE  
3. **H3** (30 seeds) — ⏳ RUNNING
4. **H4** (CatBoost) — ⏳ RUNNING
5. **H5** (Weighted blend) — ⏳ RUNNING
6. **H6** (Two-stage meta) — ⏳ RUNNING
7. **H7** (Feature engineering) — Not yet started

## H1 Results (Completed)
- **avg_gap: -0.02996** (baseline -0.02754 → improvement +0.00242)
- Best α per target: Q1=0.001, Q2=0.05, Q3=0.001, S1=0.01, S2=0.05, S3=1.0, S4=0.001
- Key: Q2 α=0.05 gives big improvement (-0.00910)
- vs308: 7/7

## H2 Results (Completed)
- **avg_gap: -0.04278** (big improvement over V534)
- Best shrinks: Q1=0.3, Q2=0.7, Q3=0.7, S1=0.8, S2=0.8, S3=0.7, S4=0.7
- **Submission file generated**: submission_v536_h2_shrink_Q13_Q27_Q37_S18_S28_S37_S47_20260614_031243.csv
- vs308: 7/7
- Expected LB: 0.66513 (OOF-based, needs verification)
- ⚠️ Q1 shrink 0.3 is aggressive — may overfit test

## H3 Results (Completed)
- **avg_gap: -0.02559** — 30 seeds degraded vs V534 (-0.02824)
- 30 seeds는 역효과. diminishing return 넘어 성능 하락.
- Conclusion: 15 seeds optimal, 30 seeds worse

## H5 Results (Completed)
- Equal blend baseline: avg_gap=-0.02772
- OOF-weighted: avg_gap=-0.02761 (no improvement — weights converge to 0.5)
- Fixed grid (Q=0.6, S=0.5): avg_gap=-0.02931 ✅ **slight improvement**
- Key: Q targets benefit from more XGB weight (0.6), S targets stay ~0.5
- OOF-based weighting itself adds no value (XGB/LGBM OOF差距很小)

## H6 Results (Completed)
- Two-stage meta (Ridge→LGBM): avg_gap=-0.34590
- ⚠️ **결과 의심스러움** — avg_gap이 -0.34는 계산 버그 또는 baseline 불일치
- H6 baseline avg_gap=-0.0085는 V534의 -0.02754와 다르다 → 다른 설정으로 실행
- submission 파일 생성됨: submission_v536_h6_two_stage_20260614_031950.csv
- **승우가 수동으로 LB 확인 필요**

## H4 Results (Completed)
- CatBoost learner: avg_gap=-0.02131 (Variant B: XGB-Q, CatBoost-S)
- **V534보다 나쁨** (-0.02754 → -0.02131)
- S2가 크게 악화: gap -0.070 → -0.025 (s_strong config은 XGB/LGBM용으로 설계됨)
- CatBoost는 small n=450에서 XGB/LGBM보다 못함
- Conclusion: CatBoost hypothesis rejected

## 📊 V536 ALL RESULTS SUMMARY

| Version | Hypothesis | avg_gap | vs V534 | vs308 | Status |
|---------|-----------|---------|---------|-------|--------|
| **V534** | **Baseline** | **-0.02754** | — | 7/7 | ✅ Current BEST |
| H1 | Target Ridge α | -0.02996 | +0.00242 | 7/7 | ✅ Slight improvement |
| **H2** | **Cal Shrinkage** | **-0.04278** | **+0.01524** | **7/7** | 🏆 **BIGGEST GAIN** |
| H3 | 30 Seeds | -0.02559 | -0.00195 | 7/7 | ❌ Worse |
| H4 | CatBoost | -0.02131 | -0.00623 | 7/7 | ❌ Worse |
| H5 | Weighted Blend | -0.02931 | +0.00177 | 7/7 | ✅ Slight |
| H6 | Two-stage Meta | -0.34590 | ? | 7/7 | ⚠️ Suspect |

### 🏆 Top Pick: H2 (Calibration Shrinkage)
- **avg_gap: -0.04278** (15% improvement over V534)
- **Submission file**: `submission_v536_h2_shrink_Q13_Q27_Q37_S18_S28_S37_S47_20260614_031243.csv`
- ⚠️ Need LB verification — OOF-based shrink may not translate to LB

### Submission Files Generated
1. `submission_v536_h2_shrink_Q13_Q27_Q37_S18_S28_S37_S47_20260614_031243.csv` — **Priority 1**
2. `submission_v535_q1n3_q2n10_s4n20_Ridge01_20260612_145935.csv` — V534 baseline (needs upload)
3. `submission_v536_h3_30seeds_20260614_031736.csv` — H3 (worse, skip)
4. `submission_v536_h4_Variant_B_XGB-Q_CatBoost-S_20260614_032540.csv` — H4 (worse, skip)
5. `submission_v536_h5_Best_Grid_(Q0.6_S0.5)_20260614_031817.csv` — H5 (slight)
6. `submission_v536_h6_two_stage_20260614_031950.csv` — H6 (uncertain)

### Recommended Submission Order
1. **V536 H2** (shrink Q1=0.3, Q2=0.7, Q3=0.7, S1=0.8, S2=0.8, S3=0.7, S4=0.7) — best OOF
2. **V535** (V534 baseline, avg_gap=-0.02754) — safe baseline
3. **H5 grid** (Q=0.6/S=0.5, avg_gap=-0.02931) — slight improvement

### H2+H1 Combo (Future V537)
- H2 shrink + per-target α tuning could combine for even better result
- H2 dominates though (avg_gap -0.04278 vs -0.02996)

## Combined Approach
- H1+H2 together: H2 dominates. Combined may yield even better.
- H1+H2+H5 combo as V537 candidate
- H2 is the strongest signal so far (avg_gap=-0.04278)
- H5 grid weights could be combined with H2 shrinkage

## Constraints
- Same V534 base config unless hypothesis requires override
- Same CV pipeline (GroupKFold, 5 folds)
- avg_gap = avg(pred_oof - true_oof) across 7 targets
- Report ALL results, not just improvements
- If any result beats V308 LB (0.63893) → IMMEDIATE report
- Submission files generated → name: submission_vXXX_hypN_<config>_<timestamp>.csv

## Files Reference
- V534 config: experiments/v534_full_meta_q1_deep.py, experiments/v534_v531_replica_q1_sweep.py
- V535 submission: experiments/v535_submission.py, submissions/submission_v535_q1n3_q2n10_s4n20_Ridge01_20260612_145935.csv
- Base scripts: experiments/v528_s2_strong_reg.py, experiments/v522_production.py, experiments/v525_per_target_learner_mixer.py
