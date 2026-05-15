# V127 External Data Research Progress

## V127 Baseline (Reproduced)
- AVG OOF: **0.61279**
- Q1: 0.649, Q2: 0.598, Q3: 0.611, S1: 0.579, S2: 0.597, S3: 0.634, S4: 0.621
- Best n_feat: Q1=15, Q2=15, Q3=10, S1=10, S2=15, S3=15, S4=10

## V07: External Proxy Features
- Added 9 per-subject proxy features from external knowledge
- External features rank in top-30 for most targets
- top15 with external: AVG OOF = **0.60291** (Δ = -0.00988)
- top30 with external: AVG OOF = 0.62077 (worse)
- **Key finding**: External features help at n=15 but hurt at higher n

### External Feature Rankings
- Q1: ext_night_light (#0), ext_health_composite (#17)
- Q2: ext_charging_z (#11), ext_total_ambience (#8), ext_screen_ratio (#10)
- Q3: ext_wifi_ble (#5)
- S2: ext_night_light (#0), ext_total_ambience (#1), ext_health_composite (#5)
- S3: ext_night_light (#0)
- S4: ext_night_light (#1), ext_activity_z (#20)

## V07 Ensemble
- Q1: Δ=-0.015 (ensemble_w0.7)
- Q3: Δ=-0.030 (ensemble_w0.7)
- S1: Δ=-0.013 (ensemble_w0.7)

## Next Steps
- [ ] V08: Target-specific external feature selection
- [ ] V09: Pseudo-labeling with external distribution matching
- [ ] V10: Domain adaptation via adversarial validation
- [ ] V11: Pretrain on external → Finetune on internal
