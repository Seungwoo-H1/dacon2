# V08: External Feature Selection Results

## Method
- External proxy features created from sleep_health_lifestyle.csv
- 9 proxy features: ext_activity_z, ext_charging_z, ext_health_composite, ext_night_light, ext_total_ambience, ext_hr_step, ext_screen_ratio, ext_wifi_ble, ext_activity_ambience, ext_step_consistency
- Target-by-target n_ext (0-8) × n_total (10-25) search

## Results

### Q1 (V53_SWEEP=deep)
- **Best**: n_ext=1, n_total=14, LL=0.65529
- **Baseline**: 0.66796
- **Δ**: -0.01267 ✓
- **Best external feature**: `ext_night_light_zscore`

### Q2 (V53_SWEEP=deep)
- **Best**: n_ext=1, n_total=23, LL=0.58886
- **Baseline**: 0.60607
- **Δ**: -0.01722 ✓
- **Best external feature**: `ext_total_ambience_zscore`

## Key Findings
1. External features improve both Q1 and Q2
2. Only 1 external feature needed per target
3. ext_night_light (night light/night hours ratio) is most valuable for Q1
4. ext_total_ambience (ambient noise) is most valuable for Q2
