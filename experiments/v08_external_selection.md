# V08: Target-Specific External Feature Selection

## Method
- 9 external proxy features created from external data (sleep health lifestyle)
- Target-by-target optimal n_ext/n_total search: n_ext=0..8, n_total=10..25
- Ranked features by LGBM gain, external vs non-external split

## Results

### Q1 (deep)
- **Best**: n_ext=1, total=14, LL=0.65529
- **Baseline**: LL=0.66796
- **Delta**: **-0.01267** (improvement)
- **Best external feature**: `ext_night_light_zscore`

### Q2 (deep)
- **Best**: n_ext=1, total=23, LL=0.58886
- **Baseline**: LL=0.60607
- **Delta**: **-0.01722** (improvement)
- **Best external feature**: `ext_total_ambience_zscore`

## Key Findings
1. **1 external feature per target** is optimal — too many external features degrade performance
2. `ext_night_light_zscore` (light at night / night hours ratio) is most valuable for Q1
3. `ext_total_ambience_zscore` (total ambient noise) is most valuable for Q2
4. External features from sleep/health proxy improve LB generalization potential
