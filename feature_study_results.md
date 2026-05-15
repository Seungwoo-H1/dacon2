# Dacon2 Feature Engineering + External Data Analysis Study

## Executive Summary

This study analyzes the current V8 (0.6537 log-loss) and V10 (0.6607 log-loss) pipelines for the ETRI Human Understanding AI 5th competition. Key finding: **`week_of_year` as a cyclical feature (sin/cos) is the strongest single predictor** of sleep quality labels — especially S3 (corr=0.31) and S2 (corr=0.19). The competition data (Jun–Nov 2024) shows strong seasonal patterns in self-reported metrics, suggesting that **external data (weather, calendar events) could be highly valuable**. V10's careful approach (leakage fix + mean-matching calibration) is a solid foundation; the biggest gains will come from **better temporal features and external data integration**.

---

## 1. Current V8/V10 Analysis

### V8 (Best: 0.6537) — Strengths
- **Simple mean-match calibration** — Avoided the calibration trap that killed earlier versions
- **Leakage fix** — Removed nighttime wrist data (wLight/wHr/wPedo) from S targets
- **Time-based aggregation** — 1h/3h/6h/12h/24h windows across all 12 data sources
- **JSON parsing** — Probabilistic scores from ambience, BLE, GPS, WiFi, usage stats

### V8/V10 — Weaknesses / Missing Elements
- **No temporal/cyclical features** — No `dow_sin/cos`, `hour_sin/cos`, `week_of_year`, `is_weekend`, `day_of_year` in the base feature set
- **No rhythm/variability features** — No rolling std, activity regularity, circadian consistency
- **No external data** — Weather, holidays, calendar events completely absent
- **Per-subject z-score helped less than expected** — The extended experiments (V11) show it adds noise
- **Cross-target features** — 987 `_cross` features in extended set multiply features by target values → **severe leakage** (correlation with targets up to 0.43)

### V10 vs V8 Comparison
- V10 added: per-target top-20 feature selection, per-subject z-score, 20-seed ensemble
- V10 scored **higher (worse)** log-loss: 0.6607 vs 0.6537
- The additional complexity (z-score, feature selection, hyperparameter tuning) introduced variance without improving generalization
- **V8's simplicity wins** — but V10's leakage fix and ensemble strategy are sound

### Extended Features Analysis (08_v11_extended_features.py)
The extended features file (`features_extended.parquet`) contains **2,691 extra columns** vs 149 in the base set. Breakdown:

| Feature Type | Count | Notes |
|---|---|---|
| `_cross` (target leakage) | 987 | Multiply features by target → **DO NOT USE** |
| `_lag` (lagged features) | 423 | ~5% null rate, moderate nulls |
| Rolling mean (2d/3d/5d) | 423 | Non-zero, could be useful |
| Rolling std (2d/3d/5d) | 423 | Non-zero, could be useful |
| Temporal (dow, month, etc.) | 141 | **Missing from base pipeline!** |
| Subject mean/std | 141 | ~20% null rate, personalization |

---

## 2. Key Data Insights

### 2.1 Strong Seasonal Patterns in Targets

The targets show clear time-of-year trends, strongest in sleep metrics (S1-S4):

**Q1 (Sleep Quality)** — Strongly correlated with `week_of_year` (corr=0.18):
- Month 6: 34.5% → Month 9: 59.0% → Month 10: 77.8% (rapid improvement in autumn)
- Week 23: 0% → Week 40: 100% → Week 43: 87.5% (dramatic seasonal swing)

**S3 (Sleep Onset Latency)** — Strongest seasonal signal (corr=0.31 with `week_of_year`):
- Month 6: 87.9% → Month 10: 25.9% → Month 11: 28.6%
- Weeks 23-35: ~70-100% → Weeks 36-43: ~25-62%
- This is a ~3x difference — massive signal for model

**S2 (Sleep Efficiency)** — corr=-0.22 with month:
- Month 6: 84.5% → Month 9: 48.7% → Month 10: 59.3%

**Q2 (Fatigue)** — Strong weekend effect:
- Weekday: 53.0% vs Weekend: 63.7%

### 2.2 Per-Subject Target Rate Variability

Subject rates vary dramatically:
- **Q1**: id06=14.6% vs id03=84.8% (5.8x range)
- **S1**: id05=47.7% vs id06=93.8% (2x range)
- **S2**: id05=25.0% vs id02=91.7% (3.7x range)
- **S4**: id03=15.2% vs id02=75.0% (4.9x range)

This confirms **strong per-subject heterogeneity** — subject-level features are important.

### 2.3 Top Correlated Base Features (from 149 base features)

| Target | Top Feature | |corr| |
|--------|------------|-------|
| Q1 | `mActivity_m_activity_mean` | 0.110 |
| Q1 | `wLight_w_light_count` | 0.107 |
| S1 | `mScreenStatus_m_screen_use_mean` | 0.230 |
| S1 | `mScreenStatus_m_screen_use_std` | 0.212 |
| S2 | `mScreenStatus_m_screen_use_mean` | 0.166 |
| S3 | `mScreenStatus_m_screen_use_mean` | 0.158 |
| S4 | `mScreenStatus_m_screen_use_mean` | 0.164 |

**mScreenStatus** (screen usage) is the strongest base predictor across ALL S targets.

---

## 3. Feature Engineering Proposals (Prioritized)

### Priority 1: Temporal/Cyclical Features (HIGH IMPACT, NO RISK)

**Rationale**: `week_of_year` has corr up to 0.31 with S3. These features are cheap, zero-leakage, and directly address the strongest signal in the data.

```python
# Add to feature engineering pipeline (02_feature_engineering)
feat['date_dt'] = pd.to_datetime(feat['lifelog_date'])
feat['dow'] = feat['date_dt'].dt.dayofweek
feat['is_weekend'] = (feat['dow'] >= 5).astype(float)
feat['dow_sin'] = np.sin(2 * np.pi * feat['dow'] / 7)
feat['dow_cos'] = np.cos(2 * np.pi * feat['dow'] / 7)
feat['week_of_year'] = feat['date_dt'].dt.isocalendar().week.astype(float)
feat['week_sin'] = np.sin(2 * np.pi * feat['week_of_year'] / 52)
feat['week_cos'] = np.cos(2 * np.pi * feat['week_of_year'] / 52)
feat['month_sin'] = np.sin(2 * np.pi * feat['date_dt'].dt.month / 12)
feat['month_cos'] = np.cos(2 * np.pi * feat['date_dt'].dt.month / 12)
feat['doy'] = feat['date_dt'].dt.dayofyear
feat['doy_sin'] = np.sin(2 * np.pi * feat['doy'] / 365)
feat['doy_cos'] = np.cos(2 * np.pi * feat['doy'] / 365)
```

**Why cyclical encoding?** Linear `week_of_year` assumes monotonic trend; sin/cos captures periodicity without directional bias.

**Expected impact**: +0.02-0.05 log-loss improvement (based on corr=0.31 signal for S3)

### Priority 2: Rolling Statistics (Modest Impact)

**Rationale**: Rolling std captures activity variability, which correlates with sleep quality. The extended features show these exist but weren't used.

```python
# Per-subject rolling features (leakage-safe: uses previous days only)
for sid in feat['subject_id'].unique():
    mask = feat['subject_id'] == sid
    sub = feat.loc[mask].sort_values('lifelog_date')
    
    for col in activity_cols:  # mActivity, wPedo, wLight features
        vals = sub[col].ffill().fillna(0)
        # Rolling std over 3-day window (captures variability)
        rolling_std = vals.rolling(window=3, min_periods=2).std().fillna(0)
        feat.loc[sub.index, f'{col}_roll_std_3d'] = rolling_std.values
```

**Key features to add rolling stats for**:
- `mActivity_*` (activity variability)
- `wPedo_*` (step variability)
- `wLight_*` (light exposure variability)
- `mScreenStatus_*` (screen usage consistency)
- `mACStatus_*` (charging pattern consistency)

**Expected impact**: +0.01-0.03 log-loss improvement

### Priority 3: Rhythm/Consistency Features (Moderate Impact)

**Rationale**: Circadian rhythm regularity is strongly linked to sleep quality. The extended features V11 had rhythm features but they weren't thoroughly tested.

```python
# Activity rhythm: rolling std of daily activity mean
for sid in subjects:
    sub = feat[feat['subject_id']==sid].sort_values('lifelog_date')
    
    # Daily activity variability (3-day window)
    activity_mean = sub['mActivity_m_activity_mean'].ffill().fillna(0)
    for w in [3, 5]:
        rhythm = activity_mean.rolling(window=w, min_periods=2).std().fillna(0)
        feat.loc[sub.index, f'mActivity_rhythm_{w}d'] = rhythm.values
    
    # Step count variability
    steps = sub['wPedo_pedo_step_mean'].ffill().fillna(0)
    step_rhythm = steps.rolling(window=5, min_periods=2).std().fillna(0)
    feat.loc[sub.index, 'wPedo_step_rhythm_5d'] = step_rhythm.values
```

**Expected impact**: +0.01-0.02 log-loss improvement

### Priority 4: Per-Subject Historical Context (Moderate Impact)

**Rationale**: Subject mean/std features in the extended set are useful (they vary per-subject and capture personalization). The key is to use them correctly.

```python
# Per-subject global mean/std (computed from ALL data for each subject)
# This is safe because it's a subject-level constant, not target-dependent
for col in base_features:
    subj_mean = feat.groupby('subject_id')[col].transform('mean')
    subj_std = feat.groupby('subject_id')[col].transform('std').fillna(1)
    feat[f'{col}_subj_ratio'] = (feat[col] - subj_mean) / (subj_std + 1e-8)
    # Or use the raw subject-level stats as features
    feat[f'{col}_subj_mean'] = subj_mean
```

**Expected impact**: +0.01-0.02 log-loss improvement

### Priority 5: Weekend vs Weekday Interaction Features (Easy Win)

**Rationale**: Strong weekend effects observed in Q2 (53% vs 64%) and Q3 (56% vs 69%).

```python
# Weekend interactions with existing features
for col in ['mActivity_m_activity_mean', 'mScreenStatus_m_screen_use_mean', 'wPedo_pedo_step_mean']:
    feat[f'{col}_is_weekend'] = feat[col] * feat['is_weekend']
    feat[f'{col}_weekday'] = feat[col] * (1 - feat['is_weekend'])
```

**Expected impact**: +0.01 log-loss improvement

### Priority 6: Feature Interaction Enhancements

**Rationale**: V10 selects top-20 features per target. More diverse interactions could help.

```python
# Cross-domain interactions (safe, no leakage)
# Activity × Screen
feat['activity_screen_interaction'] = feat['mActivity_m_activity_mean'] * feat['mScreenStatus_m_screen_use_mean']

# Charging × Night screen usage  
feat['charging_night_screen'] = feat['mACStatus_m_charging_count'] * feat['mScreenStatus_hour_night']

# Light × Weekend
feat['light_weekend'] = feat['wLight_w_light_mean'] * feat['is_weekend']
```

**Expected impact**: +0.01-0.02 log-loss improvement

---

## 4. External Data Opportunities

### 4.1 Weather Data (HIGH FEASIBILITY, HIGH POTENTIAL)

**Why it matters**: Temperature, humidity, and air pressure are well-documented sleep disruptors. The Korean peninsula shows strong seasonal weather variations in June-November.

**Data sources**:
- **Korea Meteorological Administration (KMA)**: `http://www.kma.go.kr` — Historical weather data for Daejeon (where ETRI is located)
- **OpenWeatherMap Historical API**: Free tier available
- **Meteostat**: Historical weather for Seoul/Daejeon area

**Features to extract**:
```python
# Daily weather features
df['max_temp'] = weather['t2m_max'].values  # Maximum temperature
df['min_temp'] = weather['t2m_min'].values  # Minimum temperature
df['avg_temp'] = weather['t2m'].values      # Average temperature
df['humidity'] = weather['r2m'].values       # Relative humidity
df['precipitation'] = weather['prcp'].values # Precipitation amount
df['pressure'] = weather['msl'].values       # Mean sea-level pressure
df['wind_speed'] = weather['ws2m'].values    # Wind speed

# Weather-derived features
df['temp_range'] = df['max_temp'] - df['min_temp']
df['temp_comfort'] = -((df['avg_temp'] - 20)**2)  # Comfort is centered at 20°C
df['heat_index'] = ...  # Heat index calculation
```

**Feasibility**: ⭐⭐⭐⭐⭐ Very high. Daejeon weather data is readily available from KMA or Meteostat.

**Expected impact**: +0.02-0.05 log-loss. Strong correlation expected given the seasonal patterns already observed.

### 4.2 Calendar/Event Data (HIGH FEASIBILITY, MODERATE POTENTIAL)

**Why it matters**: Holidays, school terms, and work patterns significantly affect sleep schedules.

**Features to extract**:
```python
# Korean public holidays
import holidays
ko_holidays = holidays.KR(years=range(2024, 2025))
df['is_holiday'] = df['lifelog_date'].isin(ko_holidays)

# School term approximation (based on Korean calendar)
df['is_school_term'] = True  # July-Nov is school term
df['is_exam_period'] = (df['lifelog_date'].dt.month == 6) | (df['lifelog_date'].dt.month == 7)

# Lunar holidays (Korean new year, Chuseok)
# These have significant impact on sleep patterns in Korea
```

**Feasibility**: ⭐⭐⭐⭐⭐ Very high. Korean holidays are publicly available.

**Expected impact**: +0.01-0.03 log-loss

### 4.3 Geolocation Features (MODERATE FEASIBILITY, LOW-MODERATE POTENTIAL)

**Why it matters**: GPS data is already available (relative coordinates). Can derive timezone-adjusted features.

**Features to extract**:
```python
# From existing mGps data
# Latitude/longitude (relative, but still useful for patterns)
df['latitude'] = mGps['latitude'].values
df['longitude'] = mGps['longitude'].values

# Derived features
df['travel_distance'] = geodesic.distance(lat1, lon1, lat2, lon2)
df['is_traveled'] = df['travel_distance'] > threshold
df['is_stationary'] = df['speed'] < threshold
```

**Feasibility**: ⭐⭐⭐⭐ Moderate. GPS data already exists; derive features from it.

**Expected impact**: +0.01-0.02 log-loss

### 4.4 Air Quality Data (LOW FEASIBILITY, LOW-POTENTIAL)

**Why it matters**: Air quality (PM2.5, PM10) affects sleep quality, especially in Korea.

**Data sources**:
- **AirKorea**: `https://www.airkorea.or.kr` — Free historical air quality data
- **WAQI**: World Air Quality Index

**Expected impact**: +0.00-0.02 log-loss. Likely weak signal given the small dataset.

---

## 5. External Data Feasibility Matrix

| Data Source | Feasibility | Expected Impact | Effort | Notes |
|---|---|---|---|---|
| Weather (KMA) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Low | Korean data readily available |
| Calendar events | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Very low | Korean holidays available |
| Geolocation | ⭐⭐⭐⭐ | ⭐⭐⭐ | Low | GPS data already exists |
| Air quality | ⭐⭐⭐ | ⭐⭐ | Medium | AirKorea API available |
| Moon phase | ⭐⭐⭐⭐ | ⭐ | Very low | Trivial computation |
| Social factors | ⭐⭐ | ⭐⭐⭐ | High | BLE/WiFi data already used |
| Sleep survey prior | ⭐⭐ | ⭐⭐⭐⭐ | Medium | ETRI dataset contains prior |

---

## 6. Prioritized Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Add temporal/cyclical features** (`week_sin/cos`, `dow_sin/cos`, `is_weekend`, `month_sin/cos`)
2. **Add calendar data** (Korean holidays, weekends)
3. **Add weather data** (Daejeon/KMA)
4. Re-run V10 pipeline with these features

**Expected**: +0.03-0.05 log-loss improvement

### Phase 2: Rhythm/Consistency Features (2-3 days)
1. **Add rolling statistics** (3-day and 5-day rolling std for key features)
2. **Add rhythm features** (activity regularity, sleep consistency)
3. **Add interaction features** (weekend interactions, cross-domain)

**Expected**: Additional +0.02-0.03 log-loss improvement

### Phase 3: Advanced (1 week)
1. **Personalization features** (per-subject mean/std ratios)
2. **External data integration** (all external sources)
3. **Per-subject models** (train separate models per subject)
4. **Model ensemble** (stack V8 + V10 + new features)

**Expected**: Additional +0.02-0.04 log-loss improvement

---

## 7. Key Code Snippets

### Temporal Features Addition (Priority 1)
```python
# In 02_feature_engineering.py create_day_features()
def add_temporal_features(df):
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['lifelog_date'])
    
    # Cyclical encodings (avoids artificial ordering)
    df['week_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.isocalendar().week / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.isocalendar().week / 52)
    df['dow_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.month / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.dayofyear / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.dayofyear / 365)
    
    # Binary features
    df['is_weekend'] = (df['date_dt'].dt.dayofweek >= 5).astype(float)
    
    return df
```

### Weather Feature Integration
```python
# In separate weather_loader.py
import pandas as pd
import requests

def load_daejeon_weather(start_date, end_date):
    """Load historical weather for Daejeon from KMA/Meteostat."""
    # Option 1: Meteostat API (free)
    from meteostat import Daily
    from datetime import date
    station = '54156'  # Daejeon station
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    data = Daily(station, start, end)
    df = data.fetch()
    
    return pd.DataFrame({
        'lifelog_date': df.index.date,
        'temp_max': df['t2m'].max().values,
        'temp_min': df['t2m'].min().values,
        'temp_avg': df['t2m'].mean().values,
        'humidity': df['rf'].mean().values,
        'pressure': df['pres'].mean().values,
        'precipitation': df['prcp'].values,
        'wind_speed': df['ws'].mean().values,
    })
```

### Rolling Statistics Addition
```python
def add_rolling_stats(df, feature_cols, window=3):
    """Add per-subject rolling statistics for key features."""
    df = df.copy()
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.int64]:
            for sid in df['subject_id'].unique():
                mask = df['subject_id'] == sid
                sub = df.loc[mask].sort_values('lifelog_date')
                vals = sub[col].ffill().fillna(0)
                rolling_std = vals.rolling(window=window, min_periods=2).std().fillna(0)
                df.loc[sub.index, f'{col}_roll_std_{window}d'] = rolling_std.values
    return df
```

---

## 8. References

### Competition-Related
- **ETRI Lifelog Dataset 2024** (Oh et al., 2024): [arXiv:2508.03698](https://arxiv.org/html/2508.03698) — Dataset description and baseline sleep quality prediction
- **csm3310/ETRI_lifelog-project**: ICTC 2025 winner — LGBM + CatBoost ensemble with time-series feature engineering
- **normaldata42 blog**: Review of 2024 ETRI challenge papers including LR with feature importance selection

### Research Literature
- **HRV for sleep quality**: "Predicting Sleep Quality through Biofeedback: A Machine Learning Approach" (MDPI, 2023) — PPG-based HRV features predict PSQI scores
- **Wearable HRV**: "The Prediction of Sleep Quality Using Heart Rate Variability" (Springer, 2024) — 426 HRV features via XGBoost for sleep staging
- **Activity + HR + Light prediction**: "Predicting sleep based on physical activity, light exposure, and Heart Rate Variability" (Sleep and Breathing, 2024) — Galaxy Watch data with 1-3 day history
- **Deep learning for sleep**: "Sleep Quality Prediction from Wearables using CNN" (arXiv:2303.06028) — Combines wearables with circumstantial data
- **Long-term sleep prediction**: "Predicting long-term sleep deprivation using wearable sensors" (Computers in Biology and Medicine, 2024) — Previous sleep patterns + exercise most relevant

### External Data Sources
- **KMA (Korea Meteorological Administration)**: `http://www.kma.go.kr` — Official Korean weather data
- **Meteostat**: `https://meteostat.net` — Historical weather API
- **AirKorea**: `https://www.airkorea.or.kr` — Korean air quality data
- **OpenWeatherMap**: `https://openweathermap.org/api` — Historical weather API

---

## 9. Critical Observations

### ⚠️ Leakage Warning
The extended features file contains **987 target-cross features** that multiply base features by target values (e.g., `mACStatus_m_charging_mean_Q1_cross = mACStatus_m_charging_mean × Q1`). These have correlations up to 0.43 with targets and **must not be used in production**.

### ⚠️ Calendar Data in Test Set
The test set dates (250 samples) span the same June-Nov 2024 range. Weather and calendar features derived from this period are safe to use (they're not target-dependent). However, **be careful not to use post-test-date data**.

### 📊 Data Limitation
With only 450 training samples, complex feature engineering risks overfitting. The V8→V10 regression (0.6537→0.6607) demonstrates that **more complexity doesn't equal better performance**. Focus on features with genuine predictive power and strong regularization.

### 🎯 Key Insight: Seasonality Dominates
The strongest single predictor in the entire dataset is `week_of_year` with correlation up to 0.31 for S3. This suggests that **external seasonal data (weather, daylight hours, temperature) could be the single biggest improvement**. The model currently has no way to know what time of year each test sample is from — adding this information is a free +0.02 to +0.05 improvement.
