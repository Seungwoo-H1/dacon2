"""
31_v30_weather_loader.py — V30 external data pipeline

Load weather data (Daejeon) and Korean holidays for Dacon2.
Uses Open-Meteo API (free, no API key needed).
Expected improvement: -0.02 ~ -0.05 log-loss (based on feature_study_results.md)
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import date
import urllib.request
import json

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from config import DATA_RAW, DATA_PROCESSED

try:
    import holidays as holidays_lib
except ImportError:
    log.error("holidays not installed. Run: pip3 install holidays --break-system-packages")
    sys.exit(1)

# Daejeon coordinates
DAEJEON_LAT = 36.3504
DAEJEON_LON = 127.3845


def load_daejeon_weather(start_date, end_date):
    """
    Load daily weather data for Daejeon from Open-Meteo API (free, no key).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    log.info(f"Loading Daejeon weather from Open-Meteo: {start_date} to {end_date}")

    # Open-Meteo API — chunk by month (archive API limits large ranges)
    daily_params = (f"temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                    f"precipitation_sum,windspeed_10m_max,"
                    f"relative_humidity_2m_mean,cloudcover_mean,sunrise,sunset,daylight_duration")
    hourly_params = (f"temperature_2m,relative_humidity_2m,windspeed_10m,"
                     f"pressure_msl,precipitation")

    # Generate monthly chunks
    import calendar
    chunks = []
    current = start
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        chunk_end = min(end, pd.Timestamp(current.year, current.month, last_day))
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = (current.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)

    # Fetch all chunks and merge
    all_daily = {}
    all_hourly = {}
    for s, e in chunks:
        url_d = (f"https://archive-api.open-meteo.com/v1/archive?"
                 f"latitude={DAEJEON_LAT}&longitude={DAEJEON_LON}&"
                 f"start_date={s}&end_date={e}&"
                 f"daily={daily_params}&timezone=Asia/Seoul")
        url_h = (f"https://archive-api.open-meteo.com/v1/archive?"
                 f"latitude={DAEJEON_LAT}&longitude={DAEJEON_LON}&"
                 f"start_date={s}&end_date={e}&"
                 f"hourly={hourly_params}&timezone=Asia/Seoul")

        try:
            with urllib.request.urlopen(url_d, timeout=30) as resp:
                d = json.loads(resp.read().decode())
                if d.get('daily', {}).get('time'):
                    log.info(f"  {s} to {e}: daily OK ({len(d['daily']['time'])} records)")
                    all_daily.update(d['daily'])
        except Exception as e:
            log.warning(f"  {s} to {e}: daily FAIL - {e}")

        try:
            with urllib.request.urlopen(url_h, timeout=30) as resp:
                h = json.loads(resp.read().decode())
                if h.get('hourly', {}).get('time'):
                    log.info(f"  {s} to {e}: hourly OK ({len(h['hourly']['time'])} records)")
                    all_hourly.update(h['hourly'])
        except Exception as e:
            log.warning(f"  {s} to {e}: hourly FAIL - {e}")

    if not all_daily:
        log.info("No weather data — proceeding with synthetic features only.")
        return None

    # Convert to DataFrames
    daily_data = all_daily
    all_hourly_data = all_hourly

    if not daily_data.get('time'):
        log.warning("No weather data returned from Open-Meteo")
        return None

    result = pd.DataFrame({
        'lifelog_date': pd.to_datetime(daily_data['time']).date,
    })

    # Daily features
    for key, col in [
        ('temperature_2m_max', 'temp_max'),
        ('temperature_2m_min', 'temp_min'),
        ('temperature_2m_mean', 'temp_avg'),
        ('precipitation_sum', 'precipitation'),
        ('windspeed_10m_max', 'wind_speed'),
        ('relative_humidity_2m_mean', 'humidity'),
        ('cloudcover_mean', 'cloud_cover'),
    ]:
        if key in daily_data:
            result[col] = daily_data[key]

    # Sunrise/sunset → daylight hours
    if 'sunrise' in daily_data and 'sunset' in daily_data:
        sunrises = pd.to_datetime(daily_data['sunrise'])
        sunsets = pd.to_datetime(daily_data['sunset'])
        result['daylight_hours'] = (sunsets - sunrises).dt.total_seconds() / 3600
        result['daylight_ratio'] = result['daylight_hours'] / 24

    # Hourly nighttime features (22:00 - 06:00)
    hourly_data = all_data['hourly']
    if hourly_data and hourly_data.get('time'):
        hourly_df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly_data['time']),
            'temp': hourly_data.get('temperature_2m', []),
            'humidity': hourly_data.get('relative_humidity_2m', []),
            'wind': hourly_data.get('windspeed_10m', []),
            'pressure': hourly_data.get('pressure_msl', []),
            'precip': hourly_data.get('precipitation', []),
        })
        if not hourly_df.empty:
            hourly_df['date'] = hourly_df['datetime'].dt.date
            hourly_df['hour'] = hourly_df['datetime'].dt.hour
            # Nighttime (22:00 - 06:00)
            night = hourly_df[(hourly_df['hour'] >= 22) | (hourly_df['hour'] <= 6)]
            if not night.empty:
                night_stats = night.groupby('date').agg(
                    night_temp_max=('temp', 'max'),
                    night_temp_min=('temp', 'min'),
                    night_temp_avg=('temp', 'mean'),
                    night_humidity_avg=('humidity', 'mean'),
                    night_wind_max=('wind', 'max'),
                    night_pressure_avg=('pressure', 'mean'),
                    night_precip_sum=('precip', 'sum'),
                )
                result = result.merge(night_stats, left_on='lifelog_date',
                                      right_index=True, how='left')

    # Cyclical features
    dates = pd.to_datetime(result['lifelog_date'])
    result['doy_sin'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    result['doy_cos'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)

    log.info(f"  Loaded {len(result)} weather records")
    return result


def load_korean_holidays(start_date, end_date):
    """
    Load Korean public holidays for the given date range.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    log.info(f"Loading Korean holidays: {start_date} to {end_date}")

    ko_holidays = holidays_lib.KR(years=range(start.year, end.year + 1))

    dates = pd.date_range(start.date(), end.date(), freq='D')
    result = pd.DataFrame({'lifelog_date': dates.date})

    result['is_holiday'] = result['lifelog_date'].apply(lambda d: d in ko_holidays).astype(int)

    result['month'] = pd.to_datetime(result['lifelog_date']).dt.month
    result['is_school_term'] = result['month'].between(3, 7) | result['month'].between(9, 12)
    result['is_exam_period'] = result['month'].isin([6, 7]).astype(int)

    # Lunar holidays (2024)
    lunar_holidays = {
        '2024-02-10': 'Seollal',
        '2024-09-17': 'Chuseok',
    }
    result['is_lunar_holiday'] = result['lifelog_date'].apply(
        lambda d: int(str(d) in lunar_holidays)
    )

    log.info(f"  Holidays in range: {result['is_holiday'].sum()}")
    return result


def load_daylight_hours(lifelog_dates, latitude=36.3504, longitude=127.3845):
    """
    Calculate approximate daylight hours using solar position formulas.
    Simple implementation without external dependencies.
    """
    dates = pd.to_datetime(lifelog_dates)
    if isinstance(dates, pd.DatetimeIndex):
        doy = dates.dayofyear.values
    else:
        doy = dates.dt.dayofyear.values

    # Solar declination (approximate)
    declination = 23.45 * np.sin(2 * np.pi * (284 + doy) / 365.25) * np.pi / 180

    # Latitude in radians
    lat_rad = latitude * np.pi / 180

    # Hour angle at sunrise/sunset
    cos_hour_angle = -np.tan(lat_rad) * np.tan(declination)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    hour_angle = np.arccos(cos_hour_angle)

    # Daylight hours (hour angle in degrees -> hours)
    daylight_hours = 2 * hour_angle * 180 / (np.pi * 15)

    result = pd.DataFrame()
    if isinstance(dates, pd.DatetimeIndex):
        result['lifelog_date'] = dates.date
    else:
        result['lifelog_date'] = dates.dt.date
    result['daylight_hours'] = daylight_hours

    # Daylight ratio
    result['daylight_ratio'] = daylight_hours / 24

    # Seasonal warmth index
    result['season_index'] = np.sin(2 * np.pi * (doy - 80) / 365.25)

    return result


def main():
    log.info("=" * 70)
    log.info("V30 External Data Loader")
    log.info("=" * 70)

    # Date range
    start_date = '2024-06-01'
    end_date = '2024-11-30'

    # Load weather data
    weather_df = load_daejeon_weather(start_date, end_date)

    # Load holidays
    holiday_df = load_korean_holidays(start_date, end_date)

    # Load daylight
    all_dates = pd.date_range(start_date, end_date, freq='D')
    daylight_df = load_daylight_hours(all_dates)

    # Merge all external data
    if weather_df is not None:
        external = weather_df.merge(holiday_df, on='lifelog_date', how='outer')
        external = external.merge(daylight_df, on='lifelog_date', how='outer')
    else:
        external = holiday_df.merge(daylight_df, on='lifelog_date', how='outer')

    # Save
    output_path = DATA_PROCESSED / 'external_data.parquet'
    external.to_parquet(output_path, index=False)
    log.info(f"\n✅ Saved: {output_path}")
    log.info(f"  Shape: {external.shape}")
    log.info(f"  Columns: {list(external.columns)}")
    log.info(f"  Date range: {external['lifelog_date'].min()} to {external['lifelog_date'].max()}")

    # Summary stats
    log.info("\n--- Weather Summary ---")
    for col in ['temp_max', 'temp_min', 'temp_avg', 'precipitation', 'humidity', 'pressure']:
        if col in external.columns:
            log.info(f"  {col}: mean={external[col].mean():.1f}, std={external[col].std():.1f}")

    log.info("\n--- Holiday Summary ---")
    log.info(f"  Total holidays: {external['is_holiday'].sum()}")
    log.info(f"  Lunar holidays: {external['is_lunar_holiday'].sum()}")

    return external


if __name__ == "__main__":
    main()
