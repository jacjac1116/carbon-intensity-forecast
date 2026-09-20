from datetime import datetime, timedelta, timezone
from carbon_forecast.data.ingestion import (
    CarbonIntensityClient,
    WeatherClient,
)
from carbon_forecast.data.alignment import AlignmentPipeline
from carbon_forecast.features.engineering import FeatureEngineer, shape_features
import yaml
import logging
import json
import time
from pathlib import Path
import os

logger = logging.getLogger(__name__)



def build_live_features():
    """
    Creates a live dataset to be fed into deployed model
    """

    # Obtains standard variables
    CONFIGS_PATH = Path(os.environ.get('CONFIGS_PATH', 'configs/model/lgbm_24h.yaml'))

    with open(CONFIGS_PATH) as f:
        config = yaml.safe_load(f)

    target_col = config['target_col']
    horizon = config['horizon']
    lookback_days = config['lookback_days']

    # Defines time period in UTC
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=lookback_days)
    end = today

    # Pulls carbon data from API
    carbon_client = CarbonIntensityClient()
    carbon = carbon_client.fetch(start.isoformat(), end.isoformat())

    # Pulls past and forecast weather data
    weather_client = WeatherClient()
    actual = weather_client.fetch(start_date=start.isoformat(), end_date=end.isoformat(), source='archive')
    forecast = weather_client.fetch(start_date=start.isoformat(), end_date=end.isoformat(), source='live')
    weather = actual.join(forecast, how='outer')

    # Transforms and prepares dataset
    aligned_df = AlignmentPipeline(carbon, weather).transform() 
    featured_df = FeatureEngineer(target_col=target_col).transform(aligned_df)
    model_df = shape_features(featured_df, horizon=horizon)

    # Validates dataset with schema used in model
    SCHEMAS_PATH = Path(os.environ.get('SCHEMAS_PATH', 'outputs/schemas/latest_schema.json'))

    with open(SCHEMAS_PATH) as f:
        schema = json.load(f)

    features = schema['features']

    missing_cols = [c for c in features if c not in model_df.columns]
    if missing_cols:
        raise ValueError(f'Dataset is missing columns: {missing_cols}')
    
    # Captures how fresh carbon data is
    as_of = carbon['from'].max()

    return model_df[features], as_of

if __name__ == '__main__':

    for i in range(5):
        t0 = time.perf_counter()
        df, as_of = build_live_features()
        print(f'{time.perf_counter() -t0:.2f}s')

    print(as_of)
    print(df)