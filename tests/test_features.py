import pandas as pd
from carbon_forecast.features.engineering import FeatureEngineer
from carbon_forecast.features.live import build_features
from pathlib import Path
import pytest
import yaml
import json

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS_PATH = ROOT / 'configs/model/lgbm_24h.yaml'
SCHEMA = ROOT / 'outputs/schemas/latest_schema.json'

@pytest.fixture
def sample_data():
    carbon = pd.read_parquet(FIXTURES / 'carbon_sample.parquet')
    weather = pd.read_parquet(FIXTURES / 'weather_sample.parquet')

    return carbon, weather

@pytest.fixture
def configs_data():

    with open(CONFIGS_PATH) as f:
        config = yaml.safe_load(f)

    return config

@pytest.fixture
def dataset():
    # 50 rows - enough for rolling_24h (window=48) but not 7d or 30d
    df = pd.DataFrame({
        "actual": [100.0] * 25 + [200.0] * 25,
        "london_temperature_2m": [10.0] * 50,
        "exeter_direct_radiation": [50.0] * 50,
        "glasgow_wind_speed_100m": [15.0] * 50,
        "aberdeen_wind_speed_100m": [12.0] * 50,
    }, index=pd.date_range("2024-01-01", periods=50, freq="30min"))

    return df

def test_build_features(sample_data, configs_data):
    carbon, weather = sample_data
    config = configs_data
    _, model_df = build_features(carbon, weather, config['target_col'], config['horizon'])

    with open(SCHEMA) as f:
        expected = json.load(f)['features']
    
    missing = set(expected) - set(model_df.columns)
    assert not missing, f'missing features: {missing}'

    X = model_df[expected]

    for col, dtype in X.dtypes.items():
       assert pd.api.types.is_numeric_dtype(dtype), f'{col} is {dtype}'
    
    assert len(X) > 0
    assert not X.iloc[-1].isna().any()

def test_lag_features():
    """
    Check that lag creation produces correct values
    """
    df = pd.DataFrame({
        'actual': [100, 200, 300, 400, 500]
    }, index=pd.date_range('2024-01-01', periods=5, freq='30min'))

    engineer = FeatureEngineer(target_col='actual')
    result = engineer._add_lag_features(df)

    # lag_1h = shift(2), so row index 2 should have value from row 0
    assert result['lag_1h'].iloc[2] == 100
    # First two rows should be NaN (no data to lag from)
    assert pd.isna(result['lag_1h'].iloc[0])
    assert pd.isna(result['lag_1h'].iloc[1])

def test_rolling_features(dataset):
    """
    Check that rolling creation produces correct values
    """
    
    df = dataset
    
    engineer = FeatureEngineer(target_col="actual")
    result = engineer._add_rolling_features(df)
    
    # rolling_24h column should exist
    assert "rolling_24h" in result.columns
    
    # First 42 rows should be NaN (window=48, not enough data)
    assert pd.isna(result["rolling_24h"].iloc[0])
    
    # Row 49: rolling mean over last 48 values = mix of 100s and 200s
    # Last 25 are 200, previous 23 are 100: (23*100 + 25*200) / 48 = 152.08
    assert abs(result["rolling_24h"].iloc[49] - 152.08) < 1.0
    
    # 7d and 30d should be all NaN (not enough data)
    assert result["rolling_7d"].isna().all()
    assert result["rolling_30d"].isna().all()

def test_rolling_features_min_period(dataset):
    
    df = dataset
    df.loc[df.index[30], 'actual'] = float('nan')

    engineer = FeatureEngineer(target_col="actual")
    result = engineer._add_rolling_features(df)

    assert not pd.isna(result["rolling_24h"].iloc[49])

    df.loc[df.index[30:36], 'actual'] = float('nan')

    result = engineer._add_rolling_features(df)

    assert pd.isna(result["rolling_24h"].iloc[49])