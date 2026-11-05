import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from carbon_forecast.features.engineering import FeatureEngineer
from carbon_forecast.features.live import build_features


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"

CONFIGS_PATH = ROOT / "configs/model/lgbm_24h.yaml"
SCHEMA = ROOT / "outputs/schemas/latest_schema.json"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """
    Load sample carbon intensity and weather datasets.

    These datasets are used to test feature generation using
    realistic input data.
    """
    carbon = pd.read_parquet(FIXTURES / "carbon_sample.parquet")
    weather = pd.read_parquet(FIXTURES / "weather_sample.parquet")

    return carbon, weather


@pytest.fixture
def configs_data():
    """
    Load the model configuration from the YAML file.

    Provides the target column and forecast horizon required
    by the feature-building pipeline.
    """
    with open(CONFIGS_PATH) as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture
def dataset():
    """
    Create a small synthetic dataset for testing rolling features.

    50 rows are enough to test rolling_24h (48 observations),
    but insufficient for rolling_7d or rolling_30d.

    The actual carbon intensity changes from 100 to 200 halfway
    through the dataset, allowing us to verify rolling averages.
    """
    df = pd.DataFrame({
        "actual": [100.0] * 25 + [200.0] * 25,
        "london_temperature_2m": [10.0] * 50,
        "exeter_direct_radiation": [50.0] * 50,
        "glasgow_wind_speed_100m": [15.0] * 50,
        "aberdeen_wind_speed_100m": [12.0] * 50,
    }, index=pd.date_range("2024-01-01", periods=50, freq="30min"))

    return df


# ---------------------------------------------------------------------
# Feature generation tests
# ---------------------------------------------------------------------

def test_leakage(sample_data, configs_data):
    """
    Check that feature generation does not introduce data leakage.

    Features generated at timestamp T should be identical whether
    the pipeline has access to the complete dataset or only the
    information that would have been available at timestamp T.

    Future observed weather values are hidden to simulate
    real-world forecasting conditions.
    """

    # Generate features using the complete carbon and weather datasets
    carbon, weather = sample_data
    config = configs_data

    _, model_df = build_features(
        carbon,
        weather,
        config["target_col"],
        config["horizon"]
    )

    # Define the timestamp at which we want to test for leakage
    T = pd.Timestamp("2020-02-12 12:00", tz="UTC")

    # Identify observed weather columns, excluding forecast columns
    obs_cols = [c for c in weather.columns if "_fcst_" not in c]

    # Keep only carbon observations available up to timestamp T
    carbon_cut = carbon[carbon["from"] <= T]

    # Keep weather data up to the forecast horizon after T
    weather_cut = weather.loc[:T + pd.Timedelta(hours=24)].copy()

    # Remove future observed weather values.
    # Forecast columns remain available because they represent
    # information that would have been known at prediction time.
    weather_cut.loc[weather_cut.index > T, obs_cols] = float("nan")

    # Generate features again using the restricted datasets
    _, model_df_cut = build_features(
        carbon_cut,
        weather_cut,
        config["target_col"],
        config["horizon"]
    )

    # Ensure timestamp T exists in both feature datasets
    assert T in model_df.index, (
        "Pick another T: this row is dropped even with all the data"
    )

    assert T in model_df_cut.index, (
        "Row T disappears when the future is hidden: "
        "a feature at T needs data after T"
    )

    # Features at T must be identical regardless of whether
    # future observed weather data is available
    pd.testing.assert_series_equal(model_df.loc[T], model_df_cut.loc[T])


def test_build_features(sample_data, configs_data):
    """
    Validate the output of the complete feature-building pipeline.

    Checks that:
    - All expected features are generated.
    - All features have numeric data types.
    - The resulting dataset is not empty.
    - The latest row contains no missing values.
    """

    # Generate the model features using the sample datasets
    carbon, weather = sample_data
    config = configs_data

    _, model_df = build_features(
        carbon,
        weather,
        config["target_col"],
        config["horizon"]
    )

    # Load the expected feature names from the saved schema
    with open(SCHEMA) as f:
        expected = json.load(f)["features"]

    # Check that every expected feature exists in the output
    missing = set(expected) - set(model_df.columns)

    assert not missing, f"Missing features: {missing}"

    # Select only the features defined in the schema
    X = model_df[expected]

    # Ensure every feature has a numeric data type
    for col, dtype in X.dtypes.items():
        assert pd.api.types.is_numeric_dtype(dtype), (
            f"{col} is {dtype}"
        )

    # Ensure feature generation produced at least one valid row
    assert len(X) > 0

    # The latest row should be complete, with no missing values
    assert not X.iloc[-1].isna().any()


# ---------------------------------------------------------------------
# Individual feature engineering tests
# ---------------------------------------------------------------------

def test_lag_features():
    """
    Check that lag features are generated with the correct values.

    With 30-minute data, a 1-hour lag requires shifting
    the target column by two observations.
    """

    # Create a simple dataset with known target values
    df = pd.DataFrame({
        "actual": [100, 200, 300, 400, 500]
    }, index=pd.date_range("2024-01-01", periods=5, freq="30min"))

    engineer = FeatureEngineer(target_col="actual")
    result = engineer._add_lag_features(df)

    # lag_1h = shift(2), so row 2 should contain the value from row 0
    assert result["lag_1h"].iloc[2] == 100

    # The first two rows should be NaN because there is no
    # historical data available to calculate the 1-hour lag
    assert pd.isna(result["lag_1h"].iloc[0])
    assert pd.isna(result["lag_1h"].iloc[1])


def test_rolling_features(dataset):
    """
    Check that rolling averages are calculated correctly.

    Verifies the 24-hour rolling window and confirms that
    longer rolling windows remain unavailable when there
    is insufficient historical data.
    """

    df = dataset

    engineer = FeatureEngineer(target_col="actual")
    result = engineer._add_rolling_features(df)

    # Confirm that the 24-hour rolling feature was generated
    assert "rolling_24h" in result.columns

    # The first row should be NaN because there are not yet
    # enough observations to fill the 48-period rolling window
    assert pd.isna(result["rolling_24h"].iloc[0])

    # At row 49, the rolling window contains the latest 48 values:
    # - 23 observations with a value of 100
    # - 25 observations with a value of 200
    #
    # Expected rolling mean:
    # (23 * 100 + 25 * 200) / 48 = 152.08
    assert abs(result["rolling_24h"].iloc[49] - 152.08) < 1.0

    # There are insufficient observations for 7-day and 30-day
    # rolling windows, so both features should contain only NaN
    assert result["rolling_7d"].isna().all()
    assert result["rolling_30d"].isna().all()


def test_rolling_features_min_period(dataset):
    """
    Check how rolling features handle missing historical values.

    Verifies that the rolling calculation can still produce a value
    when some observations are missing, but returns NaN when the
    number of valid observations falls below the minimum requirement.
    """

    df = dataset

    # Introduce one missing observation into the historical data
    df.loc[df.index[30], "actual"] = float("nan")

    engineer = FeatureEngineer(target_col="actual")
    result = engineer._add_rolling_features(df)

    # The rolling window should still produce a value despite
    # having one missing observation
    assert not pd.isna(result["rolling_24h"].iloc[49])

    # Introduce six consecutive missing observations
    df.loc[df.index[30:36], "actual"] = float("nan")

    result = engineer._add_rolling_features(df)

    # There are now too few valid observations to satisfy
    # the rolling window's minimum-period requirement
    assert pd.isna(result["rolling_24h"].iloc[49])