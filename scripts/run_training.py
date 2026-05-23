"""
End-to-end training pipeline for carbon intensity forecasting.

This script:
    1. Fetches and caches data from three sources
    2. Aligns datasets onto a common timeline
    3. Engineers features
    4. Trains a LightGBM model for a specified forecast horizon
    5. Evaluates against persistence baseline
"""

from carbon_forecast.data.ingestion import (
    CarbonIntensityClient,
    GenerationMixClient,
    WeatherClient,
)
from carbon_forecast.data.alignment import AlignmentPipeline
from carbon_forecast.features.engineering import FeatureEngineer
from carbon_forecast.models.lgbm import LGBMForecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
TRAIN_RATIO = 0.8
TARGET_COL = "actual"

# Forecast horizon in half-hour periods
# 2 = 1h, 48 = 24h, 96 = 48h
HORIZON = 48

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_jobs": -1,
    "random_state": 42,
}

# Columns that are targets or metadata — not features
DROP_COLS = [
    "to", "forecast", "index",
    # Raw generation (use lagged versions instead)
    "WIND", "SOLAR", "GAS", "FOSSIL", "GENERATION",
    "NUCLEAR", "IMPORTS", "COAL", "HYDRO", "BIOMASS",
    "OTHER", "STORAGE", "LOW_CARBON", "ZERO_CARBON", "RENEWABLE",
    # Ratios computed from raw generation (leaky)
    "wind_share", "solar_share", "fossil_share", "renewable_share",
]


# -----------------------------------
# DATA LOADING
# -----------------------------------

def load_or_fetch(name, path, fetch_fn):
    """Load from cache if available, otherwise fetch and save."""

    # Creates all folders needed for this path if they don't already exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        logger.info(f"Loading cached {name} from {path}")
        return pd.read_parquet(path)

    logger.info(f"Fetching {name} from API...")
    df = fetch_fn()
    df.to_parquet(path)
    logger.info(f"Saved {name} to {path}")
    return df


def fetch_all_data(start, end):
    """Fetch carbon intensity, generation mix, and weather data."""

    carbon = load_or_fetch(
        "carbon intensity",
        f"{CACHE_DIR}/carbon_{start}_{end}.parquet",
        lambda: CarbonIntensityClient().fetch(start=start, end=end),
    )

    gen = load_or_fetch(
        "generation mix",
        f"{CACHE_DIR}/gen_{start}_{end}.parquet",
        lambda: GenerationMixClient().fetch(start=start, end=end),
    )

    def fetch_weather():
        client = WeatherClient(timeout=60)
        actual = client.fetch(start_date=start, end_date=end)
        forecast = client.fetch(start_date=start, end_date=end, forecast=True)
        return actual.join(forecast)
 
    weather = load_or_fetch(
        "weather",
        f"{CACHE_DIR}/weather_{start}_{end}.parquet",
        fetch_weather,
    )
 
    return carbon, gen, weather


# -----------------------------------
# FEATURE PREPARATION
# -----------------------------------

def prepare_features(df, horizon):
    """
    Prepare feature matrix for a given forecast horizon.

    Steps:
        1. Create shifted target for the specified horizon
        2. Shift forecast weather to align with target time
        3. Remove non-feature columns
        4. Remove actual weather (only forecasts available at inference)
        5. Drop rows with NaN from lag/rolling/shift operations
    """
    df = df.copy()

    # Target: carbon intensity at T + horizon
    df["target"] = df[TARGET_COL].shift(-horizon)

    # Shift forecast weather to align with target time
    fcst_cols = [c for c in df.columns if "_fcst_" in c]
    for col in fcst_cols:
        df[f"{col}_target"] = df[col].shift(-horizon)

    # Drop columns that shouldn't be features
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Drop raw forecast weather (keep only target-aligned versions)
    df = df.drop(columns=fcst_cols)

    # Drop actual weather (not available at inference time)
    actual_weather = [
        c for c in df.columns
        if any(loc in c for loc in ["aberdeen", "glasgow", "london", "exeter"])
        and "_fcst_" not in c
        and "lag" not in c
        and "rolling" not in c
    ]
    df = df.drop(columns=actual_weather)

    # Drop NaN rows from lags, rolling, and target shift
    df = df.dropna()

    return df


# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------

def split_data(df, train_ratio):
    """Time-ordered train/test split."""
    split_idx = int(len(df) * train_ratio)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train.drop(columns=["target", TARGET_COL])
    y_train = train["target"]

    X_test = test.drop(columns=["target", TARGET_COL])
    y_test = test["target"]

    # Keep actual CI for persistence baseline
    persistence_pred = test[TARGET_COL]

    return X_train, X_test, y_train, y_test, persistence_pred


# -----------------------------------
# EVALUATION
# -----------------------------------

def evaluate(y_test, y_pred, persistence_pred, horizon):
    """Print evaluation metrics against persistence baseline."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    persistence_mae = mean_absolute_error(y_test, persistence_pred)
    persistence_rmse = np.sqrt(mean_squared_error(y_test, persistence_pred))

    hours = horizon / 2
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS — t+{hours:.0f}h forecast")
    logger.info(f"{'='*40}")
    logger.info(f"Model MAE:       {mae:.2f}")
    logger.info(f"Persistence MAE: {persistence_mae:.2f}")
    logger.info(f"Improvement:     {(1 - mae/persistence_mae)*100:.1f}%")
    logger.info(f"Model RMSE:      {rmse:.2f}")
    logger.info(f"Persistence RMSE:{persistence_rmse:.2f}")

    return {"mae": mae, "rmse": rmse, "persistence_mae": persistence_mae}


# -----------------------------------
# MAIN
# -----------------------------------

def main():
    # Fetch data
    carbon, gen, weather = fetch_all_data(START_DATE, END_DATE)

    # Align
    alignment = AlignmentPipeline(
        carbon_df=carbon,
        gen_df=gen,
        weather_df=weather,
    )
    aligned_df = alignment.transform()

    # Engineer features
    engineer = FeatureEngineer(target_col=TARGET_COL)
    featured_df = engineer.transform(aligned_df)

    # Prepare for training
    model_df = prepare_features(featured_df, horizon=HORIZON)

    # Split
    X_train, X_test, y_train, y_test, persistence_pred = split_data(
        model_df, TRAIN_RATIO
    )

    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Test samples:     {len(X_test):,}")
    logger.info(f"Features:         {X_train.shape[1]}")

    # Train
    model = LGBMForecaster(LGBM_PARAMS)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate(y_test, y_pred, persistence_pred, HORIZON)

    # Feature importance
    importance = model.feature_importance
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\nTop 15 features:")
    for name, score in sorted_imp[:15]:
        logger.info(f"  {name}: {score}")

    # Save model
    os.makedirs("outputs/models", exist_ok=True)
    model.save(f"outputs/models/lgbm_t{HORIZON}.pkl")


if __name__ == "__main__":
    main()