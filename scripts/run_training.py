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
from carbon_forecast.evaluation.stratified import StratifiedEvaluator
import yaml
import mlflow
from carbon_forecast.evaluation.metrics import EvaluationReport
import pickle
import json
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

# Build absolute path to data cache directory:
# __file__          = .../carbon-intensity-forecast/scripts/run_training.py
# abspath(__file__) = ensures full path (not relative)
# dirname() x1      = .../carbon-intensity-forecast/scripts/
# dirname() x2      = .../carbon-intensity-forecast/         (project root)
# join(... "data", "raw") = .../carbon-intensity-forecast/data/raw/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

config_path = os.path.join(PROJECT_ROOT, 'configs', 'model', 'lgbm_24h.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

START_DATE = config['data']['start_date']
END_DATE = config['data']['end_date']
TRAIN_RATIO = config['data']['train_ratio']
TARGET_COL = config['target_col']

# Forecast horizon in half-hour periods
# 2 = 1h, 48 = 24h, 96 = 48h
HORIZON = config['horizon']

LGBM_PARAMS = config['model']['params']

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

def load_or_fetch(name: str, path: str, fetch_fn: callable) -> pd.DataFrame:
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


def fetch_all_data(start: str, end: str
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
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

def prepare_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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

def split_data(df: pd.DataFrame, train_ratio: float) -> tuple:
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

def evaluate(
        y_test: pd.Series, 
        y_pred: np.array, 
        persistence_pred: pd.Series, 
        horizon: int
        ) -> dict:
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
# MLFLOW
# -----------------------------------

def import_to_mlflow(model, y_true, y_pred, X_train, TARGET_COL, stratified_results, persistence_pred, HORIZON, version):

    # Setup: Tell mflow where to save and what experiement I'm in
    MLFLOW_TRACKING_URI = '/Users/luiscanteiro/Documents/Python/Codes/mlruns'
    EXPERIMENT_NAME = 'carbon_intensity_forecasting'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    importance = model.feature_importance

    analyser = EvaluationReport(
        y_true,
        y_pred,
        persistence_pred,
        HORIZON
    )
    analysis = analyser.compute()

    with mlflow.start_run(run_name='carbon_intensity_lgb') as parent:

        with mlflow.start_run(run_name=f'{version}', nested=True):
            mlflow.log_dict(dict(sorted(importance.items(), key= lambda x: x[1], reverse=True)), 'feature_importance.json')

            stratified_results.to_csv('stratified_results.csv', index=False)
            mlflow.log_artifact('stratified_results.csv')
            os.remove('stratified_results.csv')

            mlflow.log_metrics(analysis)
            mlflow.log_params(config['model']['params'])
            mlflow.log_param('horizon', HORIZON)
            mlflow.log_param('train_ratio', TRAIN_RATIO)
           
            schema = {'features': X_train.columns.tolist(),
                      'target': TARGET_COL}
            mlflow.log_dict(schema, 'model_schema.json')

            # Save model using pickle - mlflow.sklearn.log_model has version compatibility issues
            with open('temp_model.pkl', 'wb') as f: # 'wb' write binary - saves in raw bites
                pickle.dump(model, f)
            mlflow.log_artifact('temp_model.pkl', f'model_{version}')
            os.remove('temp_model.pkl')

            # Save input example
            with open('input_example.json', 'w') as f: # 'w' - write text, json is text
                json.dump(X_train.reset_index(drop=True).iloc[:5].to_dict(), f)
            mlflow.log_artifact('input_example.json')
            os.remove('input_example.json')
    
    logger.info('Model uploaded to mlflow')


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

    evaluation_df = featured_df.loc[y_test.index]

    evaluator = StratifiedEvaluator(
        df = evaluation_df,
        y_true=y_test,
        y_pred=y_pred
    )

    stratified_results = evaluator.report()
    cols = ["slice", "condition", "n_sample", "mae", "mae_vs_global", "rmse", "r2_score", "flag"]
    print(stratified_results[cols].round(2).to_string())

    # Save model
    os.makedirs("outputs/models", exist_ok=True)
    model.save(f"outputs/models/lgbm_t{HORIZON}.pkl")

    version = datetime.now().strftime('%Y%m%d_%H%M%S')

    import_to_mlflow(
        model=model,
        y_true=y_test,
        y_pred=y_pred,
        X_train=X_train,
        TARGET_COL=TARGET_COL,
        stratified_results=stratified_results,
        persistence_pred=persistence_pred,
        HORIZON=HORIZON,
        version=version
    )


if __name__ == "__main__":
    main()