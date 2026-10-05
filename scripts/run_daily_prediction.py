from carbon_forecast.features.live import build_live_features
from carbon_forecast.data.actuals import pull_actuals
import os
from pathlib import Path
from carbon_forecast.models.lgbm import LGBMForecaster
from carbon_forecast.models.quantile import QuantileForecaster
import pandas as pd
from datetime import timedelta, datetime, timezone
import yaml
import json
from google.cloud import bigquery
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get('MODEL_DIR', 'outputs/models'))
CONFIGS_PATH = Path(os.environ.get('CONFIGS_PATH', 'configs/model/lgbm_24h.yaml'))
SCHEMAS_PATH = Path(os.environ.get('SCHEMAS_PATH', 'outputs/schemas/latest_schema.json'))
FORECAST_TABLE_ID = os.environ.get('FORECAST_TABLE_ID', 'gen-lang-client-0232862267.carbon_forecast.predictions')
ACTUALS_TABLE_ID = os.environ.get('ACTUALS_TABLE_ID', 'gen-lang-client-0232862267.carbon_forecast.actuals')
PROJECT_ID = os.environ.get('GCP_PROJECT', 'gen-lang-client-0232862267')


def upload_to_bigquery(df: pd.DataFrame, table_id: str, client: bigquery.Client) -> None:
    """
    Appends a dataset directly to existing table
    """

    # 1. Configure the insertion job to keep all existing records 
    # and simply add rows to the bottom
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    
    logger.info(f'Uploading {len(df)} rows to {table_id}...')

    # 2. Load the dataframe directly to BigQuery
    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config
    )

    # 3. Wait for the upload workflow to finish
    job.result()

    logger.info(f'Successfully appended data to {table_id}')

def write_predictions(df: pd.DataFrame, table_id: str, client: bigquery.Client) -> None:
    "Protection from duplicating data in forecasting table"

    target_time = df['predicted_at'].iloc[0]

    # 1. configure sql query with the variables used
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('target_time', 'TIMESTAMP', target_time)
        ])
    
    # 2. Create sql operation
    job = client.query(f"""
        DELETE FROM `{table_id}`
        WHERE DATE(predicted_at) = DATE(@target_time)
                 """, job_config=config)
    
    # 3. Wait for the upload workflow to finish
    job.result()

    logger.info(f'{job.num_dml_affected_rows} rows dropped from {table_id}')

def write_actuals(df: pd.DataFrame, table_id: str, client: bigquery.Client) -> None:
    "Protection from duplicating data in actuals table"

    window_start = df['target_time'].min()

    # 1. configure sql query with the variables used
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('window_start', 'TIMESTAMP', window_start)
        ])
    
    # 2. Create sql operation
    job = client.query(f"""
        DELETE FROM `{table_id}`
        WHERE target_time >= @window_start
                 """, job_config=config)
    
    # 3. Wait for the upload workflow to finish
    job.result()

    logger.info(f'{job.num_dml_affected_rows} rows dropped from {window_start} in {table_id}')


def create_forecast():
    # Get configs
    with open(CONFIGS_PATH) as f:
        config = yaml.safe_load(f)

    HORIZON = config['horizon']

    # Get model_id
    with open(SCHEMAS_PATH) as f:
        schema = json.load(f)

    model_id = schema['run_id']

    # Load models
    logger.info('Pulling models for forecast...')
    lgbm_model = LGBMForecaster.load(MODEL_DIR/ f'lgbm_t{HORIZON}.pkl')
    quantile_model = QuantileForecaster.load(MODEL_DIR/ f'lgbm_quantile_t{HORIZON}.pkl')

    # import data to use for forecast
    logger.info('Importing data for forecast...')
    df, as_of, persistence = build_live_features()


    # make predictions
    quantile_forecast = quantile_model.predict(df)
    lgbm_forecast = lgbm_model.predict(df)

    logger.info('Forecast completed! Setting table to upload to BigQuery...')

    # Join datasets
    lgbm_df = pd.DataFrame(lgbm_forecast, index=quantile_forecast.index, columns=['forecast'])
    final_forecast = lgbm_df.join(quantile_forecast)

    # Add remaining columns needed to upload in BigQuery
    final_forecast['predicted_at'] = datetime.now(timezone.utc)
    final_forecast['target_time'] = final_forecast.index + timedelta(hours= 0.5 * HORIZON)
    final_forecast['data_as_of'] = as_of
    final_forecast['model_id'] = model_id
    final_forecast['persistence'] = persistence['actual']
    final_forecast['london_fcst_temperature_2m_target'] = df['london_fcst_temperature_2m_target']
    final_forecast['exeter_fcst_direct_radiation_target'] = df['exeter_fcst_direct_radiation_target']
    final_forecast['aberdeen_fcst_wind_speed_100m_target'] = df['aberdeen_fcst_wind_speed_100m_target']
    
    final_forecast['horizon_steps'] = ((final_forecast['target_time'] - final_forecast['predicted_at'].dt.floor('30min'))
                                    .dt.total_seconds() / (30 * 60) # # gap in half-hours, now always a whole number
                                    ).astype(int) # rounds it 
    
    final_forecast = final_forecast.reset_index()

    final_df = final_forecast[['target_time', 'predicted_at', 'data_as_of',
                            'forecast', 'q10', 'q50', 'q90', 'horizon_steps', 'model_id',
                            'persistence', 'london_fcst_temperature_2m_target', 
                            'exeter_fcst_direct_radiation_target', 'aberdeen_fcst_wind_speed_100m_target']]
    return final_df

if __name__ == '__main__':

    forecast_df = create_forecast()
    actual_df = pull_actuals()

    # Initialise the BigQuery Client
    client = bigquery.Client(project=PROJECT_ID)


    write_predictions(forecast_df, FORECAST_TABLE_ID, client)
    upload_to_bigquery(forecast_df, FORECAST_TABLE_ID, client)

    write_actuals(actual_df, ACTUALS_TABLE_ID, client)
    upload_to_bigquery(actual_df, ACTUALS_TABLE_ID, client)