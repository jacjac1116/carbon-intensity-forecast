from carbon_forecast.data.ingestion import CarbonIntensityClient
from datetime import datetime, timedelta, timezone
import pandas as pd

def pull_actuals(window_days: int = 7) -> pd.DataFrame:

    carbon_client = CarbonIntensityClient()

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=window_days)

    carbon = carbon_client.fetch(start=start, end=end)

    carbon = carbon.rename(columns={'forecast': 'official_forecast',
                           'from': 'target_time'})
    
    
    missing_days = carbon.loc[carbon['actual'].isna(), 'target_time']

    if len(missing_days):
        raise ValueError(
            f'{len(missing_days)} of {len(carbon)} actuals missing, '
            f'first at {missing_days.iloc[0]}'
        )
    return carbon[['target_time', 'actual', 'official_forecast']]



if __name__ == '__main__':

    df = pull_actuals()
    print(df)