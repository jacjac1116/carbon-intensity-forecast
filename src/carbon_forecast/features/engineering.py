import pandas as pd
import logging
import numpy as np
import holidays
from carbon_forecast.data.alignment import AlignmentPipeline
from carbon_forecast.data.ingestion import (
    CarbonIntensityClient,
    GenerationMixClient,
    WeatherClient,
)

logger = logging.getLogger(__name__)

class FeatureEngineer():

    def __init__(self, target_col: str):

        self.target_col = target_col
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['lag_1h'] = df[self.target_col].shift(2) # 2 half-periods = 1h
        df['lag_24h'] = df[self.target_col].shift(2*24)
        df['lag_48h'] = df[self.target_col].shift(2*48)
        df['lag_1w'] = df[self.target_col].shift(2*24*7)

        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df['rolling_24h'] = df[self.target_col].rolling(window=48).mean()
        df['rolling_24h_std'] = df[self.target_col].rolling(window=48).std()
       
        df['rolling_7d'] = df[self.target_col].rolling(window=48*7).mean()
        df['rolling_7d_std'] = df[self.target_col].rolling(window=48*7).std()
       
        df['rolling_30d'] = df[self.target_col].rolling(window=48*30).mean()
        df['rolling_30d_std'] = df[self.target_col].rolling(window=48*30).std()
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame):

        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['Year'] = df.index.year

        uk_holidays = holidays.UnitedKingdom(subdiv = 'England')
        holidays_categories = ['new_year', 'royal', 'summer_bank_holiday', 'easter', 'christmas']
        df['bank_holiday'] = df.index.map(lambda x: uk_holidays.get(x))
        df['holiday_type'] = df['bank_holiday'].map({
            "New Year's Day": 'new_year',
            "Good Friday": 'easter',
            "Easter Monday": 'easter',
            "May Day": 'summer_bank_holiday',
            "Spring Bank Holiday": 'summer_bank_holiday',
            "Summer Bank Holiday": 'summer_bank_holiday',
            "Christmas Day": 'christmas',
            "Boxing Day": 'christmas',
            "Boxing Day (observed)": 'christmas',
            "Christmas Day (observed)": 'christmas',
            "New Year's Day (observed)": 'new_year',
            "Platinum Jubilee of Elizabeth II": 'royal',
            "State Funeral of Queen Elizabeth II": 'royal',
            "Coronation of Charles III": 'royal'
        })

        # pd.Categorical allows me to not lose any columns even if they are just nan
        df['holiday_type']= pd.Categorical(df['holiday_type'], categories=holidays_categories)
        
        # drop_first is set to False to not lose information about 'Christmas' holidays,
        # usually drop_first is set to True to avoid multicollinearity,
        # multicollinearity is a catastrophic issue for models as they can become unstable
        # and have high variance in their predictions, especially when extrapolating beyond
        # the range of the training data, which is a key requirement for this project
        dummy_holiday = pd.get_dummies(df['holiday_type'], prefix='holiday', drop_first=False)
        df = pd.concat([df, dummy_holiday], axis=1)

        
        df['season'] = df['month'].map({12:'winter', 1:'winter', 2:'winter',
                                         3:'spring', 4:'spring', 5:'spring',
                                           6:'summer', 7:'summer', 8:'summer',
                                             9:'autumn', 10:'autumn', 11:'autumn'})
        
        # drop_first is set to True to avoid multicollinearity, as the 
        # season categories are mutually exclusive and collectively exhaustive, 
        # so one of them can be inferred from the others
        dummy_season = pd.get_dummies(df['season'], drop_first=True, prefix='season')
        df = pd.concat([df, dummy_season], axis=1)

        # Hour 23 and hour 0 are 1 hour apart but numerically are 23 units apart
        # A linear model sees them as vastly different, but by using sin/cos we make
        # the data cyclical so adjacent values remain similar

        # sin(6am) = sin(6pm) but cos(6am) != cos(6pm) ->prevents ambiguous mapping by using both
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Tree based models make threshold splits:  "is hour > 16?" With raw integers
        # that's one split to capture the evening peak. With sin/cos, it would need
        # several complex splits to achieve the same. 

        # Trees can suffer with just sin/cos encoding, thus those columns won't be dropped

        df.drop(columns=['bank_holiday', 'holiday_type', 'season'], inplace=True)

        return df
    
    def _add_generation_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df["wind_share"] = df["WIND"] / df["GENERATION"]
        df["solar_share"] = df["SOLAR"] / df["GENERATION"]
        df["fossil_share"] = df["FOSSIL"] / df["GENERATION"]
        df["renewable_share"] = df["RENEWABLE"] / df["GENERATION"]
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_calendar_features(df)
        df = self._add_generation_ratios(df)

        return df

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # -----------------------------------
    # DATE RANGE
    # -----------------------------------

    start = "2021-01-01"
    end = "2021-12-31"

    # -----------------------------------
    # FETCH CARBON INTENSITY DATA
    # -----------------------------------

    carbon_client = CarbonIntensityClient()

    carbon = carbon_client.fetch(
        start=start,
        end=end
    )

    # -----------------------------------
    # FETCH GENERATION MIX DATA
    # -----------------------------------

    generation_client = GenerationMixClient()

    generation = generation_client.fetch()

    # -----------------------------------
    # FETCH WEATHER DATA
    # -----------------------------------

    weather_client = WeatherClient()

    weather = weather_client.fetch(
        start_date=start,
        end_date=end
    )

    weather_fcst = weather_client.fetch(
        start_date=start,
        end_date=end,
        forecast=True
    )

    weather_df = weather.join(weather_fcst)

    # -----------------------------------
    # ALIGN DATASETS
    # -----------------------------------

    alignment = AlignmentPipeline(
        carbon_df=carbon,
        gen_df=generation,
        weather_df=weather_df
    )

    final_df = alignment.transform()

    # -----------------------------------
    # ADD FEATURES
    # -----------------------------------

    engineer = FeatureEngineer(target_col='actual')

    final_df = engineer.transform(final_df)

    print(final_df.shape)

    print(final_df.columns)

    print(final_df.head())