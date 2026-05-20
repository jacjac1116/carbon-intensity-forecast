import pandas as pd
import logging
from carbon_forecast.data.ingestion import (
    CarbonIntensityClient,
    GenerationMixClient,
    WeatherClient,
)

logger = logging.getLogger(__name__)

class AlignmentPipeline:

    def __init__(self,
                 carbon_df: pd.DataFrame,
                 gen_df: pd.DataFrame,
                 weather_df: pd.DataFrame
                 ):
        self.carbon_df = carbon_df
        self.gen_df = gen_df
        self.weather_df = weather_df
    
    def _align(self, dfs: dict) -> pd.DataFrame:
        """
        Align all datasets onto a common datetime window.

        This method:
        - Finds overlapping time range across datasets
        - Detects timestamp gaps
        - Detects duplicate timestamps
        - Reports alignment diagnostics
        - Performs inner join on aligned timestamps

        Returns:
            pd.DataFrame:
                Fully aligned merged dataset.
        """

        # -----------------------------------
        # DATA QUALITY CHECKS
        # -----------------------------------
        for name, df in dfs.items():

            logger.info(f'Checking {name} dataset...')

            # Ensure chronological dataset
            dfs[name] = df.sort_index()

            # -----------------------------------
            # DUPLICATE TIMESTAMP CHECK
            # -----------------------------------

            duplicates = df.index.duplicated().sum()

            if duplicates > 0:
                logger.warning(
                    f'{name}: {duplicates} duplicated timestamps detected'
                )
            
            # -----------------------------------
            # GAP DETECTION
            # -----------------------------------

            expected = pd.Timedelta(minutes=30)
            freq = df.index.to_series().diff().dropna() # creates a dataset showing the time differenec between rows
            gaps = freq[freq != expected]

            if not gaps.empty:

                logger.warning(f'{name}: {len(gaps)} irregular intervals detected')

                expected_index = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq="30min"
                )

                missing_timesteps = expected_index.difference(df.index)

                logger.warning(f'Missing data on :\n{missing_timesteps}')

                logger.warning(
                    f"{name}: Largest gap = {gaps.max()}"
                )
            
            # -----------------------------------
            # BASIC DATASET SUMMARY
            # -----------------------------------

            logger.info(
                f'{name}: '
                f'{df.index.min()} -> {df.index.max()} | '
                f'{len(df):,} rows'
            )

        # -----------------------------------
        # FIND COMMON TIME WINDOW
        # -----------------------------------

        start = max(df.index.min() for df in dfs.values())
        end = min(df.index.max() for df in dfs.values())

        logger.info(
            f'Common alignment window: '
            f'{start} -> {end}'
        )

        # -----------------------------------
        # TRIM TO COMMON WINDOW
        # -----------------------------------

        carbon = self.carbon_df.loc[start:end]

        generation = self.gen_df.loc[start:end]

        weather = self.weather_df.loc[start:end]

        # -----------------------------------
        # REPORT ROW COUNTS BEFORE MERGE
        # -----------------------------------

        logger.info(
            f"Rows before merge | "
            f"carbon={len(carbon):,} | "
            f"generation={len(generation):,} | "
            f"weather={len(weather):,}"
        )

        # -----------------------------------
        # MERGE DATASETS
        # -----------------------------------

        merged = (
            carbon
            .join(generation, how="inner")
            .join(weather, how="inner")
        )

        # -----------------------------------
        # REPORT FINAL ALIGNMENT
        # -----------------------------------

        logger.info(
            f'Final merged dataset: '
            f'{len(merged):,} rows'
        )

        rows_lost = min(
            len(carbon),
            len(generation),
            len(weather)
        ) - len(merged)

        if rows_lost > 0:

            logger.warning(
                f"{rows_lost:,} timestamps lost during alignment."
            )

        return merged

        
    
    def transform(self):

        self.carbon_df = self.carbon_df.rename(columns={"from": "datetime"}).set_index("datetime")
        self.gen_df = self.gen_df.rename(columns={"DATETIME": "datetime"}).set_index("datetime")
        # weather_df already has "time" as index
        self.weather_df.index.name = "datetime"

        self.weather_df = self.weather_df.resample('30min').interpolate()

        # -----------------------------------
        # INPUT DATASETS
        # -----------------------------------

        dfs = {
            "carbon": self.carbon_df,
            "generation": self.gen_df,
            "weather": self.weather_df
        }

        merged = self._align(dfs)

        return merged




if __name__ == '__name__':

    logging.basicConfig(level=logging.INFO)

    start = '2021-01-01'
    end = '2021-12-31'

    carbon_client = CarbonIntensityClient()
    carbon = carbon_client.fetch(start=start, end=end)

    generation_client = GenerationMixClient()
    generation = generation_client.fetch()

    weather_client = WeatherClient()
    weather = weather_client.fetch(start_date=start, end_date=end)

    alignement = AlignmentPipeline(
        carbon_df=carbon,
        gen_df=generation,
        weather_df=weather)
    
    final_df = alignement.transform()

    print(final_df.shape)
    print(final_df.columns)
    print(final_df.head())
