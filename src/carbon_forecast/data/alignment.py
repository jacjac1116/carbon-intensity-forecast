import pandas as pd
import logging
from carbon_forecast.data.ingestion import (
    CarbonIntensityClient,
    GenerationMixClient,
    WeatherClient,
)

logger = logging.getLogger(__name__)

class AlignmentPipeline:
    """
    Pipeline for aligning and merging multiple energy-related datasets.

    This pipeline standardises datetime indexes, validates data quality,
    detects gaps/duplicates, aligns datasets to a common time window,
    and merges them into a single modelling-ready DataFrame.

    Datasets:
    - Carbon intensity data
    - Generation mix data
    - Weather data
    """

    def __init__(
        self,
        carbon_df: pd.DataFrame,
        gen_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ):
        """
        Initialise AlignmentPipeline.

        Args:
            carbon_df:
                Carbon intensity dataset.

            gen_df:
                Electricity generation mix dataset.

            weather_df:
                Historical weather dataset.
        """

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

            # Expected granularity
            expected = pd.Timedelta(minutes=30)

            # Calculate difference between consecutive timestamps
            #
            # Example:
            # 00:00 -> 00:30 = 30 mins
            # 00:30 -> 02:00 = 90 mins (gap)
            freq = (
                df.index
                .to_series()
                .diff()
                .dropna()
            )

            # Find intervals that are not 30 minutes
            gaps = freq[freq != expected]

            if not gaps.empty:

                logger.warning(
                    f"{name}: "
                    f"{len(gaps)} irregular intervals detected"
                )

                # -----------------------------------
                # MISSING TIMESTAMP DETECTION
                # -----------------------------------

                # Build expected continuous 30-minute timeline
                expected_index = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq="30min"
                )

                # Detect missing timestamps
                missing_timesteps = (
                    expected_index
                    .difference(df.index)
                )

                logger.warning(
                    f"{name}: Missing timestamps detected:\n"
                    f"{missing_timesteps}"
                )

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

        # Use inner joins to keep only timestamps
        # present in all datasets

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

        
    
    def transform(self) -> pd.DataFrame:
        """
        Standardise and align all datasets.

        This method:
        - Standardises datetime column names
        - Sets datetime indexes
        - Resamples weather data to 30-minute frequency
        - Aligns all datasets
        - Returns merged modelling dataset

        Returns:
            pd.DataFrame:
                Fully aligned merged dataset.
        """

        # -----------------------------------
        # STANDARDISE DATETIME COLUMNS
        # -----------------------------------

        # Carbon dataset:
        # "from" -> "datetime"
        self.carbon_df = (
            self.carbon_df
            .rename(columns={"from": "datetime"})
            .set_index("datetime")
        )

        # Generation dataset:
        # "DATETIME" -> "datetime"
        self.gen_df = (
            self.gen_df
            .rename(columns={"DATETIME": "datetime"})
            .set_index("datetime")
        )

        # Weather dataset already uses datetime index
        self.weather_df.index.name = "datetime"

        # -----------------------------------
        # RESAMPLE WEATHER DATA
        # -----------------------------------

        # Weather data is typically hourly.
        # Interpolate to 30-minute granularity
        # to align with settlement-period datasets.
        self.weather_df = (
            self.weather_df
            .resample("30min")
            .interpolate()
        )

        # -----------------------------------
        # INPUT DATASETS
        # -----------------------------------

        dfs = {
            "carbon": self.carbon_df,
            "generation": self.gen_df,
            "weather": self.weather_df
        }

        # Align and merge datasets
        merged = self._align(dfs)

        return merged


if __name__ == "__main__":

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
    # OUTPUT SUMMARY
    # -----------------------------------

    print(final_df.shape)

    print(final_df.columns)

    print(final_df.head())
