import pandas as pd
import logging
from carbon_forecast.data.base import BaseAPIClient
from datetime import datetime

logger = logging.getLogger(__name__)


class CarbonIntensityClient(BaseAPIClient):
    """
    Client for retrieving UK National Grid carbon intensity data.

    This specific client extends 'BaseAPIClient' and provides functionality
    specific to the UK Carbon Intensity API

    Main responsibilities:
    - Retrieve national carbon intensity data
    - Handle API date-range limitations through chunking
    - Reuse shared retry/error handling from BaseAPIClient
    - Convert API responses into pandas DataFrames

    Inherited functionality from BaseAPIClient:
    - Persistent HTTP session management
    - Retry handling
    - Timeout handling
    - HTTP/API error handling
    - JSON response parsing
    """

    def __init__(self, **kwargs):
        """
        Initialise Carbon Intensity API client.

        Args:
            **kwargs:
                Additional keyword arguments passed to BaseAPIClient.

                Common options:
                - retry (int):
                    Number of retry attempts.

                - timeout (int):
                    Request timeout in seconds.
        """

        # Initialise shared base client functionality
        super().__init__(
            base_url="https://api.carbonintensity.org.uk",
            **kwargs
        )

        # Set default request headers for JSON responses
        self.session.headers.update({
            "Accept": "application/json"
        })

    def _generate_ranges(
        self,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> list[dict[str, str]]:
        """
        Split large date ranges into 14-day request chunks
        because API has a maximum request window.

        Returns:
            List of dictionaries:
            [
                {'start': ..., 'end': ...},
                ...
            ]
        """
        ranges = []
        current_start = start_dt

        while current_start < end_dt:

            current_end = min(
                current_start + pd.Timedelta(days=13),
                end_dt,
            )

            ranges.append({
                "start": current_start.strftime("%Y-%m-%dT%H:%MZ"),
                "end": current_end.strftime("%Y-%m-%dT%H:%MZ"),
            })

            current_start = current_end + pd.Timedelta(minutes=30)

        return ranges

    def _parse_data(self, data: dict) -> pd.DataFrame:
        """
        Normalise nested API JSON into a clean DataFrame.
        """
        df = pd.json_normalize(data["data"])

        # Remove nested prefix (e.g. 'intensity.actual' -> 'actual')
        df.columns = df.columns.str.replace(
            "intensity.", "", regex=False
        )

        df["from"] = pd.to_datetime(df["from"], utc=True)
        df["to"] = pd.to_datetime(df["to"], utc=True)

        return df

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve carbon intensity data between two dates.

        Parameters:
            start : str
                Example: '2024-01-01'
            end : str
                Example: '2024-07-01'

        Returns:
            Combined pandas DataFrame with columns:
                from, to, forecast, actual, index
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        ranges = self._generate_ranges(start_dt, end_dt)

        dfs = []
        failed_ranges = []

        for d in ranges:

            url = (
                f"{self.base_url}"
                f"/intensity/{d['start']}/{d['end']}"
            )

            data = self._make_request(url)

            if data is None:
                failed_ranges.append(d)
                continue

            if not data or "data" not in data or not data["data"]:
                start_str = d["start"]
                end_str = d["end"]
                logger.warning(f"No data returned for {start_str} -> {end_str}")
                continue

            df = self._parse_data(data)
            dfs.append(df)

        if not dfs:
            raise ValueError("No valid data retrieved from API.")

        final_df = (
            pd.concat(dfs, ignore_index=True)
            .sort_values("from")
            .drop_duplicates(subset="from")
            .reset_index(drop=True)
        )

        if failed_ranges:
            logger.warning(f"{len(failed_ranges)} date ranges failed:")
            for r in failed_ranges:
                logger.warning(f"  {r['start']} -> {r['end']}")

        return final_df

class GenerationMixClient(BaseAPIClient):
    """
    Client for retrieving UK electricity generation mix data from NESO.

    This dataset provides time series information on electricity generation
    by fuel type (e.g. gas, coal, wind, solar, nuclear).

    Key features:
    - Pulls full generation mix dataset from NESO datastore API
    - Normalises nested JSON response into a pandas DataFrame
    - Sorts data chronologically
    - Removes redundant percentage and derived columns for cleaner analysis

    Typical use cases:
    - Energy mix analysis over time
    - Renewable penetration studies
    - Carbon intensity correlation analysis
    """

    def __init__(self, **kwargs):
        """
        Initialise GenerationMixClient.

        Args:
            **kwargs:
                Passed through to BaseAPIClient.

                Common parameters:
                - retry (int): number of retry attempts
                - timeout (int): request timeout in seconds
        """

        # NESO datastore endpoint for generation mix data
        super().__init__(
            base_url=(
                "https://api.neso.energy/api/3/action/datastore_search"
                "?resource_id=f93d1835-75bc-43e5-84ad-12472b180a98"
                "&limit=310000"
            ),
            **kwargs
        )

    def fetch(self, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Fetch generation mix data from NESO API and return as a DataFrame.

        Returns:
            pd.DataFrame:
                Cleaned and time-sorted generation mix dataset.

        Raises:
            ValueError:
                If API response contains no records or is malformed.
        """

        # Execute API request using shared base method
        data = self._make_request(self.base_url)

        # Validate response structure
        if not data or 'result' not in data or not data['result']['records']:
            logger.error("No data was found, aborting ...")
            raise ValueError("No generation mix data returned from API.")

        # Flatten JSON records into tabular structure
        df = pd.json_normalize(data['result']['records'])

        # Ensure chronological ordering for time-series analysis
        df = df.sort_values(by='DATETIME')

        # Drop percentage and derived columns to keep dataset lightweight
        # (These can be recalculated if needed for analysis)
        df = df.drop(columns=[
            '_id',
            'GAS_perc',
            'COAL_perc',
            'NUCLEAR_perc',
            'WIND_perc',
            'WIND_EMB_perc',
            'HYDRO_perc',
            'IMPORTS_perc',
            'BIOMASS_perc',
            'OTHER_perc',
            'SOLAR_perc',
            'STORAGE_perc',
            'GENERATION_perc',
            'LOW_CARBON_perc',
            'ZERO_CARBON_perc',
            'RENEWABLE_perc',
            'FOSSIL_perc',
            'WIND_EMB',
            'CARBON_INTENSITY'
        ])

        df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
        if start:
            df = df[df["DATETIME"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["DATETIME"] <= pd.Timestamp(end, tz="UTC")]

        return df
    
class WeatherClient(BaseAPIClient):
    """
    Client for retrieving historical weather data from Open-Meteo.

    This client is designed specifically for energy and carbon intensity
    forecasting applications.

    Main responsibilities:
    - Retrieve historical hourly weather data
    - Validate input date formats
    - Query multiple strategic UK locations
    - Rename weather variables with location prefixes
    - Combine all locations into a single feature table

    Default locations were selected to represent:
    - Offshore wind conditions
    - Onshore wind conditions
    - Electricity demand centres
    - Solar generation regions
    """

    def __init__(self, **kwargs):
        """
        Initialise WeatherClient.

        Args:
            **kwargs:
                Additional keyword arguments passed to BaseAPIClient.

                Common options:
                - retry (int):
                    Number of retry attempts.

                - timeout (int):
                    Request timeout in seconds.
        """
        super().__init__(
            base_url='https://archive-api.open-meteo.com/v1/archive',
            **kwargs
        )
    
    def _validate_date(self, date_str: str) -> None:
        """
        Validate date string format.

        Expected format:
            YYYY-MM-DD

        Args:
            date_str:
                Input date string.

        Raises:
            ValueError:
                If date format is invalid.
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid date format: {date_str}. "
                "Expected format: YYYY-MM-DD"
            )
    
    def fetch(self, 
              start_date: str, 
              end_date: str,
              forecast: bool = False,
              cols: list[str] = None,
              locations: dict = None
              ):
        """
        Fetch historical weather data for multiple locations.

        Args:
            start_date:
                Start date in YYYY-MM-DD format.

            end_date:
                End date in YYYY-MM-DD format.

            cols:
                List of hourly weather variables to retrieve.

                Defaults to:
                - temperature_2m
                - wind_speed_100m
                - direct_radiation
                - cloud_cover
                - pressure_msl

            locations:
                Dictionary of locations and coordinates.

                Format:
                    {
                        "location_name": [latitude, longitude]
                    }

        Returns:
            pd.DataFrame:
                Combined weather dataset indexed by UTC timestamp.

        Raises:
            ValueError:
                If date format is invalid.
        """

        # -----------------------------------
        # DEFAULT WEATHER VARIABLES
        # -----------------------------------

        # Default weather variables selected for:
        # - wind generation forecasting
        # - solar generation forecasting
        # - electricity demand modelling
        if cols is None:

            cols = [
                "temperature_2m",
                "wind_speed_100m",
                "direct_radiation",
                "cloud_cover",
                "pressure_msl"
            ]

        # -----------------------------------
        # DEFAULT ENERGY-RELEVANT LOCATIONS
        # -----------------------------------

        if locations is None:

            locations = {

                # Offshore wind generation proxy
                "aberdeen": [57.15, -2.09],

                # Onshore wind + Atlantic weather systems
                "glasgow": [55.86, -4.25],

                # Demand / population centre proxy
                "london": [51.51, -0.13],

                # Solar generation / southwest weather proxy
                "exeter": [50.72, -3.53]
            }

        # -----------------------------------
        # INPUT VALIDATION
        # -----------------------------------

        # Validate input date formats
        self._validate_date(start_date)
        self._validate_date(end_date)

        # Store location-specific DataFrames
        dfs = []

        # -----------------------------------
        # FETCH WEATHER DATA
        # -----------------------------------

        # Loop through each configured location
        for name, coords in locations.items():

            # API query parameters
            params = {

                # Geographic coordinates
                "latitude": coords[0],
                "longitude": coords[1],

                # Requested hourly weather variables
                "hourly": ",".join(cols),

                # Date range
                "start_date": start_date,
                "end_date": end_date,

                # Force UTC timestamps for alignment with
                # NESO / carbon intensity datasets
                "timezone": "UTC"
            }

            # Execute API request using shared BaseAPIClient logic
            base = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
                    if forecast
                    else self.base_url
                )

            data = self._make_request(base, params=params)

            # -----------------------------------
            # DATA TRANSFORMATION
            # -----------------------------------

            # Convert hourly weather JSON into DataFrame
            df = pd.DataFrame(data["hourly"])

            # Convert timestamps into timezone-aware datetime
            df["time"] = pd.to_datetime(
                df["time"],
                utc=True
            )

            # Rename weather columns using location prefix
            prefix = f'{name}_fcst' if forecast else name

            df = df.rename(columns={col:f'{prefix}_{col}' for col in df.columns if col != 'time'})

            # Use timestamp as DataFrame index
            df = df.set_index("time")

            # Store processed location DataFrame
            dfs.append(df)

        # -----------------------------------
        # MERGE ALL LOCATIONS
        # -----------------------------------

        # Combine all weather locations into one wide feature table
        weather_df = pd.concat(
            dfs,
            axis=1
        )

        return weather_df


    


################################################################
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO) # loggs Info and above. The hierarchy is DEBUG < INFO < WARNING < ERROR < CRITICAL. Setting INFO means you'll see INFO, WARNING, and ERROR messages but not DEBUG.
#
#   client = CarbonIntensityClient(retry=3, timeout=5)
#    df = client.fetch(start="2024-01-01", end="2024-07-07")

    client = WeatherClient()
    df = client.fetch(start_date='2023-07-23', end_date='2023-07-25')

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
