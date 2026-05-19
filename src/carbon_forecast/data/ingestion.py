import pandas as pd
import logging
from carbon_forecast.data.base import BaseAPIClient

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


################################################################
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO) # loggs Info and above. The hierarchy is DEBUG < INFO < WARNING < ERROR < CRITICAL. Setting INFO means you'll see INFO, WARNING, and ERROR messages but not DEBUG.
#
#   client = CarbonIntensityClient(retry=3, timeout=5)
#    df = client.fetch(start="2024-01-01", end="2024-07-07")

    client = GenerationMixClient()
    df = client.fetch()

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['from'].min()} -> {df['from'].max()}")
    print(f"Columns: {df.columns.tolist()}")
