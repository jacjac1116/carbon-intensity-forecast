import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CarbonIntensityClient:
    """
    Client for retrieving UK National Grid carbon intensity data.

    Features:
        - Automatic date range chunking (14-day API limit)
        - Retry handling
        - Timeout handling
        - API and HTTP error handling
        - JSON normalisation into pandas DataFrame
    """

    BASE_URL = "https://api.carbonintensity.org.uk"

    def __init__(self, retry: int = 3, timeout: int = 10):

        # Number of retry attempts per API request
        self.retry = retry

        # Request timeout in seconds
        self.timeout = timeout

        # Reuse HTTP connections for better performance
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

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

    def _make_request(self, url: str) -> dict | None:
        """
        Execute API request with retry handling.

        Returns:
            Parsed JSON dictionary if successful.
            None if all retries fail.
        """
        for attempt in range(1, self.retry + 1):

            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()

                data = r.json()

                # Check for API-level errors
                if "error" in data:
                    code = data["error"]["code"]
                    msg = data["error"]["message"]
                    logger.error(f"API Error | {code}: {msg}")
                    return None

                return data

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Timeout | Attempt {attempt}/{self.retry} | {url}"
                )

            except requests.exceptions.HTTPError as e:
                logger.error(
                    f"HTTP Error | Attempt {attempt}/{self.retry} | {e}"
                )

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request Error | Attempt {attempt}/{self.retry} | {e}"
                )

            except ValueError:
                logger.error(f"Invalid JSON response | {url}")
                return None

        logger.error(f"All {self.retry} attempts failed | {url}")
        return None

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
                f"{self.BASE_URL}"
                f"/intensity/{d['start']}/{d['end']}"
            )

            data = self._make_request(url)

            if data is None:
                failed_ranges.append(d)
                continue

            if "data" not in data or not data["data"]:
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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO) # loggs Info and above. The hierarchy is DEBUG < INFO < WARNING < ERROR < CRITICAL. Setting INFO means you'll see INFO, WARNING, and ERROR messages but not DEBUG.

    client = CarbonIntensityClient(retry=3, timeout=5)
    df = client.fetch(start="2024-01-01", end="2024-07-07")

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['from'].min()} -> {df['from'].max()}")
    print(f"Columns: {df.columns.tolist()}")
