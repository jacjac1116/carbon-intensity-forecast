from abc import ABC, abstractmethod
import pandas as pd
import requests


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients.

    This class centralises the common functionality shared across
    multiple API integrations, including:

    - Persistent HTTP session handling
    - Request timeout configuration
    - Retry logic
    - HTTP/API error handling
    - JSON response parsing

    Child classes should inherit from this class and implement
    the `fetch()` method with API-specific logic.
    """

    def __init__(
        self,
        base_url: str,
        retry: int = 3,
        timeout: int = 10
    ):
        """
        Initialise the base API client.

        Args:
            base_url:
                Base endpoint URL for the API.

            retry:
                Number of retry attempts for failed requests.

            timeout:
                Timeout (seconds) for each HTTP request.
        """

        # Root/base URL for the API
        self.base_url = base_url

        # Maximum number of retry attempts
        self.retry = retry

        # Request timeout in seconds
        self.timeout = timeout

        # Reusable HTTP session improves performance by
        # persisting TCP connections across requests
        self.session = requests.Session()

    def _make_request(self, url: str, params: dict = None) -> dict | None:
        """
        Execute a GET request with retry and error handling.

        This method:
        - Sends the request
        - Validates HTTP status codes
        - Parses JSON responses
        - Handles retries for transient failures
        - Logs request/API errors

        Args:
            url:
                Fully constructed request URL.

        Returns:
            Parsed JSON dictionary if successful.

            None if:
            - all retry attempts fail
            - response JSON is invalid
            - API returns an error payload
        """

        # Retry loop
        for attempt in range(1, self.retry + 1):

            try:
                # Send GET request
                
                r = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
            

                # Raise exception for 4xx/5xx responses
                r.raise_for_status()

                # Convert response JSON to Python dictionary
                data = r.json()

                # Optional API-level error handling
                # (depends on API response structure)
                if "error" in data:

                    code = data["error"]["code"]
                    msg = data["error"]["message"]

                    logger.error(
                        f"API Error | {code}: {msg}"
                    )

                    return None

                return data

            # Request exceeded timeout limit
            except requests.exceptions.Timeout:

                logger.warning(
                    f"Timeout | "
                    f"Attempt {attempt}/{self.retry} | "
                    f"{url}"
                )

            # HTTP response returned error status
            except requests.exceptions.HTTPError as e:

                logger.error(
                    f"HTTP Error | "
                    f"Attempt {attempt}/{self.retry} | "
                    f"{e}"
                )

            # Generic request/network failure
            except requests.exceptions.RequestException as e:

                logger.error(
                    f"Request Error | "
                    f"Attempt {attempt}/{self.retry} | "
                    f"{e}"
                )

            # Response body could not be parsed as JSON
            except ValueError:

                logger.error(
                    f"Invalid JSON response | {url}"
                )

                return None

        # All retry attempts failed
        logger.error(
            f"All {self.retry} attempts failed | {url}"
        )

        return None

    @abstractmethod
    def fetch(
        self,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        Fetch and return API data for a date range.

        This method must be implemented by all subclasses.

        Args:
            start:
                Start date/time for query.

            end:
                End date/time for query.

        Returns:
            Pandas DataFrame containing processed API data.
        """

        pass