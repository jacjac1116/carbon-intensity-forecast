"""Feature engineering - lags, rolling statistics, calendar features."""


def create_lag_features(df, target_col, lags):
    """Create lagged features from time series.

    Args:
        df: Input DataFrame.
        target_col: Column to lag.
        lags: List of lag values.

    Returns:
        DataFrame with lag features added.
    """
    pass


def create_rolling_features(df, target_col, windows):
    """Create rolling window statistics.

    Args:
        df: Input DataFrame.
        target_col: Column to compute rolling stats on.
        windows: List of window sizes.

    Returns:
        DataFrame with rolling features added.
    """
    pass
