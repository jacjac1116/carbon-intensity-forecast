import numpy as np

class AnalysisTools:
    """
    Tools the agent can call to instigate model failures
    """

    def __init__(self, y_true, y_pred, df, threshold_multiplier=2.0):
        self.y_true = y_true
        self.y_pred = y_pred
        self.df = df
        self.errors = np.abs(y_true - y_pred)
        self.mae = self.errors.mean()
        self.failure_mask = self.errors >= self.mae * threshold_multiplier

    def get_error_correlation(self, feature_name: str) -> dict:
        """
        "Analyse how a specific feature relates to model prediction errors. "
                "Returns the Pearson correlation between the feature values and absolute prediction error, "
                "along with average model error when the feature is in its lowest and highest quartiles. "
                "Also returns the error ratio and difference between low and high feature regimes. "
                "Use this tool when investigating which features are associated with larger model failures "
                "or whether model performance degrades under certain operating conditions."
        """
        if feature_name not in self.df.columns:
            raise ValueError(f'Feature "{feature_name}" not found in DataFrame.')
        
        feature_values = self.df[feature_name]
        correlation = np.corrcoef(feature_values, self.errors)[0,1] # [[corr(a,a), corr(a,b)],
                                                                    # [corr(b,a), corr(b,b)]]
        
        # Error in top vs bottom quantiles of the feature
        q25 = feature_values.quantile(0.25)
        q75 = feature_values.quantile(0.75)

        errors_when_low = self.errors[feature_values <= q25].mean()
        errors_when_high = self.errors[feature_values >= q75].mean()

        return {
            'feature': feature_name,
            'correlation_with_error': round(correlation, 4),
            'error_when_low': round(errors_when_low, 2),
            'error_when_high': round(errors_when_high, 2),
            'low_quantile': round(q25, 2),
            'high_quantile': round(q75, 2),
            'error_difference': round(errors_when_high - errors_when_low, 2),
            'error_ratio': round(errors_when_high / errors_when_low, 2) if errors_when_low > 0 else 'inf'
        }
    
    def compare_failure_vs_success(self, feature_name: str) -> dict:
        """
        Compare how a feature behaves during model failures versus normal operation. "
                "Failures are defined as predictions whose absolute error exceeds a threshold "
                "relative to overall MAE. "
                "Returns mean and standard deviation of the feature during failures and successes, "
                "plus the differences between them. "
                "Use this tool to identify whether specific conditions are overrepresented during failures, "
                "such as high demand, low wind, or extreme temperatures."
            """

        if feature_name not in self.df.columns:
            raise ValueError(f'Feature "{feature_name}" not found in DataFrame')
        
        feature_values = self.df[feature_name]
        failure = feature_values[self.failure_mask]
        success = feature_values[~self.failure_mask]

        return {
            'feature': feature_name,
            'failure_mean': round(failure.mean(), 2),
            'success_mean': round(success.mean(), 2),
            'mean_difference': round(failure.mean() - success.mean(), 2),
            'failure_std': round(failure.std(), 2),
            'success_std': round(success.std(), 2),
            'std_difference': round(failure.std() - success.std(), 2),
        }
    
    def get_episode_deep_dive(self, start: str, end: str) -> dict:
        """
        "Perform detailed analysis for a specific time window or operational event. "
                "Returns duration, mean and maximum prediction error, "
                "plus summary statistics for all features during the selected period. "
                "Use this tool to investigate major forecast failures, storms, "
                "demand spikes, unusual weather events, or anomalous behaviour."
          
        """

        mask = (self.df.index >= start) & (self.df.index <= end)
        if not mask.any():
            raise ValueError(f'No data found between {start} and {end}')
        
        episode_df = self.df[mask]
        episode_errors = self.errors[mask]

        return {
            'start': start,
            'end': end,
            'duration': (episode_df.index.max() -episode_df.index.min()).total_seconds() / 3600,
            'mean_error': round(episode_errors.mean(), 2),
            'max_error': round(episode_errors.max(), 2),
            'feature_means': episode_df.mean().round(2).to_dict(),
            'feature_stds': episode_df.std().round(2).to_dict(),
        }
    
    def get_feature_at_extremes(self, feature_name: str, percentile: float) -> dict:
        """
        "Evaluate model performance when a feature reaches unusually high values. "
                "Returns the percentile threshold, the corresponding feature value, "
                "average prediction error under extreme conditions, and the number of affected samples. "
                "Use this tool to determine whether the model struggles during extreme events "
                "such as high carbon intensity, strong wind generation, heatwaves, "
                "or unusually high electricity demand."
        """
        if feature_name not in self.df.columns:
            raise ValueError(f'Feature "{feature_name}" not found in DataFrame')
        
        feature_values = self.df[feature_name]
        threshold = feature_values.quantile(percentile)

        extreme_mask = feature_values >= threshold
        extreme_errors = self.errors[extreme_mask]

        return {
            'feature': feature_name,
            'percentile': percentile,
            'threshold_value': round(threshold, 2),
            'mean_error_at_extreme': round(extreme_errors.mean(), 2),
            'error_count_at_extreme': int(extreme_mask.sum())
        }
