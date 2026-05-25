import pandas as pd
import numpy as np
from carbon_forecast.evaluation.metrics import EvaluationReport
import logging

logger = logging.getLogger(__name__)


class FailureDetector:

    def __init__(self, y_true, y_pred, df, threshold_multiplier=2.0):

        self.y_true = y_true
        self.y_pred = y_pred
        self.df = df
        self.threshold_multiplier = threshold_multiplier

    def _detect_episodes(self):
        """
        Find contiguous failure windows.
        """
        evaluator = EvaluationReport(
            y_true=self.y_true,
            y_pred=self.y_pred,
        )
        analysis = evaluator.compute()

        error_df = pd.DataFrame({
            "actual": self.y_true,
            "predicted": self.y_pred,
            "abs_error": np.abs(self.y_true - self.y_pred),
            "error": self.y_true - self.y_pred,
            "above_threshold": np.abs(self.y_true - self.y_pred) >= analysis["mae"] * self.threshold_multiplier,
        }, index=self.df.index)

        # Flips boolean:
        # errors above threshold are 0
        # errors below threshold are 1
        # cumsum increments, consecutive Trues share the same group number
        error_df["episode_id"] = (~error_df["above_threshold"]).cumsum()

        failures = error_df[error_df["above_threshold"]]
        episodes = failures.groupby("episode_id")

        return episodes

    def _describe_conditions(self, summary: dict) -> str:
        """Generate human-readable condition description."""
        conditions = []

        # Wind
        wind_q15 = self.df["glasgow_wind_speed_100m"].quantile(0.15)
        wind_q75 = self.df["glasgow_wind_speed_100m"].quantile(0.75)
        wind_val = summary["avg_wind_gw"]

        if wind_val < wind_q15:
            conditions.append(f"Low wind ({wind_val:.1f} m/s)")
        elif wind_val > wind_q75:
            conditions.append(f"High wind ({wind_val:.1f} m/s)")

        # Temperature
        temp_q25 = self.df["london_temperature_2m"].quantile(0.25)
        temp_q75 = self.df["london_temperature_2m"].quantile(0.75)
        temp_val = summary["avg_temperature"]

        if temp_val < temp_q25:
            conditions.append(f"Cold ({temp_val:.1f}\u00b0C)")
        elif temp_val > temp_q75:
            conditions.append(f"Hot ({temp_val:.1f}\u00b0C)")

        # Demand
        demand_q75 = self.df["GENERATION"].quantile(0.75)

        if summary["avg_demand"] > demand_q75:
            conditions.append("High demand")

        # Time
        hour = summary["time_of_day"]

        if 0 <= hour < 6:
            conditions.append("Overnight")
        elif 17 <= hour < 21:
            conditions.append("Evening peak")

        # Season
        month = summary["month"]
        if month in [12, 1, 2]:
            conditions.append("Winter")
        elif month in [6, 7, 8]:
            conditions.append("Summer")

        return ", ".join(conditions) if conditions else "Normal conditions"

    def _characterise(self, episodes) -> pd.DataFrame:
        """
        Add condition summaries to each episode.
        """
        results = []

        for episode_id, episode in episodes:

            # Match episode timestamps to the full feature DataFrame
            episode_features = self.df.loc[episode.index]

            summary = {
                "episode_id": episode_id,
                "start": episode.index.min(),
                "end": episode.index.max(),
                "duration_hours": (episode.index.max() - episode.index.min()).total_seconds() / 3600,
                "mean_error": episode["abs_error"].mean(),
                "peak_error": episode["abs_error"].max(),
                "n_points": len(episode),
                # Conditions during the episode
                "avg_temperature": episode_features["london_temperature_2m"].mean(),
                "avg_wind_gw": episode_features["glasgow_wind_speed_100m"].mean(),
                "avg_pressure": episode_features["aberdeen_pressure_msl"].mean(),
                "avg_solar": episode_features["exeter_direct_radiation"].mean(),
                "avg_demand": episode_features["GENERATION"].mean(),
                "avg_carbon_intensity": episode_features["actual"].mean(),
                "time_of_day": episode.index[len(episode) // 2].hour,
                "month": episode.index[len(episode) // 2].month,
            }

            summary["conditions"] = self._describe_conditions(summary)
            results.append(summary)

        df = pd.DataFrame(results)
        return df

    def report(self) -> pd.DataFrame:

        episodes = self._detect_episodes()
        df = self._characterise(episodes)

        df = df.sort_values("mean_error", ascending=False)

        logger.info(f"\n{'='*60}")
        logger.info("FAILURE EPISODE ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Total episodes detected: {len(df)}")

        for i, (_, row) in enumerate(df.head(10).iterrows()):
            start = row["start"]
            end = row["end"]
            hours = row["duration_hours"]
            mean_err = row["mean_error"]
            peak_err = row["peak_error"]
            conds = row["conditions"]

            logger.info(f"\nEpisode {i + 1}: {start} -> {end} ({hours:.1f}h)")
            logger.info(f"  Mean Error: {mean_err:.1f} gCO2/kWh | Peak: {peak_err:.1f}")
            logger.info(f"  Conditions: {conds}")

        logger.info(f"\n{'='*60}")

        return df

        




