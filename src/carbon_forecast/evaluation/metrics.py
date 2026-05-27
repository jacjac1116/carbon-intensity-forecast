from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_pinball_loss,
)
import numpy as np
import logging
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class EvaluationReport():
    """
    Compute and report forecasting evaluation metrics.

    Supported metrics:
    - MAE
    - RMSE
    - R² score
    - Pinball loss
    - Persistence baseline comparation

    This class centralises evaluation logic to ensure:
    - consistent reporting
    - reusable metrics
    - cleaner training pipelines
    """

    def __init__(self, y_true, y_pred, baseline_pred=None, horizon=None, quantile_preds: pd.DataFrame = None):

        """
        Initialise evaluation report.

        Args:
            y_true:
                Ground truth target values.

            y_pred:
                Model predictions.

            baseline_pred:
                Optional persistence baseline predictions.

            horizon:
                Forecast horizon in half-hour periods.

                Examples:
                    2   -> 1 hour
                    48  -> 24 hours
        """

        self.y_true = y_true
        self.y_pred = y_pred
        self.baseline_pred = baseline_pred
        self.horizon = horizon
        self.quantile_preds = quantile_preds
    
    def compute(self) -> dict:
        """
        Returns all metrics as a dictionary
        """

        metrics = {
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'r2_score': r2_score(self.y_true, self.y_pred)
        }

        if self.quantile_preds is not None:
            lower = self.quantile_preds['q10']
            upper = self.quantile_preds['q90']

            # Prediction Interval Coverage Probability. Target: ~80% for 10th-90th
            metrics['picp'] = np.mean((self.y_true >= lower) & (self.y_true <= upper))
            
            # Prediction Interval Normalised Average Width. Lower is better (sharper intervals)
            metrics['pinaw'] = np.mean(upper-lower) / (self.y_true.max() - self.y_true.min())

            for col in self.quantile_preds.columns:
                q = int(col.replace('q', '')) / 100
                metrics[f"pinball_{col}"] = mean_pinball_loss(
                    self.y_true, self.quantile_preds[col], alpha=q
                )

        if self.baseline_pred is not None:
            metrics['persistence_mae'] = mean_absolute_error(self.y_true, self.baseline_pred)
            metrics['persistence_rmse'] = np.sqrt(mean_squared_error(self.y_true, self.baseline_pred))

        return metrics
    
    def summary(self) -> dict:
        """
        Return a formatted string summary
        """

        metrics = self.compute()

        hours = self.horizon / 2 if self.horizon is not None else None

        logger.info(f"\n{'='*50}")
        logger.info(f"RESULTS — t+{hours:.0f}h forecast" if hours else "RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Model MAE:       {metrics['mae']:.2f}")
        
        if 'persistence_mae' in metrics:
            logger.info(f"Persistence MAE: {metrics['persistence_mae']:.2f}")
            logger.info(f"Improvement:     {(1 - metrics['mae']/metrics['persistence_mae'])*100:.1f}%")
       
        logger.info(f"Model RMSE:      {metrics['rmse']:.2f}")

        if 'persistence_rmse' in metrics:
            logger.info(f"Persistence RMSE:{metrics['persistence_rmse']:.2f}")

        logger.info(f"{'=' * 50}")

        return metrics


        