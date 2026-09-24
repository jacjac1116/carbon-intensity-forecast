from carbon_forecast.evaluation.metrics import EvaluationReport
import pandas as pd
import numpy as np
import holidays
import logging

logger = logging.getLogger(__name__)

class StratifiedEvaluator:

    def __init__(self, 
                 df: pd.DataFrame,
                 y_true: pd.Series,
                 y_pred: np.array,
                 params: list[str]= None,
                 HOUR_BINS: dict = None,
                 FEATURE_BINS: dict = None,
                 quantile_preds: list = None):

        self.df = df
        self.y_true = y_true
        self.y_pred = pd.Series(y_pred, index = y_true.index)

        # We define lists and dicts as None as a way of defensive coding
        # if they were defined in __init__ and I was to call the class under
        # 2 different variables, if I made changes on the list of the first one
        # the second variable would have those same changes as they'd share the
        # same list object.
        # This method creates a list object for every time the class is initiated
        # making 2 instances completetly independent of each other
        if params is None:
            params = ['season', 'weather', 'time_day', 'weekday', 'carbon_intensity', 'holiday']
        self.params = params

        if HOUR_BINS is None:
            HOUR_BINS = {"overnight": (0, 5), "morning": (6, 11), "afternoon": (12, 17), "evening_peak": (18, 21), "night": (22, 24)}
        self.HOUR_BINS = HOUR_BINS

        if FEATURE_BINS is None:
            FEATURE_BINS = {
                "very_low": (0.0, 0.10),
                "low": (0.10, 0.33),
                "medium": (0.33, 0.66),
                "high": (0.66, 0.90),
                "extreme": (0.90, 1.0),
            }
        self.FEATURE_BINS = FEATURE_BINS
        self.quantile_preds = quantile_preds


    def _slicer(self, param):
        
        slicer = {}

        if param == 'season':
            slicer['autumn'] = self.df['month'].isin([9, 10, 11])
            slicer['winter'] = self.df['month'].isin([12, 1, 2])
            slicer['spring'] = self.df['month'].isin([3, 4, 5])
            slicer['summer'] = self.df['month'].isin([6, 7, 8])
        
        elif param == 'time_day':
            for key, value in self.HOUR_BINS.items():
                slicer[key] = self.df['hour'].between(value[0], value[1])
        
        elif param == 'weekday':
            slicer['weekend'] = self.df['is_weekend']==1
            slicer['week_day'] = self.df['is_weekend']==0
        
        elif param == 'holiday':
            uk_holidays = holidays.UnitedKingdom(subdiv = 'England')
            slicer['holidays'] = self.df.index.normalize().isin(uk_holidays)
        
        elif param == 'carbon_intensity':
            for bin_name, (lower, upper) in self.FEATURE_BINS.items():
                lower_val = self.df['actual'].quantile(lower)
                upper_val = self.df['actual'].quantile(upper)
                slicer[bin_name] = self.df['actual'].between(lower_val, upper_val)
        
        elif param == 'weather':
            weather_cols = ["london_fcst_temperature_2m_target",
                "exeter_fcst_direct_radiation_target",
                "aberdeen_fcst_wind_speed_100m_target"]
            for col in weather_cols:
                for bin_name, (lower, upper) in self.FEATURE_BINS.items():
                    lower_val = self.df[col].quantile(lower)
                    upper_val = self.df[col].quantile(upper)
                    slicer[f"{col}_{bin_name}"] = self.df[col].between(lower_val,upper_val)

        return slicer
    
    def analyse(self):

        results= []

        for param in self.params:
            slicer = self._slicer(param)
            for slice_name, mask in slicer.items():
                y_true_sliced = self.y_true[mask]
                y_pred_sliced = self.y_pred[mask]

                if len(y_true_sliced) == 0:
                    logger.warning(f'Skipping empty slice: {param}/{slice_name}')
                    continue

                # Slice quantile predictions if available
                q_sliced  = None
                if self.quantile_preds is not None:
                    q_sliced = self.quantile_preds.loc[mask]
                
                evaluator = EvaluationReport(
                    y_true=y_true_sliced, 
                    y_pred=y_pred_sliced,
                    quantile_preds=q_sliced)
                analysis = evaluator.compute()

                analysis['slice'] = param
                analysis['condition'] = slice_name
                analysis['n_sample'] = len(y_pred_sliced)

                results.append(analysis)

        return pd.DataFrame(results)
    
    def report(self):

        metrics = ['mae', 'r2_score', 'rmse']

        results = self.analyse()

        evaluator = EvaluationReport(y_true=self.y_true, y_pred=self.y_pred, quantile_preds=self.quantile_preds)
        global_metrics = evaluator.compute()

        for metric in metrics:

            results[f"{metric}_vs_global"] = ((results[metric] / global_metrics[metric]) - 1) * 100

        # Sort by MAE descending — worst performers first
        results = results.sort_values("mae", ascending=False)

        # Flag slices where MAE is >25% worse than global
        results["flag"] = results["mae_vs_global"] > 25

        # -----------------------------------
        # LOG SUMMARY
        # -----------------------------------

        logger.info(f"\n{'='*70}")
        logger.info("MODEL FAILURE ANALYSIS")
        logger.info(f"{'='*70}")

        logger.info("\nWorst-performing slices:\n")

        logger.info(f"Global MAE: {global_metrics['mae']:.2f}")

        for _, row in results.head(5).iterrows():
            logger.info(
                f"[{row['condition']}] "
                f"{row['slice']} | "
                f"MAE={row['mae']:.2f} | "
                f"{row['mae_vs_global']:+.1f}% vs global | "
                f"n={row['n_sample']:,}"
            )

        logger.info(f"\n{'='*70}")

        return results