from carbon_forecast.models.lgbm import LGBMForecaster
from carbon_forecast.models.base import BaseForecaster
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def quantile_sorter(preds: np.array) -> np.array:
    return np.sort(preds, axis=1)

class QuantileForecaster(BaseForecaster):
    
    def __init__(self, params: dict, quantiles: list[float] = None):

        super().__init__()
        self.params = params

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.list_models = {}

        for quantile in quantiles:
            model_params = {**self.params, 
                            'objective': 'quantile', 
                            'alpha': quantile}
            model = LGBMForecaster(model_params)

            self.list_models[quantile] = model


    def fit(self, X, y):
        
        for alpha, model in self.list_models.items():
            logger.info(f'Fitting quantile {alpha}')
            model.fit(X, y)
        self.feature_names_ = list(X.columns)

    def predict(self, X):

        predictions = []
        for model in self.list_models.values():
            predictions.append(model.predict(X))

        stacked = np.column_stack(predictions)
        ordered = quantile_sorter(stacked)
        columns = [f'q{int(alpha * 100)}' for alpha in sorted(self.list_models.keys())]

        return pd.DataFrame(ordered, columns=columns, index = X.index)

            

