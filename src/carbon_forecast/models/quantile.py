from carbon_forecast.models.lgbm import LGBMForecaster
from carbon_forecast.models.base import BaseForecaster
import logging
import pandas as pd

logger = logging.getLogger(__name__)

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

        results = {}
        for alpha, model in self.list_models.items():

            results[f'q{int(alpha * 100)}'] = model.predict(X)

        return pd.DataFrame(results, index = X.index)

            

