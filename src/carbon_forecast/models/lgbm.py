from carbon_forecast.models.base import BaseForecaster
import lightgbm as lgb

class LGBMForecaster(BaseForecaster):
    """
    LightGBM forecasting model wrapper.

    This class provides a standard forecasting interface built on top
    of LightGBM's LGBMRegressor implementation.

    Design approach:
    - Inherits from BaseForecaster to enforce a shared API
    - Uses composition to store the LightGBM model internally

    Why use composition here?
    -------------------------
    Instead of inheriting directly from lgb.LGBMRegressor,
    this class stores the model inside:

        self.model = lgb.LGBMRegressor(...)
    
    This is called composition:
    - Keeps project interface clean and controlled
    - Makes it easier to swap ML libraries later
    - Avoids exposing the entire LightGBM library
    - Reduces tight coupling to external libraries, i.e., if lgb
        updated in the future, script could break
    - Allows adding custom project-specific logic

    Inheritance is still used via BaseForecaster because
    all models should expose:
        - fit()
        - predict()
        - save()
        - load()
    
        This combination gives:
        - shared project structure (inheritance)
        - flexible model implementation (composition)
    """

    def __init__(self, params: dict):
        """
        Initialise LightGBM forecasting model.

        Args:
            params:
                Dictionary of LightGBM hyperparameters.

        Example:
            params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6
            }
        """
        super().__init__() # does nothing but good practice
        # Composition:
        # store LightGBM model internally rather than inheriting from it
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X, y):
        """
        Train LightGBM model.

        Args:
            X:
                Training feature matrix.

            y:
                Training target vector.
        """

        self.model.fit(X, y)
        self.feature_names_ = list(X.columns)

    def predict(self, X):
        """
        Generate predictions using trained model.

        Args:
            X:
                Input feature matrix.

        Returns:
            Array of predicted values.
        """

        return self.model.predict(X)

    @property
    def feature_importance(self):
        """
        Return feature importance scores.

        Why use @property?
        ------------------
        @property allows this method to be accessed like an attribute
        insteaad of a function call.

        Example:
            model.feature_importance
        
        instead of:
            model.feature_importance()

        This is useful because feature importance behaves more like 
        model metadata than an action/function.

        Returns:
            dict:
                Mapping of feature names to importance scores
        """
        return dict(
            zip(
                self.feature_names_,
                self.model.feature_importances_
            )
        )