from abc import ABC, abstractmethod
import joblib
import logging

logger = logging.getLogger(__name__)

class BaseForecaster(ABC):
    """
    Abstract base class for forecasting models.

    This class defines a standard interface for all forecasting
    models used within the project.

    Core responsabilities:
    - Model training (fit)
    - Prediction generation (predict)
    - Model persistence (save)
    - Model loading (load)

    Child classes should inherit from this base class and implement:
    - fit()
    - predict()
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f'Model saved to {path}')
    
    @classmethod
    def load(cls, path: str):
        """
        Load saved model from disk.

        This method is defined as a class method because it belongs
        to the class itself rather than an individual model instance.

        Using @classmethod allows the model to be loaded directly
        from the class without first creating an object.

        Example:
            model = XGBoostForecaster.load(
                'models/xgb_model.pkl'
                )
        
        Instead of:
            model = XGBoostForecaster()
            model.load('...')
        """
        model = joblib.load(path)
        logger.info(f'Model loaded from {path}')

        return model




