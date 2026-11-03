from carbon_forecast.models.quantile import quantile_sorter
import numpy as np

def test_quantile_sorter():
    arr = np.array([[1, 2, 3], [4, 6, 5], [5, 4, 6]])
    
    np.testing.assert_array_equal(quantile_sorter(arr), np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]]))
