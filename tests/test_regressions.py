import sys
import numpy as np
import pandas as pd
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from regressions import normal_test, calculate_residuals, has_multicolinearity, errors_autocorrelate 
from regressions import error_features_correlate, is_homoscedastic, boxcox_transform, join_dataset
sys.path.append('.')

@patch('regressions.boxcox', return_value=[[1, 2, 3], 1])
@patch('regressions.pd')

class TestRegressions(unittest.TestCase):

    def test_calculate_residuals(self, mock_pandas, mock_boxcox):
        model = Mock()
        X = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        model.predict = MagicMock(return_value=np.array([7, 8, 9]))
        residuals = calculate_residuals(model, X, y)
        model.predict.assert_called_with(X)
        self.assertTrue(np.array_equal(residuals, np.array([-3, -3, -3]))) 

    def test_has_multicolinearity(self, mock_pandas, mock_boxcox):
        X = pd.DataFrame({ 'column1': [1, 2, 3], 'column2': [1, 2, 3] })
        self.assertTrue(has_multicolinearity(X))
        X = pd.DataFrame({ 'column1': [1, 1, 1], 'column2': [1, 1, 1] })
        self.assertFalse(has_multicolinearity(X))
        X = pd.DataFrame({ 'column1': [1, 2, 3], 'column2': [3, 2, 1] })
        self.assertTrue(has_multicolinearity(X))
        X = pd.DataFrame({ 'column1': [1, 1, 1], 'column2': [1, 1, 1] })
        self.assertTrue(has_multicolinearity(X, ignore_nan=False))

    def test_normal_test(self, mock_pandas, mock_boxcox):
        np.random.seed = 1
        normally_distributed = np.random.default_rng().normal(0, 0.1, 500)
        abnormally_distributed = np.random.default_rng().gamma(2.,2., 500)
        self.assertTrue(normal_test(normally_distributed))
        self.assertFalse(normal_test(abnormally_distributed))

    def test_errors_autocorrelate(self, mock_pandas, mock_boxcox):
        residuals = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        self.assertTrue(errors_autocorrelate(residuals))
        residuals = np.array([1, 2, 1, 1, 3, 1, 4, 2, 4])
        self.assertFalse(errors_autocorrelate(residuals))

    def test_errors_features_correlate(self, mock_pandas, mock_boxcox):
        residuals = np.array([1, 2, 3])
        X = pd.DataFrame({ 'column1': [1, 2, 3], 'column2': [1, 2, 3] })
        self.assertTrue(error_features_correlate(residuals, X))
        X = pd.DataFrame({ 'column1': [1, 1, 1], 'column2': [2, 9, 1] })
        self.assertFalse(error_features_correlate(residuals, X))      

    def test_is_homoscedastic(self, mock_pandas, mock_boxcox):
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        residuals = np.array([1, 3, 1, -3, 1, 3, 1, -3, 1, 3])
        self.assertTrue(is_homoscedastic(residuals, y))
        residuals = np.array([1, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        self.assertFalse(is_homoscedastic(residuals, y))

    def test_join_dataset(self, mock_pandas, mock_boxcox):
        X_train = pd.DataFrame({ "column1": [1, 2, 3], "column2": [7, 8, 9] })
        X_test = pd.DataFrame({ "column1": [4, 5, 6], "column2": [10, 11, 12] })
        y_train = np.array([1, 2, 3])
        y_test = np.array([4, 5, 6])
        X, y = join_dataset(X_train, X_test, y_train, y_test)
        self.assertTrue(mock_pandas.concat.called)
        self.assertTrue(mock_pandas.concat.call_count, 2)
        self.assertTrue(mock_pandas.concat.call_args[0][0], [y_train, y_test])

    def test_boxcox_transform(self, mock_pandas, mock_boxcox):
        y = np.array([1, 2, 4, 8, 16, 32])
        _, _, _ = boxcox_transform(y)
        self.assertTrue(mock_boxcox.called)
        self.assertEquals(mock_boxcox.call_count, 1)
        
    
        