import sys
import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from regressions import errors_are_normal
sys.path.append('.')

@patch('regressions.pd')

class TestRegressions(unittest.TestCase):

    def test_errors_are_normal(self, mock_pandas):
        np.random.seed = 1
        normally_distributed = np.random.default_rng().normal(0, 0.1, 500)
        abnormally_distributed = np.random.default_rng().gamma(2.,2., 500)
        self.assertTrue(errors_are_normal(normally_distributed))
        self.assertFalse(errors_are_normal(abnormally_distributed))