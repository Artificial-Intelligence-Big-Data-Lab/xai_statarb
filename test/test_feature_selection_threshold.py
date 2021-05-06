from unittest import TestCase
from src.feature_selection_threshold import *
from src.config import *


class FeatureSelectionThresholdTest(TestCase):
    def setUp(self) -> None:
        self.metrics = pd.read_csv('../LIME/data/LOOC_metrics_cr_all_59.csv', parse_dates=True)

    def test_get_optimal_threshold(self):
        print('*' * 10 + 'START computing thresholds' + '*' * 10)
        threshold_row = get_optimal_threshold(self.metrics, 1, thresholds_labels)
        print(threshold_row)
        self.assertTrue(threshold_row.keys() == threshold_columns)
