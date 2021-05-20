from unittest import TestCase

from src.feature_selection_threshold import *


class FeatureSelectionThresholdTest(TestCase):
    def setUp(self) -> None:
        # self.metrics = pd.read_csv('../LIME/data/LOOC_metrics_cr_all_59.csv', parse_dates=True)
        self.metrics = pd.read_csv('../LIME/43/LOOC_metrics_cr_all.csv', parse_dates=True)

    def test_get_optimal_threshold(self):
        print('*' * 10 + 'START computing thresholds' + '*' * 10)
        threshold_row = get_optimal_threshold(self.metrics[self.metrics['walk'] == 1], 1, thresholds_labels)
        print(threshold_row)
        self.assertTrue(threshold_row.keys() == threshold_columns)


class TestThreshold(TestCase):
    def setUp(self) -> None:
        self.metrics = pd.read_csv('../LIME/43/LOOC_metrics_cr_all.csv', parse_dates=True)

    def test_get_thresholds(self):
        sut = Threshold()
        mock = sut.get_thresholds(self.metrics, 1)
        print(mock)
        self.assertTrue(list(mock.keys()) == thresholds_labels)
