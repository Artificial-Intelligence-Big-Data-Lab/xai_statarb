from unittest import TestCase

from src.feature_selection_threshold import *
from src.metrics import SelectedColumns


class TestSelectedColumns(TestCase):
    def setUp(self) -> None:
        self.metrics = pd.read_csv('../LIME/43/LOOC_metrics_cr_all.csv', parse_dates=True)

    def test_get_columns(self):
        sut = SelectedColumns('./', 11)
        mock_columns = ['Returns_1', 'Returns_2', 'Returns_3', 'Returns_4', 'Returns_5', 'Returns_21', 'Returns_63', 'Returns_126',
                        'Returns_252']
        sut.all_columns = mock_columns
        stub = Threshold()
        dfs = stub.get_thresholds(self.metrics, 1)
        print(dfs)
        columns = sut.get_columns(dfs['running'], 'Communication Services', method='running')
        print(columns)
        self.assertTrue(len(columns) < len(mock_columns))
