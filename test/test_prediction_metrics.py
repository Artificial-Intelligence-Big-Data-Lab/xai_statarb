from typing import List
from unittest import TestCase

from src.metrics import SelectedColumns
from src.walkforward import *


class TestSelectedColumnsTestCase(TestCase):

    def setUp(self) -> None:
        pass


class TestSelectedColumns(TestSelectedColumnsTestCase):

    def test_set_chosen_features(self):
        sut = SelectedColumns('./', 11)
        mock_columns: List[str] = ['a', 'b', 'c', 'd']
        sut.all_columns = mock_columns
        sut.set_chosen_features('AIR', 1, 'best', ['a', 'b', 'c'], )
        df = sut.save()
        self.assertTrue(set(df.columns) == set(mock_columns) | {'ticker', 'method', 'walk'}, 'columns must match')
        self.assertFalse(df.empty, 'Dataframe should not be empty')
        self.assertTrue(np.alltrue(df[df['ticker'] == 'AIR'][['a', 'b', 'c']]))
        self.assertFalse(np.all(df[df['ticker'] == 'AIR'][['d']]))

    def test_set_chosen_features_when_no_column_passed_throws_exception(self):
        sut = SelectedColumns('./', 11)
        mock_columns: List[str] = ['a', 'b', 'c', 'd']
        sut.all_columns = mock_columns
        with self.assertRaises(ValueError):
            sut.set_chosen_features('AIR', 1, 'best', [], )

    def test_set_chosen_features_when_features_not_passed_throws_exception(self):
        sut = SelectedColumns('./', 11)
        with self.assertRaises(ValueError):
            sut.set_chosen_features('AIR', 1, 'best', ['a'], )
