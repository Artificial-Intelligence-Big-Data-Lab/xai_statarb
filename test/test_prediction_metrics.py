import datetime
from typing import List
from unittest import TestCase

from src import *


class TestSelectedColumnsTestCase(TestCase):

    def setUp(self) -> None:
        self.walk = Walk(train=Set(1, start=datetime.datetime.strptime('2010-01-01', '%Y-%m-%d'),
                                   end=datetime.datetime.strptime('2010-01-03', '%Y-%m-%d')),
                         validation=Set(1, start=datetime.datetime.strptime('2010-01-04', '%Y-%m-%d'),
                                        end=datetime.datetime.strptime('2010-01-05', '%Y-%m-%d')),
                         test=Set(1, start=datetime.datetime.strptime('2010-01-06', '%Y-%m-%d'),
                                  end=datetime.datetime.strptime('2010-01-07', '%Y-%m-%d')))


class TestSelectedColumns(TestSelectedColumnsTestCase):

    def test_set_chosen_features(self):
        sut = SelectedColumns('./', 11)
        mock_columns: List[str] = ['a', 'b', 'c', 'd']
        sut.all_columns = mock_columns
        sut.set_chosen_features('AIR', self.walk, ['a', 'b', 'c'])
        df = sut.save()
        self.assertTrue(set(df.columns) == set(mock_columns) | {'ticker', 'walk'}, 'columns must match')
        self.assertFalse(df.empty, 'Dataframe should not be empty')
        self.assertTrue(np.alltrue(df[df['ticker'] == 'AIR'][['a', 'b', 'c']]))
        self.assertFalse(np.all(df[df['ticker'] == 'AIR'][['d']]))

    def test_set_chosen_features_when_no_column_passed_throws_exception(self):
        sut = SelectedColumns('./', 11)
        mock_columns: List[str] = ['a', 'b', 'c', 'd']
        sut.all_columns = mock_columns
        with self.assertRaises(ValueError):
            sut.set_chosen_features('AIR', self.walk, [])

    def test_set_chosen_features_when_features_not_passed_throws_exception(self):
        sut = SelectedColumns('./', 11)
        with self.assertRaises(ValueError):
            sut.set_chosen_features('AIR', self.walk, ['a'])
