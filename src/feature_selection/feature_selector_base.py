from abc import ABC

import pandas as pd


class FeatureSelectorBase(ABC):
    def __init__(self, k=1):
        self._columns = None
        self._importance = None
        self._k = k

    def transform(self, estimator, x_train, y_train, x_test, y_test):
        return self.fit_transform(estimator, x_train, y_train, x_test, y_test)

    def fit_transform(self, estimator, x_train, y_train, x_test, y_test):
        pass

    @property
    def importance(self):
        return self._importance


class BaseFeatureSelector(ABC):
    def __init__(self):
        self.importance_ = None
        self._fn_select = None

    def select(self, all_columns):
        min_row = self._fn_select(self.importance_)
        if min_row.empty:
            return min_row, all_columns
        column = min_row.index.values
        columns = set(all_columns) - set(column)
        if len(column) == len(all_columns):
            return pd.DataFrame(), all_columns
        return min_row, columns


class AllAboveZeroSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()
        self._fn_select = lambda df: df[df['feature_importance'] <= 0.0]


class SelectKBest(BaseFeatureSelector):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._fn_select = lambda df: df[df['feature_importance'] <= 0.0].sort_values(by=['feature_importance'],
                                                                                     ascending=[False]).tail(self._k)
