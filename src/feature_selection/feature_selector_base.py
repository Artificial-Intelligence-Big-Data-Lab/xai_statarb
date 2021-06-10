import itertools
from abc import ABC

import pandas as pd


class BaseFeatureSelector(ABC):
    def __init__(self, threshold=0.0):
        self._fn_select = None
        self._threshold = threshold

    def select(self, all_columns, importance_df: pd.DataFrame):
        min_row = self._fn_select(importance_df)
        if min_row.empty:
            return min_row, all_columns, "No column selected"
        column = min_row.index.values
        columns = [col for col in all_columns if col not in [column]]
        if len(column) == len(all_columns):
            return pd.DataFrame(), all_columns, "Selecting all columns"
        return min_row, columns, None

    def select_enumerate(self, all_columns, importance_df: pd.DataFrame, ks=[1]):
        min_row = self._fn_select(importance_df)
        error_msg = None
        if min_row.empty:
            min_row = importance_df.sort_values(by='feature_importance', ascending=True).head(1)
            error_msg = "No column selected"
        idx = 0
        for k in ks:
            if len(min_row) < k:
                row = min_row.copy()
                column = row.index.values
                columns = [col for col in all_columns if col not in column]
                return_row = row.copy()
                return_row["index"] = row.index.values
                yield idx, k, return_row, columns, error_msg

            for row_idx in self.__window(range(0, len(min_row)), k):
                row = min_row.iloc[list(row_idx)]
                column = row.index.values
                columns = [col for col in all_columns if col not in column]
                # return_row = pd.Series(row)
                return_row = row.copy()
                return_row["index"] = row.index.values
                yield idx, k, return_row, columns, error_msg
                idx = idx + 1

    @property
    def threshold(self):
        return self._threshold

    @staticmethod
    def __window(seq, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(itertools.islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result


class FeatureSelectorBase(ABC):
    def __init__(self, k=1, selector: BaseFeatureSelector = None):
        self._columns = None
        self._k = k
        self._original_loss = None
        self.name = None
        self._selector = selector

    def transform(self, estimator, x_train, y_train, x_test, y_test):
        return self.fit_transform(estimator, x_train, y_train, x_test, y_test)

    def fit_transform(self, estimator, x_train, y_train, x_test, y_test, k=[1]):
        pass

    @property
    def baseline_loss(self):
        return self._original_loss

    @property
    def threshold(self):
        return self._selector.threshold

    def _feature_importance_permutation(self, estimator_fn, x_test, y_test):
        """Feature importance imputation via permutation importance
        Parameters
        ----------
        estimator_fn : prediction function
            A callable function that predicts the target values
            from X.
        x_test : NumPy array, shape = [n_samples, n_features]
            Dataset, where n_samples is the number of samples and
            n_features is the number of features.
        y_test : NumPy array, shape = [n_samples]
            Target values.


        Returns
        ---------
        res, res_errors : NumPy arrays.
          The first array, mean_importance_values has shape [n_features, ] and
          contains the importance values for all features.
          The shape of the second array is [n_features, num_rounds] and contains
          the feature importance for each repetition. If num_rounds=1,
          it contains the same values as the first array, mean_importance_values."""
        pass


class AllBellowZeroSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__()
        self._fn_select = lambda df: df[df['feature_importance'] <= self._threshold].sort_values(by=['feature_importance'],
                                                                                                 ascending=[False])


class AllAboveThresholdSelector(BaseFeatureSelector):
    def __init__(self, threshold):
        super().__init__(threshold=threshold)
        self._fn_select = lambda df: df[df['feature_importance'] <= self._threshold]


class SelectKWorst(BaseFeatureSelector):
    def __init__(self, k, threshold=0.0):
        super().__init__(threshold=threshold)
        self._k = k
        self._fn_select = lambda df: df[df['feature_importance'] <= self._threshold].sort_values(
            by=['feature_importance'],
            ascending=[False]).tail(self._k)


class SelectKBest(BaseFeatureSelector):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._fn_select = lambda df: df[df['feature_importance'] <= self._threshold].sort_values(
            by=['feature_importance'],
            ascending=[False]).head(self._k)
