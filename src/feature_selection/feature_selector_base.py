from abc import ABC

import pandas as pd


class FeatureSelectorBase(ABC):
    def __init__(self, k=1):
        self._columns = None
        self._importance = None
        self._k = k
        self._original_loss = None
        self.name = None

    def transform(self, estimator, x_train, y_train, x_test, y_test):
        return self.fit_transform(estimator, x_train, y_train, x_test, y_test)

    def fit_transform(self, estimator, x_train, y_train, x_test, y_test):
        pass

    @property
    def baseline_loss(self):
        return self._original_loss

    @property
    def importance(self):
        return self._importance

    def _feature_importance_permutation(self,estimator_fn, x_test, y_test):
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


class SelectKWorst(BaseFeatureSelector):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._fn_select = lambda df: df[df['feature_importance'] <= 0.0].sort_values(by=['feature_importance'],
                                                                                     ascending=[False]).tail(self._k)


class SelectKBest(BaseFeatureSelector):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._fn_select = lambda df: df[df['feature_importance'] <= 0.0].sort_values(by=['feature_importance'],
                                                                                     ascending=[False]).head(self._k)
