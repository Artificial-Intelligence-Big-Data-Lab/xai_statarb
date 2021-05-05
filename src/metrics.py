import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src import Walk


def rmse(y, p):
    r"""Root Mean Square Error.
    .. math::
        RMSE(\mathbf{y}, \mathbf{p}) = \sqrt{MSE(\mathbf{y}, \mathbf{p})},
    with
    .. math::
        MSE(\mathbf{y}, \mathbf{p}) = |S| \sum_{i \in S} (y_i - p_i)^2
    Parameters
    ----------
    y : array-like of shape [n_samples, ]
        ground truth.
    p : array-like of shape [n_samples, ]
        predicted labels.
    Returns
    -------
    z: float
        root mean squared error.
    """
    z = y - p
    return np.sqrt(np.mean(np.multiply(z, z)))


def mda(y_cr_test: pd.DataFrame, label='label', predicted='predicted'):
    """ Mean Directional Accuracy """

    x = np.sign(y_cr_test[label] - y_cr_test[label].shift(1)) == np.sign(
        y_cr_test[predicted] - y_cr_test[label].shift(1))
    return np.count_nonzero(x.values.astype('int')) / len(x[~x.isnull()])


def hit_count(y_cr_test, label='label', predicted='predicted'):
    x = (np.sign(y_cr_test[predicted]) == np.sign(y_cr_test[label])).astype(int)
    return np.count_nonzero(x.values), np.count_nonzero(x.values) / len(x)


class MetricsSaver:
    def __init__(self, labels):
        self.__labels = labels
        self.__init_df(labels)

    def __init_df(self, labels):
        labels.extend(['baseline'])
        columns = ['MSE_' + col for col in labels]
        columns.extend(['RMSE_' + col for col in labels])
        columns.extend(['MDA_' + col for col in labels])
        columns.extend(['no_improvements_' + col for col in labels])
        columns.extend(['ticker', 'walk'])
        self.__all_columns = columns
        self.__metrics = pd.DataFrame(columns=columns)

    def set_metrics(self, ticker, walk, predictions: pd.DataFrame):
        data_row = dict(ticker=ticker, walk=walk)
        for col in self.__labels:
            pred_col = 'predicted_' + col
            data_row.update({'RMSE_' + col: rmse(predictions['label'].values, predictions[pred_col].values)})
            data_row.update({'no_improvements_' + col: hit_count(predictions[['label', pred_col]], predicted=pred_col)})
            data_row.update({'MDA_' + col: mda(predictions[['label', pred_col]], predicted=pred_col)})
            data_row.update(
                {'MSE' + col: mean_squared_error(predictions['label'].values, predictions[pred_col].values)})
        self.__metrics.append(data_row, ignore_index=True)

    def save(self, folder):
        self.__metrics[self.__all_columns].to_csv(folder + 'metrics.csv', index=False)
        self.__init_df(self.__labels)


class SelectedColumns:
    def __init__(self, save_path, test_run):
        self.__test_run = test_run
        self.__file = save_path + 'LOOC_metrics_cr_all_{0}.csv'.format(test_run)
        self.__feature_columns = None
        self.__all_columns = ['ticker', 'walk']
        self.__df = None

    @property
    def all_columns(self):
        return self.__all_columns

    @all_columns.setter
    def all_columns(self, value: []):
        if self.__feature_columns is None:
            self.__feature_columns = value
            self.__all_columns.extend(value)
        if self.__df is None:
            self.__df = pd.DataFrame(columns=self.__all_columns)

    def save(self):
        self.__df[self.__all_columns].to_csv(self.__file, index=False)
        return self.__df

    def set_chosen_features(self, ticker, walk: Walk, columns):
        if not columns:
            raise ValueError('At least one column must be selected')
        if not self.__feature_columns:
            raise ValueError('The feature names is not set')
        data_row = dict(ticker=ticker, walk=walk.train.idx)
        data_row.update(dict([(col, False) for col in self.__feature_columns]))
        data_row.update(dict([(col, True) for col in columns]))
        self.__df = self.__df.append(data_row, ignore_index=True)
