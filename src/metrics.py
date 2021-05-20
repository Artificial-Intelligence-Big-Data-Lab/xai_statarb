import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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
        label_array = labels.copy()
        label_array.extend(['baseline'])
        self.__labels = label_array

        columns = ['ticker', 'walk']
        columns.extend(['MSE_test_' + col for col in label_array])
        columns.extend(['RMSE_test_' + col for col in label_array])
        columns.extend(['MDA_test_' + col for col in label_array])
        columns.extend(['no_improvements_test_' + col for col in label_array])

        columns.extend(['MSE_validation_' + col for col in label_array])
        columns.extend(['RMSE_validation_' + col for col in label_array])
        columns.extend(['MDA_validation_' + col for col in label_array])
        columns.extend(['no_improvements_validation_' + col for col in label_array])

        self.__all_columns = columns
        self.__metrics = pd.DataFrame(columns=columns)

    def set_metrics(self, ticker, walk, validation_predictions: pd.DataFrame, predictions: pd.DataFrame):
        data_row = dict(ticker=ticker, walk=walk)
        for col in self.__labels:
            pred_col = 'predicted_' + col
            data_row.update({'RMSE_test_' + col: rmse(predictions['label'].values, predictions[pred_col].values)})
            data_row.update(
                {'no_improvements_test_' + col: hit_count(predictions[['label', pred_col]], predicted=pred_col)})
            data_row.update({'MDA_test_' + col: mda(predictions[['label', pred_col]], predicted=pred_col)})
            data_row.update(
                {'MSE_test_' + col: mean_squared_error(predictions['label'].values, predictions[pred_col].values)})

            data_row.update({'RMSE_validation_' + col: rmse(validation_predictions['label'].values,
                                                            validation_predictions[pred_col].values)})
            data_row.update({'no_improvements_validation_' + col: hit_count(validation_predictions[['label', pred_col]],
                                                                            predicted=pred_col)})
            data_row.update(
                {'MDA_validation_' + col: mda(validation_predictions[['label', pred_col]], predicted=pred_col)})
            data_row.update(
                {'MSE_validation_' + col: mean_squared_error(validation_predictions['label'].values,
                                                             validation_predictions[pred_col].values)})
        self.__metrics = self.__metrics.append(data_row, ignore_index=True)

    def save(self, folder):
        self.__metrics[self.__all_columns].to_csv(folder + '/metrics.csv', index=False)
        self.__metrics = pd.DataFrame(columns=self.__all_columns)


class SelectedColumns:

    def __init__(self, save_path, removed_feature_no=1):
        self.__file = save_path + 'LOOC_selected_columns.csv'
        self.__feature_columns = None
        self.__all_columns = ['ticker', 'walk', 'method']
        self.__df = None
        self.__removed_feature_no = removed_feature_no

    @property
    def all_columns(self):
        return self.__all_columns

    @all_columns.setter
    def all_columns(self, value: []):
        if self.__feature_columns is None:
            self.__feature_columns = value.copy()
            self.__all_columns.extend(value)
        if self.__df is None:
            self.__df = pd.DataFrame(columns=self.__all_columns)

    def save(self):
        self.__df[self.__all_columns].to_csv(self.__file, index=False)
        return self.__df

    def set_chosen_features(self, ticker, walk_idx: int, method: str, columns: []):
        if not columns:
            raise ValueError('At least one column must be selected')
        if self.__feature_columns is None:
            raise ValueError('The feature names is not set')
        data_row = dict(ticker=ticker, walk=walk_idx, method=method)
        data_row.update(dict([(col, False) for col in self.__feature_columns]))
        data_row.update(dict([(col, True) for col in columns]))
        self.__df = self.__df.append(data_row, ignore_index=True)

    def get_columns(self, df: pd.DataFrame, ticker, method):
        if df.empty:
            return self.__feature_columns

        removed_column = df[(df['ticker'] == ticker) & (df['method'] == method)]['removed_column'].values[:self.__removed_feature_no]
        if removed_column is None:
            return self.__feature_columns
        else:
            chosen_columns = [col for col in self.__feature_columns if col not in removed_column]
            return chosen_columns
