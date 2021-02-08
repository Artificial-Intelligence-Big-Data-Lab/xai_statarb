import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score


def mda(y_cr_test: pd.DataFrame):
    """ Mean Directional Accuracy """

    x = np.sign(y_cr_test['label'] - y_cr_test['label'].shift(1)) == np.sign(
        y_cr_test['predicted'] - y_cr_test['label'].shift(1))
    return np.count_nonzero(x.values.astype('int')) / len(x[~x.isnull()])


def hit_count(y_cr_test):
    x = (np.sign(y_cr_test['predicted']) == np.sign(y_cr_test['label'])).astype(int)
    return np.count_nonzero(x.values), np.count_nonzero(x.values) / len(x)


def get_prediction_performance_results(y_cr_test, show=True, prefix=''):
    results = pd.Series()
    metric_func = {
        'MSE': mean_squared_error,
        'r2_score': r2_score,
        # 'explained_variance':explained_variance_score,
        'MAE': mean_absolute_error,
        # 'MAPE':mean_absolute_percentage_error
    }
    for metric, function in metric_func.items():
        column_name = '{0}_{1}'.format(prefix, metric) if prefix != '' else metric
        results[column_name] = function(y_cr_test['label'], y_cr_test['predicted'])
    results['{0}_MDA'.format(prefix) if prefix else 'MDA'] = mda(y_cr_test)
    hc, acc = hit_count(y_cr_test)
    results['{0}_hit_count'.format(prefix) if prefix else 'hit_count'] = hc
    results['{0}_accuracy'.format(prefix) if prefix else 'accuracy'] = acc

    if show:
        print(results)
    return results
