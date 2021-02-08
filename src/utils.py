import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

import feature_selection as fs


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


def validate_data(X):
    return len(X.index.unique()) == len(X.groupby(X.index))


def check_if_processed(metrics, ticker, walk):
    if metrics.empty:
        return False
    return len(metrics[(metrics['ticker'] == ticker) & (metrics['walk'] == walk)]) == 4


def add_context_information(metric_df, context, score, transformer: fs.FeatureSelectorBase = None):
    metric_df['walk'] = context['walk']
    metric_df['model'] = context['method']
    metric_df['ticker'] = context['ticker']
    metric_df['mean_mse'] = -score['test_neg_mean_squared_error'].mean()
    metric_df['std_mse'] = score['test_neg_mean_squared_error'].std(ddof=1)
    metric_df['mean_mae'] = -score['test_neg_mean_absolute_error'].mean()
    metric_df['std_mae'] = score['test_neg_mean_absolute_error'].std(ddof=1)
    metric_df['mean_r2'] = score['test_r2'].mean()
    metric_df['std_r2'] = score['test_r2'].std(ddof=1)

    if context['method'] == 'baseline':
        for r_idx in range(0, 10):
            metric_df['removed_column{0}'.format(r_idx)] = ''
            metric_df['removed_column_imp{0}'.format(r_idx)] = ''
        return None
    elif not transformer.importance.empty:
        missing_columns = pd.DataFrame()
        for r_idx, missing_col in transformer.importance.iterrows():
            missing_col_dict = dict(walk=context['walk'], method=context['method'], ticker=context['ticker']
                                    , removed_column=missing_col['index']
                                    , feature_importance=missing_col['feature_importance'], CI=missing_col['ci_fixed']
                                    , error=missing_col['errors']
                                    , baseline_error=metric_df['MSE']
                                    , std_err=missing_col['std_errors'])
            missing_columns = missing_columns.append(missing_col_dict, ignore_index=True)
            metric_df['removed_column{0}'.format(r_idx)] = missing_col['index']
            metric_df['removed_column_imp{0}'.format(r_idx)] = missing_col['feature_importance']
        return missing_columns
    else:
        return pd.DataFrame()
