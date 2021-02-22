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


def get_prediction_performance_results(y_cr_test, show=True, suffix=''):
    results = pd.Series()
    metric_func = {
        'MSE': mean_squared_error,
        'r2_score': r2_score,
        # 'explained_variance':explained_variance_score,
        'MAE': mean_absolute_error,
        # 'MAPE':mean_absolute_percentage_error
    }
    for metric, function in metric_func.items():
        results[metric] = function(y_cr_test['label'], y_cr_test['predicted'])
    results['MDA'] = mda(y_cr_test)
    hc, acc = hit_count(y_cr_test)
    results['hit_count'] = hc
    results['accuracy'] = acc
    # results = results.add_suffix(suffix)
    # results.name = suffix.replace("_", "")
    if show:
        print(results)
    return results


def validate_data(X):
    return len(X.index.unique()) == len(X.groupby(X.index))


def check_if_processed(metrics, ticker, walk):
    if metrics.empty:
        return False
    return len(metrics[(metrics['ticker'] == ticker) & (metrics['walk'] == walk)]) == 4


def add_metrics_information(metric_original: pd.Series, context: dict, score,
                            transformer: fs.FeatureSelectorBase = None, copy_to: pd.Series = None):

    method_suffix = "_baseline" if context['method'] == 'baseline' else "_pi"

    final_series = metric_original.copy().add_suffix(method_suffix)
    if copy_to is not None:
        final_series = copy_to.append(final_series)

    if context['method'] == 'baseline':
        final_series['walk'] = context['walk']
        final_series['ticker'] = context['ticker']
    else:
        final_series['model'] = context['method']

    metric_series = add_score_to_metrics(score, method_suffix)

    metric_series['selection_error'] = context['selection_error'] \
        if 'selection_error' in context.keys() and context['selection_error'] is not None else None

    if context['method'] != 'baseline':
        for col in context['all_columns']:
            metric_series['{0}_count'.format(col)] = ''
            metric_series['{0}_CI'.format(col)] = ''
            metric_series['{0}_FI'.format(col)] = ''
            metric_series['{0}_error'.format(col)] = ''
            metric_series['{0}_std_error'.format(col)] = ''

    if transformer is not None and not transformer.importance.empty:
        for r_idx, missing_col in transformer.importance.iterrows():
            metric_series['{0}_CI'.format(missing_col['index'])] = missing_col['ci_fixed']
            metric_series['{0}_FI'.format(missing_col['index'])] = missing_col['feature_importance']
            metric_series['{0}_count'.format(missing_col['index'])] = missing_col['success_count']
            metric_series['{0}_error'.format(missing_col['index'])] = missing_col['errors']
            metric_series['{0}_std_error'.format(missing_col['index'])] = missing_col['std_errors']
    final_series = final_series.append(pd.Series(metric_series))
    return final_series


def add_score_to_metrics(score, method_suffix=''):
    metric_series = pd.Series()
    metric_series['mean_mse'] = -score['test_neg_mean_squared_error'].mean()
    metric_series['std_mse'] = score['test_neg_mean_squared_error'].std(ddof=1)
    metric_series['mean_mae'] = -score['test_neg_mean_absolute_error'].mean()
    metric_series['std_mae'] = score['test_neg_mean_absolute_error'].std(ddof=1)
    metric_series['mean_r2'] = score['test_r2'].mean()
    metric_series['std_r2'] = score['test_r2'].std(ddof=1)
    if method_suffix != '':
        metric_series = metric_series.add_suffix(method_suffix)
    return metric_series.to_dict()


def add_context_information(metric_series: pd.Series, context: dict, score, transformer: fs.FeatureSelectorBase = None):

    metric_series = metric_series.append(pd.Series(add_score_to_metrics(score)))
    metric_series['walk'] = context['walk']
    metric_series['model'] = context['method']
    metric_series['ticker'] = context['ticker']
    metric_series['selection_error'] = context['selection_error'] \
        if 'selection_error' in context.keys() and context['selection_error'] is not None else None

    for col in context['all_columns']:
        metric_series['{0}_count'.format(col)] = ''

    if context['method'] == 'baseline':
        for r_idx in range(0, 10):
            metric_series['removed_column{0}'.format(r_idx)] = ''
            metric_series['removed_column_imp{0}'.format(r_idx)] = ''
        return metric_series, None
    elif not transformer.importance.empty:
        missing_columns = pd.DataFrame()
        for r_idx, missing_col in transformer.importance.iterrows():
            missing_col_dict = dict(walk=context['walk'], method=context['method'], ticker=context['ticker']
                                    , removed_column=missing_col['index']
                                    , feature_importance=missing_col['feature_importance'], CI=missing_col['ci_fixed']
                                    , error=missing_col['errors']
                                    , baseline_error=transformer.baseline_loss
                                    , std_err=missing_col['std_errors']
                                    , success_count=missing_col['success_count'])
            missing_columns = missing_columns.append(missing_col_dict, ignore_index=True)
            metric_series['removed_column{0}'.format(r_idx)] = missing_col['index']
            metric_series['removed_column_imp{0}'.format(r_idx)] = missing_col['feature_importance']
            metric_series['{0}_count'.format(missing_col['index'])] = missing_col['success_count']
        return metric_series, missing_columns
    else:
        return metric_series, pd.DataFrame()
