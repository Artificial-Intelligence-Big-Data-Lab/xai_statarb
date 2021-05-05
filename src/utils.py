import sys
import time
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import mda, hit_count


def init_prediction_df(ticker, X_cr_test, y_cr_test, b_y_test):
    predictions_df = X_cr_test.copy()
    predictions_df = pd.concat([predictions_df, y_cr_test], axis=1)
    predictions_df = predictions_df.join(b_y_test)
    predictions_df['ticker'] = ticker
    return predictions_df


def get_prediction_performance_results(y_cr_test, show=True, suffix=''):
    results = pd.Series()
    metric_func = {
        'MSE': mean_squared_error,
        # 'r2_score': r2_score,
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


def add_metrics_information(metric_original: pd.Series, context: dict, score, importance_series: pd.Series = None,
                            copy_to=None):
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

    if context['method'] != 'baseline':
        metric_series['selection_error'] = context['selection_error'] \
            if context['method'] != 'baseline' and 'selection_error' in context.keys() and context[
            'selection_error'] is not None else None
        # for col in context['all_columns']:
        #     metric_series['{0}_count'.format(col)] = ''
        #     metric_series['{0}_CI'.format(col)] = ''
        #     metric_series['{0}_FI'.format(col)] = ''
        #     metric_series['{0}_error'.format(col)] = ''
        #     metric_series['{0}_std_error'.format(col)] = ''
        #     metric_series['threshold'] = ''
        metric_series['index'] = ''
        metric_series['removed_count'] = ''
        metric_series['removed_CI'] = ''
        metric_series['removed_FI'] = ''
        metric_series['removed_error'] = ''
        metric_series['removed_std_error'] = ''
        metric_series['removed_column'] = ''

    if importance_series is not None and len(importance_series) != 0:
        metric_series['index'] = context["index"]
        missing_col = pd.Series(importance_series)
        metric_series['removed_column'] = missing_col['index']
        metric_series['removed_CI'] = missing_col['ci_fixed']
        metric_series['removed_FI'] = missing_col['feature_importance']
        metric_series['removed_count'] = missing_col['success_count']
        metric_series['removed_error'] = missing_col['errors']
        metric_series['removed_std_error'] = missing_col['std_errors']
    final_series = final_series.append(pd.Series(metric_series))
    return final_series


def add_score_to_metrics(score, method_suffix=''):
    metric_series = pd.Series()
    if not score:
        return metric_series.to_dict()
    metric_series['mean_mse'] = -score['test_neg_mean_squared_error'].mean()
    metric_series['std_mse'] = score['test_neg_mean_squared_error'].std(ddof=1)
    metric_series['mean_mae'] = -score['test_neg_mean_absolute_error'].mean()
    metric_series['std_mae'] = score['test_neg_mean_absolute_error'].std(ddof=1)
    # metric_series['mean_r2'] = score['test_r2'].mean()
    # metric_series['std_r2'] = score['test_r2'].std(ddof=1)
    if method_suffix != '':
        metric_series = metric_series.add_suffix(method_suffix)
    return metric_series.to_dict()


def add_context_information(metric_series: pd.Series, context: dict, score, importance_series: pd.Series = None,
                            baseline_loss=None):
    metric_series = metric_series.append(pd.Series(add_score_to_metrics(score)))
    metric_series['walk'] = context['walk']
    metric_series['model'] = context['method']
    metric_series['ticker'] = context['ticker']
    metric_series['selection_error'] = context['selection_error'] \
        if 'selection_error' in context.keys() and context['selection_error'] is not None else None

    # for col in context['all_columns']:
    #     metric_series['{0}_count'.format(col)] = ''

    if context['method'] == 'baseline':
        # for r_idx in range(0, 10):
        #     metric_series['removed_column{0}'.format(r_idx)] = ''
        #     metric_series['removed_column_imp{0}'.format(r_idx)] = ''
        metric_series['index'] = ''
        return metric_series, None
    elif importance_series is not None and len(importance_series) != 0:

        missing_columns = pd.DataFrame()
        missing_col = pd.Series(importance_series)
        r_idx = context["index"]
        missing_col_dict = dict(walk=context['walk'], method=context['method'], ticker=context['ticker']
                                , removed_column=missing_col['index']
                                , feature_importance=missing_col['feature_importance'], CI=missing_col['ci_fixed']
                                , error=missing_col['errors']
                                , baseline_error=baseline_loss
                                , std_err=missing_col['std_errors']
                                , success_count=missing_col['success_count']
                                , index=context["index"]
                                )
        missing_columns = missing_columns.append(missing_col_dict, ignore_index=True)

        # metric_series['removed_column{0}'.format(r_idx)] = missing_col['index']
        # metric_series['removed_column_imp{0}'.format(r_idx)] = missing_col['feature_importance']
        # metric_series['{0}_count'.format(missing_col['index'])] = missing_col['success_count']
        metric_series['index'] = context["index"]
        return metric_series, missing_columns
    else:
        return metric_series, pd.DataFrame()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__qualname__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            f = "stdout"
            safe_print("""[INFO] start_method  %r - %2.2f [ms]""" % (method.__qualname__, (te - ts) * 1000), file=f,
                       flush=True)

        return result

    return timed


def safe_print(*objects, **kwargs):
    """Safe print function for backwards compatibility."""
    # Get stream
    file = kwargs.pop('file', sys.stdout)
    if isinstance(file, str):
        file = getattr(sys, file)

    # Get flush
    flush = kwargs.pop('flush', False)

    # Print
    print(*objects, file=file, **kwargs)

    # Need to flush outside print function for python2 compatibility
    if flush:
        file.flush()


def print_info(message='', **kwargs):
    safe_print('[INFO] {0} {1}'.format(time.strftime("%d-%b-%Y %H:%M:%S"), message), **kwargs)


def print_time(t0, message='', **kwargs):
    """Utility function for printing time"""
    if len(message) > 0:
        message += ' | '

    m, s = divmod(time() - t0, 60)
    h, m = divmod(m, 60)

    safe_print(message + '%02d:%02d:%02d' % (h, m, s), **kwargs)
