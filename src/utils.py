import sys
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error

from metrics import *


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


def compute_yearly_drawdown(series, window=252):
    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = series.rolling(window, min_periods=1).max()
    Daily_Drawdown = series / Roll_Max

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    return Daily_Drawdown, Max_Daily_Drawdown


def compute_buy_and_hold_companies(df_input: pd.DataFrame, active=5,
                                   sort_by_labels={'Expected': 'Expected_OC_perc', 'Predicted': 'Predicted_OC_perc',
                                                   'Metric': {'Name': None, 'Order': None}}, industry_information=None,
                                   prediction_method=None, mask=None):
    active_range = set(range(0, active, 1))
    dftot = df_input.copy()

    if industry_information is not None and len(industry_information) != 0:
        dftot = dftot[dftot['Title'].isin(industry_information)]

    df_unuseful = dftot.groupby('Date').filter(lambda x: len(x) < active)

    print(df_unuseful.index.unique())
    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)

    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False)
    dftot_1['Rank'] = dftot_1.groupby('Date')[sort_by_labels['Expected']].rank(ascending=False, method='dense')
    # print "QUIDFTOT_1:",dftot_1
    peggiori_exp = dftot_1.groupby('Date').nth(active_range)

    if prediction_method is not None:
        peggiori_exp.to_csv('{0}_peggiori_expected.csv'.format(prediction_method))

    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True)
    dftot_2['Rank'] = -dftot_2.groupby('Date')[sort_by_labels['Expected']].rank(ascending=True, method='dense')
    migliori_exp = dftot_2.groupby('Date').nth(active_range)

    if prediction_method is not None:
        migliori_exp.to_csv('{0}_migliori_expected.csv'.format(prediction_method))

    incr_bydate_exp = migliori_exp.groupby('Date').sum()
    # print "incr_bydate_exp",incr_bydate_exp
    decr_bydate_exp = peggiori_exp.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    # print "decr_bydate_exp",decr_bydate_exp

    valore_giornaliero_exp = (- incr_bydate_exp['Expected_OC_perc'] + decr_bydate_exp['Expected_OC_perc']) / (
            2 * active) * 100
    valore_giornaliero_exp.index = pd.to_datetime(valore_giornaliero_exp.index)

    if prediction_method is not None:
        valore_giornaliero_exp.to_csv('{0}_valore_giornaliero_expected.csv'.format(prediction_method))

    if mask is not None:
        dftot = dftot[dftot.apply(mask, axis=1)]

    if ('Metric' in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric'][
        'Name'] is not None:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']],
                                    ascending=[False, sort_by_labels['Metric']['Order']])
    else:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[False])

    dftot_3['Rank'] = dftot_3.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=False, method='dense')
    peggiori_pred = dftot_3.groupby('Date').nth(active_range)

    if prediction_method is not None:
        peggiori_pred.to_csv('{0}_peggiori_predicted.csv'.format(prediction_method))

    # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    if ('Metric' in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric'][
        'Name'] is not None:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']],
                                    ascending=[True, sort_by_labels['Metric']['Order']])
    else:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[True])

    dftot_4['Rank'] = - dftot_4.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=True, method='dense')
    migliori_pred = dftot_4.groupby('Date').nth(active_range)

    if prediction_method is not None:
        migliori_pred.to_csv('{0}_migliori_predicted.csv'.format(prediction_method))

    # calcolo del valore percentuale sui valori predetti
    incr_bydate_pred = migliori_pred.groupby('Date').sum()
    decr_bydate_pred = peggiori_pred.groupby('Date').sum()

    valore_giornaliero_pred = (- incr_bydate_pred['Expected_OC_perc'] + decr_bydate_pred['Expected_OC_perc']) / (
            2 * active) * 100
    valore_giornaliero_pred.index = pd.to_datetime(valore_giornaliero_pred.index)

    return valore_giornaliero_pred, valore_giornaliero_exp, migliori_exp, migliori_pred, peggiori_exp, peggiori_pred


def compute_buy_and_hold_companies2(dftot: pd.DataFrame, active=5,
                                    sort_by_labels={'Expected': 'Expected_OC_perc', 'Predicted': 'Predicted_OC_perc'}):
    active_range = set(range(0, active, 1))

    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)
    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False);
    # print "QUIDFTOT_1:",dftot_1
    migliori_exp = dftot_1.groupby('Date').nth(active_range);
    # print "MIGL:",migliori_exp

    # Calcolo dei 5 titoli peggiori sui risultati attesi (expected)
    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True);
    peggiori_exp = dftot_2.groupby('Date').nth(active_range);

    dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=False);
    migliori_pred = dftot_3.groupby('Date').nth(active_range);

    # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=True);
    peggiori_pred = dftot_4.groupby('Date').nth(active_range);

    incr_bydate_exp = migliori_exp.groupby('Date').sum()
    # print "incr_bydate_exp",incr_bydate_exp
    decr_bydate_exp = peggiori_exp.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    # print "decr_bydate_exp",decr_bydate_exp

    valore_giornaliero_exp = (incr_bydate_exp['Expected_OC_perc'] - decr_bydate_exp['Expected_OC_perc']) / (
            2 * active) * 100
    valore_giornaliero_exp.index = pd.to_datetime(valore_giornaliero_exp.index)

    # calcolo del valore percentuale sui valori predetti
    incr_bydate_pred = migliori_pred.groupby('Date').sum()
    decr_bydate_pred = peggiori_pred.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    # decr_bydate_pred = (decr_bydate_pred)
    # print "incr_bydate_pred",incr_bydate_pred
    # print "decr_bydate_pred",decr_bydate_pred

    valore_giornaliero_pred = (incr_bydate_pred['Expected_OC_perc'] - decr_bydate_pred['Expected_OC_perc']) / (
            2 * active) * 100
    valore_giornaliero_pred.index = pd.to_datetime(valore_giornaliero_pred.index)

    return valore_giornaliero_pred, valore_giornaliero_exp, migliori_exp, migliori_pred, peggiori_exp, peggiori_pred


def compute_ranks_peggiori(df_input: pd.DataFrame,
                           sort_by_labels={'Expected': 'Expected_OC_perc', 'Predicted': 'Predicted_OC_perc',
                                           'Metric': {'Name': None, 'Order': None}}, industry_information=None,
                           prediction_method=None, mask=None):
    dftot = df_input.copy()

    if industry_information is not None and len(industry_information) != 0:
        dftot = dftot[dftot['Title'].isin(industry_information)]

    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False)
    dftot_1['Rank'] = dftot_1.groupby('Date')[sort_by_labels['Expected']].rank(ascending=False, method='dense')

    if mask is not None:
        dftot = dftot[dftot.apply(mask, axis=1)]

    if ('Metric' in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric'][
        'Name'] is not None:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']],
                                    ascending=[False, sort_by_labels['Metric']['Order']])
    else:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[False])

    dftot_3['Rank'] = dftot_3.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=False, method='dense')
    if 'Date' in dftot_1.columns:
        dftot_1.set_index('Date', inplace=True)

    if 'Date' in dftot_1.columns:
        dftot_3.set_index('Date', inplace=True)
    return dftot_1, dftot_3


def compute_ranks_migliori(df_input: pd.DataFrame, active=5,
                           sort_by_labels={'Expected': 'Expected_OC_perc', 'Predicted': 'Predicted_OC_perc',
                                           'Metric': {'Name': None, 'Order': None}}, industry_information=None,
                           prediction_method=None, mask=None):
    dftot = df_input.copy()

    if industry_information is not None and len(industry_information) != 0:
        dftot = dftot[dftot['Title'].isin(industry_information)]

    df_unuseful = dftot.groupby('Date').filter(lambda x: len(x) < active)

    print(df_unuseful.index.unique())
    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)

    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True)
    dftot_2['Rank'] = -dftot_2.groupby('Date')[sort_by_labels['Expected']].rank(ascending=True, method='dense')

    if mask is not None:
        dftot = dftot[dftot.apply(mask, axis=1)]

        # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    if ('Metric' in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric'][
        'Name'] is not None:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']],
                                    ascending=[True, sort_by_labels['Metric']['Order']])
    else:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[True])

    dftot_4['Rank'] = - dftot_4.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=True, method='dense')

    return dftot_2, dftot_4
