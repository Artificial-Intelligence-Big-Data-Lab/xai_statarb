import numpy as np
import pandas as pd

threshold_columns = {'walk', 'threshold_best', 'error_best', 'no_improvements_best', 'ratio_best',
                     'threshold_worst', 'error_worst', 'no_improvements_worst', 'ratio_worst',
                     'threshold_running', 'error_running', 'no_improvements_running', 'ratio_running'}
thresholds_labels = ['best', 'worst', 'running']


def get_is_lower(x, th):
    y = th[th['walk'] == x.walk]['value'].values[0]
    return x['removed_FI'] <= y


def get_error(x, th, error_label='MSE'):
    y = th[th['walk'] == x.walk]['value'].values[0]
    label_pi = "{0}_pi".format(error_label)
    baseline_pi = "{0}_baseline".format(error_label)
    return x[label_pi] if x['removed_FI'] <= y else x[baseline_pi]


def get_column(x, th):
    y = th[th['walk'] == x.walk]['value'].values[0]
    return x['removed_column'] if x['removed_FI'] <= y else None


def get_metrics(metrics, thresholds, label='worst', error_label='MSE'):
    th = pd.DataFrame()
    th['walk'] = thresholds['walk']
    th['value'] = thresholds['threshold_{0}'.format(label)]

    if label != 'running':
        worst = label == 'worst'
        indexes = metrics.sort_values(['walk', 'ticker', 'removed_FI'], ascending=[True, True, worst]) \
            .groupby(by=['walk', 'ticker'], as_index=False).nth(0)[['walk', 'ticker', 'removed_FI', 'removed_column']]
        indexes.dropna(inplace=True)
        test_df = metrics[metrics.index.isin(indexes.index)]
    else:
        test_df = metrics.copy()
        test_df['is_lower'] = metrics.apply(lambda x: get_is_lower(x, th), axis=1)
        test_df = test_df[test_df['is_lower']].sort_values(['walk', 'ticker', 'removed_FI'], ascending=[True, True, False]) \
            .groupby(by=['walk', 'ticker'], as_index=False).nth(0)
        all_companies = metrics.groupby(by=['walk', 'ticker'], as_index=False).nth(0)
        df3 = all_companies.merge(test_df, left_on=['walk', 'ticker'], right_on=['walk', 'ticker'], how='left',
                                  suffixes=('_all', '_selected'))
        df3 = df3[['walk', 'ticker', 'MSE_baseline_all', 'MSE_baseline_selected', 'MSE_pi_selected', 'MSE_pi_all',
                   'removed_column_selected', 'removed_FI_selected', 'is_lower']]
        df3.rename(inplace=True, columns={'MSE_baseline_all': 'MSE_baseline', 'MSE_pi_selected': 'MSE_pi',
                                          'removed_column_selected': 'removed_column',
                                          'removed_FI_selected': 'removed_FI'})
        test_df = df3.copy()

    df1 = test_df.copy()

    df1['{0}'.format(error_label)] = test_df.apply(lambda x: get_error(x, th, error_label='MSE'), axis=1)
    df1['removed_column'] = test_df.apply(lambda x: get_column(x, th), axis=1)
    df1['method'] = label
    return df1


def get_errors_df_by_walk_5(metrics_df, thresholds, walk, metric='MSE', worst=False):
    metrics = metrics_df[metrics_df['walk'] == walk].copy()
    label_baseline, label_pi = "{0}_baseline".format(metric), "{0}_pi".format(metric)
    metrics['error_diff'] = (metrics[label_baseline] - metrics[label_pi]).astype(float)
    metrics['positive'] = ((metrics[label_baseline] - metrics[label_pi]) > 0).astype(int)

    indexes = metrics.sort_values(['walk', 'ticker', 'removed_FI'], ascending=[True, True, worst]) \
        .groupby(by=['walk', 'ticker'], as_index=False).nth(0)[['walk', 'ticker', 'removed_FI', 'removed_column']]

    indexes.dropna(inplace=True)
    test_df = metrics[metrics.index.isin(indexes.index)]
    errors = pd.DataFrame()

    for idx, th in enumerate(thresholds):
        df = test_df[test_df['removed_FI'] <= th].copy()
        intermediate = pd.concat([
            df.groupby(by='walk').agg({'error_diff': 'mean', 'positive': 'sum'}).rename(
                columns={'error_diff': 'error_diff_avg', 'positive': 'positive_count'}),
            df.groupby(by='walk').agg({'error_diff': 'sum', 'positive': 'count'}).rename(
                columns={'error_diff': 'error_diff_sum', 'positive': 'removed_count'}),
        ], axis=1).reset_index()

        intermediate['threshold'] = th
        intermediate['index'] = idx
        errors = pd.concat([errors, intermediate], sort=False)
    return errors


def get_errors_df_by_walk_3(metrics_df, thresholds, walk, metric='MSE', worst=False):
    metrics = metrics_df[metrics_df['walk'] == walk].copy()
    label_baseline, label_pi = "{0}_baseline".format(metric), "{0}_pi".format(metric)
    metrics['error_diff'] = (metrics[label_baseline] - metrics[label_pi]).astype(float)
    metrics['positive'] = ((metrics[label_baseline] - metrics[label_pi]) > 0).astype(int)
    errors = pd.DataFrame()
    metrics = metrics.sort_values(by=['walk', 'ticker', 'removed_FI'], ascending=[True, True, worst])
    for idx, th in enumerate(thresholds):
        test = metrics.groupby(by=['walk', 'ticker'], as_index=False).apply(lambda x: x[x['removed_FI'] <= th].head(1))

        if test.empty:
            intermediate = pd.DataFrame(data=np.ones((1, 5)) * np.NAN, index=[0],
                                        columns=['walk', 'error_diff_avg', 'error_diff_sum',
                                                 'positive_count', 'removed_count'])
            intermediate['walk'] = walk
        else:
            intermediate = pd.concat([
                test.groupby(by='walk').agg({'error_diff': 'mean', 'positive': 'sum'}).rename(
                    columns={'error_diff': 'error_diff_avg', 'positive': 'positive_count'}),
                test.groupby(by='walk').agg({'error_diff': 'sum', 'positive': 'count'}).rename(
                    columns={'error_diff': 'error_diff_sum', 'positive': 'removed_count'}),
            ], axis=1).reset_index()

        intermediate['threshold'] = th
        intermediate['index'] = idx
        errors = pd.concat([errors, intermediate])
    return errors


def get_optimal_threshold(metrics_all, walk, labels):
    df_worst = get_errors_df_by_walk_5(metrics_all, np.arange(0.0, -0.03, -0.0001), walk, worst=True)
    df_best = get_errors_df_by_walk_5(metrics_all, np.arange(0.0, -0.03, -0.0001), walk, worst=False)
    df_running = get_errors_df_by_walk_3(metrics_all, np.arange(0.0, -0.03, -0.0001), walk, worst=False)
    idx_worst = get_optimal_threshold_strategy(df_worst)
    idx_best = get_optimal_threshold_strategy(df_best)
    idx_running = get_optimal_threshold_strategy(df_running)
    return {'walk': walk, 'threshold_best': None if idx_best is None else df_best.iloc[idx_best]['threshold']
        , 'error_best': None if idx_best is None else df_best.iloc[idx_best]['error_diff_avg']
        , 'no_improvements_best': None if idx_best is None else df_best.iloc[idx_best]['positive_count']
        , 'ratio_best': None if idx_best is None else df_best.iloc[idx_best]['positive_count'] / df_best.iloc[idx_best]['removed_count']
        , 'threshold_worst': None if idx_worst is None else df_worst.iloc[idx_worst]['threshold']
        , 'error_worst': None if idx_worst is None else df_worst.iloc[idx_worst]['error_diff_avg']
        , 'no_improvements_worst': None if idx_worst is None else df_worst.iloc[idx_best]['positive_count']
        , 'ratio_worst': None if idx_worst is None else df_worst.iloc[idx_best]['positive_count'] / df_worst.iloc[idx_best]['removed_count']
        , 'threshold_running': None if idx_running is None else df_running.iloc[idx_running]['threshold']
        , 'error_running': None if idx_running is None else df_running.iloc[idx_running]['error_diff_avg']
        , 'no_improvements_running': None if idx_running is None else df_running.iloc[idx_running]['positive_count']
        , 'ratio_running': None if idx_running is None else df_running.iloc[idx_running]['positive_count'] / df_running.iloc[idx_running][
            'removed_count']
            }


def get_optimal_threshold_strategy(metrics_df):
    try:
        if metrics_df.empty:
            return None
        th_index = np.nanargmax(metrics_df['error_diff_avg'].values)
        return th_index
    except ValueError:
        return None


class Threshold:
    def __init__(self, thresholds_path=None):
        self.__thresholds = pd.DataFrame(columns=threshold_columns)
        if thresholds_path is not None:
            self.__thresholds_path = thresholds_path + 'LOOC_thresholds.csv'
        else:
            self.__thresholds_path = None

    def get_thresholds(self, metrics_all: pd.DataFrame, idx: int):
        print('*' * 10 + 'START computing thresholds' + '*' * 10)
        threshold_row = get_optimal_threshold(metrics_all[metrics_all.walk == idx], idx, thresholds_labels)
        self.__thresholds = self.__thresholds.append(threshold_row, ignore_index=True)
        if self.__thresholds_path is not None:
            self.__thresholds.to_csv(self.__thresholds_path, index=False)
        print('*' * 10 + 'END computing thresholds' + '*' * 10)

        dfs = dict([(th_label, (get_metrics(metrics_all[metrics_all.walk == idx], self.__thresholds, th_label))) for th_label
                    in thresholds_labels])
        return dfs
