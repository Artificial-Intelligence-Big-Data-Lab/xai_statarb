import numpy as np
import pandas as pd


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

        intermediate = df.groupby(by='walk').agg(
            error_diff_avg=pd.NamedAgg(column="error_diff", aggfunc="mean"),
            error_diff_sum=pd.NamedAgg(column="error_diff", aggfunc="sum"),
            positive_count=pd.NamedAgg(column="positive", aggfunc="sum"),
            removed_count=pd.NamedAgg(column="positive", aggfunc="count"),
        ).reset_index()

        intermediate['threshold'] = th
        intermediate['index'] = idx

        baseline = test_df[test_df['removed_FI'] <= 0].groupby(by='walk').agg(
            error_diff_avg=pd.NamedAgg(column="error_diff", aggfunc="mean"),
            error_diff_sum=pd.NamedAgg(column="error_diff", aggfunc="sum"),
            positive_count=pd.NamedAgg(column="positive", aggfunc="sum"),
            removed_count=pd.NamedAgg(column="positive", aggfunc="count"),
        ).reset_index()

        intermediate = intermediate.merge(baseline, left_on=['walk'], right_on=['walk'], suffixes=('', '_baseline'),
                                          how='right')
        errors = pd.concat([errors, intermediate])

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
            intermediate = test.groupby(by='walk').agg(
                error_diff_avg=pd.NamedAgg(column="error_diff", aggfunc="mean"),
                error_diff_sum=pd.NamedAgg(column="error_diff", aggfunc="sum"),
                positive_count=pd.NamedAgg(column="positive", aggfunc="sum"),
                removed_count=pd.NamedAgg(column="positive", aggfunc="count"),
            ).reset_index()

        intermediate['threshold'] = th
        intermediate['index'] = idx
        errors = pd.concat([errors, intermediate])
    return errors


def get_optimal_threshold(df_worst, df_best, df_running, walk):
    idx_worst = get_optimal_threshold_strategy(df_worst)
    idx_best = get_optimal_threshold_strategy(df_best)
    idx_running = get_optimal_threshold_strategy(df_running)
    return {'walk': walk, 'threshold_best': df_best.iloc[idx_best]['threshold']
        , 'error_best': df_best.iloc[idx_best]['error_diff_avg']
        , 'no_improvements_best': df_best.iloc[idx_best]['positive_count']
        , 'ratio_best': df_best.iloc[idx_best]['positive_count'] / df_best.iloc[idx_best]['removed_count']
        , 'threshold_worst': df_worst.iloc[idx_worst]['threshold']
        , 'error_worst': df_worst.iloc[idx_worst]['error_diff_avg']
        , 'no_improvements_worst': df_worst.iloc[idx_best]['positive_count']
        , 'ratio_worst': df_worst.iloc[idx_best]['positive_count'] / df_worst.iloc[idx_best]['removed_count']
        , 'threshold_running': df_running.iloc[idx_running]['threshold']
        , 'error_running': df_running.iloc[idx_running]['error_diff_avg']
        , 'no_improvements_running': df_running.iloc[idx_running]['positive_count']
        , 'ratio_running': df_running.iloc[idx_running]['positive_count'] / df_running.iloc[idx_running][
            'removed_count']
            }


def get_optimal_threshold_strategy(metrics_df):
    th_index = metrics_df['error_diff_avg'].argmax()
    return th_index
