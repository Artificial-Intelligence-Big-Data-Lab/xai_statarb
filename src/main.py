import datetime
import os
import random
import time

import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import feature_selection as fs
from utils import *
from walkforward import WalkForward


def get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test, context, columns=None,
                      get_cross_validation_results=True):
    if columns is not None:
        X_train, y_train = X_cr_train[columns].copy(), y_cr_train.copy()
        X_test, y_test = X_cr_test[columns].copy(), y_cr_test.copy()
    else:
        X_train, y_train = X_cr_train.copy(), y_cr_train.copy()
        X_test, y_test = X_cr_test.copy(), y_cr_test.copy()

    print('train', X_train.shape, y_train.shape)
    print('test', X_test.shape, y_test.shape)

    regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1, oob_score=True,
                                      random_state=42)
    # regressor = Pipeline([('scaler', MinMaxScaler(feature_range=(-1, 1)))
    #                          , ('svc', linear_model.TweedieRegressor( alpha=0.001))])
    # # regressor = ExtraTreesRegressor(n_estimators=350, max_samples=0.4, max_features=1, oob_score=True, bootstrap=True, random_state=42)
    # regressor.fit(X,y.values.ravel())
    # # save_path= './LIME/models/{0}_cr_{1}_{2}_{3}.joblib'.format(context["ticker"],context["method"],context["start"].strftime("%Y-%m-%d"),context["end"].strftime("%Y-%m-%d"))
    # # joblib.dump(regressor,save_path)

    if get_cross_validation_results:
        X, y = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])

        cv = TimeSeriesSplit(max_train_size=int(2 * len(X) / 3), n_splits=10)
        score = cross_validate(regressor, X.values, y.values.ravel(),
                               scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'], n_jobs=-1,
                               verbose=0, cv=cv)
    regressor.fit(X_train.values, y_train.values.ravel())
    y_hat = regressor.predict(X_test.values)
    y_test['predicted'] = y_hat.reshape(-1, 1)
    print('test', X_test.shape, y_test.shape)
    return regressor, y_test, score if get_cross_validation_results else None


def get_cumulative_returns(data: pd.DataFrame, ticker):
    indices = [1, 2, 3, 4, 5, 21, 63, 126, 252]
    clr_df = pd.DataFrame()
    for i in indices:
        columns_name = 'Returns_{0}'.format(i)
        dummy_ret = pd.DataFrame((data['Close'].shift(1) - data['Open'].shift(i)) / data['Open'].shift(i),
                                 columns=[columns_name])
        clr_df = pd.concat([clr_df, dummy_ret], axis=1)
    clr_df['ticker'] = ticker
    clr_df.reset_index(inplace=True)
    clr_df.set_index(['Date', 'ticker'], inplace=True)
    return clr_df


def get_target(data_df: pd.DataFrame, ticker):
    """

    Parameters
    ----------
    ticker : string
    """
    label_df = pd.DataFrame(data=(data_df['Close'] - data_df['Open']) / data_df['Open'], columns=['label'])
    label_df['ticker'] = ticker
    label_df.reset_index(inplace=True)
    label_df.set_index(['Date', 'ticker'], inplace=True)
    return label_df


def validate_data(X):
    return len(X.index.unique()) == len(X.groupby(X.index))


def check_if_processed(metrics, ticker, walk):
    if metrics.empty:
        return False
    return len(metrics[(metrics['ticker'] == ticker) & (metrics['walk'] == walk)]) == 4


def add_score_to_metrics(metric_df, score):
    """

    Parameters
    ----------
    score: dictionary with cross-validation measures
    metric_df : pandas DataFrame
    """
    metric_df['mean_mse'] = -score['test_neg_mean_squared_error'].mean()
    metric_df['std_mse'] = score['test_neg_mean_squared_error'].std(ddof=1)
    metric_df['mean_mae'] = -score['test_neg_mean_absolute_error'].mean()
    metric_df['std_mae'] = score['test_neg_mean_absolute_error'].std(ddof=1)
    metric_df['mean_r2'] = score['test_r2'].mean()
    metric_df['std_r2'] = score['test_r2'].std(ddof=1)
    return


if __name__ == "__main__":
    constituents = pd.read_csv('../LIME/data/constituents.csv')
    tickers = constituents['Ticker']
    num_stocks = len(tickers)

    random.seed(30)
    test = 38
    features_no = 2
    METRICS_OUTPUT_PATH = '../LIME/data/LOOC_metrics_cr_{0}.csv'.format(test)
    REMOVED_COLUMNS_PATH = '../LIME/data/LOOC_missing_columns_cr_{0}.csv'.format(test)

    metrics = pd.read_csv(METRICS_OUTPUT_PATH) if os.path.exists(METRICS_OUTPUT_PATH) else pd.DataFrame()

    missing_columns = pd.read_csv(REMOVED_COLUMNS_PATH) if os.path.exists(REMOVED_COLUMNS_PATH) else pd.DataFrame()

    wf = WalkForward(datetime.datetime.strptime('2011-01-01', '%Y-%m-%d'),
                     datetime.datetime.strptime('2018-01-01', '%Y-%m-%d'), 4, no_walks=1)

    methods = {
        # 'fi': RFFeatureImportanceSelector(features_no),
        'pi': fs.PermutationImportanceSelector(features_no, seed=42),
        # 'pi2': fs.PISelector(features_no, seed=42),
        'pi_all': fs.PermutationImportanceSelector(seed=42),
        # 'pi2_all': fs.PISelector(seed=42)
        # 'sp': get_least_important_feature_by_sp
    }

    for idx, train_set, test_set in wf.get_walks():
        print('*' * 20, idx, '*' * 20)
        print(train_set.start, train_set.end)
        print(test_set.start, test_set.end)
        print('*' * 20)

        start_test = test_set.start

        for ticker in tickers[:2]:

            print('*' * 20, ticker, '*' * 20)

            if check_if_processed(metrics, ticker, idx):
                print("ticker {0} ALREADY PROCESSED".format(ticker))
                continue

            start_time = time.perf_counter()

            data = yf.download(ticker, train_set.start, test_set.end)  # ,'2018-01-01')
            feature_df = get_cumulative_returns(data, ticker)
            label_df = get_target(data, ticker)
            df1 = pd.concat([feature_df, label_df], axis=1)
            df1.dropna(inplace=True)
            appl_df1 = df1.loc[[i for i in df1.index if i[1] == ticker]].copy()
            # print (appl_df1.head())
            if appl_df1.empty:
                continue

            appl_df1.reset_index(inplace=True)
            if 'Date' in appl_df1.columns:
                appl_df1.set_index('Date', inplace=True)

            X_cr_train = appl_df1.loc[:start_test][[c for c in df1.columns if c not in ['Date', 'ticker', 'label']]]
            y_cr_train = appl_df1.loc[:start_test][['label']]

            X_cr_test = appl_df1.loc[start_test:][[c for c in df1.columns if c not in ['Date', 'ticker', 'label']]]
            y_cr_test = appl_df1.loc[start_test:][['label']]

            if len(X_cr_train) == 0 or len(y_cr_test) == 0:
                continue
            print('{0} train {1} {2}'.format(ticker, X_cr_train.index.min(), X_cr_train.index.max()))
            print('{0} test {1} {2}'.format(ticker, X_cr_test.index.min(), X_cr_test.index.max()))

            baseline, b_y_cr_test, score = get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test,
                                                             context={'walk': idx, 'ticker': ticker,
                                                                      'method': 'baseline', 'start': train_set.start,
                                                                      'end': train_set.end},
                                                             get_cross_validation_results=True)

            metric_single_baseline = get_prediction_performance_results(b_y_cr_test, False)
            metric_single_baseline['walk'] = idx
            metric_single_baseline['model'] = 'baseline'
            metric_single_baseline['ticker'] = ticker
            add_score_to_metrics(metric_single_baseline, score)

            for r_idx in range(0, len(X_cr_test.columns)):
                metric_single_baseline['removed_column{0}'.format(r_idx)] = ''
                metric_single_baseline['removed_column_imp{0}'.format(r_idx)] = ''

            metrics = metrics.append(metric_single_baseline, ignore_index=True)

            for method, transformer in methods.items():
                columns = transformer.fit_transform(baseline, X_cr_train, y_cr_train, X_cr_test, y_cr_test)
                looc_fi_regressor, looc_y_cr_test, score_looc = get_fit_regressor(X_cr_train, y_cr_train, X_cr_test,
                                                                                  y_cr_test,
                                                                                  context={'walk': idx,
                                                                                           'ticker': ticker,
                                                                                           'method': method,
                                                                                           'start': train_set.start,
                                                                                           'end': train_set.end},
                                                                                  columns=columns)
                metrics_fi_looc = get_prediction_performance_results(looc_y_cr_test, False)
                metrics_fi_looc['walk'] = idx
                metrics_fi_looc['model'] = method
                metrics_fi_looc['ticker'] = ticker
                missing_col_dict = {}
                for r_idx, missing_col in enumerate(transformer.importance):
                    metrics_fi_looc['removed_column{0}'.format(r_idx)] = missing_col[0]
                    metrics_fi_looc['removed_column_imp{0}'.format(r_idx)] = missing_col[1]
                    missing_col_dict['removed_column{0}'.format(r_idx)] = missing_col[0]

                add_score_to_metrics(metrics_fi_looc, score_looc)

                missing_col_dict.update(dict({'walk': idx, 'method': method, 'ticker': ticker}))
                missing_columns = missing_columns.append(missing_col_dict, ignore_index=True)
                missing_columns.to_csv(REMOVED_COLUMNS_PATH, index=False)

                metrics = metrics.append(metrics_fi_looc, ignore_index=True)
                metrics.to_csv(METRICS_OUTPUT_PATH, index=False)

            end_time = time.perf_counter()
            print('{0} took {1} s'.format(ticker, end_time - start_time))
    print('*' * 20, 'DONE', '*' * 20)
    metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
