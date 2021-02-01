import datetime
import time

import joblib
import lime
import lime.lime_tabular
import pandas as pd
import yfinance as yf
from lime import submodular_pick
from sklearn.ensemble import RandomForestRegressor

from utils import *


def get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test, context, columns=None):
    if columns is not None:
        X, y = X_cr_train[columns].copy(), y_cr_train.copy()
        X_test, y_test = X_cr_test[columns].copy(), y_cr_test.copy()
    else:
        X, y = X_cr_train.copy(), y_cr_train.copy()
        X_test, y_test = X_cr_test.copy(), y_cr_test.copy()

    print('train', X.shape, y.shape)
    print('test', X_test.shape, y_test.shape)
    # print('columns X:', X.columns,X_test.columns)
    # print('columns y:', y.columns,y_test.columns)

    # regressor = RandomForestRegressor(n_estimators=500, max_depth=20, max_features=1, oob_score=True, random_state=42)
    regressor = RandomForestRegressor(n_estimators=350, max_samples=0.4, max_features=1, oob_score=True,
                                      random_state=42)
    regressor.fit(X, y.values.ravel())
    save_path = './LIME/models/{0}_cr_{1}_{2}_{3}.joblib'.format(context["ticker"], context["method"],
                                                                 context["start"].strftime("%Y-%m-%d"),
                                                                 context["end"].strftime("%Y-%m-%d"))
    joblib.dump(regressor, save_path)
    y_hat = regressor.predict(X_test)
    # y_test['label']=y_cr_test['label'].values
    y_test['predicted'] = y_hat.reshape(-1, 1)
    print('test', X_test.shape, y_test.shape)
    return regressor, y_test


def get_least_important_feature_by_fi(X_cr_train, y_cr_train, X_cr_test, y_cr_test, regressor, walk=1):
    ####***************feature importance***********************
    print('*' * 20, 'feature importance', '*' * 20)
    all_columns = X_cr_train.columns
    feat_imp_s = pd.DataFrame(
        {'features': all_columns, "feature_importance": regressor.feature_importances_}).sort_values(
        'feature_importance', ascending=False)
    min_row = feat_imp_s['feature_importance'].values.argmin()
    column = feat_imp_s.iloc[min_row].features
    columns = set(all_columns) - set([column])
    return columns, column, None


def get_least_important_feature_by_pi(X_cr_train, y_cr_train, X_cr_test, y_cr_test, regressor, walk=1):
    ####***************permutation feature importance***********************
    print('*' * 20, 'permutation importance', '*' * 20)
    permutation_importance_s, all_feat_imp_df, order_column = compute_permutation_importance(X_cr_test, y_cr_test,
                                                                                             regressor,
                                                                                             metric=mean_squared_error)
    min_row = permutation_importance_s['permutation_importance'].values.argmin()
    column = permutation_importance_s.iloc[min_row].features
    columns = set(X_cr_test.columns) - set([column])
    return columns, column, None


def get_least_important_feature_by_sp(X_cr_train, y_cr_train, X_cr_test, y_cr_test, regressor, walk=1):
    # ####***************LIME feature importance***********************
    print('*' * 20, 'LIME feature importance', '*' * 20)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_cr_train.values,
                                                            training_labels=y_cr_train.values,
                                                            feature_names=X_cr_train.columns.tolist(),
                                                            verbose=False, mode='regression'
                                                            , discretize_continuous=False
                                                            , random_state=123)
    sp_obj_cr = submodular_pick.SubmodularPick(lime_explainer, X_cr_test.values, regressor.predict, num_features=9,
                                               num_exps_desired=100)
    W_s = pd.DataFrame([dict(this.as_list(label=0)) for this in sp_obj_cr.explanations])
    rank_w_s = W_s[X_cr_test.columns].abs().rank(1, ascending=False, method='first')
    rank_w_s_median = rank_w_s.median().sort_values(ascending=False)
    rank_w_s_mean = rank_w_s.mean().sort_values(ascending=False)
    min_row, mean_row = rank_w_s_median.index[0], rank_w_s_mean.index[0]
    columns = set(X_cr_test.columns) - set([min_row])
    return columns, min_row, mean_row


def get_cumulative_returns(data: pd.DataFrame):
    indices = [1, 2, 3, 4, 5, 21, 63, 126, 252]
    clr_df = pd.DataFrame()
    for i in indices:
        columns_name = 'Returns_{0}'.format(i)
        item = (data['Close'].shift(1) - data['Open'].shift(i)) / data['Open'].shift(i)
        item.columns = pd.MultiIndex.from_product([[columns_name], item.columns])
        dummy = item.unstack().reset_index().pivot(columns=["level_0"])
        dummy_ret = pd.DataFrame()
        dummy_ret['Date'] = dummy.loc[:, ('Date', columns_name)].values.ravel()
        dummy_ret['ticker'] = dummy.loc[:, ('level_1', columns_name)].values.ravel()
        dummy_ret.set_index(['Date', 'ticker'], inplace=True)
        dummy_ret[columns_name] = dummy.loc[:, [(0, columns_name)]].values.ravel()
        clr_df = pd.concat([clr_df, dummy_ret], axis=1)
    return clr_df


def get_target(data: pd.DataFrame):
    dummy_df = (data['Close'] - data['Open']) / data['Open']
    dummy_df.columns = pd.MultiIndex.from_product([['label'], dummy_df.columns])
    dummy = dummy_df.unstack().reset_index().pivot(columns=["level_0"])
    label_df = pd.DataFrame()
    label_df['Date'] = dummy.loc[:, ('Date', 'label')].values.ravel()
    label_df['ticker'] = dummy.loc[:, ('level_1', 'label')].values.ravel()
    label_df.set_index(['Date', 'ticker'], inplace=True)
    label_df['label'] = dummy.loc[:, [(0, 'label')]].values.ravel()
    label_df.head()
    return label_df


def validate_data(X):
    return len(X.index.unique()) == len(X.groupby(X.index))


methods = {
    'fi': get_least_important_feature_by_fi,
    'pi': get_least_important_feature_by_pi,
    'sp': get_least_important_feature_by_sp
}

if __name__ == "__main__":
    constituents = pd.read_csv('./LIME/data/constituents.csv')
    tickers = constituents['Ticker']
    num_stocks = len(tickers)

    metrics = pd.DataFrame()
    feat_imp = pd.DataFrame()
    W = pd.DataFrame()
    rank_w = pd.DataFrame()
    permutation_importance = pd.DataFrame()
    missing_columns = pd.DataFrame()
    test = 18

    wf = WalkForward(datetime.datetime.strptime('2011-01-01', '%Y-%m-%d'),
                     datetime.datetime.strptime('2018-01-01', '%Y-%m-%d'), 4, no_walks=3)

    for idx, train_set, test_set in wf.get_walks():
        print('*' * 20, idx, '*' * 20)
        print(train_set.start, train_set.end)
        print(test_set.start, test_set.end)
        print('*' * 20)
        data = yf.download(list(tickers), train_set.start, test_set.end)  # ,'2018-01-01')
        feature_df = get_cumulative_returns(data)
        label_df = get_target(data)
        df1 = pd.concat([feature_df, label_df], axis=1)
        df1.dropna(inplace=True)
        start_test = test_set.start

        for ticker in tickers:
            print('*' * 20, ticker, '*' * 20)
            start_time = time.perf_counter()
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

            baseline, b_y_cr_test = get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test,
                                                      context={'walk': idx, 'ticker': ticker, 'method': 'baseline',
                                                               'start': train_set.start, 'end': train_set.end})

            metric_single_baseline = get_prediction_performance_results(b_y_cr_test, False)
            metric_single_baseline['walk'] = idx
            metric_single_baseline['model'] = 'baseline'
            metric_single_baseline['ticker'] = ticker
            metric_single_baseline['removed_column'] = ''
            metrics = metrics.append(metric_single_baseline, ignore_index=True)

            for method, func in methods.items():
                columns, missing_col_median, missing_col_mean = func(X_cr_train, y_cr_train, X_cr_test, y_cr_test,
                                                                     baseline, walk=idx)
                looc_fi_regressor, looc_y_cr_test = get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test,
                                                                      context={'walk': idx, 'ticker': ticker,
                                                                               'method': method,
                                                                               'start': train_set.start,
                                                                               'end': train_set.end}, columns=columns)
                metrics_fi_looc = get_prediction_performance_results(looc_y_cr_test, False)
                metrics_fi_looc['walk'] = idx
                metrics_fi_looc['model'] = method
                metrics_fi_looc['ticker'] = ticker
                metrics_fi_looc['removed_column'] = missing_col_median
                missing_columns = missing_columns.append(
                    {'walk': idx, 'method': method, 'ticker': ticker, 'removed_column_median': missing_col_median,
                     'removed_column_mean': missing_col_mean}, ignore_index=True)
                metrics = metrics.append(metrics_fi_looc, ignore_index=True)
                metrics.to_csv('/LIME/data/LOOC_metrics_cr_{0}.csv'.format(test), index=False)
                missing_columns.to_csv('/LIME/data/LOOC_missing_columns_cr_{0}.csv'.format(test), index=False)
            end_time = time.perf_counter()
            print('{0} took {1} s'.format(ticker, end_time - start_time))

    metrics.to_csv('/LIME/data/LOOC_metrics_cr_{0}.csv'.format(test), index=False)
