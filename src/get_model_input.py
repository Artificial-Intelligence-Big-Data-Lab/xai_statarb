import os

import numpy as np
import pandas as pd
import ta as ta
import yfinance as yf

from walkforward import Walk


def disparity(close, n_pmav):
    return pd.Series(close * 100 / close.rolling(n_pmav).mean(), name='disp_{0}'.format(n_pmav))


def get_cumulative_returns(data: pd.DataFrame):
    indices = [1, 2, 3, 4, 5, 21, 63, 126, 252]
    clr_df = pd.DataFrame()
    if len(data) < 252:
        return clr_df
    for i in indices:
        columns_name = 'Returns_{0}'.format(i)
        dummy_ret = pd.DataFrame((data['Close'].shift(1) - data['Open'].shift(i)) / data['Open'].shift(i),
                                 columns=[columns_name])
        clr_df = pd.concat([clr_df, dummy_ret], axis=1)
    clr_df.reset_index(inplace=True)
    clr_df.set_index(['Date'], inplace=True)
    return clr_df


def get_technical_indicators(data: pd.DataFrame):
    if len(data) < 252:
        return pd.DataFrame()
    wr = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=14)
    roc = ta.momentum.roc(data['Close'], window=12)
    rsi = ta.momentum.rsi(data['Close'], window=14)
    accd = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'],
                                           data['Volume'])
    macd = ta.trend.macd(data['Close'])
    ema = ta.trend.ema_indicator(data['Close'])
    stock_k = ta.momentum.stochrsi_k(data['Close'], window=14)
    disp5 = disparity(data['Close'], 5)
    disp10 = disparity(data['Close'], 10)
    ti_df = pd.concat([wr.williams_r(), roc, rsi, accd.acc_dist_index(), macd, ema, stock_k, disp5, disp10], axis=1)
    ti_df = ti_df.shift(1)
    ti_df.reset_index(inplace=True)
    ti_df.set_index(['Date'], inplace=True)
    return ti_df


def get_target(data_df: pd.DataFrame):
    """

    Parameters
    ----------
    data_df: OHCLV dataframe from source
    """
    label_df = pd.DataFrame(data=(data_df['Close'] - data_df['Open']) / data_df['Open'], columns=['label'])
    label_df.reset_index(inplace=True)
    label_df.set_index(['Date'], inplace=True)
    return label_df


def get_data_from_file(ticker: str, folder, date_start, date_end):
    data_df = pd.read_csv(folder + ticker + ".csv", parse_dates=True) if os.path.exists(
        folder + ticker + ".csv") else pd.DataFrame()

    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.set_index('Date', inplace=True)
        data_df = data_df[date_start:date_end]
        data_df.reset_index(inplace=True)
    return data_df


class CompanyFeatures:
    def __init__(self, constituents: pd.DataFrame, folder_output: str, feature_type='cr', prediction_type='company'):
        """
        Parameters
        ----------
        folder_output : str
        """
        self.folder = folder_output
        self.__feature_type = feature_type
        self.__prediction_type = prediction_type
        self.__constituents = constituents

    def get_entities(self):
        if self.__prediction_type == 'company':
            for idx in range(len(self.__constituents)):
                row = pd.DataFrame([self.__constituents.iloc[idx]])
                yield row['Ticker'].values[0], row
        else:
            for sector, group in self.__constituents.groupby(by=['Sector'], as_index=False):
                yield sector, group

    def get_features(self, constituents_batch, walk: Walk):

        x_train, y_train = pd.DataFrame(), pd.DataFrame()
        x_validation, y_validation = pd.DataFrame(), pd.DataFrame()
        x_test, y_test = pd.DataFrame(), pd.DataFrame()

        for ticker in constituents_batch['Ticker'].values:
            x_df, y_df = self.__get_data_for_ticker(ticker, folder=self.folder, date_start=walk.train.start, date_end=walk.test.end)

            if x_df.empty or y_df.empty:
                continue

            start_test = walk.test.start
            validation_start = walk.validation.start

            X_cr_train, y_cr_train = x_df.loc[:validation_start], y_df.loc[:validation_start]
            X_cr_train = X_cr_train.reset_index().set_index(['Date', 'ticker'])
            y_cr_train = y_cr_train.reset_index().set_index(['Date', 'ticker'])

            X_cr_validation, y_cr_validation = x_df.loc[validation_start:start_test], y_df.loc[validation_start:start_test]
            X_cr_validation = X_cr_validation.reset_index().set_index(['Date', 'ticker'])
            y_cr_validation = y_cr_validation.reset_index().set_index(['Date', 'ticker'])

            X_cr_test, y_cr_test = x_df.loc[start_test:], y_df.loc[start_test:]
            X_cr_test = X_cr_test.reset_index().set_index(['Date', 'ticker'])
            y_cr_test = y_cr_test.reset_index().set_index(['Date', 'ticker'])

            x_train, y_train = pd.concat([x_train, X_cr_train], sort=False), pd.concat([y_train, y_cr_train],
                                                                                       sort=False)
            x_validation = pd.concat([x_validation, X_cr_validation], sort=False)
            y_validation = pd.concat([y_validation, y_cr_validation], sort=False)

            x_test, y_test = pd.concat([x_test, X_cr_test], sort=False), pd.concat([y_test, y_cr_test], sort=False)

            print('{0} train {1} {2}'.format(ticker, X_cr_train.index.min(), X_cr_train.index.max()))
            print('{0} test {1} {2}'.format(ticker, X_cr_validation.index.min(), X_cr_validation.index.max()))
            print('{0} test {1} {2}'.format(ticker, X_cr_test.index.min(), X_cr_test.index.max()))

        if not (x_train.empty and x_validation.empty and x_test.empty) and (len(constituents_batch['Ticker'].values) > 1):
            x_train, y_train, x_validation, y_validation, x_test, y_test = self.__sanitize_data(x_train, y_train,
                                                                                                x_validation, y_validation,
                                                                                                x_test, y_test)
        print('{0} train {1}'.format(x_train.index.min(), x_train.index.max()))
        print('{0} test {1}'.format(x_validation.index.min(), x_validation.index.max()))
        print('{0} test {1}'.format(x_test.index.min(), x_test.index.max()))

        return x_train, y_train, x_validation, y_validation, x_test, y_test

    def __get_data_online(self, ticker: str, date_start, date_end):
        data = yf.download(ticker, start=date_start, end=date_end)
        if data is None or len(data) == 0:
            return pd.DataFrame()
        if self.__feature_type == 'cr':
            feature_df = get_cumulative_returns(data)
        elif 'cr' in self.__feature_type and 'ti' in self.__feature_type:
            feature_df1 = get_cumulative_returns(data)
            feature_df2 = get_technical_indicators(data)
            if feature_df1.empty or feature_df2.empty:
                feature_df = pd.DataFrame()
            else:
                feature_df = pd.concat([feature_df1, feature_df2], axis=1, sort=False)
        else:
            feature_df = get_technical_indicators(data)

        label_df = get_target(data)
        if feature_df.empty or label_df.empty:
            return pd.DataFrame()
        df1 = pd.concat([feature_df, label_df], axis=1)
        df1.dropna(inplace=True)
        df1.loc[:, 'ticker'] = ticker
        # data_df = df1.loc[[i for i in df1.index if i[1] == ticker]].copy()
        df1.reset_index(inplace=True)
        return df1

    def __get_data_for_ticker(self, ticker: str, folder: str, date_start, date_end):
        """

        Returns
        -------
        pandas DataFrame
        """
        data_df = get_data_from_file(ticker, folder, date_start, date_end)

        if data_df.empty:
            data_df = self.__get_data_online(ticker, date_start, date_end)

        if data_df.empty:
            return data_df, data_df

        data_df = data_df[~data_df.isin([np.nan, np.inf, -np.inf]).any(1)]

        if not data_df.empty:
            data_df.to_csv("{0}{1}.csv".format(folder, ticker), index=False)

        if 'Date' in data_df.columns:
            data_df.set_index(['Date'], inplace=True)
        X = data_df[[c for c in data_df.columns if c not in ['Date', 'label']]]
        y = data_df[['ticker', 'label']]
        return X, y

    @staticmethod
    def __sanitize_data(x_train, y_train, x_validation, y_validation, x_test, y_test):
        tickers_train = x_train.index.get_level_values('ticker').unique()
        tickers_validation = x_validation.index.get_level_values('ticker').unique()
        tickers_test = x_test.index.get_level_values('ticker').unique()
        interset = set(tickers_train).intersection(tickers_validation).intersection(tickers_test)

        x_train.drop(set(tickers_train).difference(interset), level='ticker', inplace=True)
        y_train.drop(set(tickers_train).difference(interset), level='ticker', inplace=True)

        x_validation.drop(set(tickers_validation).difference(interset), level='ticker', inplace=True)
        y_validation.drop(set(tickers_validation).difference(interset), level='ticker', inplace=True)

        x_test.drop(set(tickers_test).difference(interset), level='ticker', inplace=True)
        y_test.drop(set(tickers_test).difference(interset), level='ticker', inplace=True)

        x = x_train.groupby(level=['ticker']).size()
        to_discard_train = x[x < x.max()].index.values

        x = x_validation.groupby(level=['ticker']).size()
        to_discard_validation = x[x < x.max()].index.values

        x = x_test.groupby(level=['ticker']).size()
        to_discard_test = x[x < x.max()].index.values

        to_discard = set(to_discard_train) | set(to_discard_validation) | set(to_discard_test)

        x_train.drop(to_discard, level='ticker', inplace=True)
        y_train.drop(to_discard, level='ticker', inplace=True)

        x_validation.drop(to_discard, level='ticker', inplace=True)
        y_validation.drop(to_discard, level='ticker', inplace=True)

        x_test.drop(to_discard, level='ticker', inplace=True)
        y_test.drop(to_discard, level='ticker', inplace=True)

        return x_train, y_train, x_validation, y_validation, x_test, y_test
