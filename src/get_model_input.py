import os

import numpy as np
import pandas as pd
import ta as ta
import yfinance as yf

from walkforward import Walk


def disparity(close, n_pmav):
    return pd.Series(close * 100 / close.rolling(n_pmav).mean(), name='disp_{0}'.format(n_pmav))


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


def get_technical_indicators(data: pd.DataFrame, ticker):
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
    ti_df.loc[:, 'ticker'] = ticker
    ti_df.set_index(['Date', 'ticker'], inplace=True)
    return ti_df


def get_target(data_df: pd.DataFrame, ticker):
    """

    Parameters
    ----------
    data_df: OHCLV dataframe from source
    ticker : string
    """
    label_df = pd.DataFrame(data=(data_df['Close'] - data_df['Open']) / data_df['Open'], columns=['label'])
    label_df['ticker'] = ticker
    label_df.reset_index(inplace=True)
    label_df.set_index(['Date', 'ticker'], inplace=True)
    return label_df


def get_data_from_file(ticker: str, folder, date_start, date_end):
    data_df = pd.read_csv(folder + ticker + ".csv", parse_dates=True) if os.path.exists(
        folder + ticker + ".csv") else pd.DataFrame()

    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df.set_index('Date', inplace=True)
        data_df = data_df[date_start:date_end]

    return data_df


class CompanyFeatures:
    def __init__(self, folder_output: str, feature_type='cr'):
        """
        Parameters
        ----------
        folder_output : str
        """
        self.folder = folder_output
        self.__feature_type = feature_type

    def get_features(self, ticker: str, walk: Walk):
        x_df, y_df = self.__get_data_for_ticker(ticker, folder=self.folder, date_start=walk.train.start,
                                                date_end=walk.test.end)

        if x_df.empty or y_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        start_test = walk.test.start
        validation_start = walk.validation.start

        X_cr_train, y_cr_train = x_df.loc[:validation_start], y_df.loc[:validation_start]
        X_cr_validation, y_cr_validation = x_df.loc[validation_start:start_test], y_df.loc[
                                                                                  validation_start:start_test]
        X_cr_test, y_cr_test = x_df.loc[start_test:], y_df.loc[start_test:]

        print('{0} train {1} {2}'.format(ticker, X_cr_train.index.min(), X_cr_train.index.max()))
        print('{0} test {1} {2}'.format(ticker, X_cr_validation.index.min(), X_cr_validation.index.max()))
        print('{0} test {1} {2}'.format(ticker, X_cr_test.index.min(), X_cr_test.index.max()))

        return X_cr_train, y_cr_train, X_cr_validation, y_cr_validation, X_cr_test, y_cr_test

    def __get_data_online(self, ticker: str, date_start, date_end):
        data = yf.download(ticker, start=date_start, end=date_end)
        if data is None or len(data) == 0:
            return pd.DataFrame()
        if self.__feature_type == 'cr':
            feature_df = get_cumulative_returns(data, ticker)
        elif self.__feature_type == 'lr':
            feature_df = get_cumulative_returns(data, ticker)
        else:
            feature_df = get_technical_indicators(data, ticker)

        label_df = get_target(data, ticker)
        df1 = pd.concat([feature_df, label_df], axis=1)
        df1.dropna(inplace=True)
        data_df = df1.loc[[i for i in df1.index if i[1] == ticker]].copy()
        data_df.reset_index(inplace=True)
        return data_df

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
            data_df.set_index('Date', inplace=True)
        X = data_df[[c for c in data_df.columns if c not in ['Date', 'ticker', 'label']]]
        y = data_df[['label']]
        return X, y
