import pandas as pd
import yfinance as yf


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
    data_df: OHCLV dataframe from source
    ticker : string
    """
    label_df = pd.DataFrame(data=(data_df['Close'] - data_df['Open']) / data_df['Open'], columns=['label'])
    label_df['ticker'] = ticker
    label_df.reset_index(inplace=True)
    label_df.set_index(['Date', 'ticker'], inplace=True)
    return label_df


def get_data_for_ticker(ticker: str, date_start, date_end):
    """

    Returns
    -------
    pandas DataFrame
    """
    data = yf.download(ticker, start=date_start, end=date_end)  # ,'2018-01-01')
    feature_df = get_cumulative_returns(data, ticker)
    label_df = get_target(data, ticker)
    df1 = pd.concat([feature_df, label_df], axis=1)
    df1.dropna(inplace=True)
    data_df = df1.loc[[i for i in df1.index if i[1] == ticker]].copy()

    if data_df.empty:
        return data_df

    data_df.reset_index(inplace=True)
    if 'Date' in data_df.columns:
        data_df.set_index('Date', inplace=True)
    X = data_df[[c for c in data_df.columns if c not in ['Date', 'ticker', 'label']]]
    y = data_df[['label']]
    return X, y
