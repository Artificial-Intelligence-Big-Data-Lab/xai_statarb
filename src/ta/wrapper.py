"""
.. module:: wrapper
   :synopsis: Wrapper of Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)
"""

import pandas as pd

from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)


def add_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Accumulation Distribution Index
    df["{0}volume_adi".format(colprefix)] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()

    # On Balance Volume
    df["{0}volume_obv".format(colprefix)] = OnBalanceVolumeIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).on_balance_volume()

    # Chaikin Money Flow
    df["{0}volume_cmf".format(colprefix)] = ChaikinMoneyFlowIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).chaikin_money_flow()

    # Force Index
    df["{0}volume_fi".format(colprefix)] = ForceIndexIndicator(
        close=df[close], volume=df[volume], window=13, fillna=fillna
    ).force_index()

    # Money Flow Indicator
    df["{0}volume_mfi".format(colprefix)] = MFIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        volume=df[volume],
        window=14,
        fillna=fillna,
    ).money_flow_index()

    # Ease of Movement
    indicator_eom = EaseOfMovementIndicator(
        high=df[high], low=df[low], volume=df[volume], window=14, fillna=fillna
    )
    df["{0}volume_em".format(colprefix)] = indicator_eom.ease_of_movement()
    df["{0}volume_sma_em".format(colprefix)] = indicator_eom.sma_ease_of_movement()

    # Volume Price Trend
    df["{0}volume_vpt".format(colprefix)] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()

    # Negative Volume Index
    df["{0}volume_nvi".format(colprefix)] = NegativeVolumeIndexIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).negative_volume_index()

    # Volume Weighted Average Price
    df["{0}volume_vwap".format(colprefix)] = VolumeWeightedAveragePrice(
        high=df[high],
        low=df[low],
        close=df[close],
        volume=df[volume],
        window=14,
        fillna=fillna,
    ).volume_weighted_average_price()

    return df


def add_volatility_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Average True Range
    df["{0}volatility_atr".format(colprefix)] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    ).average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], window=20, window_dev=2, fillna=fillna
    )
    df["{0}volatility_bbm".format(colprefix)] = indicator_bb.bollinger_mavg()
    df["{0}volatility_bbh".format(colprefix)] = indicator_bb.bollinger_hband()
    df["{0}volatility_bbl".format(colprefix)] = indicator_bb.bollinger_lband()
    df["{0}volatility_bbw".format(colprefix)] = indicator_bb.bollinger_wband()
    df["{0}volatility_bbp".format(colprefix)] = indicator_bb.bollinger_pband()
    df["{0}volatility_bbhi".format(colprefix)] = indicator_bb.bollinger_hband_indicator()
    df["{0}volatility_bbli".format(colprefix)] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    indicator_kc = KeltnerChannel(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    )
    df["{0}volatility_kcc".format(colprefix)] = indicator_kc.keltner_channel_mband()
    df["{0}volatility_kch".format(colprefix)] = indicator_kc.keltner_channel_hband()
    df["{0}volatility_kcl".format(colprefix)] = indicator_kc.keltner_channel_lband()
    df["{0}volatility_kcw".format(colprefix)] = indicator_kc.keltner_channel_wband()
    df["{0}volatility_kcp".format(colprefix)] = indicator_kc.keltner_channel_pband()
    df["{0}volatility_kchi".format(colprefix)] = indicator_kc.keltner_channel_hband_indicator()
    df["{0}volatility_kcli".format(colprefix)] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], window=20, offset=0, fillna=fillna
    )
    df["{0}volatility_dcl".format(colprefix)] = indicator_dc.donchian_channel_lband()
    df["{0}volatility_dch".format(colprefix)] = indicator_dc.donchian_channel_hband()
    df["{0}volatility_dcm".format(colprefix)] = indicator_dc.donchian_channel_mband()
    df["{0}volatility_dcw".format(colprefix)] = indicator_dc.donchian_channel_wband()
    df["{0}volatility_dcp".format(colprefix)] = indicator_dc.donchian_channel_pband()

    # Ulcer Index
    df["{0}volatility_ui".format(colprefix)] = UlcerIndex(
        close=df[close], window=14, fillna=fillna
    ).ulcer_index()

    return df


def add_trend_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # MACD
    indicator_macd = MACD(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df["{0}trend_macd".format(colprefix)] = indicator_macd.macd()
    df["{0}trend_macd_signal".format(colprefix)] = indicator_macd.macd_signal()
    df["{0}trend_macd_diff".format(colprefix)] = indicator_macd.macd_diff()

    # SMAs
    df["{0}trend_sma_fast".format(colprefix)] = SMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).sma_indicator()
    df["{0}trend_sma_slow".format(colprefix)] = SMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).sma_indicator()

    # EMAs
    df["{0}trend_ema_fast".format(colprefix)] = EMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).ema_indicator()
    df["{0}trend_ema_slow".format(colprefix)] = EMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).ema_indicator()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(
        high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
    )
    df["{0}trend_adx".format(colprefix)] = indicator_adx.adx()
    df["{0}trend_adx_pos".format(colprefix)] = indicator_adx.adx_pos()
    df["{0}trend_adx_neg".format(colprefix)] = indicator_adx.adx_neg()

    # Vortex Indicator
    indicator_vortex = VortexIndicator(
        high=df[high], low=df[low], close=df[close], window=14, fillna=fillna
    )
    df["{0}trend_vortex_ind_pos".format(colprefix)] = indicator_vortex.vortex_indicator_pos()
    df["{0}trend_vortex_ind_neg".format(colprefix)] = indicator_vortex.vortex_indicator_neg()
    df["{0}trend_vortex_ind_diff".format(colprefix)] = indicator_vortex.vortex_indicator_diff()

    # TRIX Indicator
    df["{0}trend_trix".format(colprefix)] = TRIXIndicator(
        close=df[close], window=15, fillna=fillna
    ).trix()

    # Mass Index
    df["{0}trend_mass_index".format(colprefix)] = MassIndex(
        high=df[high], low=df[low], window_fast=9, window_slow=25, fillna=fillna
    ).mass_index()

    # CCI Indicator
    df["{0}trend_cci".format(colprefix)] = CCIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=20,
        constant=0.015,
        fillna=fillna,
    ).cci()

    # DPO Indicator
    df["{0}trend_dpo".format(colprefix)] = DPOIndicator(
        close=df[close], window=20, fillna=fillna
    ).dpo()

    # KST Indicator
    indicator_kst = KSTIndicator(
        close=df[close],
        roc1=10,
        roc2=15,
        roc3=20,
        roc4=30,
        window1=10,
        window2=10,
        window3=10,
        window4=15,
        nsig=9,
        fillna=fillna,
    )
    df["{0}trend_kst".format(colprefix)] = indicator_kst.kst()
    df["{0}trend_kst_sig".format(colprefix)] = indicator_kst.kst_sig()
    df["{0}trend_kst_diff".format(colprefix)] = indicator_kst.kst_diff()

    # Ichimoku Indicator
    indicator_ichi = IchimokuIndicator(
        high=df[high],
        low=df[low],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=fillna,
    )
    df["{0}trend_ichimoku_conv".format(colprefix)] = indicator_ichi.ichimoku_conversion_line()
    df["{0}trend_ichimoku_base".format(colprefix)] = indicator_ichi.ichimoku_base_line()
    df["{0}trend_ichimoku_a".format(colprefix)] = indicator_ichi.ichimoku_a()
    df["{0}trend_ichimoku_b".format(colprefix)] = indicator_ichi.ichimoku_b()
    indicator_ichi_visual = IchimokuIndicator(
        high=df[high],
        low=df[low],
        window1=9,
        window2=26,
        window3=52,
        visual=True,
        fillna=fillna,
    )
    df["{0}trend_visual_ichimoku_a".format(colprefix)] = indicator_ichi_visual.ichimoku_a()
    df["{0}trend_visual_ichimoku_b".format(colprefix)] = indicator_ichi_visual.ichimoku_b()

    # Aroon Indicator
    indicator_aroon = AroonIndicator(close=df[close], window=25, fillna=fillna)
    df["{0}trend_aroon_up".format(colprefix)] = indicator_aroon.aroon_up()
    df["{0}trend_aroon_down".format(colprefix)] = indicator_aroon.aroon_down()
    df["{0}trend_aroon_ind".format(colprefix)] = indicator_aroon.aroon_indicator()

    # PSAR Indicator
    indicator_psar = PSARIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        step=0.02,
        max_step=0.20,
        fillna=fillna,
    )
    # df[f'{colprefix}trend_psar'] = indicator.psar()
    df["{0}trend_psar_up".format(colprefix)] = indicator_psar.psar_up()
    df["{0}trend_psar_down".format(colprefix)] = indicator_psar.psar_down()
    df["{0}trend_psar_up_indicator".format(colprefix)] = indicator_psar.psar_up_indicator()
    df["{0}trend_psar_down_indicator".format(colprefix)] = indicator_psar.psar_down_indicator()

    # Schaff Trend Cycle (STC)
    df["{0}trend_stc".format(colprefix)] = STCIndicator(
        close=df[close],
        window_slow=50,
        window_fast=23,
        cycle=10,
        smooth1=3,
        smooth2=3,
        fillna=fillna,
    ).stc()

    return df


def add_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """

    # Relative Strength Index (RSI)
    df["{0}momentum_rsi".format(colprefix)] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()

    # Stoch RSI (StochRSI)
    indicator_srsi = StochRSIIndicator(
        close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df["{0}momentum_stoch_rsi".format(colprefix)] = indicator_srsi.stochrsi()
    df["{0}momentum_stoch_rsi_k".format(colprefix)] = indicator_srsi.stochrsi_k()
    df["{0}momentum_stoch_rsi_d".format(colprefix)] = indicator_srsi.stochrsi_d()

    # TSI Indicator
    df["{0}momentum_tsi".format(colprefix)] = TSIIndicator(
        close=df[close], window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df["{0}momentum_uo".format(colprefix)] = UltimateOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window1=7,
        window2=14,
        window3=28,
        weight1=4.0,
        weight2=2.0,
        weight3=1.0,
        fillna=fillna,
    ).ultimate_oscillator()

    # Stoch Indicator
    indicator_so = StochasticOscillator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=14,
        smooth_window=3,
        fillna=fillna,
    )
    df["{0}momentum_stoch".format(colprefix)] = indicator_so.stoch()
    df["{0}momentum_stoch_signal".format(colprefix)] = indicator_so.stoch_signal()

    # Williams R Indicator
    df["{0}momentum_wr".format(colprefix)] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    ).williams_r()

    # Awesome Oscillator
    df["{0}momentum_ao".format(colprefix)] = AwesomeOscillatorIndicator(
        high=df[high], low=df[low], window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # KAMA
    df["{0}momentum_kama".format(colprefix)] = KAMAIndicator(
        close=df[close], window=10, pow1=2, pow2=30, fillna=fillna
    ).kama()

    # Rate Of Change
    df["{0}momentum_roc".format(colprefix)] = ROCIndicator(
        close=df[close], window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator
    indicator_ppo = PercentagePriceOscillator(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df["{0}momentum_ppo".format(colprefix)] = indicator_ppo.ppo()
    df["{0}momentum_ppo_signal".format(colprefix)] = indicator_ppo.ppo_signal()
    df["{0}momentum_ppo_hist".format(colprefix)] = indicator_ppo.ppo_hist()

    # Percentage Volume Oscillator
    indicator_pvo = PercentageVolumeOscillator(
        volume=df[volume], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df["{0}momentum_ppo".format(colprefix)] = indicator_pvo.pvo()
    df["{0}momentum_ppo_signal".format(colprefix)] = indicator_pvo.pvo_signal()
    df["{0}momentum_ppo_hist".format(colprefix)] = indicator_pvo.pvo_hist()

    return df


def add_others_ta(
    df: pd.DataFrame, close: str, fillna: bool = False, colprefix: str = ""
) -> pd.DataFrame:
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    # Daily Return
    df["{0}others_dr".format(colprefix)] = DailyReturnIndicator(
        close=df[close], fillna=fillna
    ).daily_return()

    # Daily Log Return
    df["{0}others_dlr".format(colprefix)] = DailyLogReturnIndicator(
        close=df[close], fillna=fillna
    ).daily_log_return()

    # Cumulative Return
    df["{0}others_cr".format(colprefix)] = CumulativeReturnIndicator(
        close=df[close], fillna=fillna
    ).cumulative_return()

    return df


def add_all_ta_features(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
        colprefix(str): Prefix column names inserted

    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
    )
    df = add_volatility_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    df = add_trend_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    df = add_momentum_ta(
        df=df,
        high=high,
        low=low,
        close=close,
        volume=volume,
        fillna=fillna,
        colprefix=colprefix,
    )
    df = add_others_ta(df=df, close=close, fillna=fillna, colprefix=colprefix)
    return df
