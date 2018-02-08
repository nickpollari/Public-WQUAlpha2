"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

AlphaLab is a module which contains functions which can be
used to create alpha signals. AlphaLab functions are imported
by a strategy and used by the strategy to determine the signal allocation
to a particular ticker.
"""

import pandas as pd


def calc_ewma(price_series, com):
    """
    calculate the exponentional moving average
    of a price series.

    Args:
        price_series (pd.Series): Pandas series of prices
        com (int): center of mass to use for the exponential function

    Returns:
        pd.Series: Exponentionally weighted average
    """

    if not isinstance(price_series, pd.Series):
        price_series = pd.Series(price_series)

    ewma = price_series.ewm(com=com).mean()
    ewma.name = 'ewma_%s' % com
    return ewma


def calc_true_range(ohlc_df):
    """
    calculate the true range of a security ohlc dataframe.
    The ohlc dataframe must contain the following columns;
    'close', 'high', 'low'.

    Args:
        ohlc_df (pd.DataFrame): Pandas dataframe containing open, high, low, close columns

    Returns:
        pd.Series: pandas series containing the True Range value
    """
    tr1 = ohlc_df['high'].divide(ohlc_df['low']).subtract(1.0)
    tr2 = ohlc_df['high'].divide(ohlc_df['close'].shift(1)).subtract(1.0).abs()
    tr3 = ohlc_df['low'].divide(ohlc_df['close'].shift(1)).subtract(1.0).abs()
    ret_val = pd.concat([tr1, tr2, tr3], axis=1).dropna().max(axis=1)
    return ret_val
