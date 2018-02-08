"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The Position Sizer classes are used by the Portfolio
to determine the position size of the signal that is passed
to the portfolio
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class PositionSizer(object):
    """
    Abstract Base Class of Position Sizers to be used by the Portfolio
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def determine_position_size(self):
        """
        Abstract Method for determining the position size of the trade
        based on the portfolio state and the intended signal
        """
        raise NotImplementedError("Should Implement determine_position_size()")


class PercentEquityModel(PositionSizer):
    def __init__(self, st_equity_risk_pct):
        """
        Standard Position Sizing Model where we would invest
        st_equity_risk_pct (a percentage) percent of our
        current equity into this ticker to begin with

        Args:
            st_equity_risk_pct (float): percentage of equity to risk on the trade. ie: 0.01 (1%% of equity)
        """
        self.name = 'PEM'
        self.st_equity_risk_pct = st_equity_risk_pct

    def determine_position_size(self, current_nav):
        """
        Determine the position size of the trade

        Args:
            current_nav (float): current nav state of the portfolio. ie: 150 ($150 of NAV)

        Returns:
            float: the number of dollars to invest in this trade
        """
        pos_size_dollars = self.st_equity_risk_pct * current_nav
        return pos_size_dollars


class PercentVolatilityModel(PositionSizer):
    def __init__(self, st_equity_risk_pct, target_vol):
        """
        Position Sizing Model where we calculate the realized volatility
        of the asset and compare it to a target realized volatility. We
        invest a percentage of our equity. We start with an allocation
        of st_equity_risk_pct (a percentage) percent of our equity and
        then we scale this up or down based on the ratio of the realized
        volatility compared to the targeted volatility.

        Args:
            st_equity_risk_pct (float): percentage of equity to risk on the trade. ie: 0.01 (1%% of equity)
            target_vol (float): The targeted annualized volatility we want. ie: 0.15 (15%% annualized volatility)
        """
        self.name = 'PVM'
        self.target_vol = target_vol
        self.st_equity_risk_pct = st_equity_risk_pct
        # set a maximum number of days to use when calculating realized volatility
        self._max_size_roll_window = 63
        # set a minimum number of days to use when calculating realized volatility
        self._min_size_roll_window = 21

    def determine_position_size(self, ohlc_df, current_nav):
        """
        Determine the position size of the trade

        Args:
            ohlc_df (pd.DataFrame): DataFrame containing 'open', 'high' 'low', 'close' data of ticker
            current_nav (float): current nav state of the portfolio. ie: 150 ($150 of NAV)

        Returns:
            float: the number of dollars to invest in this trade
        """
        # get the initial position size in dollars
        pos_size_dollars = self.st_equity_risk_pct * current_nav

        # get the shape of the dataframe that was passed
        df_size = ohlc_df.shape[0]
        # determine what half of its size is to use for the volatility window
        half_size = int(float(df_size) / 2.0)
        # use a window size equal to the minimum of the _max_size_roll_window and the maximum of half the size of the
        # dataframe or the _min_size_roll_window. This ensures that our window size is never less than 21 days (1 month)
        # and never greater than 63 days (1 quarter)
        window_size = min(self._max_size_roll_window, max(half_size, self._min_size_roll_window))
        # calculate the realized volatility
        realized_vol = ohlc_df['return'].ewm(com=window_size).std()
        # annualize the realized volatility
        realized_vol = realized_vol.multiply(np.sqrt(252))
        # scale the number of dollars to invest in this trade by the target volatility
        # as compared to the realized volatility
        position_size_scaler = round(self.target_vol / realized_vol.iloc[-1], 6)
        pos_size_dollars *= position_size_scaler
        return pos_size_dollars


class MarketMoneyModel(PositionSizer):
    def __init__(self, st_equity_risk_pct, profit_risk_pct):
        """
        Position Sizing Model where we calculate our position size based on some
        percentage of our initial starting capital and then we increase our
        position size by some number of dollars that represents
        a percentage (profit_risk_pct) of our total profit so far

        Args:
            st_equity_risk_pct (float): percentage of initial equity to risk on the trade. ie: 0.01 (1%% of initial equity)
            profit_risk_pct (float): percentage of total profits to risk on the trade. ie: 0.5 (50%% of total profits)
        """
        self.name = 'MMM'
        self.st_equity_risk_pct = st_equity_risk_pct
        self.profit_risk_pct = profit_risk_pct

    def determine_position_size(self, initial_capital, current_nav):
        """
        Determine the position size of the trade

        Args:
            initial_capital (float): Number of dollars we initially started with. ie: 150 ($100 of initial equity)
            current_nav (float): current nav state of the portfolio. ie: 150 ($150 of NAV)

        Returns:
            float: the number of dollars to invest in this trade
        """
        # set up an initial position size in dollars based on my initial capital
        pos_size_dollars = self.st_equity_risk_pct * initial_capital
        # calculate my current profits
        curr_profits = current_nav - initial_capital
        # if my current profits are positive then lets look to increase our position size
        if curr_profits > 0:
            # increase our position size by the number of dollars equal to
            # our profit * our profit_risk_pct (percentage)
            pos_size_dollars += (self.profit_risk_pct * curr_profits)
        return pos_size_dollars


class MultiTierModel(PositionSizer):
    def __init__(self, st_equity_risk_pct, pct_return_tier_dict):
        """
        Position Sizing Model where we calculate our position size
        based on a set of "threshold levels" of cumulative return for the strategy.
        As our cumulative return exceeds these "threshold levels" we increase
        our position size, otherwise we don't

        Args:
            st_equity_risk_pct (float): percentage of equity to risk on the trade. ie: 0.01 (1%% of equity)
            pct_return_tier_dict (dict): dictionary containing the "threshold levels" of cumulative return and their
                                         corresponding increases in percentage of equity to trade.
                                         ie:
                                         pct_return_tier_dict = {0.03 : 1.1,   # if 0.05 (5 pct) > cumulative return > 0.03 (3 pct) then increase our position size by 10%
                                                                 0.05 : 1.25,  # if 0.1 (10 pct) > cumulative return > 0.05 (5 pct) then increase our position size by 25%
                                                                 0.1 : 1.5,    # if 0.15 (15 pct) > cumulative return > 0.1 (10 pct) then increase our position size by 50%
                                                                 0.15 : 2.0,   # if 0.2 (20 pct) > cumulative return > 0.15 (15 pct) then increase our position size by 100%
                                                                 0.2 : 2.5}    # if cumulative return >= 0.2 (20 pct) then increase our position size by 150%
        """
        self.name = 'MTM'
        self.st_equity_risk_pct = st_equity_risk_pct
        self.pct_return_tier_set = sorted(pct_return_tier_dict.items(), key = lambda x: x[0])

    def determine_position_size(self, initial_capital, current_nav):
        """
        Determine the position size of the trade

        Args:
            initial_capital (float): Number of dollars we initially started with. ie: 150 ($100 of initial equity)
            current_nav (float): current nav state of the portfolio. ie: 150 ($150 of NAV)

        Returns:
            float: the number of dollars to invest in this trade
        """
        pos_size_dollars = self.st_equity_risk_pct * current_nav
        profit_pct = (current_nav / initial_capital) - 1.0
        scalers = [x[1] for x in self.pct_return_tier_set if profit_pct > x[0]]
        if bool(scalers):
            pos_size_dollars *= scalers[-1]
        return pos_size_dollars
