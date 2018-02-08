"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The DataHandler module is what contains all of the data
that will be used in the backtest
"""


from abc import ABCMeta, abstractmethod
from collections import defaultdict
import datetime as dt
import pandas as pd
from Events import MarketEvent
from pandas_datareader.google.daily import GoogleDailyReader
from pandas_datareader import data

# because pandas datareader url is set to http://www.google.com/finance/historical
# in the package, it is making use of the most up to date google finance api
# url. The problem is that the new google finance api only returns
# 1 year worth of data. To get around that we will replace the new
# api url with the old one which will work until google
# finally permenently disconnects it.
GoogleDailyReader.url = 'http://finance.google.com/finance/historical'


class DataHandler(object):
    """
    Abstract Base Class which contains and fetches all the data that the backtesting engine
    would utilize

    Attributes:
        continue_running (bool): This is set to False when we are out of data to force the Backtester to end
    """

    __metaclass__ = ABCMeta

    def __new__(cls, *args, **kwargs):
        """
        Factory method for base/subtype creation. Simply creates an
        (new-style class) object instance and sets a base property.
        """
        instance = object.__new__(cls, *args, **kwargs)
        # set DataHandler to continue running to begin with
        instance.continue_running = True

        return instance

    @abstractmethod
    def _load_timeseries_data(self):
        """
        Abstract method for loading data from a source
        """
        raise NotImplementedError("Should Implement _load_timeseries_data()")

    def update_data(self, push_market_event = True):
        """
        The main function which asks the DataHandler to get the latest
        set of bars and push them onto the 'latest_ticker_data' dictionary.

        Args:
            push_market_event (bool, optional): If set to False then doesn't push a MarketEvent onto the queue
        """
        # give a temporary first date incase we stop iterating
        curr_dt = dt.datetime(1970, 1, 1)
        for ticker, ticker_gen in self._ticker_data.items():
            try:
                # get next bar from generator
                next_bar = next(ticker_gen)
                # append latest bar to latest_ticker_data
                self.latest_ticker_data[ticker].append(next_bar)
                # set the new curr_dt value
                curr_dt = max(next_bar[0], curr_dt)
            except StopIteration:
                # If I stopped then I am out of data and the backtest does not need to continue
                self.continue_running = False
                push_market_event = False

        if push_market_event:
            self._event_queue.put(MarketEvent(curr_dt))

    def get_latest_bars(self, ticker, N=1):
        """
        Gets the latest bars that have been appened to
        latest_ticker_data from the 'update_data' function

        Args:
            ticker (string): ticker as a string
            N (int, optional): number of latest bars to get, 0 will get all

        Returns:
            list of tuples: list of tuples of latest bars where [0] is the timestamp and [1] is the bar
        """
        # get the latest bars that I have appened for the ticker
        idx_bars_list_of_tuples = self.latest_ticker_data[ticker][-N:]
        return idx_bars_list_of_tuples

    def get_latest_dataframe(self, ticker, N=1):
        """
        Gets the latest bars that have been appened to
        latest_ticker_data from the 'update_data' function

        Args:
            ticker (string): ticker as a string
            N (int, optional): number of latest bars to get, 0 will get all

        Returns:
            pd.DataFrame: DataFrame of the latest bars
        """
        # get the latest bars
        bars = self.get_latest_bars(ticker, N=N)

        if bool(bars):
            # split into two through reverse zip
            idx, l_series = zip(*bars)
            # define arbitrary function
            _func = lambda x: x.values
            # execute mapping
            vals_list = map(_func, l_series)
            # instantiate a dataframe
            df = pd.DataFrame(index = idx, data = vals_list, columns = l_series[0].index)
        else:
            df = pd.DataFrame()
        return df


class GoogleDataHandler(DataHandler):
    def __init__(self, event_queue, ticker_list, trade_st_dt, trade_ed_dt, load_data_days_offset = 365, **kwargs):
        """
        DataHandler Class which retrieves its data from Google

        Args:
            event_queue (Queue.queue): queue object which stores each event
            ticker_list (list): list of tickers to get data for
            trade_st_dt (dt.datetime): the start date of the backtest
            trade_ed_dt (dt.datetime): the end date of the backtest
            load_data_days_offset (int, optional): how many days prior to trade_st_dt do I need to get data from?
            **kwargs: any additional keyword arguments
        """
        self._event_queue = event_queue
        self._trade_st_dt = trade_st_dt
        self._trade_ed_dt = trade_ed_dt
        self._data_st_dt = trade_st_dt - dt.timedelta(days = load_data_days_offset)
        self._ticker_data = dict()  # stores all the raw data
        self.ticker_list = ticker_list  # stores a list of the tickers
        self.latest_ticker_data = defaultdict(list)  # stores the latest bars

        self._load_timeseries_data()

    def _load_timeseries_data(self):
        """
        Loads timeseries data from Google
        """
        ticker_df_list = list()
        index_union = None
        for ticker in self.ticker_list:
            # get ticker data from Google
            ticker_df = data.DataReader(ticker,
                                        data_source='google',
                                        start = self._data_st_dt,
                                        end = self._trade_ed_dt)
            ticker_df.columns = [str(x).lower() for x in ticker_df.columns]
            if index_union is None:
                index_union = ticker_df.index
            else:
                index_union = index_union.union(ticker_df.index).sort_values()

            ticker_df_list.append((ticker, ticker_df))

        # determine how much of the data needs to be loaded and pushed
        # to latest_ticker_data at the start (because self._data_st_dt < self._trade_st_dt)
        autoload_idx = index_union[index_union < self._trade_st_dt]
        futureload_idx = index_union.difference(autoload_idx)

        for ticker, ticker_df in ticker_df_list:
            ticker_df = ticker_df.reindex(index_union)
            ticker_df['volume'] = ticker_df['volume'].fillna(0)
            ticker_df = ticker_df.ffill().bfill()
            ticker_df['return'] = ticker_df['close'].pct_change()

            autoload_df = ticker_df.loc[autoload_idx, :]
            futureload_df = ticker_df.loc[futureload_idx, :]

            # add the generator for the future data
            self._ticker_data[ticker] = futureload_df.iterrows()
            # load all the data in the autoload_df to the latest_ticker_data
            autoload_gen = autoload_df.iterrows()
            while True:
                try:
                    next_bar = next(autoload_gen)
                    self.latest_ticker_data[ticker].append(next_bar)
                except StopIteration:
                    break
