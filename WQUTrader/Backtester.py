"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The Backtester module is what contains the main loop
for executing a backtest. We have two "While" loops to
run the backtest.

The first "While" loop is responsible for
asking the DataHandler to update the data so that it contains
the latest bars.

The second "while" loop is responsible for pulling down the latest events
from the queue and executing the relevant methods with their respective
classes.

The cycle works as such:
1. DataHandler updates data --> Push MarketEvent on to queue
2. MarketEvent triggers Portfolio to update NAV and market value
3. MarketEvent triggers Strategy to use the latest data and
   create signals. Strategy --> Push SignalEvent on to queue
4. SignalEvent triggers Portfolio to process the signal as it
   relates to a ticker. Convert the signal into an order to buy
   or sell a ticker. Portfolio --> Push OrderEvent on to queue
5. OrderEvent triggers ExecutionHandler to process the order
   and send it out for execution to the broker. Upon receiving
   the fill information the ExecutionHandler creates a FillEvent.
   ExecutionHandler --> Push FillEvent on to queue
6. FillEvent triggers the Portfolio to process the fill. The portfolio
   updates the cash position of the portfolio and the holdings of the
   portfolio. This is also where commissions impact the portfolio.
"""

import datetime as dt
import pandas as pd

# some Python2 vs. Python3 queue importing magic
try:
    import Queue as queue
except:
    import queue


class Backtester(object):
    def __init__(self, start_datetime, end_datetime, ticker_list,
                       Strategy, initial_capital,
                       DataHandlerClass, PortfolioClass,
                       ExecutionHandlerClass, **kwargs):
        """
        The main class for backtesting.
        This is the main engine of the backtesting engine.
        This is responsible for instantiating the DataHandler,
        the Portfolio, and the ExecutionHandler.

        Args:
            start_datetime (dt.datetime): First Date of the backtest
            end_datetime (dt.datetime): Last Date of the backtest
            ticker_list (list): list of tickers to backtest with
            Strategy (Strategy.Strategy): Strategy Class
            initial_capital (int): Initial capital dollars, ie: 100 means $100
            DataHandlerClass (DataHandler.DataHandler.__class__): DataHandler Class not instantiated yet
            PortfolioClass (Portfolio.Portfolio.__class__): Portfolio Class not instantiated yet
            ExecutionHandlerClass (ExecutionHandler.ExecutionHandler.__class__): ExecutionHandler Class not instantiated yet
            **kwargs: Description
        """
        # create a queue to use with everybody
        self._event_queue = queue.Queue()
        # store any and all kwargs to use for instantiated the classes
        self._kwargs = kwargs
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.ticker_list = ticker_list
        self.Strategy = Strategy
        self.initial_capital = initial_capital
        # set these to None until the Backtester instantiates them
        self.DataHandler = None
        self.Portfolio = None
        self.ExecutionHandler = None

        self._init_classes(DataHandlerClass, PortfolioClass, ExecutionHandlerClass)

        self.trades_df = None
        self.nav_df = None
        self.quantity_df = None

    def generate_output(self):
        """
        Generates the following;
        - trades_df (stores all trades made)
        - nav_df (stores all the NAV info)
        - quantity_df (stores a running total of the number of shares held on each day)
        """
        self.trades_df = pd.DataFrame(self.Portfolio.trades)
        self.nav_df = pd.DataFrame(self.Portfolio.nav)

        temp_quantity_df = self.trades_df.pivot(index = 'timestamp',
                                                columns = 'ticker',
                                                values = 'quantity')
        temp_quantity_df.reindex(self.nav_df['date'].tolist())
        self.quantity_df = temp_quantity_df.fillna(0.0).cumsum()

    def _init_classes(self, DataHandlerClass, PortfolioClass, ExecutionHandlerClass):
        """
        Instantiates the classes with their respective inputs

        Args:
            DataHandlerClass (DataHandler.DataHandler.__class__): DataHandler Class not instantiated yet
            PortfolioClass (Portfolio.Portfolio.__class__): Portfolio Class not instantiated yet
            ExecutionHandlerClass (ExecutionHandler.ExecutionHandler.__class__): ExecutionHandler Class not instantiated yet
        """
        # instantiate DataHandler
        self.DataHandler = DataHandlerClass(self._event_queue,
                                            self.ticker_list,
                                            self.start_datetime,
                                            self.end_datetime,
                                            **self._kwargs)
        # instantiate Portfolio
        self.Portfolio = PortfolioClass(self._event_queue,
                                        self.ticker_list,
                                        self.DataHandler,
                                        self.initial_capital,
                                        **self._kwargs)
        # instantiate ExecutionHandler
        self.ExecutionHandler = ExecutionHandlerClass(self._event_queue,
                                                      self.DataHandler)

        # initialize the starting holdings and nav of the portfolio
        self.Portfolio.initialize_nav_holdings(self.start_datetime - dt.timedelta(days=1))
        # assign the event queue to the strategy
        self.Strategy._event_queue = self._event_queue
        # assign the DataHandler to the strategy
        self.Strategy.DataHandler = self.DataHandler

    def begin_backtest(self):
        # main outer loop
        while True:
            # check if the DataHandler still has data to run
            if self.DataHandler.continue_running:
                # update DataHandler Data
                self.DataHandler.update_data()
            else:
                break

            # main inner loop
            while True:
                try:
                    # fetch event from the event queue
                    event = self._event_queue.get(False)
                except queue.Empty:
                    # if queue is empty then nothing left to do on this time interval
                    break
                else:

                    # get the current event type
                    current_event_type = event.get_type()

                    # execute relevant functions depending on event type
                    if current_event_type == 'MARKET':
                        self.Portfolio.process_market_event(event)
                        self.Strategy.create_signals()

                    elif current_event_type == 'SIGNAL':
                        self.Portfolio.process_signal_event(event)

                    elif current_event_type == 'ORDER':
                        self.ExecutionHandler.process_order_event(event)

                    elif current_event_type == 'FILL':
                        self.Portfolio.process_fill_event(event)
