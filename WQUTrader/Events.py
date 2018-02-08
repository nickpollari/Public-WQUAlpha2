"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The Events are the individual events that current
during any 1 time-interval (such as 1 day). These
events with their specific type are what force the
backtester to call certain functions on certain
classes.
"""


from abc import ABCMeta


class Event(object):
    """
    Abstract Base Class of an Event with a default function
    to get the event type
    """

    __metaclass__ = ABCMeta

    def get_type(self):
        """
        Gets the event type

        Returns:
            string: name of the event type
        """
        return self.event_type


class MarketEvent(Event):
    def __init__(self, timestamp):
        """
        MarketEvent is triggered at the beginning of each new time interval
        by the DataHandler when it updates its data from the datasource
        to signal to the portfolio to update its NAV and to the strategy
        to calculate new signals based on the latest data.

        Args:
            timestamp (dt.datetime): the dt.datetime of the current time interval in the backtest
        """
        self.event_type = 'MARKET'
        self.timestamp = timestamp


class OrderEvent(Event):
    def __init__(self, ticker, quantity, side):
        """
        OrderEvent is triggered by the portfolio after receiving a signal on a ticker
        and contains the information on the order to be passed to the execution handler.

        Args:
            ticker (string): Ticker to trade
            quantity (int): the number of shares/contracts to trade
            side (str): orders are either to 'BUY' if quantity > 0 or 'SELL' if not
        """
        self.event_type = 'ORDER'
        self.ticker = ticker
        self.quantity = quantity
        self.side = side


class SignalEvent(Event):
    def __init__(self, ticker, signal):
        """
        SignalEvent is triggered by the strategy after it has processed
        a MarketEvent and used the latest data available to create new signals
        for the tickers we are trading. SignalEvent triggers the portfolio
        to convert the signal into an order.

        Args:
            ticker (string): Ticker the signal applies to
            signal (float): float representing whether we want to be long (1.0), short (-1.0), nothing (0.0) or
                            anywhere inbetween as a percentage of equity prior to position-sizing decisions
                            made at the protfolio level
        """
        self.event_type = 'SIGNAL'
        self.ticker = ticker
        self.signal = signal


class FillEvent(Event):
    def __init__(self, timestamp, ticker, quantity, price, commission, slippage, side):
        """
        The FillEvent is triggered by the ExecutionHandler after it has processed an order,
        sent it to a broker, and recieved the fill information on that order.

        Args:
            timestamp (dt.datetime): dt.datetime of the fill
            ticker (string): the ticker which was traded
            quantity (int): the number of shares traded
            price (float): the price the trade was executed at
            commission (float): the commissions paid on this order
            slippage (float): the slippage incurred on this order
            side (string): orders are either to 'BUY' if quantity > 0 or 'SELL' if not
        """
        self.event_type = 'FILL'
        self.timestamp = timestamp
        self.ticker = ticker

        if side == 'BUY':
            self.quantity = int(abs(quantity))
        elif side == 'SELL':
            self.quantity = int(-1.0 * abs(quantity))

        self.price = price
        self.commission = commission
        self.slippage = slippage
        self.side = side
