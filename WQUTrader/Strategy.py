"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The Strategy class is what contains the logic
of any strategy. The Strategy class is triggered
by a MarketEvent which notifies the strategy that we have
new data to use. The strategy calculates the new signals
and then pushes a SignalEvent on to the event queue
for the Portfolio to process.
"""

from abc import ABCMeta, abstractmethod
from Events import SignalEvent


class Strategy(object):
    """
    Abstract Base Class of the Strategy
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_signals(self):
        """
        executes strategy logic to create signals for the tickers
        """
        raise NotImplementedError("Should Implement create_signals()")

    def create_signal_event(self, ticker, signal):
        """
        creates a signal event based on the ticker and the signal value
        and pushes the SignalEvent on to the event queue

        Args:
            ticker (string): Ticker we want to be trading
            signal (float): a float value for how much the strategy wants to invest.
                            ie:
                                1.0 means invest 1x whatvever my portfolio will let me invest in this ticker
                                -1.0 means invest -1x whatvever my portfolio will let me invest in this ticker
        """
        self._event_queue.put(SignalEvent(ticker, signal))
