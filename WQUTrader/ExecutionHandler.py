"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The ExecutionHandler is the responsible party
for communicating with the broker about
orders that are being posted and fills
that are being received.
"""


from abc import ABCMeta, abstractmethod
from Events import FillEvent


class ExecutionHandler(object):
    """
    Abstract Base Class for the Execution Handling
    which is done with the broker
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def process_order_event(self):
        """
        Abstract method for processing order events
        """
        raise NotImplementedError("Should Implement process_order_event()")


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, event_queue, DataHandler):
        """
        This is a simulated relationship with a broker for
        executing securities.

        Args:
            event_queue (Queue.queue): queue object which allows for events to be processed
            DataHandler (DataHandler.DataHandler): DataHandler class which contains the latest ticker data
        """
        self._event_queue = event_queue
        self.DataHandler = DataHandler
        # slippage in bps for each trade
        self._slippage = 0.0005
        # commission in $ for unit that is traded
        self._commission = 0.01

    def process_order_event(self, event):
        """
        Processes an OrderEvent from the portfolio and sends the
        order to a simulated broker whereby we receive the fill
        information from and then push a FillEvent on to the
        events queue

        Args:
            event (Event.Event): OrderEvent Class
        """
        ticker = event.ticker
        quantity = event.quantity
        side = event.side

        # get the latest data from the DataHandler
        timestamp, latest_bar = self.DataHandler.get_latest_bars(ticker)[-1]
        latest_price = latest_bar['close']

        # calculate the executed price to determine the slippage amount
        if side == 'BUY':
            price_impact = 1.0 + self._slippage
        elif side == 'SELL':
            price_impact = 1.0 - self._slippage

        # Calculate the fill price based on the latest price
        fill_price = (latest_price * price_impact)
        # calculate the commissions paid
        commission = abs(quantity * self._commission)
        # calculate the slippage impact
        slippage = abs(fill_price - latest_price) * abs(quantity)
        # create a fill event
        fill_event = FillEvent(timestamp, ticker, quantity, fill_price, commission, slippage, side)
        # put the FillEvent on to the queue
        self._event_queue.put(fill_event)
