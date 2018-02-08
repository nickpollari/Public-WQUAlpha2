"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The Portfolio Class is the life and blood
of the backtesting engine. Here is where we
store the NAV, current positions, interpret our signals
from the strategy, ask the ExecutionHandler to execute our
orders, and use the Fills from the ExecutionHandler to
book our trades.
"""


from Events import OrderEvent
from collections import defaultdict
import PositionSizers


class Portfolio(object):
    def __init__(self, events_que, ticker_list, DataHandler, initial_capital, PositionSizer = None, **kwargs):
        """
        This is the Portfolio Class which;
        1. Gets triggered by a MarketEvent to update our market values and NAV with the latest prices
        2. Gets triggered by a SignalEvent to convert the strategy signals into orders for the ExecutionHandler
        3. Gets triggered by a FillEvent to update the cash position of the portfolio based on the fills from the ExecutionHandler
        4. Triggers an OrderEvent after processing a SignalEvent
        5. Is responsible for tracking the portfolio NAV
        6. Is responsible for tracking each trade made

        Args:
            events_que (queue.queue): event queue to store events
            ticker_list (list): list of tickers I will be trading
            DataHandler (DataHandler.DataHandler): DataHandler class which stores all the ticker data
            initial_capital (float): initial starting capital for the backtest. 1000.0 means a starting NAV of $1000.0
            PositionSizer (None, optional): PositionSizer.PositionSizer class for position sizing. If left out then defaults to PercentEquity model
            **kwargs: Any and all kwargs which may be needed or used
        """
        self._event_queue = events_que
        self._ticker_list = ticker_list
        self._initial_capital = initial_capital
        self.DataHandler = DataHandler

        # create a default dictionary which holds the current quantity of
        # shares/contracts we own for each ticker
        self.current_positions = defaultdict(int)
        # create a dictionary which holds the current portfolio NAV state
        self.current_portfolio_state = dict()
        # generate an empty list for the trades and nav to store data over time
        self.trades = list()
        self.nav = list()
        # if the PositionSizer is None then we will equally weight our
        # positions
        if PositionSizer is None:
            position_size = 1.0 / len(self._ticker_list)
            PositionSizer = PositionSizers.PercentEquityModel(position_size)

        self.PositionSizer = PositionSizer

    def initialize_nav_holdings(self, first_dt):
        """
        Initializes the nav holdings for the first date

        Args:
            first_dt (dt.datetime): The starting date of the backtest minus 1 day
        """

        nav_dict = {'cash' : self._initial_capital, 'mv' : 0.0,
                    'date' : first_dt, 'nav' : self._initial_capital,
                    'commission' : 0.0, 'slippage' : 0.0,
                    'mv_long' : 0.0, 'mv_short' : 0.0}
        self.current_portfolio_state = dict(nav_dict)
        self.nav.append(dict(nav_dict))

    def get_open_positions(self):
        """
        get all the open positions in current positions

        Returns:
            dict: dictionary of positions like so; ticker : # of shares
        """
        open_positions = {k : v for (k, v) in self.current_positions.items() if v != 0}
        return open_positions

    def process_market_event(self, event):
        """
        Processes a MarketEvent from the DataHandler and
        updates the current NAV and market values of the portfolio

        Args:
            event (Event.MarketEvent): MarketEvent
        """
        # get open positions
        open_positions = self.get_open_positions()
        curr_dt = event.timestamp
        # set the current market value to 0 so we can increment it
        # based on our open positions
        curr_mv = 0.0
        curr_mv_long = 0.0
        curr_mv_short = 0.0

        # iterate through our open positions and update our market value
        # using the latest prices
        for ticker, quantity in open_positions.items():
            # get latest bar and latest price
            latest_bar = self.DataHandler.get_latest_bars(ticker)[-1]
            latest_price = latest_bar[1]['close']
            # set the ticker market value and increment our current market value
            ticker_mv = latest_price * quantity
            curr_mv += ticker_mv
            # if I am long then increment the current market value which is long
            if quantity > 0:
                curr_mv_long += ticker_mv
            # if I am short then increment the current market value which is short
            else:
                curr_mv_short += ticker_mv

        # update the current portfolio state
        self.current_portfolio_state['mv'] = curr_mv
        self.current_portfolio_state['mv_long'] = curr_mv_long
        self.current_portfolio_state['mv_short'] = curr_mv_short
        self.current_portfolio_state['date'] = curr_dt

        # get my current cash amount and my current slippage cost and calculate my new NAV
        curr_cash = self.current_portfolio_state['cash']
        curr_slip = self.current_portfolio_state['slippage']
        self.current_portfolio_state['nav'] = curr_cash - curr_slip + curr_mv
        self.nav.append(dict(self.current_portfolio_state))

    def determine_position_size(self, ohlc_df):
        """
        Utilize the PositionSizer class to determine the position
        size of the ticker in the portfolio for this trade

        Args:
            ohlc_df (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close'

        Returns:
            float: The number of dollars to invest in this trade

        Raises:
            Exception: Raise an Exception if the PositionSizer name is not
                       recognized and thus I cannot calculate the number of dollars to invest for this trade
        """
        pos_sizer_name = self.PositionSizer.name
        current_nav = self.current_portfolio_state['nav']

        if pos_sizer_name == 'PEM':
            pos_size_inputs = tuple([current_nav])
        elif self.PositionSizer.name == 'PVM':
            pos_size_inputs = (ohlc_df, current_nav)
        elif self.PositionSizer.name in ['MMM', 'MTM']:
            pos_size_inputs = (self._initial_capital, current_nav)
        else:
            raise Exception("Position Sizer Name '%s' is not recognized" % pos_sizer_name)
        # calculate the number of dollars to invest in this trade
        dollars_to_invest = self.PositionSizer.determine_position_size(*pos_size_inputs)
        return dollars_to_invest

    def process_signal_event(self, event):
        """
        Processes a SignalEvent about a ticker and
        creates an OrderEvent to trade the ticker

        Args:
            event (Event.OrderEvent): OrderEvent containing the order details
        """
        ticker = event.ticker
        # get the current quantity
        current_quantity = self.current_positions.get(ticker, 0)
        # get the latest prices of the ticker
        latest_data = self.DataHandler.get_latest_dataframe(ticker, 0)
        latest_price = latest_data.iloc[-1]['close']
        # calculate the number of dollars to invest in the trade
        dollars_to_invest = self.determine_position_size(latest_data)
        # calculate the intended quantity of the ticker that we want
        intended_quantity = int((dollars_to_invest * event.signal) / latest_price)
        # take the difference between the intended quantity and the current
        # quantity to determine how much to buy or sell
        trade_quantity = int(intended_quantity - current_quantity)
        if trade_quantity > 0:
            side = 'BUY'
        else:
            side = 'SELL'

        # if I am going to trade at all then create an OrderEvent
        if trade_quantity != 0:
            order_event = OrderEvent(ticker, trade_quantity, side)
            self._event_queue.put(order_event)

    def process_fill_event(self, event):
        """
        Processes a FillEvent about a ticker
        from the ExecutionHandler which contains
        all the details on the fill of an order

        Args:
            event (Event.FillEvent): FillEvent which contains all the fill information of an order
        """
        # determine how my cash is going to be impacted
        # if I am buying then my cash is reduced and if I am selling then
        # my cash is increased
        cash_impact = event.quantity * event.price * -1.0
        # adjust the current portfolio state cash for the cash impact and the commission
        self.current_portfolio_state['cash'] += cash_impact
        self.current_portfolio_state['cash'] -= event.commission
        # adjust the current quantity of this ticker that I am invested in
        self.current_positions[event.ticker] += event.quantity
        # store the trade
        self.trades.append(event.__dict__)
        # adjust the commission and slippage metrics to keep track of the cumulative costs
        self.current_portfolio_state['commission'] += event.commission
        self.current_portfolio_state['slippage'] += event.slippage
