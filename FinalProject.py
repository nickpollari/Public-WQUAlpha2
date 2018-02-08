import datetime as dt
import pandas as pd
import numpy as np
import WQUTrader.Strategy as Strategy
import WQUTrader.Backtester as Backtester
import WQUTrader.DataHandler as DataHandler
import WQUTrader.Portfolio as Portfolio
import WQUTrader.ExecutionHandler as ExecutionHandler
import WQUTrader.PositionSizers as PositionSizers
import WQUTrader.PyramidModels as PyramidModels
import WQUTrader.Output as Output
import WQUTrader.AlphaLab as AlphaLab
import itertools
import matplotlib.pyplot as plt
import sys


############################
# ####### SETTINGS ####### #
############################

# create all the static input variables
# http://money.cnn.com/data/dow30/
# list of tickers to trade
ticker_list = ['MMM', 'AXP', 'AAPL', 'BA',
               'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
               'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC',
               'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',
               'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT']

# $1,000,000 starting capital
init_capital = 1000000

# create the date related inputs
in_sample_s_dt = dt.datetime(2000, 1, 1)  # In Sample Start
in_sample_e_dt = dt.datetime(2010, 12, 31)  # In Sample End
out_sample_s_dt = dt.datetime(2011, 1, 1)  # Out of Sample Start
out_sample_e_dt = dt.datetime.today()  # Out of Sample End


############################
# ##### SETTINGS END ##### #
############################


class EMA_Strategy(Strategy.Strategy):
    def __init__(self, ticker_list, ema_com, PyramidModel):
        """
        This Strategy executes a trend following system whereby
        we enter into a long position when the current
        price of a ticker is greater than 0.5ATR (lookback equal to ema_com)
        above the EMA price (with lookback equal to ema_com)
        and a short position if the exact opposite is true

        Args:
            ticker_list (list): list of tickers interested in trading
            ema_com (int): center of mass for the EMA calculation
            PyramidModel (PyramidModel.PyramidModel): PyramidModel class to scale signals
        """
        self.ticker_list = ticker_list
        self.ema_com = ema_com
        # set current signals for each ticker to 0
        self.current_signals = dict(zip(ticker_list, np.zeros(len(ticker_list))))
        # create a dictionary to store any entry prices when I enter a position
        self.enter_prices = dict()
        self.PyramidModel = PyramidModel

    def determine_open_position(self, current_price, current_ema, current_atr):
        """
        Determine whether or not I should consider opening a position
        in a ticker

        Args:
            current_price (float): Current price of the ticker
            current_ema (float): Current EMA price of the ticker
            current_atr (float): Current ATR value of the ticker

        Returns:
            float: returns signal for entering into a position
        """
        signal = 0.0
        # calculate half the current ATR value
        current_atr_half = current_atr * 0.5
        # create the cutoff values for entering a long / short position
        enter_long_pr = current_ema * (1.0 + current_atr_half)
        enter_short_pr = current_ema * (1.0 - current_atr_half)
        # if the current price is greater than the enter long cutoff then use signal of 1.0
        if (current_price > enter_long_pr):
            signal = 1.0
        # if the current price is less than the enter short cutoff then use the signal of -1.0
        elif (current_price < enter_short_pr):
            signal = -1.0
        return signal

    def calculate_rebalance_signal(self, current_signal, prev_cum_profit, curr_cum_profit):
        """
        Determine what the new signal for the ticker should be because
        we are rebalancing based on the latest data. This is what
        utilizes the Pyramid Model

        Args:
            current_signal (float): The current signal for the ticker
            prev_cum_profit (float): The amount of cumulative profit I have made on this ticker prior to today
            curr_cum_profit (float): The amount of cumulative profit I have made on this ticker including today

        Returns:
            float: the scaled signal value
        """
        # create tuple of input and bool for scaling
        signal_scale_inputs = tuple([current_signal])
        scale_signal = False

        # if we are doing the reflective pyramid we need to do some calculations
        if self.PyramidModel.name == 'RPM':
            scale_signal = True
            cum_prof_diff = curr_cum_profit - prev_cum_profit
            cum_profit_up = cum_prof_diff > 0
            signal_scale_inputs = tuple([current_signal, cum_profit_up])
        # if we are not doing the reflecting pyramid then we just
        # scale up the signal if our current cumulative profit is positive
        elif curr_cum_profit > 0:
            scale_signal = True

        # if we are going to scale the signal then call the Pyramid Model
        if scale_signal:
            new_signal = self.PyramidModel.scale_signal(*signal_scale_inputs)
        else:
            new_signal = current_signal
        return new_signal

    def create_signals(self):
        """
        This method creates the signals for the tickers in self.ticker_list
        utilizing the latest data from the DataHandler. After calculating
        the latest signals the strategy will create a SignalEvent
        to push that information on to the event queue
        """
        # the number of latest bars to get from the DataHandler
        num_bars_to_fetch = self.ema_com + 10

        for ticker in self.ticker_list:
            try:
                # set bool to False. We don't trade unless we need to
                submit_signal_to_trade = False
                # get the latest ticker data bars and convert to DataFrame
                ticker_df = self.DataHandler.get_latest_dataframe(ticker, num_bars_to_fetch)
                # get the latest price
                current_pr = ticker_df.iloc[-1]['close']
                # calculate the EMA of the ticker prices
                ticker_ema = AlphaLab.calc_ewma(ticker_df['close'], self.ema_com)
                # calculate the True Range of the ticker prices
                ticker_tr = AlphaLab.calc_true_range(ticker_df)
                # calculate the Average True Range of the ticker prices
                ticker_atr = ticker_tr.rolling(self.ema_com).mean()
                # get the current EMA, ATR, and Signal
                current_ema = ticker_ema.iloc[-1]
                current_atr = ticker_atr.iloc[-1]
                current_signal = self.current_signals.get(ticker)
                # if I currently am not invested with this ticker then lets look
                # to see if we should enter a trade
                if current_signal == 0.0:
                    # check to enter the trade
                    new_signal = self.determine_open_position(current_pr,
                                                              current_ema,
                                                              current_atr)
                    # If I am not going to open up a trade then continue to the next ticker
                    if new_signal == 0.0:
                        continue
                    else:
                        # I am going to enter a position then lets record the enter price
                        # and lets set the submit_signal_to_trade bool to True so I will
                        # create a signal event for this ticker with its new signal
                        self.enter_prices[ticker] = current_pr
                        submit_signal_to_trade = True

                else:
                    # if my current signal is != to 0 then I must currently have
                    # a trade on for this ticker.
                    close_trade = False
                    # check if the current price is higher or lower than the current
                    # EMA value
                    curr_pr_higher_lower = current_pr > current_ema
                    # If I am long and the current price is not greater than the EMA
                    # then lets close the position
                    if (current_signal > 0.0) & (not curr_pr_higher_lower):
                        # check to exit the long position
                        close_trade = True
                    # If I am short and the current price is greater than the EMA
                    # then lets close the position
                    elif (current_signal < 0.0) & (curr_pr_higher_lower):
                        # check to exit the short position
                        close_trade = True

                    if close_trade:
                        # If I am closing the position then my new signal
                        # must be set to 0
                        new_signal = 0.0
                        self.enter_prices.pop(ticker)
                    else:
                        # If I am not closing my trade then I need to
                        # rebalance the trade and use the Pyramid Model
                        curr_signal_sign = np.sign(current_signal)
                        enter_price = self.enter_prices.get(ticker)
                        previous_pr = ticker_df.iloc[-2]['close']

                        prev_cum_profit = ((previous_pr / enter_price) - 1.0) * curr_signal_sign
                        curr_cum_profit = ((current_pr / enter_price) - 1.0) * curr_signal_sign
                        # call Pyramid Model for the scaled signal
                        new_signal = self.calculate_rebalance_signal(current_signal,
                                                                     prev_cum_profit,
                                                                     curr_cum_profit)

                    # if my new signal and the current signal are not the same
                    # then we need to rebalance the ticker in the portfolio
                    if new_signal != current_signal:
                        submit_signal_to_trade = True

                # check to see if we should submit this new signal
                if submit_signal_to_trade:
                    self.current_signals[ticker] = new_signal
                    self.create_signal_event(ticker, new_signal)
            except:
                pass


class BuyHold_Strategy(Strategy.Strategy):
    def __init__(self, ticker_list):
        """
        This Strategy executes a simple Buy and Hold
        with Daily Rebalancing

        Args:
            ticker_list (list): list of tickers interested in trading
        """
        self.ticker_list = ticker_list

    def create_signals(self):
        """
        This method creates the signals for the tickers in self.ticker_list
        utilizing the latest data from the DataHandler. After calculating
        the latest signals the strategy will create a SignalEvent
        to push that information on to the event queue
        """

        for ticker in self.ticker_list:
            self.create_signal_event(ticker, 1.0)


def run_ema_backtest(ParameterCombo):
    # extract components of the backtest run
    EMA_CenterOfMass = ParameterCombo['EMA_Lookback']
    PyramidClass, PyramidParams = ParameterCombo['PyramidModels']
    PositionSizerClass, PositionSizerParams = ParameterCombo['PositionSizers']
    SampleStartDt = ParameterCombo['SampleStartDt']
    SampleEndDt = ParameterCombo['SampleEndDt']
    TickerList = ParameterCombo['TickerList']
    InitialCapital = ParameterCombo['InitialCapital']
    # instantiate the PyramidObject and the PositionSizerObject
    PyramidObject = PyramidClass(*PyramidParams)
    PositionSizerObject = PositionSizerClass(*PositionSizerParams)

    # create a name for the backtest
    BacktestName = '%iEMA - %s - %s' % (EMA_CenterOfMass, PyramidObject.name, PositionSizerObject.name)
    # create a generic string for printing the status of the backtest
    backtest_string = "\nEMA Lookback: %i\nPyramid Class: %s\nPositionSizer Class: %s\n" % (EMA_CenterOfMass, PyramidObject.name, PositionSizerObject.name)
    # instantiate the StrategyObject for executing the Strategy Logic
    StrategyObject = EMA_Strategy(TickerList, EMA_CenterOfMass, PyramidObject)
    # instantiate the backtesting engine
    backtester = Backtester.Backtester(SampleStartDt, SampleEndDt, TickerList, StrategyObject,
                                      InitialCapital, DataHandler.GoogleDataHandler,
                                      Portfolio.Portfolio, ExecutionHandler.SimulatedExecutionHandler,
                                      PositionSizer = PositionSizerObject)
    print "%s - Beginning Backtest%s" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), backtest_string)
    sys.stdout.flush()
    # begin the backtest
    backtester.begin_backtest()
    print "%s - Finished Running Backtest%s" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), backtest_string)
    sys.stdout.flush()
    # generate the backtest output
    backtester.generate_output()
    return backtester, BacktestName


def run_buy_hold_backtest(ParameterCombo):
    # extract components of the backtest run
    PositionSizerClass, PositionSizerParams = ParameterCombo['PositionSizers']
    SampleStartDt = ParameterCombo['SampleStartDt']
    SampleEndDt = ParameterCombo['SampleEndDt']
    TickerList = ParameterCombo['TickerList']
    InitialCapital = ParameterCombo['InitialCapital']
    BacktestName = ParameterCombo['BacktestName']
    # instantiate the PositionSizerObject
    PositionSizerObject = PositionSizerClass(*PositionSizerParams)

    # instantiate the StrategyObject for executing the Strategy Logic
    StrategyObject = BuyHold_Strategy(TickerList)
    # instantiate the backtesting engine
    backtester = Backtester.Backtester(SampleStartDt, SampleEndDt, TickerList, StrategyObject,
                                      InitialCapital, DataHandler.GoogleDataHandler,
                                      Portfolio.Portfolio, ExecutionHandler.SimulatedExecutionHandler,
                                      PositionSizer = PositionSizerObject)
    print "%s - Beginning Buy and Hold Backtest - %s" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), BacktestName)
    sys.stdout.flush()
    # begin the backtest
    backtester.begin_backtest()
    print "%s - Finished Running Buy and Hold Backtest - %s" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), BacktestName)
    sys.stdout.flush()
    # generate the backtest output
    backtester.generate_output()
    return backtester, BacktestName


def create_bar_plot(series):
    # create plot
    f_, a_ = plt.subplots()
    # create x axis labels
    x_axis = [str(x) for x in series.index]
    # create y axis values
    y_axis = [round(x, 2) for x in series.tolist()]
    # create x axis ticks
    ind = np.arange(len(x_axis))
    # add to plot
    a_.bar(ind, y_axis)
    # set x ticks and labels and rotate
    a_.set_xticks(ind)
    a_.set_xticklabels(x_axis)
    for label in a_.get_xticklabels():
        label.set_rotation(90)
    # set y label
    a_.set_ylabel(series.name)
    # set x label
    a_.set_xlabel('Strategy Name (EMA Days - PyramidModel - PositionSizer)')
    # set tot;e
    a_.set_title('%s Comparison' % series.name)
    return f_, a_


def create_line_plot(dataframe, ylabel, title):
    # create line plot
    f_, a_ = plt.subplots()
    # plot the columns
    for col in dataframe:
        a_.plot(dataframe.index, dataframe[col].values, label = col)
    # set the y label
    a_.set_ylabel(ylabel)
    # set the title
    a_.set_title(title)
    # position legend outside the image
    box = a_.get_position()
    a_.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    a_.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return f_, a_


def sort_and_score_series(series):
    # get the series values and sort them
    sorted_series = series.sort_values()
    # score the values
    new_series = pd.Series(index = sorted_series.index, data = range(1, len(sorted_series) + 1), name = series.name)
    return new_series


def get_ticker_correlations(backtester_object):
    # extract ticker pair by pair correlations
    returns_df_list = list()
    for ticker in backtester_object.ticker_list:
        # get the ticker data from DataHandler
        ticker_df = backtester_object.DataHandler.get_latest_dataframe(ticker, 0)
        # get the returns
        ticker_returns = ticker_df['return']
        ticker_returns.name = ticker
        returns_df_list.append(ticker_returns)
    # create a DataFrame for the returns
    returns_df = pd.concat(returns_df_list, axis=1)
    # return the correlated returns
    return returns_df.corr()


def find_N_least_correlated_tickers(correlation_df, N):
    # get the combination of tickers
    ticker_pairs = itertools.combinations(correlation_df.index.tolist(), 2)
    # create a dictionary of all the pairs and their values
    correlation_dict = {pair : correlation_df.loc[pair[0], pair[1]] for pair in ticker_pairs}
    # convert the dictionary to a pandas series
    correlation_series = pd.Series(correlation_dict).drop_duplicates().sort_values()
    # select the least correlated tickers
    tickers_to_select = list()
    for ticker_pair in correlation_series.index:
        ticker1, ticker2 = ticker_pair
        if ticker1 not in tickers_to_select:
            tickers_to_select.append(ticker1)

        if len(tickers_to_select) < 4:
            if ticker2 not in tickers_to_select:
                tickers_to_select.append(ticker2)
        else:
            break
    return tickers_to_select


if __name__ == '__main__':
    print "%s - Program Launched" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # create a dictionary which has the MultiTierModel Thresholds
    PositionSizer_MultiTierModel_Thresholds = {0.03 : 1.1, 0.05 : 1.25, 0.1 : 1.5, 0.15 : 2.0, 0.2 : 2.5}
    # create a dictionary which stores all the possible elements of the various model parameter sets
    ParameterSet = {'TickerList' : [ticker_list],
                    'InitialCapital' : [init_capital],
                    'SampleStartDt' : [in_sample_s_dt],
                    'SampleEndDt' : [in_sample_e_dt],
                    'EMA_Lookback' : [45, 21],
                    'PyramidModels' : [(PyramidModels.ReflectingPyramidModel, tuple([2.0])),
                                       (PyramidModels.UprightPyramidModel, tuple([2.0])),
                                       (PyramidModels.InvertedPyramidModel, tuple([2.0]))],
                    'PositionSizers' : [(PositionSizers.PercentVolatilityModel, tuple([0.01, 0.2])),
                                        (PositionSizers.MarketMoneyModel, tuple([0.01, 0.5])),
                                        (PositionSizers.MultiTierModel, tuple([0.01, PositionSizer_MultiTierModel_Thresholds]))]
                    }

    print "%s - Creating All Possible Parameter Combinations" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # create all the combinations of the parameters
    ParameterCombinations = (dict(itertools.izip(ParameterSet, x)) for x in itertools.product(*ParameterSet.itervalues()))

    # create a dictionary to store all the output
    InSampleOutputDict = dict()
    # In Sample Backtesting Object
    InSampleBacktesterObj = None
    print "%s - Beginning In Sample Backtests" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # These could all be done at once through multiprocessing
    # but because Python has different behavior depending on the
    # OS it is running on, I will leave that out for now.
    for ParameterCombo in ParameterCombinations:
        BacktesterObj, BacktestName = run_ema_backtest(ParameterCombo)
        if InSampleBacktesterObj is None:
            InSampleBacktesterObj = BacktesterObj
        InSampleOutputDict[BacktestName] = {'nav_df' : BacktesterObj.nav_df,
                                            'trades_df' : BacktesterObj.trades_df,
                                            'components' : ParameterCombo}

    print "%s - Creating In Sample Wealth Paths" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # Create Wealthpaths
    WealthPathList = list()
    for BacktestName, BacktestOutputDict in InSampleOutputDict.items():
        nav_df = BacktestOutputDict['nav_df'].set_index('date')
        wealthpath = nav_df['nav'].divide(nav_df['nav'].iloc[0])
        wealthpath.name = BacktestName
        WealthPathList.append(wealthpath)
    InSampleWealthPath_Df = pd.concat(WealthPathList, axis=1)

    # Create Wealthpath Plot
    InSampleWealthPathFig, InSampleWealthPathAx = create_line_plot(InSampleWealthPath_Df,
                                                                   'Growth of $1',
                                                                   'In Sample WealthPath Chart Growth of $1')
    InSampleWealthPathFig.savefig('InSample - WealthPaths.png')

    print "%s - Creating In Sample Return Statistics and Charts" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # create insample output stats
    InSampleOutput_Obj = Output.Output(InSampleWealthPath_Df.pct_change())
    InSampleOutput_Df = InSampleOutput_Obj.generate_output()
    InSampleOutput_Df.to_csv('InSampleStats.csv')

    # create all the bar charts
    BarChartColumns = ['Annualized Return', 'Sharpe Ratio',
                       'Sortino Ratio', 'Win Rate',
                       'Trade Expectancy', 'Max Drawdown',
                       'Lake Ratio', 'Win Loss Ratio', 'Gain to Pain Ratio']
    BarChartsList = list()
    for BarChartCol in BarChartColumns:
        BarChartDataset = InSampleOutput_Df.loc[BarChartCol, :]
        if 'Ratio' not in BarChartCol:
            BarChartDataset.name = BarChartCol + ' (%)'
            BarChartDataset = BarChartDataset.multiply(100.0)

        fig, ax = create_bar_plot(BarChartDataset)
        filename = 'InSample - %s.png' % BarChartCol
        fig.tight_layout()
        fig.savefig(filename)
        BarChartsList.append((fig, ax))

    print "%s - Selecting The Best In Sample Model" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # sort and score the in sample strategies based on their metrics
    MetricsToScore = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Trade Expectancy', 'Win Rate']
    ScoredMetrics = list()
    for Metric in MetricsToScore:
        ScoredSeries = sort_and_score_series(InSampleOutput_Df.loc[Metric, :])
        ScoredMetrics.append(ScoredSeries)

    # select the backtest to use for out of sample
    InSampleScoredBacktests = pd.concat(ScoredMetrics, axis=1).sum(axis=1).sort_values()
    BestInSampleBacktestName = InSampleScoredBacktests.index[-1]
    BestInSampleParameters = InSampleOutputDict[BestInSampleBacktestName]['components']
    BestInSampleParameters['SampleStartDt'] = out_sample_s_dt
    BestInSampleParameters['SampleEndDt'] = out_sample_e_dt

    print "%s - Running Out Of Sample with Best In Sample Model" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # run the out of sample backtest
    BestInSampleBacktesterObj, BestInSampleBacktestName = run_ema_backtest(BestInSampleParameters)
    BestInSampleBacktestNav = BestInSampleBacktesterObj.nav_df.set_index('date')['nav']
    BestInSampleBacktestNav.name = BestInSampleBacktestName

    # generate output statistics for out of sample period
    OutSampleOutput_Obj = Output.Output(BestInSampleBacktestNav.pct_change())
    OutSampleOutput_Df = OutSampleOutput_Obj.generate_output()
    OutSampleOutput_Df.to_csv('OutSampleStats.csv')

    # create out of sample wealthpath to plot
    OutSampleWealthPath_Df = pd.DataFrame(BestInSampleBacktestNav)
    OutSampleWealthPath_Df.columns = [BestInSampleBacktestName]
    OutSampleWealthPath_Df = OutSampleWealthPath_Df.divide(OutSampleWealthPath_Df.iloc[0])
    OutSampleWealthPath_Df = pd.DataFrame(OutSampleWealthPath_Df)

    OutSampleWealthPathFig, OutSampleWealthPathAx = create_line_plot(OutSampleWealthPath_Df,
                                                                     'Growth of $1',
                                                                     'Out Sample WealthPath Chart Growth of $1')
    OutSampleWealthPathFig.savefig('OutSample - WealthPaths.png')

    print "%s - Running Full Period Simulations" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # now we get the in sample ticker correlations
    Correlation_Df = get_ticker_correlations(InSampleBacktesterObj)
    # now we find the 10 least correlated tickers
    LeastCorrelatedTickers = find_N_least_correlated_tickers(Correlation_Df, 10)

    # create some new parameter groups for backtesting
    FullPeriodParameters = [{'TickerList' : ['DIA'],
                             'InitialCapital' : init_capital,
                             'SampleStartDt' : in_sample_s_dt,
                             'SampleEndDt' : out_sample_e_dt,
                             'PositionSizers' : (PositionSizers.PercentEquityModel, tuple([1.0])),
                             'BacktestName' : 'Buy & Hold DIA Index'
                             },
                             {'TickerList' : LeastCorrelatedTickers,
                              'InitialCapital' : init_capital,
                              'SampleStartDt' : in_sample_s_dt,
                              'SampleEndDt' : out_sample_e_dt,
                              'PositionSizers' : (PositionSizers.PercentEquityModel, tuple([1.0 / len(LeastCorrelatedTickers)])),
                              'BacktestName' : 'Buy & Hold 10 Least Correlated'
                              }]

    # get full period results for the Buy and Hold Strategies
    FullPeriod = dict()
    for ParameterCombo in FullPeriodParameters:
        BacktesterObj, BacktestName = run_buy_hold_backtest(ParameterCombo)
        FullPeriod[BacktestName] = {'nav_df' : BacktesterObj.nav_df,
                                    'trades_df' : BacktesterObj.trades_df,
                                    'components' : None}

    # get full period results for the Best In Sample Strategy
    BestInSampleParameters['SampleStartDt'] = in_sample_s_dt
    FullPeriodBacktesterObj, BacktestName = run_ema_backtest(BestInSampleParameters)
    FullPeriod[BacktestName] = {'nav_df' : FullPeriodBacktesterObj.nav_df,
                                'trades_df' : FullPeriodBacktesterObj.trades_df,
                                'components' : None}

    print "%s - Creating Full Period Wealth Paths and Return Stats" % (dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # Create Full Period Wealthpaths
    FullPeriodWealthPathList = list()
    for BacktestName, BacktestOutputDict in FullPeriod.items():
        nav_df = BacktestOutputDict['nav_df'].set_index('date')
        wealthpath = nav_df['nav'].divide(nav_df['nav'].iloc[0])
        wealthpath.name = BacktestName
        FullPeriodWealthPathList.append(wealthpath)

    FullPeriodWealthPath_Df = pd.concat(FullPeriodWealthPathList, axis=1)
    # Create Wealthpath Plot
    FullPeriodWealthPathFig, FullPeriodWealthPathAx = create_line_plot(FullPeriodWealthPath_Df,
                                                                       'Growth of $1',
                                                                       'Full Period WealthPath Chart Growth of $1')
    FullPeriodWealthPathFig.savefig('FullPeriod - WealthPaths.png')

    # generate output statistics for full sample period
    FullPeriodOutput_Obj = Output.Output(FullPeriodWealthPath_Df.pct_change())
    FullPeriodOutput_Df = FullPeriodOutput_Obj.generate_output()
    FullPeriodOutput_Df.to_csv('FullPeriodStats.csv')
    print "Finished Running"
