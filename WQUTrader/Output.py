import pandas as pd
import datetime as dt
import numpy as np


class Output(object):
    def __init__(self, returns_df, date_freq = 'D', **kwargs):
        self.returns_df = returns_df if isinstance(returns_df, pd.DataFrame) else pd.DataFrame(returns_df)
        self.wealthpaths = self.returns_df.apply(self._calc_wealthpath)

        self._date_freq = str(date_freq).upper()
        self.kwargs = {str(k).lower() : v for (k, v) in kwargs.items()}
        if self._date_freq == 'D':
            self._freq = 252
        elif self._date_freq == 'M':
            self._freq = 12

        self.monthly_wealthpaths = self.returns_df.apply(self._calc_monthly_wealthpath)

    def _calc_annualized_return(self, series):
        avg_daily_return = series.mean()
        ann_return = avg_daily_return * self._freq
        return ann_return

    def _calc_annualized_std_dev(self, series):
        series_std = series.std()
        ann_std = series_std * (np.sqrt(self._freq))
        return ann_std

    def _calc_annualized_downside_dev(self, series):
        downside_std = series[series < 0].std()
        ann_std = downside_std * (np.sqrt(self._freq))
        return ann_std

    def _calc_sortino(self, ann_returns, ann_downside_stds):
        mar = self.kwargs.get('mar', 0.0)
        sortino = ann_returns.subtract(mar).divide(ann_downside_stds)
        return sortino

    def _calc_sharpe(self, ann_returns, ann_stds):
        sharpe = ann_returns.divide(ann_stds)
        return sharpe

    def _calc_hwm(self, wealthpath):
        hwm = wealthpath.expanding().max()
        return hwm

    def _calc_wealthpath(self, series):
        if series.iloc[0] != 0:
            first_dt = series.index[0]
            set_dt = first_dt - dt.timedelta(days = 1)
            series.ix[set_dt] = 0.0
            series = series.sort_index()

        cum_prod = (1.0 + series).cumprod()
        return cum_prod

    def _calc_monthly_wealthpath(self, series):
        first_dt = series.index[0]
        prev_month = first_dt - pd.tseries.offsets.MonthEnd(1)
        series.ix[prev_month] = 0.0
        cum_prod = (1.0 + series.sort_index()).cumprod()

        if self._freq == 252:
            cum_prod = cum_prod.resample('M').last()

        return cum_prod

    def _calc_drawdowns(self, wealthpath):
        hwm = self._calc_hwm(wealthpath)
        drawdowns = wealthpath.divide(hwm).subtract(1.0)
        return drawdowns

    def _calc_lake_ratios(self, hwm, wps):
        lakes = hwm.subtract(wps)
        mountains = hwm.subtract(lakes)
        lake_ratios = lakes.sum() / mountains.sum()
        return lake_ratios

    def _best_month(self, monthly_wp_series):
        rets = monthly_wp_series.pct_change().dropna()
        return rets.max()

    def _worst_month(self, monthly_wp_series):
        rets = monthly_wp_series.pct_change().dropna()
        return rets.min()

    def _profitable_months_pct(self, monthly_wp_series):
        rets = monthly_wp_series.pct_change().dropna()
        pos_months = rets[rets > 0]
        perc_prof_months = float(len(pos_months)) / float(len(rets))
        return perc_prof_months

    def _calc_gain_to_pain_ratio(self, series):
        total_return_series = (1.0 + series).cumprod().subtract(1.0)
        total_return = total_return_series.iloc[-1]

        loss_returns_series = self.__get_loss_returns(series).abs()
        if not loss_returns_series.empty:
            total_loss_return_series = (1.0 + loss_returns_series).cumprod().subtract(1.0)
            total_loss_return = total_loss_return_series.iloc[-1]

            gpr = total_return / total_loss_return
        else:
            gpr = np.nan
        return gpr

    def __get_win_returns(self, series):
        win_returns = series[series >= 0.0]
        return win_returns

    def __get_loss_returns(self, series):
        loss_returns = series[series < 0.0]
        return loss_returns

    def _calc_win_rate(self, series):
        win_returns = self.__get_win_returns(series)
        rate = float(len(win_returns)) / float(len(series))
        return rate

    def _calc_loss_rate(self, series):
        loss_returns = self.__get_loss_returns(series)
        rate = float(len(loss_returns)) / float(len(series))
        return rate

    def _calc_avg_win_return(self, series):
        win_returns = self.__get_win_returns(series)
        avg = win_returns.mean()
        return avg

    def _calc_avg_loss_return(self, series):
        loss_returns = self.__get_loss_returns(series)
        avg = loss_returns.mean()
        return avg

    def _calc_winloss_ratio(self, series):
        wins = self.__get_win_returns(series)
        losses = self.__get_loss_returns(series)
        if len(losses) == 0.0:
            wl_ratio = np.nan
        else:
            wl_ratio = len(wins) / len(losses)
        return wl_ratio

    def _calc_expectancy(self, win_rates, avg_win, loss_rates, avg_loss):
        w_win = win_rates.multiply(avg_win)
        w_loss = loss_rates.multiply(avg_loss)
        exp = w_win.subtract(w_loss)
        return exp

    def generate_output(self):
        hwms = self.wealthpaths.apply(self._calc_hwm)
        lake_ratios = self._calc_lake_ratios(hwms, self.wealthpaths)
        lake_ratios.name = "Lake Ratio"

        drawdowns = self.wealthpaths.apply(self._calc_drawdowns)
        max_dds = drawdowns.min()
        max_dds.name = "Max Drawdown"

        ann_returns = self.returns_df.apply(self._calc_annualized_return)
        ann_returns.name = "Annualized Return"

        ann_stds = self.returns_df.apply(self._calc_annualized_std_dev)
        ann_stds.name = "Annualized Std Dev"

        ann_downside_std = self.returns_df.apply(self._calc_annualized_downside_dev)
        ann_downside_std.name = "Annualized Downside Dev"

        sharpes = self._calc_sharpe(ann_returns, ann_stds)
        sharpes.name = "Sharpe Ratio"

        sortino = self._calc_sortino(ann_returns, ann_downside_std)
        sortino.name = "Sortino Ratio"

        win_rates = self.returns_df.apply(self._calc_win_rate)
        win_rates.name = "Win Rate"

        loss_rates = self.returns_df.apply(self._calc_loss_rate)
        loss_rates.name = "Loss Rate"

        avg_win_returns = self.returns_df.apply(self._calc_avg_win_return)
        avg_win_returns.name = "Avg Win Return"

        avg_loss_returns = self.returns_df.apply(self._calc_avg_loss_return)
        avg_loss_returns.name = "Avg Loss Return"

        win_loss_ratio = self.returns_df.apply(self._calc_winloss_ratio)
        win_loss_ratio.name = "Win Loss Ratio"

        expectancy = self._calc_expectancy(win_rates, avg_win_returns, loss_rates, avg_loss_returns)
        expectancy.name = "Trade Expectancy"

        gpr = self.returns_df.apply(self._calc_gain_to_pain_ratio)
        gpr.name = 'Gain to Pain Ratio'

        best_months = self.monthly_wealthpaths.apply(self._best_month)
        best_months.name = "Best Month Return"

        worst_months = self.monthly_wealthpaths.apply(self._worst_month)
        worst_months.name = "Worst Month Return"

        pct_prof_months = self.monthly_wealthpaths.apply(self._profitable_months_pct)
        pct_prof_months.name = "Percent Profitable Months"

        output_df = pd.concat([lake_ratios, max_dds, ann_returns,
                               ann_stds, sharpes, win_rates,
                               loss_rates, avg_win_returns,
                               avg_loss_returns, expectancy,
                               gpr, ann_downside_std, sortino,
                               best_months, worst_months,
                               pct_prof_months, win_loss_ratio], axis = 1).round(4)

        return output_df.T.sort_index()
