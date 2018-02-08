"""
Author - Nick Pollari (nickpollari@gmail.com)
Last Update: 10/21/2017

The PyramidModel classes are used by the strategy
to determine the signal that is passed
to the portfolio for creating position size.

A signal of 1.0 means that the strategy wants to invest
1x whatever the portfolio will let it invest.

A signal of 1.5 means that the strategy wants to invest
1.5x whatever the portfolio will let it invest.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class PyramidModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def scale_signal(self):
        """
        Abstract Method for scaling the signal of the strategy
        """
        raise NotImplementedError("Should Implement scale_signal()")


class StandardPyramidModel(PyramidModel):
    def __init__(self):
        """
        Standard PyramidModel which involes no scaling of the signal
        before passing it on the Portfolio
        """
        self.name = 'STD'
        self.scaler = 1.0

    def scale_signal(self, current_signal):
        """
        Scales the signal of the strategy by some factor

        Args:
            current_signal (float): The current signal of the ticker

        Returns:
            float: A scaled signal value to pass to the portfolio
        """
        return current_signal * self.scaler


class UprightPyramidModel(PyramidModel):
    def __init__(self, max_signal):
        """
        http://www.investopedia.com/articles/trading/09/pyramid-trading.asp

        Upright Pyramid Model to scale the signal.
        As our investment produces positive returns
        we scale up our signal by half of the amount
        that it was previously scaled up by

        Args:
            max_signal (float): A maximum signal so that the strategy cannot request the
                                portfolio to invest more than 'max_signal' times the amount of
                                dollars into this trade as the portfolio would let it
        """
        self.name = 'UPM'
        self.max_signal = max_signal
        # determine all possible signals
        self._possible_signals = self.determine_possible_signals()

    def determine_possible_signals(self):
        """
        This determines all possible signal values so that, given
        the current signal, we can determine easily what the next signal
        will be if we are looking to increase the signal in our trade.

        Returns:
            list: list of all possible signal values
        """
        # set a base signal of 1.0
        curr_signal = 1.0
        all_sigs = [curr_signal]
        # execute while loop to create a list of possible signals
        while True:
            # if the current signal is 1.0 then the difference
            # between the current signal of 1.0 and the previous
            # signal of 0 is a total of 1.0
            if curr_signal == 1.0:
                signal_diff = curr_signal
            else:
                # find out how much we increased the signal last time
                signal_diff = all_sigs[-1] - all_sigs[-2]

            # calculate how much we should add on to our signal
            signal_addon = round(signal_diff / 2.0, 2)
            # calculate the new signal based on how much signal I added
            # on previously and the current signal as compared to what
            # the maximum allowed signal is
            new_signal = min(curr_signal + signal_addon, self.max_signal)
            # append this possible new signal to the all_sigs list
            all_sigs.append(new_signal)
            curr_signal = new_signal
            # if we have reached what the maximum possible signal is
            # then break the while loop
            if new_signal == self.max_signal:
                break
        return all_sigs

    def scale_signal(self, current_signal):
        """
        Scales the signal of the strategy by some factor

        Args:
            current_signal (float): The current signal of the ticker

        Returns:
            float: A scaled signal value to pass to the portfolio
        """
        # get the sign of the current signal
        curr_signal_sign = np.sign(current_signal)
        # if the absolute value of the current signal
        # is less than the maximum signal allowed then
        # lets look to increase the signal along the
        # scale, otherwise dont
        if (abs(current_signal) < self.max_signal):
            # find the index of the current signal among the possible signals
            idx_curr_signal = self._possible_signals.index(abs(round(current_signal, 2)))
            # the new signal is the next signal in line from the list of possible signals
            new_signal = self._possible_signals[idx_curr_signal + 1]
        else:
            new_signal = abs(current_signal)
        # return the new signal adjusted for the appropriate sign on
        return round(new_signal * curr_signal_sign, 2)


class InvertedPyramidModel(PyramidModel):
    def __init__(self, max_signal, max_n_steps = 8):
        """
        http://www.investopedia.com/articles/trading/09/pyramid-trading.asp

        Inverted Pyramid Model to scale the signal.

        As our investment produces positive returns
        we scale up our signal by an equal amount at
        each step until we reach our max_signal

        Args:
            max_signal (float): A maximum signal so that the strategy cannot request the
                                portfolio to invest more than 'max_signal' times the amount of
                                dollars into this trade as the portfolio would let it
            max_n_steps (int, optional): The maximum number of intervals with which to increase our signal
        """
        self.name = 'IPM'
        self.max_signal = max_signal
        self.max_steps = max_n_steps
        # determine all possible signals
        self._possible_signals = self.determine_possible_signals()

    def determine_possible_signals(self):
        """
        This determines all possible signal values so that, given
        the current signal, we can determine easily what the next signal
        will be if we are looking to increase the signal in our trade.

        Returns:
            list: list of all possible signal values
        """
        # set initial current signal
        curr_signal = 1.0
        # find the difference between the maximum signal and the current signal
        diff_max_signal = self.max_signal - curr_signal
        # calculate the incrementations of the signal for each step
        signal_increment = diff_max_signal / self.max_steps
        # perform the signal incrementing
        all_sigs = [round(curr_signal + (n * signal_increment), 4) for n in range(9)]
        return all_sigs

    def scale_signal(self, current_signal):
        """
        Scales the signal of the strategy by some factor

        Args:
            current_signal (float): The current signal of the ticker

        Returns:
            float: A scaled signal value to pass to the portfolio
        """
        # set the signal index from list to increment equal to 0 (no increment)
        signal_idx_inc = 0
        # get the sign of the current signal
        curr_signal_sign = np.sign(current_signal)
        # get the index from the _possible_signals list of the current signal
        idx_curr_signal = self._possible_signals.index(abs(round(current_signal, 4)))
        # if I am not at the last possible signal then set the increment to 1 position
        if (idx_curr_signal < (len(self._possible_signals) - 1)):
            signal_idx_inc = 1
        # get the next signal
        new_signal = self._possible_signals[idx_curr_signal + signal_idx_inc]
        return round(new_signal * curr_signal_sign, 4)


class ReflectingPyramidModel(PyramidModel):
    def __init__(self, max_signal):
        """
        http://www.investopedia.com/articles/trading/09/pyramid-trading.asp

        Reflecting Pyramid Model to scale the signal.

        As our investment produces positive returns
        we scale up our signal by an equal to half
        the amounnt it was increased previously. We do this
        until we reach the 'half-way' point of expected profit
        (where our position is maximized) and then we begin
        to decrement the amount in the signal as we approach
        our 'full profit target'. We only increase our position when the current
        cumulative profits are increasing.

        Args:
            max_signal (float): A maximum signal so that the strategy cannot request the
                                portfolio to invest more than 'max_signal' times the amount of
                                dollars into this trade as the portfolio would let it
        """
        self.name = 'RPM'
        self.max_signal = max_signal
        # determine all the possible signals
        self._possible_signals = self.determine_possible_signals()

    def determine_possible_signals(self):
        """
        This determines all possible signal values so that, given
        the current signal, we can determine easily what the next signal
        will be if we are looking to increase the signal in our trade.

        Returns:
            list: list of all possible signal values
        """
        # set current signal
        curr_signal = 1.0
        all_sigs = [curr_signal]
        while True:
            # if current signal is 1 then the amount it has been incremented by is equal to 1
            if curr_signal == 1.0:
                signal_diff = curr_signal
            else:
                # find the amount the signal has been incremented by
                signal_diff = all_sigs[-1] - all_sigs[-2]
            # add on to the signal value
            signal_addon = round(max(signal_diff / 2.0, 0.15), 4)
            # calculate the new signal
            new_signal = min(curr_signal + signal_addon, self.max_signal)
            all_sigs.append(round(new_signal, 4))
            curr_signal = new_signal
            if new_signal == self.max_signal:
                break
        return all_sigs

    def scale_signal(self, current_signal, cum_profit_up = True):
        """
        Scales the signal of the strategy by some factor

        Args:
            current_signal (float): The current signal of the ticker

        Returns:
            float: A scaled signal value to pass to the portfolio
        """
        # get the current signal sign
        curr_signal_sign = np.sign(current_signal)
        # get the index of the current signal in the list of _possible_signals
        idx_curr_signal = self._possible_signals.index(abs(round(current_signal, 4)))
        # set the index increment to 0
        signal_idx_inc = 0
        # if my cumulative profit has increased and my current signal is less than the maximum signal
        # then set the index increment to 1 for a higher signal
        if (cum_profit_up) & (abs(current_signal) < self.max_signal):
            signal_idx_inc = 1
        # else set it to -1 if my cumulative profit is not increasing and my
        # current signal is greater than 1.0, so I move back one index slot
        elif (not cum_profit_up) & (abs(current_signal) > 1.0):
            signal_idx_inc = -1
        # calculate the new signal
        new_signal = self._possible_signals[idx_curr_signal + signal_idx_inc]
        return round(new_signal * curr_signal_sign, 4)
