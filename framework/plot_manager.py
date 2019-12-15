
import matplotlib.pyplot as plt


class PlotManager(object):

    def __init__(self, max_intervals, interval_length):
        """

        :param max_intervals:
        :param interval_length:
        """
        self._max_intervals = max_intervals
        self._interval_length = interval_length

        self._train_epoch_score_log = list()
        self._eval_score_log = list()

        self._train_x_values = list()
        self._eval_x_values = list()

        self._ax = plt.subplot(111)

    def add_train_score(self, epoch_idx, score):
        """

        :param epoch_idx:
        :param score:
        :return:
        """
        # store data
        self._train_epoch_score_log.append(score)
        self._train_x_values.append(epoch_idx)

        # trim history
        self._train_epoch_score_log = self._train_epoch_score_log[-(self._max_intervals * self._interval_length):]
        self._train_x_values = self._train_x_values[-(self._max_intervals * self._interval_length):]

    def add_eval_score(self, epoch_idx, score):
        """

        :param epoch_idx:
        :param score:
        :return:
        """
        # store data
        self._eval_score_log.append(score)
        self._eval_x_values.append(epoch_idx)

        # trim history
        self._eval_score_log = self._eval_score_log[-self._max_intervals:]
        self._eval_x_values = self._eval_x_values[-self._max_intervals:]

    def update_plots(self):
        """

        :return:
        """
        # reset ax
        self._ax.cla()

        # plot train values
        if len(self._train_epoch_score_log) > 1:
            self._ax.plot(self._train_x_values, self._train_epoch_score_log, label='train')

        # plot eval values
        if len(self._eval_score_log) > 1:
            self._ax.plot(self._eval_x_values, self._eval_score_log, label='eval')

        # add legend
        self._ax.legend()
