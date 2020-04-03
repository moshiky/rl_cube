import time

import matplotlib.pyplot as plt

from utils.singleton import Singleton


class PlotManager(metaclass=Singleton):
    """
    Manages plots using matplotlib.
    """

    def __init__(self, max_values=500):
        """
        Init global instance.
        """
        self._fig = plt.figure(figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k', constrained_layout=True)
        self._axs_dict = dict()
        self._series_dict = dict()
        self._max_values = max_values

        plt.ion()
        plt.show()

    def add_series_values(self, series_name, x_vals, y_vals):
        """

        :param series_name:
        :param x_vals:
        :param y_vals:
        :return:
        """
        # validate shapes
        assert len(x_vals) == len(y_vals), 'must be same size!'

        # init series dict if needed
        if series_name not in self._series_dict.keys():
            self._series_dict[series_name] = {'x': list(), 'y': list()}

        # store new values
        series_dict = self._series_dict[series_name]
        series_dict['x'] += x_vals
        series_dict['y'] += y_vals

        if len(series_dict['x']) > self._max_values:
            series_dict['x'] = series_dict['x'][-self._max_values:]
            series_dict['y'] = series_dict['y'][-self._max_values:]

        # re-plot current data
        self._plot_stored_series()

    def _plot_stored_series(self):
        """

        :return:
        """
        # get num plots
        num_plots = len(self._series_dict.keys())

        # check whether we need to create new axs or not
        if num_plots > len(self._axs_dict.keys()):

            for ax in self._axs_dict.values():
                self._fig._remove_ax(ax)

            # add axes
            next_id = 1
            for sn in self._series_dict.keys():
                self._axs_dict[sn] = self._fig.add_subplot(num_plots, 1, next_id)
                next_id += 1

        # plot data
        for sn, ax in self._axs_dict.items():
            ax.clear()
            ax.plot(self._series_dict[sn]['x'], self._series_dict[sn]['y'])
            ax.set_title(sn)

        plt.draw()
        plt.pause(0.001)
