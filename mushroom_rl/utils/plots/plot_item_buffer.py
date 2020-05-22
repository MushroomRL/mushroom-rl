import random
from itertools import product

import numpy as np
from PyQt5.QtGui import QPen
from pyqtgraph import PlotItem, PlotDataItem, mkPen, mkColor, mkQApp


class PlotItemBuffer(PlotItem):
    """
    Class that represents a plot of any type of variable stored in a data
    buffer.

    """
    def __init__(self, name, plot_item_data_params, *args, **kwargs):
        """
        Constructor.

        Args:
            name (str): name of plot. Also show in the title section of the
                plot;
            plot_item_data_params (list): dictionary of ``DataBuffer`` and
                respective parameters;
            *args: positional arguments to be passed to ``PlotItem``
                class;
            **kwargs: dictionary to be passed to ``PlotItem`` class.

        """
        mkQApp()
        self.data_buffers = list()
        self._plot_drawing_params = list()
        self._curves_names = []
        self._refresh_activated = True

        self.name = name

        if len(plot_item_data_params) == 1 and ("pen" not in plot_item_data_params[0]):
            plot_item_data_params[0]["pen"] = mkPen(color='r')

        if len(plot_item_data_params) > 1:
            for single_params in plot_item_data_params:
                colors = self.get_n_colors(len(plot_item_data_params))
                if "pen" not in single_params:
                    color = colors[plot_item_data_params.index(single_params)]
                    color = list(color)
                    color.append(255)
                    single_params["pen"] = mkPen(color=mkColor(tuple(color)))
                if "name" not in single_params:
                    single_params["name"] = single_params["data_buffer"].name

        for single_params in plot_item_data_params:
            self.data_buffers.append(single_params["data_buffer"])
            self._curves_names.append(single_params["data_buffer"].name)
            single_params.pop("data_buffer", None)
            self._plot_drawing_params.append(single_params)

        kwargs['title'] = self.name

        super().__init__(*args, **kwargs)

        if not isinstance(self.data_buffers, list):
            self.data_buffers = [self.data_buffers]

        self.plot_data_items_list = list()
        if self._plot_drawing_params is not None:
            for data_buffer, plot_drawing_param in zip(
                    self.data_buffers, self._plot_drawing_params):
                self.plot_data_items_list.append(
                    PlotDataItem(data_buffer.get(), **plot_drawing_param)
                )
        else:
            for data_buffer, plot_drawing_param in zip(
                    self.data_buffers, self._plot_drawing_params):
                self.plot_data_items_list.append(
                    PlotDataItem(data_buffer.get(), **plot_drawing_param)
                )

        if self.len_curves > 1:
            self.addLegend()

    def refresh(self):
        """
        Refresh buffer only if activated in window.

        """
        if self._refresh_activated:
            for curve, data_buffer in zip(
                    self.plot_data_items_list, self.data_buffers):
                curve.setData(data_buffer.get())

    def draw(self, item):
        """
        Draw curve.

        Args:
            item (PlotDataItem): curve item associated with a DataBuffer to be
                drawn.

        """
        if item not in self.listDataItems():
            self.addItem(item)

    def erase(self, item):
        """
        Erase curve in case it is drawn.

        Args:
             item (PlotDataItem): curve item associated with a data buffer to be
                erased.

        """
        index_item = self.plot_data_items_list.index(item)

        try:
            self.legend.removeItem(self._curves_names[index_item])
        except:
            pass

        if item in self.listDataItems():
            self.removeItem(item)

    def refresh_state(self, state):
        """
        Setter.

        Args:
            state (bool): whether to refresh state or not.

        """
        self._refresh_activated = state

    @property
    def curves_names(self):
        """
        Returns:
             List of curves names.

        """
        return self._curves_names

    @property
    def len_curves(self):
        """
        Returns:
            The number of curves.

        """
        return len(self.plot_data_items_list)

    @staticmethod
    def get_n_colors(number_of_colors, min_value=50, max_value=256):
        """
        Get n very distinct colors. The color vector is a 3D vector.
        To calculate the different colors, it is considered that the volume of
        RGB space is divided into n parts, and the coordinates of each center is
        the value of the color.

        Args:
            number_of_colors (int): number of colors to get;
            min_value (int, 50): minimum value of each component of the color;
            max_value (int, 256): maximum value of each component of the color.

        Returns:
            List of RGB tuples that represent each color.

        """
        total_area = (max_value - min_value) ** 3
        area_per_color = total_area / number_of_colors
        side_of_area_per_color = int(area_per_color ** (1 / 3))
        onedpoints = np.arange(min_value, max_value, side_of_area_per_color)

        colors = random.sample(list(product(onedpoints, repeat=3)),
                               number_of_colors)

        return colors


class PlotItemBufferLimited(PlotItemBuffer):
    """
    This class represents the plots with limits on the variables.

    """
    def __init__(self, title, plot_buffers, maxs=None, mins=None,
                 dotted_limits=None):
        """
        Contructor.

        Args:
            title (str): name of the plot. Also is show in the title section by
                default;
            plot_buffers (DataBuffer): List of DataBuffers for each curve;
            maxs (list, None): list of maximums values to draw horizontal
                lines;
            mins (list, None): list of minimum values to draw horizontal
                lines;
            dotted_limits (list, None): list of booleans. If True, the
                corresponding limit is dotted; otherwise, it is printed as a
                solid line.

        """
        curves_params = []

        error_msg = ""
        if maxs is not None:
            if len(maxs) != len(plot_buffers):
                error_msg += "maxs"
        if mins is not None:
            if len(mins) != len(plot_buffers):
                error_msg += " and mins"

        if error_msg != "":
            raise TypeError(
                "Size of {} parameter(s) doesnt correspond to the number of"
                "plot_buffers".format(error_msg)
            )

        for i in plot_buffers:
            curves_params.append(dict(data_buffer=i, name=i.name))

        super().__init__(title, curves_params)
        self._maxs_exist = False
        self._mins_exist = False

        if isinstance(maxs, list):
            self._maxs_exist = bool(maxs) or bool(dotted_limits)
        if isinstance(mins, list):
            self._mins_exist = bool(mins) or bool(dotted_limits)

        self._maxs_vals = maxs
        self._mins_vals = mins

        if dotted_limits is not None:
            for i in range(len(dotted_limits)):
                if dotted_limits[i]:
                    self._maxs_vals[i] = 1
                    self._mins_vals[i] = -1

        self._max_line_items = []
        self._min_line_items = []

        if self._maxs_vals is not None:
            for i in range(len(self._maxs_vals)):
                if self._maxs_vals[i] == np.inf or self._maxs_vals[i] == -np.inf:
                    self._maxs_vals[i] = None
                self._max_line_items.append(None)
        if self._mins_vals is not None:
            for i in range(len(self._mins_vals)):
                if self._mins_vals[i] == np.inf or self._mins_vals[i] == -np.inf:
                    self._mins_vals[i] = None
                self._min_line_items.append(None)

        self._dotted_limits = dotted_limits

    def draw(self, item):
        super().draw(item)
        i_item = self.plot_data_items_list.index(item)

        pen = item.opts['pen']

        if self._dotted_limits is not None:
            if self._dotted_limits[i_item]:
                pen = QPen(pen)
                pen.setDashPattern([6, 12, 6, 12])

        if self._maxs_exist:
            if self._maxs_vals[i_item] is not None:
                self._max_line_items[i_item] = self.addLine(
                    y=self._maxs_vals[i_item], pen=pen
                )

        if self._mins_exist:
            if self._mins_vals[i_item] is not None:
                self._min_line_items[i_item] = self.addLine(
                    y=self._mins_vals[i_item], pen=pen
                )

    def erase(self, item):
        super().erase(item)
        i_item = self.plot_data_items_list.index(item)

        if self._maxs_exist:
            self.removeItem(self._max_line_items[i_item])

        if self._mins_exist:
            self.removeItem(self._min_line_items[i_item])
