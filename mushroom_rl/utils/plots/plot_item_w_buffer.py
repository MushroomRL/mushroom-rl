import random
from itertools import product

import numpy as np
from pyqtgraph import PlotItem, PlotDataItem, mkPen, mkColor, mkQApp


class PlotItemWBuffer(PlotItem):
    """
    Class that represents a plot of any type of variable.
    """

    def __init__(self, name, plot_item_data_params, *args, **kwargs):
        """
        Constructor.

        Args:
            name (str) : Name of plot. Also show in the title section of the plot;
            plot_item_data_params (list) : Dictionary of DataBuffers and respective parameters;
            *args (list) : Positional arguments of PlotItem;
            **kwargs (dict) : Kwargs of PlotItem.

        """
        mkQApp()
        self.data_buffers = list()
        self._plot_drawing_params = list()
        self._curves_names = []
        self._refresh_activated = True

        self.name = name

        # If there is only one DataBuffer then the color of the curve is red by default
        if len(plot_item_data_params) == 1 and ("pen" not in plot_item_data_params[0]):
            plot_item_data_params[0]["pen"] = mkPen(color='r')

        # For more than one curve, distinct colors for each curve is chosen
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

        # Creation of the curves
        for single_params in plot_item_data_params:
            self.data_buffers.append(single_params["data_buffer"])
            self._curves_names.append(single_params["data_buffer"].name)
            single_params.pop("data_buffer", None)
            self._plot_drawing_params.append(single_params)

        kwargs['title'] = self.name

        super().__init__(*args, **kwargs)

        if not isinstance(self.data_buffers, list):
            self.data_buffers = [self.data_buffers]

        self.PlotDataItemsList = list()
        if self._plot_drawing_params is not None:
            for data_buffer, plot_drawing_param in zip(self.data_buffers, self._plot_drawing_params):
                self.PlotDataItemsList.append(PlotDataItem(data_buffer.get(), **plot_drawing_param))
        else:
            for data_buffer, plot_drawing_param in zip(self.data_buffers, self._plot_drawing_params):
                self.PlotDataItemsList.append(PlotDataItem(data_buffer.get(), **plot_drawing_param))

        # Legend added based on the name of the buffer
        if self.len_curves > 1:
            self.addLegend()

    def refresh(self):
        """
        Method to refresh buffer only if activated in Window.
        """
        if self._refresh_activated:
            for curve, data_buffer in zip(self.PlotDataItemsList, self.data_buffers):
                curve.setData(data_buffer.get())

    def draw(self, item):
        """
        Draw curve.

        Args:
            item (PlotDataItem) : Curve Item associated with a DataBuffer to be drawn,
                in case it is not drawn .
        """
        if item not in self.listDataItems():
            self.addItem(item)

    def erase(self, item: PlotDataItem):
        """
        Erase curve in case it is drawn.

        Args:
             item (PlotDataItem) : Curve Item associated with a DataBuffer to be erased.
        """
        index_item = self.PlotDataItemsList.index(item)

        try:
            # Delete Legend
            self.legend.removeItem(self._curves_names[index_item])
        except:
            pass

        if item in self.listDataItems():
            # Erase Curve
            self.removeItem(item)

    def refresh_state(self, state):
        """
        Setter that determines whether this plot should be refreshed.

        Args:
            state (bool) : State to which the flag should change to.
        """
        self._refresh_activated = state

    @property
    def curves_names(self):
        """
        Get names of each curve.

        Returns:
             List of curves names.
        """
        return self._curves_names

    @property
    def len_curves(self):
        """
        Get number of curves in plot.

        Returns:
            Integer that represent the number of curves.
        """
        return len(self.PlotDataItemsList)

    @staticmethod
    def get_n_colors(number_of_colors, min=50, max=256):
        """

        Method that gets the n very distinct colors. The color vector is a 3D vector.
        To calculate the different colors it is considered that the volume of 3D space, each
        component between 0 and 255, is divided into n parts, and the coordinates of each
        center of the homogeneus parts is the value of the color.

        Args:
            number_of_colors (int) : Number of colors to get;
            min (int, 50) : Minimum value of each component of the color;
            max (int, 256) : Maximum value of each component of the color;

        Returns:
            List of 3D tuples that represent each color.
        """
        total_area = (max - min) ** 3
        area_per_color = total_area / number_of_colors
        side_of_area_per_color = area_per_color ** (1 / 3)
        onedpoints = np.arange(min, max, int(side_of_area_per_color))

        # In case there are more colors made than needed due to odd numbers
        colors = random.sample(list(product(onedpoints, repeat=3)), number_of_colors)

        return colors


class _Limited(PlotItemWBuffer):
    """
    This class represents the plots that need horizontal lines to specify the limits of the variables.
    """

    def __init__(self, title, plot_buffers, maxs=None, mins=None):
        """
        Contructor.

        Args:
            title (str) : Name of the plot. Also is show in the title section by default;
            plot_buffers ([DataBuffer]) : List of DataBuffers for each curve;
            maxs (list, None) : List of maximums values to draw horizontal lines;
            mins (list, None) : List of minimums values to draw horizontal lines.

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
                "Size of {} parameter(s) doesnt correspond to the number of plot_buffers".format(error_msg))

        for i in plot_buffers:
            curves_params.append(dict(data_buffer=i, name=i.name))

        super().__init__(title, curves_params)
        self._maxs_exist = False
        self._mins_exist = False

        if isinstance(maxs, list):
            self._maxs_exist = bool(maxs)
        elif isinstance(maxs, np.ndarray):
            self._maxs_exist = maxs.any()
        if isinstance(mins, list):
            self._mins_exist = bool(mins)
        elif isinstance(mins, np.ndarray):
            self._mins_exist = mins.any()

        self._maxs_vals = maxs
        self._mins_vals = mins

        self._maxlineitems = []
        self._minlineitems = []

        if self._maxs_vals is not None:
            for i in range(len(self._maxs_vals)):
                if self._maxs_vals[i] == np.inf or self._maxs_vals[i] == -np.inf:
                    self._maxs_vals[i] = None
                self._maxlineitems.append(None)
        if self._mins_vals is not None:
            for i in range(len(self._mins_vals)):
                if self._mins_vals[i] == np.inf or self._mins_vals[i] == -np.inf:
                    self._mins_vals[i] = None
                self._minlineitems.append(None)

    def draw(self, item):
        """
        Method to draw plot and respective horizontal lines from the limits of the variables.

        Args:
            item (PlotDataItem) : PlotDataItem that represents the curve to draw.
        """
        super().draw(item)
        i_item = self.PlotDataItemsList.index(item)

        if self._maxs_exist:
            if self._maxs_vals[i_item] != None:
                self._maxlineitems[i_item] = self.addLine(y=self._maxs_vals[i_item], pen=item.opts['pen'])

        if self._mins_exist:
            if self._mins_vals[i_item] != None:
                self._minlineitems[i_item] = self.addLine(y=self._mins_vals[i_item], pen=item.opts['pen'])

    def erase(self, item):
        """
        Method to erase plot and respective horizontal lines from the limits of the variables.

        Args:
            item (PlotDataItem) : PlotDataItem that represents the curve to erase.
        """
        super().erase(item)
        i_item = self.PlotDataItemsList.index(item)

        if self._maxs_exist:
            self.removeItem(self._maxlineitems[i_item])

        if self._mins_exist:
            self.removeItem(self._minlineitems[i_item])
