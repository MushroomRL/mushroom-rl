import math

import numpy as np
from pyqtgraph.Qt import QtCore
from pyqtgraph import PlotItem, PlotDataItem, InfiniteLine, LabelItem, LegendItem, mkPen, mkColor, mkQApp


class PlotItemBuffer(PlotItem):
    """
    Class that represents a plot of any type of variable stored in a data
    buffer.

    """
    PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
               '#bcbd22', '#17becf']
    LINE_STYLES = [QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine,
                   QtCore.Qt.PenStyle.DashDotLine]
    _TEXT_COLOR = '#333333'
    _AXIS_COLOR = '#999999'
    _LEGEND_TEXT_SIZE = '9pt'

    def __init__(self, name, plot_item_data_params, *args, **kwargs):
        """
        Constructor.

        Args:
            name (str): name of plot. Also shown in the title section of the
                plot;
            plot_item_data_params (list): list of dictionaries, one per curve,
                each holding the ``DataBuffer`` under ``data_buffer`` and the
                ``PlotDataItem`` parameters of the curve. Curves without a
                ``pen`` take the one given by ``default_pen``. The dictionaries
                are not modified;
            *args: positional arguments to be passed to ``PlotItem``
                class;
            **kwargs: dictionary to be passed to ``PlotItem`` class.

        """
        mkQApp()

        self.name = name
        self.data_buffers = [params['data_buffer'] for params in plot_item_data_params]
        self._curves_names = [data_buffer.name for data_buffer in self.data_buffers]
        self._refresh_activated = True

        super().__init__(*args, **kwargs)
        self._apply_style()

        self.plot_data_items_list = [PlotDataItem(data_buffer.get(), **drawing_params) for data_buffer, drawing_params
                                     in zip(self.data_buffers, self._drawing_params(plot_item_data_params))]

        if self.len_curves > 1:
            # a row of its own below the bottom axis, so that the legend does not cover the curves
            self.legend = LegendItem(labelTextColor=self._TEXT_COLOR, labelTextSize=self._LEGEND_TEXT_SIZE)
            self.layout.addItem(self.legend, 4, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            self.layout.setRowMinimumHeight(4, LabelItem('', size=self._LEGEND_TEXT_SIZE).boundingRect().height())

    def refresh(self):
        """
        Refresh buffer only if activated in window.

        """
        if self._refresh_activated:
            for curve, data_buffer in zip(self.plot_data_items_list, self.data_buffers):
                curve.setData(data_buffer.get())
            self.recomputeAverages()

    def draw(self, item):
        """
        Draw curve, applying the transformations currently selected in the
        plot options menu.

        Args:
            item (PlotDataItem): curve item associated with a DataBuffer to be
                drawn.

        """
        if item not in self.listDataItems():
            self.addItem(item)
            item.setDerivativeMode(self.ctrl.derivativeCheck.isChecked())
            item.setPhasemapMode(self.ctrl.phasemapCheck.isChecked())
            self._update_legend_layout()

    def erase(self, item):
        """
        Erase curve in case it is drawn.

        Args:
             item (PlotDataItem): curve item associated with a data buffer to be
                erased.

        """
        self.removeItem(item)
        self._update_legend_layout()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_legend_layout()

    def refresh_state(self, state):
        """
        Setter.

        Args:
            state (bool): whether to refresh state or not.

        """
        self._refresh_activated = state

    @classmethod
    def default_pen(cls, index):
        """
        Pen of the curve in the given position. The curves take the colors of
        ``PALETTE`` in order, and move to the next style of ``LINE_STYLES``
        every time the colors are used up.

        Args:
            index (int): position of the curve in the plot.

        Returns:
            The pen of the curve.

        """
        color = cls.PALETTE[index % len(cls.PALETTE)]
        style = cls.LINE_STYLES[(index // len(cls.PALETTE)) % len(cls.LINE_STYLES)]
        return mkPen(color=color, width=1, style=style)

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

    def _drawing_params(self, plot_item_data_params):
        drawing_params = [{key: value for key, value in params.items() if key != 'data_buffer'}
                          for params in plot_item_data_params]

        for i, (params, data_buffer) in enumerate(zip(drawing_params, self.data_buffers)):
            params.setdefault('pen', self.default_pen(i))
            params.setdefault('antialias', True)
            params.setdefault('name', data_buffer.name)

        return drawing_params

    def _update_legend_layout(self):
        if self.legend is not None:
            entries = self.legend.items
            spacing = self.legend.layout.horizontalSpacing()
            entry_width = max((sample.width() + label.width() + 2 * spacing for sample, label in entries), default=1)
            n_columns = max(1, min(len(entries), int(self.vb.width() // entry_width)))

            # force the legend to lay its entries out again, also when only the entries changed
            self.legend.columnCount = 0
            self.legend.setColumnCount(n_columns)

            n_rows = math.ceil(len(entries) / n_columns)
            row_height = max((label.height() for _, label in entries), default=0)
            size = QtCore.QSizeF(n_columns * entry_width, n_rows * row_height)
            self.legend.setMinimumSize(size)
            self.legend.setMaximumSize(size)

            # the legend moves itself to the origin when its entries change, the layout puts it back in its row
            self.layout.invalidate()
            self.layout.activate()

    def _apply_style(self):
        self.setTitle(self.name, color=self._TEXT_COLOR, size='11pt', bold=True)
        self.showGrid(x=True, y=True, alpha=0.15)
        for axis_name in ('left', 'bottom'):
            axis = self.getAxis(axis_name)
            axis.setPen(self._AXIS_COLOR)
            axis.setTextPen(self._TEXT_COLOR)
            if axis.labelText:
                axis.setLabel(axis.labelText, color=self._TEXT_COLOR)


class PlotItemBufferLimited(PlotItemBuffer):
    """
    This class represents the plots with limits on the variables.

    """
    def __init__(self, title, plot_buffers, maxs=None, mins=None,
                 dotted_limits=None, **kwargs):
        """
        Constructor.

        Args:
            title (str): name of the plot. Also shown in the title section by
                default;
            plot_buffers (list): list of DataBuffers, one for each curve;
            maxs (list, None): list of maximum values to draw horizontal
                lines. An element that is None or infinite draws no line;
            mins (list, None): list of minimum values to draw horizontal
                lines. An element that is None or infinite draws no line;
            dotted_limits (list, None): list of booleans. If True, the
                corresponding limits are drawn as dotted lines at 1 and -1,
                in place of the values in ``maxs`` and ``mins``;
            **kwargs: dictionary to be passed to ``PlotItem`` class.

        Raises:
            ValueError: if ``maxs``, ``mins`` or ``dotted_limits`` does not have one element per curve.

        """
        self._check_sizes(len(plot_buffers), maxs=maxs, mins=mins, dotted_limits=dotted_limits)

        super().__init__(title, [dict(data_buffer=data_buffer, name=data_buffer.name) for data_buffer in plot_buffers],
                         **kwargs)

        n_curves = len(plot_buffers)
        self._dotted_limits = [False] * n_curves if dotted_limits is None else list(dotted_limits)
        self._limits = [self._curve_limits(i, maxs, mins) for i in range(n_curves)]
        self._limit_lines = [[] for _ in range(n_curves)]

        for check in self._transform_checks:
            check.toggled.connect(self._update_limit_lines_visibility)

    def draw(self, item):
        if item not in self.listDataItems():
            super().draw(item)
            i_item = self.plot_data_items_list.index(item)

            pen = mkPen(color=mkColor(item.opts['pen'].color()), width=1)
            if self._dotted_limits[i_item]:
                pen.setDashPattern([6, 12, 6, 12])

            for value in self._limits[i_item]:
                line = _LimitLine(value, pen)
                self.addItem(line)
                self._limit_lines[i_item].append(line)

            self._update_limit_lines_visibility()

    def erase(self, item):
        super().erase(item)
        i_item = self.plot_data_items_list.index(item)

        for line in self._limit_lines[i_item]:
            self.removeItem(line)
        self._limit_lines[i_item] = []

    def _curve_limits(self, i_curve, maxs, mins):
        if self._dotted_limits[i_curve]:
            return [1, -1]

        limits = [values[i_curve] for values in (maxs, mins) if values is not None]
        return [value for value in limits if value is not None and np.isfinite(value)]

    def _update_limit_lines_visibility(self):
        hidden = any(check.isChecked() for check in self._transform_checks)
        for lines in self._limit_lines:
            for line in lines:
                line.set_hidden(hidden)

    @staticmethod
    def _check_sizes(n_curves, **limits):
        wrong = [name for name, values in limits.items() if values is not None and len(values) != n_curves]
        if wrong:
            raise ValueError(f"Size of {' and '.join(wrong)} doesn't correspond to the number of plot_buffers")

    @property
    def _transform_checks(self):
        return self.ctrl.fftCheck, self.ctrl.derivativeCheck, self.ctrl.phasemapCheck, self.ctrl.subtractMeanCheck


class _LimitLine(InfiniteLine):
    """
    Horizontal line marking a limit of a curve, following the logarithmic
    scale of the y axis and hidden while the curves show a transformed
    signal.

    """
    def __init__(self, value, pen):
        """
        Constructor.

        Args:
            value (float): value of the limit;
            pen (QPen): pen used to draw the line.

        """
        super().__init__(pos=value, angle=0, pen=pen)
        self._value = value
        self._log_y = False
        self._hidden = False

    def setLogMode(self, x_state, y_state):
        """
        Set whether the y axis is in logarithmic scale.

        Args:
            x_state (bool): whether the x axis is in logarithmic scale;
            y_state (bool): whether the y axis is in logarithmic scale.

        """
        self._log_y = y_state
        self._update()

    def set_hidden(self, hidden):
        """
        Set whether the line is hidden.

        Args:
            hidden (bool): whether to hide the line.

        """
        self._hidden = hidden
        self._update()

    def _update(self):
        if self._log_y:
            drawable = self._value > 0
            self.setValue(np.log10(self._value) if drawable else 0)
        else:
            drawable = True
            self.setValue(self._value)

        self.setVisible(drawable and not self._hidden)
