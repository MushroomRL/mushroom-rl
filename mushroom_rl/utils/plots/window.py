import time

from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QTreeWidgetItem
from PyQt5 import QtCore

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


class Window(QtGui.QSplitter):
    """
    This class is used creating windows for plotting.

    """
    WhiteBrush = QBrush(QColor(255, 255, 255))
    GreenBrush = QBrush(QColor(0, 255, 0))

    def __init__(self, plot_list, track_if_deactivated=False,
                 title="None", size=(800, 600), fullscreen=False,
                 update_freq=10):
        """
        Constructor.

        Args:
            plot_list (list): list of plots to add to the window;
            track_if_deactivated (bool, False): flag that determines if the
                respective data buffers should track the variable, when plot is
                not active. Should be used when data buffers in the plot do not
                need to be tracked;
            title (str, "None"): title of the plot;
            size (tuple, (800, 600)): size of the window in pixels;
            fullscreen (bool, False): fullscreen flag;
            update_freq (int, 10): frequency(Hz) of update of the plots.

        """
        pg.mkQApp()

        self._title = title
        self._size = size
        self._fullscreen = fullscreen
        self._track_if_deactivated = track_if_deactivated

        self.plot_list = plot_list

        if not isinstance(self.plot_list, list):
            self.plot_list = [self.plot_list]

        super().__init__(QtCore.Qt.Horizontal)

        self._activation_widget = QtGui.QTreeWidget()
        self._activation_widget.setHeaderLabels(["Plots"])

        self._activation_widget.itemClicked.connect(self.clicked)

        self._dependencies = dict()
        self._activation_items = dict()

        listitem = QTreeWidgetItem(self._activation_widget, ["ALL"])
        listitem.setBackground(0, Window.WhiteBrush)

        for plot_instance in self.plot_list:
            listitem_parent = QTreeWidgetItem(self._activation_widget, [plot_instance.name])
            listitem_parent.setBackground(0, Window.WhiteBrush)

            self._activation_items[plot_instance.name] = listitem_parent
            self._dependencies[plot_instance.name] = [self, plot_instance]

            for i in range(len(plot_instance.data_buffers)):
                listitem = QTreeWidgetItem(listitem_parent, [plot_instance.data_buffers[i].name])
                listitem.setBackground(0, Window.WhiteBrush)

                self._activation_items[plot_instance.data_buffers[i].name] = listitem
                self._dependencies[plot_instance.data_buffers[i].name] = [
                    plot_instance, plot_instance.plot_data_items_list[i]
                ]

        self._GraphicsWindow = pg.GraphicsWindow(title=title)

        self.addWidget(self._activation_widget)
        self.addWidget(self._GraphicsWindow)

        self.timecounter = time.perf_counter()
        self.timeinterval = (1.0 / update_freq)

        self.refresh()

        for plot in self.plot_list:
            self._deactivate_buffer_plots(plot)

    def draw(self, item):
        """
        Draw ``PlotItem`` on the plot widget.

        Args:
            item (PlotItem): plot item to be drawn.

        """
        self._GraphicsWindow.addItem(item)
        self._GraphicsWindow.nextRow()

    def erase(self, item):
        """
        Remove ``PlotItem`` from the plot widget.

        Args:
            item (PlotItem): plot item to be removed.

        """
        self._GraphicsWindow.removeItem(item)

    def refresh(self):
        """
        Refresh all plots if the refresh timer allows it.

        """
        if time.perf_counter() - self.timecounter > self.timeinterval:
            self.timecounter = time.perf_counter()

            for plot_instance in self.plot_list:
                plot_instance.refresh()

            QtGui.QGuiApplication.processEvents()

    def activate(self, item):
        """
        Activate the plots and data buffers connected to the given item.

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` that represents the
             plots to be activated.

        """
        if not self.check_activated(item):
            item.setBackground(0, Window.GreenBrush)
            callback_func_params = self._dependencies[item.text(0)]
            callback_func_params[0].draw(callback_func_params[1])
            if isinstance(callback_func_params[0], Window):
                self._activate_buffer_plots(callback_func_params[1])

    def deactivate(self, item):
        """
        Deactivate the plots and DataBuffers connected to the given item

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` that represents the
             plots to be deactivated.

        """
        if self.check_activated(item):
            item.setBackground(0, Window.WhiteBrush)
            callback_func_params = self._dependencies[item.text(0)]
            if isinstance(callback_func_params[0], Window):
                for curve_name in callback_func_params[1].curves_names:
                    self.deactivate(self._activation_items[curve_name])
                self._deactivate_buffer_plots(callback_func_params[1])

            callback_func_params[0].erase(callback_func_params[1])

    def clicked(self, item):
        """
        Callback to be used when plot activation widget object is clicked.

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` clicked.

        """
        if item.text(0) == "ALL":
            if self.check_activated(item):
                item.setBackground(0, Window.WhiteBrush)
                for activation_item_key in self._activation_items:
                    activation_item = self._activation_items[
                        activation_item_key
                    ]
                    self.deactivate(activation_item)

            else:
                item.setBackground(0, Window.GreenBrush)
                for activation_item_key in self._activation_items:
                    activation_item = self._activation_items[
                        activation_item_key
                    ]
                    self.activate(activation_item)

        else:
            if self.check_activated(item):
                self.deactivate(item)
            else:
                self.activate(item)

    def _deactivate_buffer_plots(self, plot):
        plot.refresh_state(False)
        if not self._track_if_deactivated[self.plot_list.index(plot)]:
            for buffer in plot.data_buffers:
                buffer.enable_tracking(False)

    def _activate_buffer_plots(self, plot):
        for buffer in plot.data_buffers:
            plot.refresh_state(True)
            buffer.enable_tracking(True)

    @staticmethod
    def check_activated(item):
        """
        Check if ``QTreeWidgetItem`` is activated.

        Args:
            item (QTreeWidgetItem): ``QTreeWidgetItem`` to be analysed.

        Returns:
            If activated returns True, else False.

        """
        return (item.background(0).color().getRgb()[0] == 0
                and item.background(0).color().getRgb()[2] == 0)
