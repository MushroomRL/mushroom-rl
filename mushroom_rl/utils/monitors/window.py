import time

from pyqtgraph.Qt.QtGui import QGuiApplication
from pyqtgraph.Qt.QtWidgets import QTreeWidget, QTreeWidgetItem, QSplitter
from pyqtgraph.Qt import QtCore

import pyqtgraph as pg


class Window(QSplitter):
    """
    This class is used creating windows for plotting.

    """
    _TREE_STYLE = """
        QTreeWidget { background: #f7f7f7; color: #333333; border: none; font-size: 10pt; }
        QTreeWidget::item { padding: 3px; }
        QHeaderView::section { background: #ececec; color: #333333; border: none; padding: 5px;
                               font-weight: bold; }
    """
    _SPLITTER_STYLE = "QSplitter::handle { background: #dddddd; }"

    def __init__(self, plot_list, track_if_deactivated=False,
                 title="None", size=(800, 600), fullscreen=False,
                 update_freq=10, layout=None):
        """
        Constructor.

        Args:
            plot_list (list): list of plots to add to the window. All the
                plots and curves are active at start;
            track_if_deactivated (bool or list, False): flag that determines if
                the respective data buffers should track the variable, when plot
                is not active. Either one flag for all the plots, or a list with
                one flag per plot;
            title (str, "None"): title of the window;
            size (tuple, (800, 600)): size of the window in pixels;
            fullscreen (bool, False): fullscreen flag;
            update_freq (int, 10): frequency(Hz) of update of the plots;
            layout (list, None): list of rows, each a list of indices into
                ``plot_list`` of the plots shown side by side in that row. If
                None, every plot gets a row of its own.

        """
        pg.mkQApp()

        self._title = title
        self._size = size
        self._fullscreen = fullscreen

        self.plot_list = plot_list if isinstance(plot_list, list) else [plot_list]

        if isinstance(track_if_deactivated, bool):
            track_if_deactivated = [track_if_deactivated] * len(self.plot_list)
        self._track_if_deactivated = track_if_deactivated

        self._slots = self._compute_slots(layout)

        super().__init__(QtCore.Qt.Horizontal)
        self.setWindowTitle(title)
        self.setStyleSheet(self._SPLITTER_STYLE)
        self.setHandleWidth(3)

        self._activation_widget = QTreeWidget()
        self._activation_widget.setHeaderLabels(["Plots"])
        self._activation_widget.setStyleSheet(self._TREE_STYLE)

        self._dependencies = dict()
        self._activation_items = dict()

        self._all_item = self._new_tree_item(self._activation_widget, "ALL")

        for plot_instance in self.plot_list:
            plot_item = self._new_tree_item(self._activation_widget, plot_instance.name)
            self._activation_items[plot_instance.name] = plot_item
            self._dependencies[plot_instance.name] = [self, plot_instance]

            if plot_instance.len_curves == 1:
                # a single curve is shown together with its plot, without an entry of its own
                plot_instance.draw(plot_instance.plot_data_items_list[0])
            else:
                for data_buffer, curve in zip(plot_instance.data_buffers, plot_instance.plot_data_items_list):
                    curve_item = self._new_tree_item(plot_item, data_buffer.name)
                    self._activation_items[data_buffer.name] = curve_item
                    self._dependencies[data_buffer.name] = [plot_instance, curve]

        self._activation_widget.expandAll()
        self._activation_widget.itemChanged.connect(self._item_changed)

        self._GraphicsWindow = pg.GraphicsLayoutWidget(title=title)
        self._GraphicsWindow.setBackground('w')

        self.addWidget(self._activation_widget)
        self.addWidget(self._GraphicsWindow)
        self.setSizes([190, max(size[0] - 190, 1)])
        self.resize(*size)

        if fullscreen:
            self.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)

        self.timecounter = time.perf_counter()
        self.timeinterval = (1.0 / update_freq)

        self.refresh()

        for plot in self.plot_list:
            self._deactivate_buffer_plots(plot)

        for item in self._activation_items.values():
            self.activate(item)

    def draw(self, item):
        """
        Draw ``PlotItem`` on the plot widget, in its position of the layout.

        Args:
            item (PlotItem): plot item to be drawn.

        """
        row, col, colspan = self._slots[self.plot_list.index(item)]
        self._GraphicsWindow.addItem(item, row=row, col=col, colspan=colspan)

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

            QGuiApplication.processEvents()

    def activate(self, item):
        """
        Activate the plots and data buffers connected to the given item.

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` that represents the
             plots to be activated.

        """
        if not self.check_activated(item):
            self._set_checked(item, True)
            callback_func_params = self._dependencies[item.text(0)]
            callback_func_params[0].draw(callback_func_params[1])
            if isinstance(callback_func_params[0], Window):
                self._activate_buffer_plots(callback_func_params[1])
            self._update_all_item()

    def deactivate(self, item):
        """
        Deactivate the plots and DataBuffers connected to the given item

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` that represents the
             plots to be deactivated.

        """
        if self.check_activated(item):
            self._set_checked(item, False)
            callback_func_params = self._dependencies[item.text(0)]
            if isinstance(callback_func_params[0], Window):
                self._deactivate_buffer_plots(callback_func_params[1])

            callback_func_params[0].erase(callback_func_params[1])
            self._update_all_item()

    def clicked(self, item):
        """
        Toggle the given item, as clicking its checkbox does. The "ALL" item
        toggles the plots, leaving the state of their curves unchanged.

        Args:
             item (QTreeWidgetItem): ``QTreeWidgetItem`` clicked.

        """
        if item is self._all_item:
            activate = self._all_item.checkState(0) != QtCore.Qt.CheckState.Checked
            for activation_item in self._plot_items:
                if activate:
                    self.activate(activation_item)
                else:
                    self.deactivate(activation_item)
        elif self.check_activated(item):
            self.deactivate(item)
        else:
            self.activate(item)

    def _item_changed(self, item, column):
        wanted = item.checkState(0) == QtCore.Qt.CheckState.Checked
        # restore the state before the click, then toggle through the activation logic
        self._set_checked(item, not wanted)
        self.clicked(item)

    def _set_checked(self, item, checked):
        self._set_state(item, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def _set_state(self, item, state):
        blocked = self._activation_widget.blockSignals(True)
        item.setCheckState(0, state)
        self._activation_widget.blockSignals(blocked)

    def _set_all_state(self, state):
        self._set_state(self._all_item, state)

    def _update_all_item(self):
        n_active = sum(self.check_activated(item) for item in self._plot_items)
        if n_active == 0:
            self._set_all_state(QtCore.Qt.CheckState.Unchecked)
        elif n_active == len(self._plot_items):
            self._set_all_state(QtCore.Qt.CheckState.Checked)
        else:
            self._set_all_state(QtCore.Qt.CheckState.PartiallyChecked)

    def _deactivate_buffer_plots(self, plot):
        plot.refresh_state(False)
        if not self._track_if_deactivated[self.plot_list.index(plot)]:
            for buffer in plot.data_buffers:
                buffer.enable_tracking(False)

    def _activate_buffer_plots(self, plot):
        for buffer in plot.data_buffers:
            plot.refresh_state(True)
            buffer.enable_tracking(True)

    def _compute_slots(self, layout):
        if layout is None:
            layout = [[i] for i in range(len(self.plot_list))]

        n_columns = max(len(row) for row in layout)
        slots = dict()
        for i_row, row in enumerate(layout):
            colspan = n_columns // len(row)
            for i_col, i_plot in enumerate(row):
                slots[i_plot] = (i_row, i_col * colspan, colspan)

        return slots

    @staticmethod
    def check_activated(item):
        """
        Check if ``QTreeWidgetItem`` is activated.

        Args:
            item (QTreeWidgetItem): ``QTreeWidgetItem`` to be analysed.

        Returns:
            If activated returns True, else False.

        """
        return item.checkState(0) == QtCore.Qt.CheckState.Checked

    @staticmethod
    def _new_tree_item(parent, text):
        item = QTreeWidgetItem(parent, [text])
        item.setFlags((item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable) & ~QtCore.Qt.ItemFlag.ItemIsAutoTristate)
        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        return item

    @property
    def _plot_items(self):
        return [self._activation_items[plot.name] for plot in self.plot_list]
