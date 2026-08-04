Monitors
========

Live plotting widgets, built on ``pyqtgraph`` and ``PySide6``: a ``Window`` holds a set of ``PlotItemBuffer``
items, each backed by a ``DataBuffer``, and is updated while an experiment runs. They are driven by
:class:`~mushroom_rl.utils.callbacks.dataset_monitor.DatasetMonitor`, so the usual way to display them is to pass
that callback to ``Core`` rather than to build a window by hand.

They require the ``monitors`` extra (``pip install mushroom_rl[monitors]``).

.. automodule:: mushroom_rl.utils.monitors.databuffer

.. automodule:: mushroom_rl.utils.monitors.plot_item_buffer

.. automodule:: mushroom_rl.utils.monitors.window

.. automodule:: mushroom_rl.utils.monitors.common_plots
