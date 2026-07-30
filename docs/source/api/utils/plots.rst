Plots
=====

Two different things live under similar names, and it is worth stating which is which.

``mushroom_rl.utils.plots`` is a package of **live plotting** widgets, built on ``pyqtgraph`` and ``PySide6``: a
``Window`` holds a set of ``PlotItemBuffer`` items, each backed by a ``DataBuffer``, and is updated while an
experiment runs — it is what :class:`~mushroom_rl.utils.callbacks.plot_dataset.PlotDataset` drives. It requires
the ``plots`` extra (``pip install mushroom_rl[plots]``).

``mushroom_rl.utils.plot`` is a single module of **matplotlib** helpers to plot results *after* a run, computing
the mean and the confidence interval of a set of runs and drawing them as a shaded band.

Live plotting
-------------

.. automodule:: mushroom_rl.utils.plots.databuffer

.. automodule:: mushroom_rl.utils.plots.plot_item_buffer

.. automodule:: mushroom_rl.utils.plots.window

.. automodule:: mushroom_rl.utils.plots.common_plots

Result plotting
---------------

.. automodule:: mushroom_rl.utils.plot
    :private-members:
