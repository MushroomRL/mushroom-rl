from mushroom_rl.utils.plots import PlotItemBuffer, DataBuffer
from mushroom_rl.utils.plots.plot_item_buffer import PlotItemBufferLimited


class RewardPerStep(PlotItemBuffer):
    """
    Class that represents a plot for the reward at every step.

    """
    def __init__(self, plot_buffer):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): data buffer to be used.

        """
        title = "Step_Reward"
        curves_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, curves_params)


class RewardPerEpisode(PlotItemBuffer):
    """
    Class that represents a plot for the accumulated reward per episode.

    """
    def __init__(self, plot_buffer):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): data buffer to be used.

        """
        title = "Episode_Reward"
        curves_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, curves_params)


class Actions(PlotItemBufferLimited):
    """
    Class that represents a plot for the actions.

    """
    def __init__(self, plot_buffers, maxs=None, mins=None):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): data buffer to be used;
            maxs(list, None): list of max values of each data buffer plotted.
                If an element is None, no max line is drawn;
            mins(list, None): list of min values of each data buffer plotted.
                If an element is None, no min line is drawn.

        """
        title = "Actions"
        super().__init__(title, plot_buffers, maxs=maxs, mins=mins)


class Observations(PlotItemBufferLimited):
    """
    Class that represents a plot for the observations.

    """
    def __init__(self, plot_buffers, maxs=None, mins=None, dotted_limits=None):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): data buffer to be used;
            maxs(list, None): list of max values of each data buffer plotted.
                If an element is None, no max line is drawn;
            mins(list, None): list of min values of each data buffer plotted.
                If an element is None, no min line is drawn.
            dotted_limits (list, None): list of booleans. If True, the
                corresponding limit is dotted; otherwise, it is printed as a
                solid line.

        """
        title = "Observations"
        super().__init__(title, plot_buffers, maxs=maxs, mins=mins,
                         dotted_limits=dotted_limits)


class LenOfEpisodeTraining(PlotItemBuffer):
    """
    Class that represents a plot for the length of the episode.

    """
    def __init__(self, plot_buffer):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): data buffer to be used;

        """
        title = "Len of Episode"
        plot_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, plot_params)
