from mushroom_rl.utils.plots import PlotItemWBuffer, DataBuffer
from mushroom_rl.utils.plots.plot_item_w_buffer import _Limited


class RewardPerStep(PlotItemWBuffer):
    """
    Class that represents a plot for the instant undiscounted reward.
    """
    def __init__(self, plot_buffer):
        """
        Constructor.
        Args:
            plot_buffer (DataBuffer): DataBuffer registering the instant reward.
        """
        title = "Step_Reward"
        curves_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, curves_params)


class RewardPerEpisode(PlotItemWBuffer):
    """
    Class that represents a plot for the accumulated reward per episode.
    """
    def __init__(self, plot_buffer):
        """
        Constructor.
        Args:
            plot_buffer (DataBuffer): DataBuffer registering the reward per Episode.
        """
        title = "Episode_Reward"
        curves_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, curves_params)


class Actions(_Limited):
    """
    Class that represents a plot for the Actions.
    """
    def __init__(self, plot_buffers, maxs=None, mins=None):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): DataBuffer registering the actions.
            maxs(list, None) : List of maxs of each DataBuffer plotted (each action) that
                allows for plotting of a horizontal line if max exists. If a element of maxs
                is None then no max horizontal line is drawn, e. g. [2, 3, None, ...]
            mins(list, None) : List of mins of each DataBuffer plotted (each action) that
                allows for plotting of a horizontal line if min exists. If a element of mins
                is None then no min horizontal line is drawn, e. g. [2, 3, None, ...]
        """
        title = "Actions"
        super().__init__(title, plot_buffers, maxs=maxs, mins=mins)


class Observations(_Limited):
    """
    Class that represents a plot for the observations.
    """
    def __init__(self, plot_buffers, maxs=None, mins=None):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): DataBuffer registering the observations.
            maxs(list, None) : List of maxs of each DataBuffer plotted (each observation) that
                allows for plotting of a horizontal line if max exists. If a element of maxs
                is None then no max horizontal line is drawn, e. g. [2, 3, None, ...]
            mins(list, None) : List of mins of each DataBuffer plotted (each observation) that
                allows for plotting of a horizontal line if min exists. If a element of mins
                is None then no min horizontal line is drawn, e. g. [2, 3, None, ...]
        """
        title = "Observations"
        super().__init__(title, plot_buffers, maxs=maxs, mins=mins)


class LenOfEpisodeTraining(PlotItemWBuffer):
    """
    Class that represents a plot for the length of the episode.
    """
    def __init__(self, plot_buffer):
        """
        Constructor.

        Args:
            plot_buffer (DataBuffer): DataBuffer registering the length of the episode.
        """
        title = "Len of Episode"
        plot_params = [dict(data_buffer=plot_buffer)]
        super().__init__(title, plot_params)
