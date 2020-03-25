__extras__ = []

try:
    from mushroom_rl.utils.callbacks.plot_dataset import PlotDataset
    __extras__.append('PlotDataset')
except ImportError:
    pass

from .callback import Callback
from .collect_dataset import CollectDataset
from .collect_max_q import CollectMaxQ
from .collect_q import CollectQ
from .collect_parameters import CollectParameters

__all__ = ['Callback', 'CollectDataset', 'CollectQ', 'CollectMaxQ',
           'CollectParameters'] + __extras__
