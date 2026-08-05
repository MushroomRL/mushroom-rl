__extras__ = []

try:
    from mushroom_rl.utils.callbacks.dataset_monitor import DatasetMonitor  # noqa: F401
    __extras__.append('DatasetMonitor')
except ImportError:
    pass

from .callback import Callback, CallbackList
from .collect_dataset import CollectDataset
from .collect_max_q import CollectMaxQ
from .collect_q import CollectQ
from .collect_parameters import CollectParameters

__all__ = ['Callback', 'CallbackList', 'CollectDataset', 'CollectQ', 'CollectMaxQ',
           'CollectParameters']

__all__ += __extras__
