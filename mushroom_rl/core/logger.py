import re
import numpy as np
from datetime import datetime

from pathlib import Path


class Logger(object):
    """
    This class implements the logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.

    """
    def __init__(self, append=False, results_dir=None, seed=None):
        """
        Constructor.

        Args:
            append (bool, False): If true, the logger will append the new
                data logged to the one already existing in the directory;
            results_dir (string, None): name of the logging directory. If
                not specified, a time-stamped directory is created inside
                a 'log' folder;
            seed (int, None): seed for the current run. It can be optionally
                specified to add a seed suffix for each data file logged.

        """
        if results_dir is None:
            results_dir = Path('.', 'logs') / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            results_dir = Path(results_dir)

        print('Logging in folder: ' + results_dir.name)
        results_dir.mkdir(parents=True, exist_ok=True)

        self._results_dir = results_dir
        self._seed = str(seed) if seed is not None else None
        self._data_dict = dict()

        if append:
            self._load_numpy()

    def log_numpy(self, **kwargs):
        """
        Log scalars into numpy arrays.

        Args:
            **kwargs: set of named scalar values to be saved. The argument name
                will be used to identify the given quantity and as base file name.

        """
        for name, data in kwargs.items():
            if name not in self._data_dict:
                self._data_dict[name] = list()

            self._data_dict[name].append(data)

            filename = name + self._seed_suffix + '.npy'
            path = self._results_dir / filename

            current_data = np.array(self._data_dict[name])
            np.save(path, current_data)

    def log_agent(self, agent, epoch=None, full_save=False):
        """
        Log agent into log folder.

        Args:
            agent (Agent): The agent to be saved;
            epoch (int, None): optional epoch number to
                be added to the agent file currently saved;
            full_save (bool, False): wheter to save the full
                data from the agent or not.

        """
        epoch_suffix = '' if epoch is None else '-' + str(epoch)

        filename = 'agent' + self._seed_suffix + epoch_suffix + '.msh'
        path = self._results_dir / filename
        agent.save(path, full_save=full_save)

    def _load_numpy(self):
        for file in self._results_dir.iterdir():
            if file.is_file() and file.suffix == '.npy':
                if file.stem.endswith(self._seed_suffix):
                    name = re.split(r'-\d+$', file.stem)[0]
                    data = np.load(str(file)).tolist()
                    self._data_dict[name] = data

    @property
    def _seed_suffix(self):
        return '' if self._seed is None else '-' + str(self._seed)

