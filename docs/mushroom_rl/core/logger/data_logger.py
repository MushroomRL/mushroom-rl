import re
import pickle
import numpy as np

from pathlib import Path


class DataLogger(object):
    """
    This class implements the data logging functionality. It can be used to create
    automatically a log directory, save numpy data array and the current agent.

    """
    def __init__(self, results_dir, suffix='', append=False):
        """
        Constructor.

        Args:
            results_dir (Path): path of the logging directory;
            suffix (string): optional string to add a suffix to each
                data file logged;
            append (bool, False): If true, the logger will append the new
                data logged to the one already existing in the directory.

        """
        self._results_dir = results_dir
        self._suffix = suffix
        self._data_dict = dict()

        self._best_J = -np.inf

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

            filename = name + self._suffix + '.npy'
            path = self._results_dir / filename

            current_data = np.array(self._data_dict[name])
            np.save(path, current_data)

    def log_numpy_array(self, **kwargs):
        """
        Log numpy arrays.

        Args:
            **kwargs: set of named arrays to be saved. The argument name
                will be used to identify the given quantity and as base file name.

        """
        for name, data in kwargs.items():
            filename = name + self._suffix + '.npy'
            path = self._results_dir / filename

            np.save(path, data)

    def log_agent(self, agent, epoch=None, full_save=False):
        """
        Log agent into the log folder.

        Args:
            agent (Agent): The agent to be saved;
            epoch (int, None): optional epoch number to
                be added to the agent file currently saved;
            full_save (bool, False): whether to save the full
                data from the agent or not.

        """
        epoch_suffix = '' if epoch is None else '-' + str(epoch)

        filename = 'agent' + self._suffix + epoch_suffix + '.msh'
        path = self._results_dir / filename
        agent.save(path, full_save=full_save)

    def log_best_agent(self, agent, J, full_save=False):
        """
        Log the best agent so far into the log folder. The agent
        is logged only if the current performance is better
        than the performance of the previously stored agent.

        Args:
            agent (Agent): The agent to be saved;
            J (float): The performance metric of the current agent;
            full_save (bool, False): whether to save the full
                data from the agent or not.

        """

        if J >= self._best_J:
            self._best_J = J

            filename = 'agent' + self._suffix + '-best.msh'
            path = self._results_dir / filename
            agent.save(path, full_save=full_save)

    def log_dataset(self, dataset):
        filename = 'dataset' + self._suffix + '.pkl'
        path = self._results_dir / filename

        with path.open(mode='wb') as f:
            pickle.dump(dataset, f)

    @property
    def path(self):
        """
        Property to return the path to the current logging directory

        """
        return self._results_dir

    def _load_numpy(self):
        for file in self._results_dir.iterdir():
            if file.is_file() and file.suffix == '.npy':
                if file.stem.endswith(self._suffix):
                    name = re.split(r'-\d+$', file.stem)[0]
                    data = np.load(str(file)).tolist()
                    self._data_dict[name] = data
