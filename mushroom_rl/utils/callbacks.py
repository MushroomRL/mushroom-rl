import pickle
from copy import deepcopy

import numpy as np

from mushroom_rl.utils import plots
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.table import EnsembleTable


class Callback(object):
    """
    Interface for all basic callbacks. Implements a list in which it is possible to store data and
    methods to query and clean the content stored by the callback.

    """

    def __init__(self):
        self._data_list = list()

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        raise NotImplementedError

    def get(self):
        """
        Returns:
             The current collected data as a list.

        """
        return self._data_list

    def clean(self):
        """
        Deletes the current stored data list

        """
        self._data_list = list()


class CollectDataset(Callback):
    """
    This callback can be used to collect samples during the learning of the
    agent.

    """

    def __call__(self, dataset):
        """
        Add samples to the samples list.

        Args:
            dataset (list): the samples to collect.

        """
        self._data_list += dataset


class CollectQ(Callback):
    """
    This callback can be used to collect the action values in all states at the
    current time step.

    """

    def __init__(self, approximator):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use to
                predict the action values.

        """
        self._approximator = approximator

        super().__init__()

    def __call__(self, **kwargs):
        """
        Add action values to the action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._data_list.append(deepcopy(np.mean(qs, 0)))
        else:
            self._data_list.append(deepcopy(self._approximator.table))


class CollectMaxQ(Callback):
    """
    This callback can be used to collect the maximum action value in a given
    state at each call.

    """

    def __init__(self, approximator, state):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use;
            state (np.ndarray): the state to consider.

        """
        self._approximator = approximator
        self._state = state

        super().__init__()

    def __call__(self, **kwargs):
        """
        Add maximum action values to the maximum action-values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        q = self._approximator.predict(self._state)
        max_q = np.max(q)

        self._data_list.append(max_q)


class CollectParameters(Callback):
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """

    def __init__(self, parameter, *idx):
        """
        Constructor.

        Args:
            parameter (Parameter): the parameter whose values have to be
                collected;
            *idx (list): index of the parameter when the ``parameter`` is
                tabular.

        """
        self._parameter = parameter
        self._idx = idx

        super().__init__()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """
        value = self._parameter.get_value(*self._idx)
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._data_list.append(value)


class PlotDataset(CollectDataset):
    """
    This callback is used for plotting the values of the actions, observations, reward per step,
    reward per episode, episode length only for the training.

    """

    def __init__(self, mdp, window_size=1000, update_freq=10):
        """
        Constructor.

        Args:
            mdp (Environment): Environment used to extract additional parameters
                like observation space limits, etc.;
            window_size (int): Number of steps plotted in the windowed plots.
                Only action, observation and reward per step plots are affected.
                The other are always adding information;
            update_freq (int): Frequency(Hz) that window should be updated.
                This update frequency is not accurate because the refresh
                method of the window runs sequentially with the rest of the script.
                So this update frequency is only relevant if the frequency of refresh
                calls is too high, avoiding excessive updates.

        """

        super().__init__()
        # create buffers
        self.action_buffers_list = []
        for i in range(mdp.info.action_space.shape[0]):
            self.action_buffers_list.append(
                plots.DataBuffer('Action_' + str(i), window_size))

        self.observation_buffers_list = []
        for i in range(mdp.info.observation_space.shape[0]):
            self.observation_buffers_list.append(
                plots.DataBuffer('Observation_' + str(i), window_size))

        self.instant_reward_buffer = \
            plots.DataBuffer("Instant_reward", window_size)

        self.training_reward_buffer = \
            plots.common_buffers.EndOfEpisodeBuffer("Episode_reward")

        self.episodic_len_buffer_training = \
            plots.common_buffers.EndOfEpisodeBuffer("Episode_len")

        if isinstance(mdp.info.action_space, Box):
            high_actions = mdp.info.action_space.high.tolist()
            low_actions = mdp.info.action_space.low.tolist()
        else:
            high_actions = None
            low_actions = None

        # create plots
        actions_plot = plots.common_plots.Actions(self.action_buffers_list,
                                                  maxs=high_actions,
                                                  mins=low_actions)

        if isinstance(mdp.info.observation_space, Box):
            high_mdp = mdp.info.observation_space.high.tolist()
            low_mdp = mdp.info.observation_space.low.tolist()
        else:
            high_mdp = None
            low_mdp = None

        observation_plot = plots.common_plots.Observations(self.observation_buffers_list,
                                                           maxs=high_mdp,
                                                           mins=low_mdp)

        step_reward_plot = plots.common_plots.RewardPerStep(self.instant_reward_buffer)

        training_reward_plot = plots.common_plots.RewardPerEpisode(self.training_reward_buffer)

        episodic_len_training_plot = \
            plots.common_plots.LenOfEpisodeTraining(self.episodic_len_buffer_training)

        # create window
        self.plot_window = plots.Window(
            plot_list=[training_reward_plot, episodic_len_training_plot,
                       step_reward_plot, actions_plot, observation_plot],
            title="EnvironmentPlot",
            track_if_deactivated=[True, True, False, False, False],
            update_freq=update_freq)

    def __call__(self, dataset):
        """
        Add samples to DataBuffers and refresh window.

        Args:
            dataset (list): the samples to collect.

        """

        for sample in dataset:

            obs = sample[0]
            action = sample[1]
            reward = sample[2]
            absorbing = sample[4]

            for i in range(len(action)):
                self.action_buffers_list[i].update([action[i]])

            for i in range(obs.size):
                self.observation_buffers_list[i].update([obs[i]])

            self.instant_reward_buffer.update([reward])
            self.training_reward_buffer.update([[reward, absorbing]])
            self.episodic_len_buffer_training.update([[1, absorbing]])

            self.plot_window.refresh()

    def get_state(self):
        """
        Returns:
             The dictionary of data in each DataBuffer in tree structure associated with the plot name.

        """
        data = dict(plot_data={plot.name: {buffer.name: buffer.get()}
                               for plot in self.plot_window.plot_list
                               for buffer in plot.data_buffers})

        return data

    def set_state(self, data):
        """
        Set the state of the DataBuffers to resume the plots.

        Args:
            data (dict): data of each plot and databuffer.

        """

        normalize_data = data["plot_data"]
        for plot_name, buffer_dict in normalize_data.items():
            # could use keys to find if plots where in dicts instead of list
            for plot in self.plot_window.plot_list:
                if plot.name == plot_name:

                    # could use keys to find if plots where in dicts instead of list
                    for buffer_name, buffer_data in buffer_dict.items():
                        for buffer in plot.data_buffers:
                            if buffer.name == buffer_name:
                                buffer.set(buffer_data)

    def save_state(self, path):
        """
        Save the data in the plots given a path.

        Args:
            path (str): path to save the data.

        """
        data = self.get_state()
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=3)

    def load_state(self, path):
        """
        Load the data to the plots given a path.

        Args:
            path (str): path to load the data.

        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.set_state(data)
