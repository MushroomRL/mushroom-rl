import pickle

from mushroom_rl.utils.callbacks.collect_dataset import CollectDataset
import mushroom_rl.utils.plots as plots
from mushroom_rl.utils.spaces import Box


class PlotDataset(CollectDataset):
    """
    This callback is used for plotting the values of the actions, observations, reward per step,
    reward per episode, episode length only for the training.
    """

    def __init__(self, mdp_info, window_size=1000, update_freq=10, show=True):
        """
        Constructor.

        Args:
            mdp_info (MDPInfo): mdp_info object to extract information
                about action/observation_spaces.
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
        for i in range(mdp_info.action_space.shape[0]):
            self.action_buffers_list.append(
                plots.DataBuffer('Action_' + str(i), window_size))

        self.observation_buffers_list = []
        for i in range(mdp_info.observation_space.shape[0]):
            self.observation_buffers_list.append(
                plots.DataBuffer('Observation_' + str(i), window_size))

        self.instant_reward_buffer = \
            plots.DataBuffer("Instant_reward", window_size)

        self.training_reward_buffer = \
            plots.common_buffers.EndOfEpisodeBuffer("Episode_reward")

        self.episodic_len_buffer_training = \
            plots.common_buffers.EndOfEpisodeBuffer("Episode_len")

        if isinstance(mdp_info.action_space, Box):
            high_actions = mdp_info.action_space.high.tolist()
            low_actions = mdp_info.action_space.low.tolist()
        else:
            high_actions = None
            low_actions = None

        # create plots
        actions_plot = plots.common_plots.Actions(self.action_buffers_list,
                                                  maxs=high_actions,
                                                  mins=low_actions)

        if isinstance(mdp_info.observation_space, Box):
            high_mdp = mdp_info.observation_space.high.tolist()
            low_mdp = mdp_info.observation_space.low.tolist()
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

        if show:
            self.plot_window.show()

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
            last = sample[5]

            for i in range(len(action)):
                self.action_buffers_list[i].update([action[i]])

            for i in range(obs.size):
                self.observation_buffers_list[i].update([obs[i]])

            self.instant_reward_buffer.update([reward])
            self.training_reward_buffer.update([[reward, last]])
            self.episodic_len_buffer_training.update([[1, last]])

        self.plot_window.refresh()

    def get_state(self):
        """
        Returns:
             The dictionary of data in each DataBuffer in tree structure associated with the plot name.
        """
        data = {plot.name: {buffer.name: buffer.get()}
                for p_i, plot in enumerate(self.plot_window.plot_list)
                for buffer in plot.data_buffers
                if self.plot_window._track_if_deactivated[p_i]}

        return data

    def set_state(self, data):
        """
        Set the state of the DataBuffers to resume the plots.
        Args:
            data (dict): data of each plot and databuffer.
        """
        for plot_name, buffer_dict in data.items():
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