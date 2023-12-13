import pickle
import numpy as np

from mushroom_rl.utils.callbacks.callback import Callback
from mushroom_rl.utils.plots import DataBuffer, Window, Actions,\
    LenOfEpisodeTraining, Observations, RewardPerEpisode, RewardPerStep
from mushroom_rl.rl_utils.spaces import Box


class PlotDataset(Callback):
    """
    This callback is used for plotting the values of the actions, observations,
    reward per step, reward per episode, episode length only for the training.

    """
    def __init__(self, mdp_info, obs_normalized=False, window_size=1000,
                 update_freq=10, show=True):

        """
        Constructor.

        Args:
            mdp_info (MDPInfo): information of the environment;
            obs_normalized (bool, False): whether observation needs to be
                normalized or not;
            window_size (int, 1000): number of steps plotted in the windowed
                plots. Only action, observation and reward per step plots are
                affected. The other are always adding information;
            update_freq (int, 10): Frequency(Hz) that window should be updated.
                This update frequency is not accurate because the refresh
                method of the window runs sequentially with the rest of the
                script. So this update frequency is only relevant if the
                frequency of refresh calls is too high, avoiding excessive
                updates;
            show (bool, True): whether to show the window or not.

        """
        super().__init__()

        self.action_buffers_list = []
        for i in range(mdp_info.action_space.shape[0]):
            self.action_buffers_list.append(DataBuffer('Action_' + str(i), window_size))

        self.observation_buffers_list = []
        for i in range(mdp_info.observation_space.shape[0]):
            self.observation_buffers_list.append(DataBuffer('Observation_' + str(i), window_size))

        self.instant_reward_buffer = DataBuffer("Instant_reward", window_size)

        self.training_reward_buffer = DataBuffer("Episode_reward")

        self.episodic_len_buffer_training = DataBuffer("Episode_len")

        if isinstance(mdp_info.action_space, Box):
            high_actions = mdp_info.action_space.high.tolist()
            low_actions = mdp_info.action_space.low.tolist()
        else:
            high_actions = None
            low_actions = None

        actions_plot = Actions(self.action_buffers_list, maxs=high_actions, mins=low_actions)

        dotted_limits = None
        if isinstance(mdp_info.observation_space, Box):
            high_mdp = mdp_info.observation_space.high.tolist()
            low_mdp = mdp_info.observation_space.low.tolist()
            if obs_normalized:
                dotted_limits = []
                for i in range(len(high_mdp)):
                    if abs(high_mdp[i]) == np.inf:
                        dotted_limits.append(True)
                    else:
                        dotted_limits.append(False)
                        
                    high_mdp[i] = 1
                    low_mdp[i] = -1
        else:
            high_mdp = None
            low_mdp = None

        observation_plot = Observations(self.observation_buffers_list, maxs=high_mdp, mins=low_mdp,
                                        dotted_limits=dotted_limits)

        step_reward_plot = RewardPerStep(self.instant_reward_buffer)

        training_reward_plot = RewardPerEpisode(self.training_reward_buffer)

        episodic_len_training_plot = LenOfEpisodeTraining(self.episodic_len_buffer_training)

        self.plot_window = Window(
            plot_list=[training_reward_plot, episodic_len_training_plot,
                       step_reward_plot, actions_plot, observation_plot],
            title="EnvironmentPlot",
            track_if_deactivated=[True, True, False, False, False],
            update_freq=update_freq)

        if show:
            self.plot_window.show()

        self._episode_reward = 0.
        self._episode_lenght = 0

    def __call__(self, sample):
        obs = sample[0]
        action = sample[1]
        reward = sample[2]
        last = sample[5]

        for i in range(len(action)):
            self.action_buffers_list[i].update([action[i]])

        for i in range(obs.size):
            self.observation_buffers_list[i].update([obs[i]])

        self.instant_reward_buffer.update([reward])

        self._episode_reward += reward
        self._episode_lenght += 1

        if last:
            self.training_reward_buffer.update([self._episode_reward])
            self.episodic_len_buffer_training.update([self._episode_lenght])
            self._episode_reward = 0.
            self._episode_lenght = 0

        self.plot_window.refresh()
