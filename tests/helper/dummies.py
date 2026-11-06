import numpy as np

from mushroom_rl.core import Agent, Box, MDPInfo, VectorizedEnvironment
from mushroom_rl.policy import Policy


class ZeroPolicy(Policy):
    def __init__(self, action_dim=1, backend='numpy'):
        self._dim = action_dim
        super().__init__(backend)

    def draw_action(self, state):
        return np.zeros(np.shape(state)[:-1] + (self._dim,))


class RecordingAgent(Agent):
    def __init__(self, mdp_info):
        super().__init__(mdp_info, ZeroPolicy(mdp_info.action_space.shape[0]), backend='numpy')
        self.fit_rewards = list()

    def fit(self, dataset):
        self.fit_rewards += np.asarray(dataset.reward).tolist()


class CountingVecEnv(VectorizedEnvironment):
    def __init__(self, n_envs, horizon):
        mdp_info = MDPInfo(Box(0, 1e6, shape=(3,)), Box(-1e6, 1e6, shape=(1,)), 0.99, horizon)
        self._t = np.full(n_envs, -1.)
        self._ep_t = np.zeros(n_envs)
        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._t[env_mask] += 1
        self._ep_t[env_mask] = 0
        return self._obs(), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        self._t[env_mask] += 1
        self._ep_t[env_mask] += 1
        return self._obs(), np.zeros(self._n_envs), np.zeros(self._n_envs, dtype=bool), [{}] * self._n_envs

    def _obs(self):
        return np.stack([np.arange(self._n_envs), self._t, self._ep_t], axis=1).astype(float)


class FakeLogger:
    def __init__(self):
        self.calls = list()
        self.messages = list()

    def log_training(self, prefix=None, **kwargs):
        self.calls.append({(prefix + '/' + name if prefix else name): value for name, value in kwargs.items()})

    def advance_step(self):
        pass

    def weak_line(self, debug=False):
        pass

    def debug(self, message):
        self.messages.append(message)

    def keys(self):
        return {key for call in self.calls for key in call}
