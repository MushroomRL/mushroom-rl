import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.approximators import Approximator
from mushroom_rl.core import Agent, Box, Discrete, MDPInfo, VectorizedEnvironment
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.preprocessors import Preprocessor


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


class ChainVecEnv(VectorizedEnvironment):
    def __init__(self, n_envs, device):
        super().__init__(MDPInfo(Box(-10., 10., shape=(2,)), Discrete(3), .9, 10, backend='torch', device=device),
                         n_envs)
        self._state = torch.zeros(n_envs, 2, device=device)
        self._t = torch.zeros(n_envs, device=device)

    def reset_all(self, env_mask, state=None):
        self._state[env_mask] = torch.rand(self.number, 2, device=self._state.device)[env_mask]
        self._t[env_mask] = 0
        return self._state.clone(), [{}] * self.number

    def step_all(self, env_mask, action):
        self._state[env_mask] += .1 * (action[env_mask].float() - 1.)
        self._t[env_mask] += 1
        return self._state.clone(), (action[:, 0] == 2).float(), (self._t >= 10) & env_mask, [{}] * self.number


class FixedQ:
    def __init__(self, table):
        self.table = table
        self.n_actions = table.shape[1]

    def predict(self, state):
        return self.table[state[..., 0]]


class FlattenExtraTrees(Approximator):
    def __init__(self, input_shape, output_shape, **params):
        super().__init__(input_shape, output_shape)
        self._models = [ExtraTreesRegressor(**params) for _ in range(output_shape[0])]
        self.fit_state = None

    def fit(self, x, a, q):
        self.fit_state = x
        x = x.reshape(len(x), -1)
        for i, m in enumerate(self._models):
            mask = a[:, 0] == i
            m.fit(x[mask], q[mask])

    def predict(self, x):
        if self.fit_state is None:
            return np.zeros((len(x), len(self._models)))
        else:
            x = x.reshape(len(x), -1)
            return np.stack([m.predict(x) for m in self._models], axis=-1)


class RecordingPreprocessor(Preprocessor):
    def __init__(self, scale):
        super().__init__('torch')
        self.scale = scale
        self.shapes = list()

    def __call__(self, state):
        self.shapes.append(tuple(state.shape))

        return state * self.scale


class FrameRecorder:
    def __init__(self, fps=None, **kwargs):
        self.fps = fps
        self.frames = list()
        self.masks = list()
        self.n_stops = 0

    def __call__(self, frame, mask=None):
        self.frames.append(frame)
        self.masks.append(mask)

    def stop(self):
        self.n_stops += 1


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
