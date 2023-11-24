import numpy as np
import torch

from mushroom_rl.core import Agent, VectorCore, VectorizedEnvironment, MDPInfo
from mushroom_rl.rl_utils import Box
from mushroom_rl.policy import Policy


class DummyPolicy(Policy):
    def __init__(self, action_shape, backend):
        self._dim = action_shape[0]
        self._backend = backend
        super().__init__()

    def draw_action(self, state, policy_state):
        if self._backend == 'torch':
            return torch.randn(state.shape[0], self._dim), None
        elif self._backend == 'numpy':
            return np.random.randn(state.shape[0], self._dim), None
        else:
            raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, mdp_info, backend):
        policy = DummyPolicy(mdp_info.action_space.shape, backend)
        super().__init__(mdp_info, policy, backend=backend)

    def fit(self, dataset):

        assert len(dataset.episodes_length) == 20


class DummyVecEnv(VectorizedEnvironment):
    def __init__(self, backend):
        n_envs = 10
        state_dim = 3

        horizon = 100
        gamma = 0.99

        observation_space = Box(0, 200, shape=(3,))
        action_space = Box(0, 200, shape=(2,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, backend=backend)

        if backend == 'torch':
            self._state = torch.empty(n_envs, state_dim)
        elif backend == 'numpy':
            self._state = np.empty((n_envs, state_dim))
        else:
            raise NotImplementedError

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._state[env_mask] = torch.randint(size=(env_mask.sum(), self._state.shape[1]), low=2, high=200).float()
        return self._state, [{}]*self._n_envs

    def step_all(self, env_mask, action):
        self._state[env_mask] -= 1

        if self.info.backend == 'torch':
            reward = torch.zeros(self._state.shape[0])
        elif self.info.backend == 'numpy':
            reward = torch.zeros(self._state.shape[0])
        else:
            raise NotImplementedError

        done = (self._state == 0).any(1)

        return self._state, reward, done & env_mask, [{}] * self._n_envs


def run_exp(env_backend, agent_backend):
    torch.random.manual_seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)

    core = VectorCore(agent, env)

    dataset = core.evaluate(n_steps=2000)
    assert len(dataset) == 2000

    dataset = core.evaluate(n_episodes=20)
    assert len(dataset.episodes_length) == 20

    core.learn(n_steps=10000, n_episodes_per_fit=20)


def test_vectorized_env_():
    run_exp(env_backend='torch', agent_backend='torch')
    run_exp(env_backend='torch', agent_backend='numpy')
    run_exp(env_backend='numpy', agent_backend='torch')
    run_exp(env_backend='numpy', agent_backend='numpy')
