import numpy as np
import torch

from mushroom_rl.core import Agent, VectorCore, VectorizedEnvironment, MDPInfo
from mushroom_rl.rl_utils import Box
from mushroom_rl.policy import Policy
from mushroom_rl.utils import TorchUtils


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
        print(f'\t* samples={len(dataset)}, episodes={len(dataset.episodes_length)}')
        assert len(dataset.episodes_length) == 20 or len(dataset) == 150


class DummyEpisodicAgent(Agent):
    def __init__(self, mdp_info, backend):
        self._backend = backend
        policy = DummyPolicy(mdp_info.action_space.shape, backend)
        super().__init__(mdp_info, policy, is_episodic=True, backend=backend)
        self._counter = 0

    def fit(self, dataset):
        assert len(dataset.theta_list) == 5

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        n_envs = len(start_mask)
        current_count = self._counter
        self._counter += 1
        if self._backend == 'torch':
            return None, torch.ones(n_envs, 2) * current_count
        elif self._backend == 'numpy':
            return None, np.ones((n_envs, 2)) * current_count
        else:
            raise NotImplementedError


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
            self._state = torch.empty(n_envs, state_dim).to(TorchUtils.get_device())
        elif backend == 'numpy':
            self._state = np.empty((n_envs, state_dim))
        else:
            raise NotImplementedError

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        if self.info.backend == 'torch':
            self._state[env_mask] = torch.randint(size=(env_mask.sum(), self._state.shape[1]),
                                                  low=2, high=200).float().to(TorchUtils.get_device())
        elif self.info.backend == 'numpy':
            self._state[env_mask] = np.random.randint(size=(env_mask.sum(), self._state.shape[1]),
                                                      low=2, high=200).astype(float)
        return self._state, [{}]*self._n_envs

    def step_all(self, env_mask, action):
        self._state[env_mask] -= 1

        if self.info.backend == 'torch':
            reward = torch.zeros(self._state.shape[0]).to(TorchUtils.get_device())
        elif self.info.backend == 'numpy':
            reward = np.zeros(self._state.shape[0])
        else:
            raise NotImplementedError

        done = (self._state == 0).any(1)

        return self._state, reward, done & env_mask, [{}] * self._n_envs


def run_exp(env_backend, agent_backend):
    torch.random.manual_seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)

    core = VectorCore(agent, env)

    print('- evaluate n_steps=2000')
    dataset = core.evaluate(n_steps=2000)
    assert len(dataset) == 2000

    print('- evaluate n_episodes=20')
    dataset = core.evaluate(n_episodes=20)
    assert len(dataset.episodes_length) == 20

    print('- learn n_steps=10000 n_episodes_per_fit=20')
    core.learn(n_steps=10000, n_episodes_per_fit=20)

    print('- learn n_steps=10000 n_steps_per_fit=150')
    core.learn(n_steps=10000, n_steps_per_fit=150)

    print('- learn n_episode=100 n_steps_per_fit=150')
    core.learn(n_episodes=100, n_steps_per_fit=150)

    print('- learn n_episode=100 n_episodes_per_fit=20')
    core.learn(n_episodes=100, n_episodes_per_fit=20)


def run_exp_episodic(env_backend, agent_backend):
    torch.random.manual_seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyEpisodicAgent(env.info, agent_backend)

    core = VectorCore(agent, env)

    print('- evaluate n_episodes=20')
    dataset = core.evaluate(n_episodes=20)
    assert len(dataset.episodes_length) == 20
    assert len(dataset.theta_list) == 20

    print('- learn n_episodes=25 n_episodes_per_fit=5')
    core.learn(n_episodes=25, n_episodes_per_fit=5)
    

def test_vectorized_core():
    print('# CPU test')
    run_exp(env_backend='torch', agent_backend='torch')
    run_exp(env_backend='torch', agent_backend='numpy')
    run_exp(env_backend='numpy', agent_backend='torch')
    run_exp(env_backend='numpy', agent_backend='numpy')

    run_exp_episodic(env_backend='torch', agent_backend='torch')
    run_exp_episodic(env_backend='torch', agent_backend='numpy')
    run_exp_episodic(env_backend='numpy', agent_backend='torch')
    run_exp_episodic(env_backend='numpy', agent_backend='numpy')

    if torch.cuda.is_available():
        print('# Testing also cuda')
        TorchUtils.set_default_device('cuda')
        run_exp(env_backend='torch', agent_backend='torch')
        run_exp(env_backend='torch', agent_backend='numpy')
        run_exp_episodic(env_backend='torch', agent_backend='torch')
        run_exp_episodic(env_backend='torch', agent_backend='numpy')
        TorchUtils.set_default_device('cpu')

