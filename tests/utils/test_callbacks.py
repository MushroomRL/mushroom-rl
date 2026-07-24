import numpy as np
import pytest

from mushroom_rl.core import Core, Agent, MDPInfo, Box, VectorizedEnvironment
from mushroom_rl.core.dataset import VectorizedDataset
from mushroom_rl.environments import GridWorld
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.policy import EpsGreedy, Policy
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter
from mushroom_rl.utils.callbacks import Callback, CollectDataset, CollectQ, CollectMaxQ, CollectParameters


def episode_lengths_from_flags(last_flags):
    lengths = list()
    length = 0
    for flag in last_flags:
        length += 1
        if flag:
            lengths.append(length)
            length = 0
    return lengths


class ZeroPolicy(Policy):
    def __init__(self, action_shape):
        self._dim = action_shape[0]
        super().__init__()

    def draw_action(self, state):
        return np.zeros((state.shape[0], self._dim))


class CountdownAgent(Agent):
    def __init__(self, mdp_info):
        policy = ZeroPolicy(mdp_info.action_space.shape)
        super().__init__(mdp_info, policy, backend='numpy')

    def fit(self, dataset):
        pass


class CountdownVectorizedEnv(VectorizedEnvironment):
    def __init__(self):
        n_envs = 4
        observation_space = Box(0, 10, shape=(1,))
        action_space = Box(-1, 1, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, 0.99, 50, backend='numpy')

        self._lengths = np.array([3, 4, 5, 6])
        self._state = np.zeros((n_envs, 1))

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        idxs = np.arange(self._n_envs)[env_mask]
        self._state[idxs, 0] = self._lengths[idxs]

        return self._state.copy(), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        self._state[env_mask, 0] -= 1
        reward = np.ones(self._n_envs)
        absorbing = self._state[:, 0] <= 0

        return self._state.copy(), reward, absorbing & env_mask, [{}] * self._n_envs


class RecordFlattenedLast(Callback):
    def __init__(self):
        self._flags = list()

    def __call__(self, dataset):
        if isinstance(dataset, VectorizedDataset):
            dataset = dataset.flatten()
        if dataset is None:
            return
        self._flags += dataset.array_backend.to_numpy(dataset.last).astype(bool).tolist()

    def get(self):
        return self._flags

    def clean(self):
        self._flags = list()


def test_callback_base_raises():
    callback = Callback()

    with pytest.raises(NotImplementedError):
        callback(None)
    with pytest.raises(NotImplementedError):
        callback.get()
    with pytest.raises(NotImplementedError):
        callback.clean()


def test_collect_dataset():
    np.random.seed(42)
    callback = CollectDataset()

    mdp = GridWorld.from_size(4, 4, (2, 2), goal_reward=10.)

    eps = Parameter(0.1)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.2)
    agent = SARSA(mdp.info, pi, alpha)

    last_flags = list()

    def record_last(sample):
        last_flags.append(bool(sample[5]))

    core = Core(agent, mdp, callbacks_fit=[callback], callback_step=record_last)

    core.learn(n_steps=10, n_steps_per_fit=1, quiet=True)

    dataset = callback.get()
    assert len(dataset) == 10
    core.learn(n_steps=5, n_steps_per_fit=1, quiet=True)
    dataset = callback.get()
    assert len(dataset) == 15

    assert np.array_equal(dataset.last, np.array(last_flags))
    assert dataset.n_episodes == sum(last_flags) + (0 if last_flags[-1] else 1)
    assert np.array_equal(dataset.episodes_length, np.array(episode_lengths_from_flags(last_flags)))

    callback.clean()
    dataset = callback.get()
    assert len(dataset) == 0


def test_collect_dataset_small_initial_capacity():
    np.random.seed(42)
    callback = CollectDataset(initial_capacity=2)

    mdp = GridWorld.from_size(4, 4, (2, 2), goal_reward=10.)

    eps = Parameter(0.1)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.2)
    agent = SARSA(mdp.info, pi, alpha)

    last_flags = list()

    def record_last(sample):
        last_flags.append(bool(sample[5]))

    core = Core(agent, mdp, callbacks_fit=[callback], callback_step=record_last)

    core.learn(n_steps=20, n_steps_per_fit=1, quiet=True)

    dataset = callback.get()
    assert len(dataset) == 20
    assert dataset.capacity >= 20
    assert np.array_equal(dataset.last, np.array(last_flags))
    assert dataset.n_episodes == sum(last_flags) + (0 if last_flags[-1] else 1)
    assert np.array_equal(dataset.episodes_length, np.array(episode_lengths_from_flags(last_flags)))


def test_collect_dataset_vectorized():
    np.random.seed(42)
    callback = CollectDataset(initial_capacity=2)
    reference = RecordFlattenedLast()

    env = CountdownVectorizedEnv()
    agent = CountdownAgent(env.info)

    core = Core(agent, env, callbacks_fit=[callback, reference])

    core.learn(n_episodes=40, n_episodes_per_fit=3, quiet=True)

    dataset = callback.get()
    reference_flags = reference.get()
    last = dataset.array_backend.to_numpy(dataset.last).astype(bool).tolist()

    assert last == reference_flags
    assert len(dataset) == len(reference_flags)
    assert dataset.n_episodes == sum(reference_flags) + (0 if reference_flags[-1] else 1)
    assert np.array_equal(dataset.episodes_length, np.array(episode_lengths_from_flags(reference_flags)))


def test_collect_Q():
    np.random.seed(42)
    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)

    eps = Parameter(0.1)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.1)
    agent = SARSA(mdp.info, pi, alpha)

    callback_q = CollectQ(agent.Q)
    callback_max_q = CollectMaxQ(agent.Q, np.array([2]))

    core = Core(agent, mdp, callbacks_fit=[callback_q, callback_max_q])

    core.learn(n_steps=1000, n_steps_per_fit=1, quiet=True)

    V_test = np.array([3.17651767, 6.45911513, 1.18433269, 0.78816385])
    V = callback_q.get()[-1]

    assert np.allclose(V[0, :], V_test)

    V_max = np.array([np.max(x[2, :], axis=-1) for x in callback_q.get()])
    max_q = np.array(callback_max_q.get())

    assert np.allclose(V_max, max_q)

    callback_q.clean()
    callback_max_q.clean()
    assert callback_q.get() == []
    assert callback_max_q.get() == []


def test_collect_parameter():
    np.random.seed(42)
    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)

    eps = DecayParameter(value=1, exp=.5, shape=mdp.info.observation_space.size)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.1)
    agent = SARSA(mdp.info, pi, alpha)

    callback_eps = CollectParameters(eps, 1)

    core = Core(agent, mdp, callbacks_fit=[callback_eps])

    core.learn(n_steps=30, n_steps_per_fit=1, quiet=True)

    eps_test = np.array([1.0] * 14 + [0.7071067811865475] * 10 + [0.5773502691896258] * 3 + [0.5] * 3)
    eps = callback_eps.get()

    assert np.allclose(eps, eps_test)

    callback_eps.clean()
    assert callback_eps.get() == []
