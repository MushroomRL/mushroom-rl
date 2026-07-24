from mushroom_rl.core import Core
from mushroom_rl.environments import GridWorld
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter
from mushroom_rl.utils.callbacks import CollectDataset, CollectQ, CollectMaxQ, CollectParameters
import numpy as np


def episode_lengths_from_flags(last_flags):
    lengths = list()
    length = 0
    for flag in last_flags:
        length += 1
        if flag:
            lengths.append(length)
            length = 0
    return lengths


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
