from mushroom_rl.core import Core
from mushroom_rl.environments import GridWorld
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, DecayParameter
from mushroom_rl.utils.callbacks import *
import numpy as np


def test_collect_dataset():
    np.random.seed(42)
    callback = CollectDataset()

    mdp = GridWorld(4, 4, (2, 2))

    eps = Parameter(0.1)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.2)
    agent = SARSA(mdp.info, pi, alpha)

    core = Core(agent, mdp, callbacks_fit=[callback])

    core.learn(n_steps=10, n_steps_per_fit=1, quiet=True)

    dataset = callback.get()
    assert len(dataset) == 10
    core.learn(n_steps=5, n_steps_per_fit=1, quiet=True)
    dataset = callback.get()
    assert len(dataset) == 15

    callback.clean()
    dataset = callback.get()
    assert len(dataset) == 0


def test_collect_Q():
    np.random.seed(42)
    mdp = GridWorld(3, 3, (2, 2))

    eps = Parameter(0.1)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.1)
    agent = SARSA(mdp.info, pi, alpha)

    callback_q = CollectQ(agent.Q)
    callback_max_q = CollectMaxQ(agent.Q, np.array([2]))

    core = Core(agent, mdp, callbacks_fit=[callback_q, callback_max_q])

    core.learn(n_steps=1000, n_steps_per_fit=1, quiet=True)

    V_test = np.array([2.89138932, 0.59676854, 1.83726064, 5.85391026])
    V = callback_q.get()[-1]

    assert np.allclose(V[0, :], V_test)

    V_max = np.array([np.max(x[2, :], axis=-1) for x in callback_q.get()])
    max_q = np.array(callback_max_q.get())

    assert np.allclose(V_max, max_q)


def test_collect_parameter():
    np.random.seed(42)
    mdp = GridWorld(3, 3, (2, 2))

    eps = DecayParameter(value=1, exp=.5,
                         size=mdp.info.observation_space.size)
    pi = EpsGreedy(eps)
    alpha = Parameter(0.1)
    agent = SARSA(mdp.info, pi, alpha)

    callback_eps = CollectParameters(eps, 1)

    core = Core(agent, mdp, callbacks_fit=[callback_eps])

    core.learn(n_steps=10, n_steps_per_fit=1, quiet=True)

    eps_test = np.array([1.0, 1.0, 1.0, 1.0, 0.7071067811865475, 0.7071067811865475, 0.7071067811865475,
                         0.7071067811865475, 0.7071067811865475, 0.5773502691896258])
    eps = callback_eps.get()

    print(eps)

    assert np.allclose(eps, eps_test)
