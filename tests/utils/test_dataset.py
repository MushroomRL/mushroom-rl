from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.environments import GridWorld
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy

from mushroom_rl.utils.dataset import *


def test_dataset_utils():
    np.random.seed(88)

    mdp = GridWorld(3, 3, (2,2))
    epsilon = Parameter(value=0.)
    alpha = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    agent = SARSA(mdp.info, pi, alpha)
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=10)

    J = compute_J(dataset, mdp.info.gamma)
    J_test = np.array([1.16106307e-03, 2.78128389e-01, 1.66771817e+00, 3.09031544e-01,
                       1.19725152e-01, 9.84770902e-01, 1.06111661e-02, 2.05891132e+00,
                       2.28767925e+00, 4.23911583e-01])
    assert np.allclose(J, J_test)

    L = compute_episodes_length(dataset)
    L_test = np.array([87, 35, 18, 34, 43, 23, 66, 16, 15, 31])
    assert np.array_equal(L, L_test)

    dataset_ep = select_first_episodes(dataset, 3)
    J = compute_J(dataset_ep, mdp.info.gamma)
    assert np.allclose(J, J_test[:3])

    L = compute_episodes_length(dataset_ep)
    assert np.allclose(L, L_test[:3])

    samples = select_random_samples(dataset, 2)
    s, a, r, ss, ab, last = parse_dataset(samples)
    s_test = np.array([[6.], [1.]])
    a_test = np.array([[0.], [1.]])
    r_test = np.zeros(2)
    ss_test = np.array([[3], [4]])
    ab_test = np.zeros(2)
    last_test = np.zeros(2)
    assert np.array_equal(s, s_test)
    assert np.array_equal(a, a_test)
    assert np.array_equal(r, r_test)
    assert np.array_equal(ss, ss_test)
    assert np.array_equal(ab, ab_test)
    assert np.array_equal(last, last_test)

    s0 = get_init_states(dataset)
    s0_test = np.zeros((10, 1))
    assert np.array_equal(s0, s0_test)

    index = np.sum(L_test[:2]) + L_test[2]//2
    min_J, max_J, mean_J, n_episodes = compute_metrics(dataset[:index], mdp.info.gamma)
    assert min_J == 0.0011610630703530948
    assert max_J == 0.2781283894436937
    assert mean_J == 0.1396447262570234
    assert n_episodes == 2


