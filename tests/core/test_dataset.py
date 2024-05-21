import numpy as np
import torch

from mushroom_rl.core import Core, Dataset
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.environments import GridWorld
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy


def generate_dataset(mdp, n_episodes):
    epsilon = Parameter(value=0.)
    alpha = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    agent = SARSA(mdp.info, pi, alpha)
    core = Core(agent, mdp)

    return core.evaluate(n_episodes=n_episodes)


def test_dataset():
    np.random.seed(42)
    mdp = GridWorld(3, 3, (2, 2))
    dataset = generate_dataset(mdp, 10)

    assert dataset.n_episodes == 10

    J = dataset.compute_J(mdp.info.gamma)
    J_test = np.array([4.304672100000001, 2.287679245496101, 3.138105960900001,  0.13302794647291147,
                       7.290000000000001,   1.8530201888518416, 1.3508517176729928, 0.011790184577738602,
                       1.3508517176729928, 7.290000000000001])
    assert np.allclose(J, J_test)

    L = dataset.episodes_length
    L_test = np.array([9, 15, 12, 42, 4, 17, 20, 65, 20, 4])
    assert np.array_equal(L, L_test)

    dataset_ep = dataset.select_first_episodes(3)
    J = dataset_ep.compute_J(mdp.info.gamma)
    assert np.allclose(J, J_test[:3])

    L = dataset_ep.episodes_length
    assert np.allclose(L, L_test[:3])

    samples = dataset.select_random_samples(2)
    s, a, r, ss, ab, last = samples.parse()
    s_test = np.array([[5.], [6.]])
    a_test = np.array([[3.], [0.]])
    r_test = np.zeros(2)
    ss_test = np.array([[5], [3]])
    ab_test = np.zeros(2)
    last_test = np.zeros(2)
    assert np.array_equal(s, s_test)
    assert np.array_equal(a, a_test)
    assert np.array_equal(r, r_test)
    assert np.array_equal(ss, ss_test)
    assert np.array_equal(ab, ab_test)
    assert np.array_equal(last, last_test)

    s0 = dataset.get_init_states()
    s0_test = np.zeros((10, 1))
    assert np.array_equal(s0, s0_test)

    index = np.sum(L_test[:2]) + L_test[2]//2
    min_J, max_J, mean_J, median_J, n_episodes = dataset[:index].compute_metrics(mdp.info.gamma)
    assert min_J == 2.287679245496101
    assert max_J == 4.304672100000001
    assert mean_J == 3.296175672748051
    assert median_J == 3.296175672748051
    assert n_episodes == 2


def test_dataset_creation():
    np.random.seed(42)

    mdp = GridWorld(3, 3, (2, 2))
    dataset = generate_dataset(mdp, 5)

    parsed = tuple(dataset.parse())
    parsed_torch = (torch.from_numpy(array) for array in parsed)

    print(len(parsed))

    new_numpy_dataset = Dataset.from_array(*parsed, gamma=mdp.info.gamma)
    new_list_dataset = Dataset.from_array(*parsed, gamma=mdp.info.gamma, backend='list')
    new_torch_dataset = Dataset.from_array(*parsed, gamma=mdp.info.gamma, backend='torch')

    assert vars(dataset).keys() == vars(new_numpy_dataset).keys()
    assert vars(dataset).keys() == vars(new_list_dataset).keys()
    assert vars(dataset).keys() == vars(new_torch_dataset).keys()

    assert new_numpy_dataset.n_episodes == dataset.n_episodes
    assert new_list_dataset.n_episodes == dataset.n_episodes
    assert new_torch_dataset.n_episodes == dataset.n_episodes

    for array_1, array_2 in zip(parsed, new_numpy_dataset.parse()):
        assert np.array_equal(array_1, array_2)

    for array_1, array_2 in zip(parsed, new_list_dataset.parse(to='numpy')):
        assert np.array_equal(array_1, array_2)

    for array_1, array_2 in zip(parsed_torch, new_torch_dataset.parse(to='torch')):
        assert torch.equal(array_1, array_2)


def test_dataset_loading(tmpdir):
    np.random.seed(42)

    mdp = GridWorld(3, 3, (2, 2))
    dataset = generate_dataset(mdp, 20)

    path = tmpdir / 'dataset_test.msh'
    dataset.save(path)

    new_dataset = dataset.load(path)

    assert vars(dataset).keys() == vars(new_dataset).keys()

    assert np.array_equal(dataset.state, new_dataset.state) and \
            np.array_equal(dataset.action, new_dataset.action) and \
            np.array_equal(dataset.reward, new_dataset.reward) and \
            np.array_equal(dataset.next_state, new_dataset.next_state) and \
            np.array_equal(dataset.absorbing, new_dataset.absorbing) and \
            np.array_equal(dataset.last, new_dataset.last)

    assert dataset._dataset_info.gamma == new_dataset._dataset_info.gamma


