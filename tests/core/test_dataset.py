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
    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)
    dataset = generate_dataset(mdp, 10)

    assert dataset.n_episodes == 10

    J = dataset.compute_J(mdp.info.gamma)
    J_test = np.array([5.3144100000000005, 5.3144100000000005, 6.561, 0.25031555049932436,
                       1.6677181699666577, 3.486784401000001, 1.0941898913151242, 3.874204890000001,
                       1.5009463529699918, 0.033813919135227306])
    assert np.allclose(J, J_test)

    L = dataset.episodes_length
    L_test = np.array([7, 7, 5, 36, 18, 11, 22, 10, 19, 55])
    assert np.array_equal(L, L_test)

    dataset_ep = dataset.select_first_episodes(3)
    J = dataset_ep.compute_J(mdp.info.gamma)
    assert np.allclose(J, J_test[:3])

    L = dataset_ep.episodes_length
    assert np.allclose(L, L_test[:3])

    samples = dataset.select_random_samples(2)
    s, a, r, ss, ab, last = samples.parse()
    s_test = np.array([[1.], [2.]])
    a_test = np.array([[0.], [3.]])
    r_test = np.zeros(2)
    ss_test = np.array([[1], [2]])
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

    index = np.sum(L_test[:3]) + L_test[3]//2
    min_J, max_J, mean_J, median_J, n_episodes = dataset[:index].compute_metrics(mdp.info.gamma)
    assert min_J == 5.3144100000000005
    assert max_J == 6.561
    assert mean_J == 5.72994
    assert median_J == 5.3144100000000005
    assert n_episodes == 3


def test_dataset_creation():
    np.random.seed(42)

    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)
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

    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)
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

    assert len(dataset.info) == len(new_dataset.info)
    for key in dataset.info:
        assert np.array_equal(dataset.info[key], new_dataset.info[key])


def test_list_dataset_compute_j_metrics():
    np.random.seed(42)

    mdp = GridWorld.from_size(3, 3, (2, 2), goal_reward=10.)
    dataset = generate_dataset(mdp, 5)

    parsed = tuple(dataset.parse())
    list_dataset = Dataset.from_array(*parsed, gamma=mdp.info.gamma, backend='list')

    assert np.allclose(list_dataset.compute_J(), dataset.compute_J())
    assert np.allclose(list_dataset.compute_J(mdp.info.gamma), dataset.compute_J(mdp.info.gamma))

    metrics_numpy = dataset.compute_metrics(mdp.info.gamma)
    metrics_list = list_dataset.compute_metrics(mdp.info.gamma)

    assert np.allclose(metrics_list[:4], metrics_numpy[:4])
    assert metrics_list[4] == metrics_numpy[4]


def test_from_array_list_backend():
    states = np.arange(12).reshape(6, 2).astype(float)
    actions = np.zeros((6, 1))
    rewards = np.ones(6)
    next_states = states + 1
    absorbings = np.zeros(6)
    lasts = np.array([0, 0, 1, 0, 0, 1])

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                 backend='list', gamma=0.9)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert len(dataset) == 6
    assert dataset._dataset_info.state_shape == (2,)
    assert dataset._dataset_info.action_shape == (1,)
    assert not dataset.is_stateful
    assert np.array_equal(dataset.compute_J(), np.array([3.0, 3.0]))

    for original, restored in zip((states, actions, rewards, next_states, absorbings, lasts),
                                  dataset.parse(to='numpy')):
        assert np.array_equal(original, restored)


def test_from_array_list_backend_stateful():
    states = np.arange(8).reshape(4, 2).astype(float)
    actions = np.zeros((4, 1))
    rewards = np.ones(4)
    next_states = states + 1
    absorbings = np.zeros(4)
    lasts = np.array([0, 0, 0, 1])
    policy_states = np.arange(4).reshape(4, 1).astype(float)
    policy_next_states = policy_states + 1

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                 policy_state=policy_states, policy_next_state=policy_next_states,
                                 backend='list', gamma=0.9)

    assert dataset.is_stateful
    assert dataset._dataset_info.policy_state_shape == (1,)
    assert np.array_equal(np.array(dataset.policy_state), policy_states)
    assert np.array_equal(np.array(dataset.policy_next_state), policy_next_states)


def test_from_array_list_backend_ragged():
    states = [np.array([0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])]
    actions = [{'a': 0}, {'a': 1}, {'a': 2}]
    rewards = [1.0, 1.0, 1.0]
    absorbings = [False, False, True]
    lasts = [False, False, True]

    dataset = Dataset.from_array(states, actions, rewards, states, absorbings, lasts,
                                 backend='list', gamma=0.9)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 1
    assert dataset._dataset_info.state_shape == (1,)
    assert dataset._dataset_info.action_shape == ()
    assert np.array_equal(dataset.state[1], np.array([0.0, 1.0]))
    assert dataset.action[2] == {'a': 2}
    assert np.array_equal(dataset.compute_J(), np.array([3.0]))


def test_dataset_policy_backend_split():
    n = 4
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float)
    next_states = states + 1
    absorbings = np.zeros(n, dtype=bool)
    lasts = np.array([False, False, False, True])
    policy_states = np.arange(n).reshape(n, 1).astype(float)
    policy_next_states = policy_states + 1

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                 policy_state=policy_states, policy_next_state=policy_next_states,
                                 backend='numpy', policy_backend='torch', gamma=0.9)

    assert dataset.is_stateful
    assert isinstance(dataset.state, np.ndarray)
    assert isinstance(dataset.policy_state, torch.Tensor)

    s, a, r, ss, ab, last = dataset.parse()
    assert isinstance(s, np.ndarray)

    ps, pns = dataset.parse_policy_state()
    assert isinstance(ps, torch.Tensor)
    assert torch.equal(ps, torch.from_numpy(policy_states))
    assert torch.equal(pns, torch.from_numpy(policy_next_states))

    ps_np, pns_np = dataset.parse_policy_state(to='numpy')
    assert isinstance(ps_np, np.ndarray)
    assert np.array_equal(ps_np, policy_states)


def test_dataset_add_marks_boundary_last():
    n = 3
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float)
    absorbings = np.zeros(n, dtype=bool)
    lasts = np.zeros(n, dtype=bool)

    a = Dataset.from_array(states, actions, rewards, states, absorbings, lasts, gamma=0.9)
    b = Dataset.from_array(states, actions, rewards, states, absorbings, lasts, gamma=0.9)

    result = a + b

    assert len(result) == 2 * n
    assert bool(result.last[n - 1])
    assert not bool(a.last[n - 1])


def test_dataset_save_load_policy_split(tmpdir):
    n = 5
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float)
    next_states = states + 1
    absorbings = np.zeros(n, dtype=bool)
    lasts = np.array([False, False, True, False, True])
    policy_states = np.arange(n).reshape(n, 1).astype(float)
    policy_next_states = policy_states + 1

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                 policy_state=policy_states, policy_next_state=policy_next_states,
                                 backend='numpy', policy_backend='torch', gamma=0.9)

    path = tmpdir / 'dataset_split.msh'
    dataset.save(path)
    new_dataset = Dataset.load(path)

    assert vars(dataset).keys() == vars(new_dataset).keys()
    assert new_dataset.is_stateful
    assert isinstance(new_dataset.policy_state, torch.Tensor)
    assert np.array_equal(new_dataset.state, states)
    assert torch.equal(new_dataset.policy_state, torch.from_numpy(policy_states))
    assert new_dataset.n_episodes == 2
