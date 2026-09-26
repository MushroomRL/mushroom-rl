import numpy as np
import pytest
import torch

from mushroom_rl.core import Agent, Core, Dataset, MDPInfo, VectorizedEnvironment
from mushroom_rl.core.spaces import Box
from mushroom_rl.core.dataset_info import DatasetInfo
from mushroom_rl.core.extra_info import ExtraInfo
from mushroom_rl.algorithms.value import SARSA
from mushroom_rl.environments import GridWorld
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.policy import EpsGreedy, StatefulPolicy

from mushroom_rl.core._impl.layout import StreamLayout, CodedLayout
from mushroom_rl.core._impl.history_state import HistoryState


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
    last_test = np.ones(2)
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
    metrics = dataset[:index].compute_metrics(mdp.info.gamma)
    assert metrics['min_J'] == 5.3144100000000005
    assert metrics['max_J'] == 6.561
    assert metrics['mean_J'] == 5.72994
    assert metrics['median_J'] == 5.3144100000000005
    assert metrics['n_episodes'] == 3


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

    assert metrics_list.keys() == metrics_numpy.keys()
    for name in metrics_numpy:
        assert np.allclose(metrics_list[name], metrics_numpy[name])


def test_compute_j_skips_incomplete_episode():
    states = np.arange(12).reshape(6, 2).astype(float)
    actions = np.zeros((6, 1))
    rewards = np.ones(6)
    next_states = states + 1
    absorbings = np.zeros(6, dtype=bool)
    lasts = np.array([False, False, True, False, False, False])

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts, gamma=0.9)

    assert np.array_equal(dataset.compute_J(), np.array([3.0]))
    assert np.array_equal(dataset.compute_J(skip_incomplete=False), np.array([3.0, 3.0]))
    assert np.array_equal(dataset.undiscounted_return, np.array([3.0]))
    assert dataset.n_episodes == 1
    assert np.array_equal(dataset.episodes_length, np.array([3]))
    assert dataset.compute_metrics() == dict(min_J=3.0, max_J=3.0, mean_J=3.0, median_J=3.0, n_episodes=1)


def test_compute_j_without_complete_episodes():
    states = np.arange(6).reshape(3, 2).astype(float)
    actions = np.zeros((3, 1))
    rewards = np.ones(3)
    next_states = states + 1
    absorbings = np.zeros(3, dtype=bool)
    lasts = np.zeros(3, dtype=bool)

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts, gamma=0.9)

    assert len(dataset.compute_J()) == 0
    assert len(dataset.undiscounted_return) == 0
    assert np.array_equal(dataset.compute_J(skip_incomplete=False), np.array([3.0]))
    assert dataset.n_episodes == 0
    assert len(dataset.episodes_length) == 0
    assert dataset.compute_metrics() == dict(n_episodes=0)


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


def test_list_dataset_parse_to_torch():
    states = np.arange(12).reshape(6, 2).astype(float)
    actions = np.zeros((6, 1))
    rewards = np.ones(6)
    next_states = states + 1
    absorbings = np.zeros(6)
    lasts = np.array([0, 0, 1, 0, 0, 1])

    list_dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                      backend='list', gamma=0.9)
    numpy_dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                       gamma=0.9)

    for from_list, from_numpy in zip(list_dataset.parse(to='torch'), numpy_dataset.parse(to='torch')):
        assert from_list.dtype == from_numpy.dtype
        assert torch.equal(from_list, from_numpy)

    for from_list, from_numpy in zip(list_dataset.parse(to='numpy'), numpy_dataset.parse(to='numpy')):
        assert from_list.dtype == from_numpy.dtype
        assert np.array_equal(from_list, from_numpy)


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


def test_dataset_add_leaves_last_untouched():
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
    assert np.array_equal(result.last, np.zeros(2 * n, dtype=bool))
    assert np.array_equal(a.last, np.zeros(n, dtype=bool))
    assert np.array_equal(b.last, np.zeros(n, dtype=bool))


def build_info_dataset(n, first_reward, capacity):
    states = np.arange(n * 2).reshape(n, 2).astype(float)
    actions = np.arange(n).reshape(n, 1).astype(float)
    rewards = np.arange(n).astype(float) + first_reward
    next_states = states + 1
    absorbings = np.zeros(n, dtype=bool)
    lasts = np.zeros(n, dtype=bool)
    lasts[-1] = True

    extras = ExtraInfo(1, 'numpy')
    for i in range(n):
        extras.append_step({'idx': float(i + first_reward)})
    extras.append_episode({'ep': float(first_reward)})
    extras.append_theta(np.array([float(first_reward)]))

    dataset = Dataset.from_array(states, actions, rewards, next_states, absorbings, lasts,
                                 extras=extras, gamma=0.9)
    dataset.reserve(capacity)
    return dataset


def test_dataset_iadd_matches_add_in_place():
    a = build_info_dataset(3, 0.0, capacity=8)
    b = build_info_dataset(2, 10.0, capacity=8)

    reference = a + b

    a += b

    assert len(a) == 5
    assert np.array_equal(a.last, reference.last)
    assert np.array_equal(a.reward, reference.reward)
    assert np.array_equal(a.info['idx'], reference.info['idx'])
    assert np.array_equal(a.episode_info['ep'], reference.episode_info['ep'])
    assert np.array_equal(np.array(a.theta_list), np.array(reference.theta_list))


def test_dataset_iadd_in_place_when_capacity_available():
    a = build_info_dataset(3, 0.0, capacity=8)
    b = build_info_dataset(2, 10.0, capacity=8)

    original = a
    a += b

    assert a is original
    assert len(a) == 5


def test_dataset_iadd_falls_back_when_over_capacity():
    a = build_info_dataset(3, 0.0, capacity=3)
    b = build_info_dataset(2, 10.0, capacity=3)

    reference = a + b

    original = a
    a += b

    assert a is not original
    assert len(a) == 5
    assert np.array_equal(a.last, reference.last)
    assert np.array_equal(a.reward, reference.reward)
    assert np.array_equal(a.info['idx'], reference.info['idx'])
    assert np.array_equal(a.episode_info['ep'], reference.episode_info['ep'])


def test_dataset_capacity_and_reserve():
    a = build_info_dataset(3, 0.0, capacity=4)

    assert a.capacity == 4

    a.reserve(2)
    assert a.capacity == 4

    a.reserve(16)
    assert a.capacity == 16
    assert len(a) == 3
    assert np.array_equal(a.reward, np.array([0.0, 1.0, 2.0]))


def test_dataset_capacity_none_for_list_backend():
    states = np.arange(6).reshape(3, 2).astype(float)
    actions = np.zeros((3, 1))
    rewards = np.ones(3)
    absorbings = np.zeros(3, dtype=bool)
    lasts = np.array([False, False, True])

    dataset = Dataset.from_array(states, actions, rewards, states, absorbings, lasts,
                                 backend='list', gamma=0.9)

    assert dataset.capacity is None
    dataset.reserve(100)
    assert dataset.capacity is None
    assert len(dataset) == 3


def test_dataset_reserve_grows_agent_data():
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

    assert dataset.capacity == n

    dataset.reserve(16)

    assert dataset.capacity == 16
    assert dataset._agent_data.capacity == 16
    assert torch.equal(dataset.policy_state, torch.from_numpy(policy_states))
    assert torch.equal(dataset.policy_next_state, torch.from_numpy(policy_next_states))


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


def test_to_backend_converts_the_extra_info():
    extras = ExtraInfo(1, 'torch')
    extras.append_step({'x': torch.tensor(1.)})
    extras.append_step({'x': torch.tensor(2.)})

    n_steps = 2
    state = torch.zeros(n_steps, 1)
    action = torch.zeros(n_steps, 1)
    reward = torch.zeros(n_steps)
    next_state = torch.zeros(n_steps, 1)
    absorbing = torch.zeros(n_steps, dtype=torch.bool)
    last = torch.zeros(n_steps, dtype=torch.bool)
    last[-1] = True

    dataset = Dataset.from_array(state, action, reward, next_state, absorbing, last, extras=extras, backend='torch')
    converted = dataset.to_backend('numpy')

    assert isinstance(converted.state, np.ndarray)
    assert isinstance(converted.info['x'], np.ndarray)
    assert np.array_equal(converted.info['x'], np.array([1., 2.]))


def make_stream(states, lasts, continuing=False):
    n = len(states)
    states = np.array(states, dtype=float)[:, None]
    return Dataset.from_array(states, np.zeros((n, 1)), np.zeros(n), states + 0.5, np.zeros(n, dtype=bool),
                              np.array(lasts, dtype=bool), gamma=0.5, continuing=continuing)


def test_walk_back_and_forward_stop_at_the_episode_ends():
    dataset = make_stream([0, 1, 2, 3, 4, 5], [0, 0, 1, 0, 0, 1])

    back, back_valid = dataset.walk_back(np.array([1, 4, 5]), 3)
    forward, forward_valid = dataset.walk_forward(np.array([0, 3]), 3)

    assert np.array_equal(back, np.array([[1, 0, 0, 0], [4, 3, 3, 3], [5, 4, 3, 3]]))
    assert np.array_equal(back_valid, np.array([[True, True, False, False], [True, True, False, False],
                                                [True, True, True, False]]))
    assert np.array_equal(forward, np.array([[0, 1, 2, 2], [3, 4, 5, 5]]))
    assert np.array_equal(forward_valid, np.array([[True, True, True, False], [True, True, True, False]]))


def test_boundary_codes_are_stored_only_for_a_break():
    stitched = make_stream([0, 1], [0, 0]) + make_stream([2], [1], continuing=True)
    closed_then_fresh = make_stream([0, 1], [0, 1]) + make_stream([2], [1])
    open_then_fresh = make_stream([0, 1], [0, 0]) + make_stream([2], [1])

    assert isinstance(make_stream([0, 1, 2], [0, 0, 1])._layout, StreamLayout)
    assert isinstance(stitched._layout, StreamLayout)
    assert isinstance(closed_then_fresh._layout, StreamLayout)
    assert isinstance(open_then_fresh._layout, CodedLayout)
    assert np.array_equal(stitched.last_or_boundary, np.array([False, False, True]))
    assert np.array_equal(closed_then_fresh.last_or_boundary, np.array([False, True, True]))
    assert np.array_equal(open_then_fresh.last_or_boundary, np.array([False, True, True]))


def test_contiguous_without_joins_is_the_dataset_itself():
    dataset = make_stream([0, 1, 2], [0, 0, 1])
    stitched = make_stream([0, 1], [0, 0]) + make_stream([2], [1], continuing=True)

    assert dataset.contiguous() is dataset
    assert stitched.contiguous() is stitched


def test_to_backend_keeps_the_horizon_and_the_discount_factor():
    dataset = Dataset.from_array(np.zeros((3, 2)), np.zeros((3, 1)), np.ones(3), np.zeros((3, 2)),
                                 np.zeros(3, dtype=bool), np.array([False, False, True]), horizon=50, gamma=0.5)

    converted = dataset.to_backend('torch')

    assert converted._dataset_info.horizon == 50
    assert converted._dataset_info.gamma == 0.5
    assert torch.allclose(converted.discounted_return, torch.tensor([1.75]))


def test_to_backend_returns_the_dataset_when_the_resolved_device_matches():
    dataset = make_stream([0, 1, 2], [0, 0, 1]).to_backend('torch')

    assert dataset.to_backend('torch', device='cpu') is dataset
    assert dataset.to_backend('torch') is dataset


def make_list_stream(states, lasts, continuing=False):
    n = len(states)
    states = list(np.array(states, dtype=float)[:, None])
    return Dataset.from_array(states, [np.zeros(1)] * n, list(np.zeros(n)), states, [False] * n,
                              [bool(last) for last in lasts], gamma=0.5, backend='list', continuing=continuing)


def test_contiguous_of_a_joined_list_dataset():
    joined = make_list_stream([1, 2, 3], [0, 1, 0]) + make_list_stream([5, 6], [1, 0])

    glued = joined.contiguous()

    assert np.array_equal(np.array(glued.state)[:, 0], np.array([1., 2., 3., 5., 6.]))
    assert np.array_equal(glued.last_or_boundary, np.array([False, True, True, True, True]))


def test_list_dataset_converted_to_torch_keeps_int8_boundary_codes():
    joined = make_list_stream([0, 1], [0, 0]) + make_list_stream([2, 3], [0, 1])

    converted = joined.to_backend('torch')
    glued = converted.contiguous()

    assert converted._layout.array().dtype == torch.int8
    assert torch.equal(glued.state[:, 0], torch.tensor([0., 1., 2., 3.]))


def test_history_state_takes_row_indices_from_another_device():
    if not torch.cuda.is_available():
        return

    entries = HistoryState('numpy', None, np.array([0, 3]), {'obs_history': np.array([[1.], [2.]])})

    view = entries.get_view(torch.tensor([3, 1], device='cuda'), 5)
    kept = entries.drop(torch.tensor([0], device='cuda'), 5)

    assert np.array_equal(view.positions, np.array([0]))
    assert np.array_equal(view.windows('obs_history'), np.array([[2.]]))
    assert np.array_equal(kept.positions, np.array([3]))
    assert np.array_equal(kept.windows('obs_history'), np.array([[2.]]))


class CudaCountingVecEnv(VectorizedEnvironment):
    def __init__(self):
        super().__init__(MDPInfo(Box(-1000, 1000, shape=(1,)), Box(-1000, 1000, shape=(1,)), 0.9, 100,
                                 backend='torch', device='cuda'), 3)
        self._s = torch.zeros(3, 1, device='cuda')
        self._t = torch.zeros(3, device='cuda')

    def reset_all(self, env_mask, state=None):
        self._s[env_mask] = 100. * (1 + torch.arange(3, device='cuda')[env_mask, None].float())
        self._t[env_mask] = 0
        return self._s.clone(), [{}] * 3

    def step_all(self, env_mask, action):
        self._s[env_mask] += 1
        self._t[env_mask] += 1
        return self._s.clone(), torch.ones(3, device='cuda'), (self._t >= 4) & env_mask, [{}] * 3


class StateTrackingPolicy(StatefulPolicy):
    def __init__(self):
        super().__init__((1,))

    def reset_vectorized(self, start_mask):
        self._policy_state = np.zeros((len(start_mask), 1))
        return self._policy_state

    def _draw_action(self, state, policy_state, **kwargs):
        return state[:, :1], state[:, :1].copy()


class FitCollectingAgent(Agent):
    def __init__(self, mdp_info):
        super().__init__(mdp_info, StateTrackingPolicy(), backend='numpy')
        self.fits = list()

    def fit(self, dataset):
        self.fits.append(dataset)


def test_policy_states_follow_row_indices_from_the_env_device():
    if not torch.cuda.is_available():
        return

    env = CudaCountingVecEnv()
    agent = FitCollectingAgent(env.info)
    Core(agent, env).learn(n_steps=30, n_steps_per_fit=7, quiet=True)

    view = agent.fits[1][torch.tensor([3, 0], device='cuda')]
    glued = (agent.fits[1] + agent.fits[2]).contiguous()

    assert np.array_equal(view.policy_next_state[:, 0], np.array([203., 103.]))
    assert torch.equal(glued.state[:, 0].cpu(), torch.tensor([103., 100., 101., 102., 202., 203., 200., 201., 202.,
                                                              302., 303., 300., 301., 302.]))
    assert np.array_equal(glued.policy_next_state[:, 0], glued.state[:, 0].cpu().numpy())


def test_history_state_takes_boolean_row_masks():
    entries = HistoryState('torch', None, torch.tensor([0, 3]), {'obs_history': torch.tensor([[1.], [2.]])})

    torch_view = entries.get_view(torch.tensor([False, True, False, True, True]), 5)
    numpy_view = entries.get_view(np.array([False, True, False, True, True]), 5)

    for view in (torch_view, numpy_view):
        assert torch.equal(view.positions, torch.tensor([1]))
        assert torch.equal(view.windows('obs_history'), torch.tensor([[2.]]))


def test_integer_index_reads_only_the_stored_steps():
    info = DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None, horizon=10,
                       gamma=0.9, state_shape=(1,), state_dtype=np.float64, action_shape=(1,),
                       action_dtype=np.float64, policy_state_shape=None)
    dataset = Dataset(info, n_steps=10)
    for value in (1., 2., 3.):
        dataset.append((np.array([value]), np.zeros(1), value, np.array([value]), False, value == 3.), {})

    assert dataset[-1][0][0] == 3.
    assert dataset[-3][0][0] == 1.
    with pytest.raises(IndexError):
        dataset[3]
    with pytest.raises(IndexError):
        dataset[-4]
