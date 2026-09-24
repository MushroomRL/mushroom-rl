import numpy as np
import torch

from mushroom_rl.core.dataset import DatasetInfo, VectorizedDataset


def make_info():
    return DatasetInfo(env_backend='numpy', agent_backend='torch', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(2,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=(1,), n_envs=2)


def append_steps(dataset, n_steps):
    mask = np.array([True, True])
    for t in range(n_steps):
        state = np.full((2, 2), float(t))
        action = np.full((2, 1), float(t))
        reward = np.ones(2)
        next_state = state + 1
        absorbing = np.zeros(2, dtype=bool)
        last = np.zeros(2, dtype=bool)
        policy_state = torch.full((2, 1), float(t))
        policy_next_state = policy_state + 1
        step = (state, action, reward, next_state, absorbing, last, policy_state, policy_next_state)
        dataset.append_vectorized(step, [{}, {}], mask)


def test_vectorized_dataset_flatten_cross_backend():
    dataset = VectorizedDataset(make_info(), n_steps=10)
    append_steps(dataset, 3)

    assert isinstance(dataset.mask, np.ndarray)
    assert dataset.mask.shape == (3, 2)

    flat = dataset.flatten()

    assert len(flat) == 6
    assert isinstance(flat.state, np.ndarray)
    assert isinstance(flat.policy_state, torch.Tensor)
    assert flat.policy_state.shape == (6, 1)
    assert flat.is_stateful


def test_vectorized_dataset_clear_residual_carry():
    dataset = VectorizedDataset(make_info(), n_steps=10)
    append_steps(dataset, 3)

    dataset.flatten(5)
    n_carry = dataset.clear(keep_leftovers=True)

    assert int(n_carry) == 1
    assert len(dataset) == 1
    assert dataset._agent_data is not None
    assert dataset.mask.sum() == 1
    assert np.array_equal(dataset.mask, np.array([[False, True]]))
    assert np.array_equal(dataset.state[0], np.full((2, 2), 2.0))


def test_vectorized_dataset_flatten_keeps_last_untouched_at_block_ends():
    info = DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None, n_envs=3)
    dataset = VectorizedDataset(info, n_steps=12)

    active_steps = [4, 2, 3]
    for t in range(4):
        mask = np.array([t < steps for steps in active_steps])
        step = (np.full((3, 1), float(t)), np.zeros((3, 1)), np.arange(3) + 10. * t,
                np.full((3, 1), t + 1.), np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))
        dataset.append_vectorized(step, [{}] * 3, mask)

    flat = dataset.flatten()

    assert np.array_equal(np.asarray(flat.reward), np.array([0., 10., 20., 30., 1., 11., 2., 12., 22.]))
    assert np.array_equal(np.asarray(flat.last).astype(bool), np.zeros(9, dtype=bool))


def test_vectorized_dataset_flatten_keeps_episode_ends_inside_a_block():
    info = DatasetInfo(env_backend='numpy', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=10, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=None, n_envs=3)
    dataset = VectorizedDataset(info, n_steps=12)

    active_steps = [4, 2, 3]
    episode_ends = {(1, 0), (0, 2)}
    for t in range(4):
        mask = np.array([t < steps for steps in active_steps])
        last = np.array([(t, e) in episode_ends for e in range(3)])
        step = (np.full((3, 1), float(t)), np.zeros((3, 1)), np.arange(3) + 10. * t,
                np.full((3, 1), t + 1.), np.zeros(3, dtype=bool), last)
        dataset.append_vectorized(step, [{}] * 3, mask)

    flat = dataset.flatten()

    assert np.array_equal(np.asarray(flat.last).astype(bool),
                          np.array([False, True, False, False, False, False, True, False, False]))


def make_infinite_horizon_info():
    return DatasetInfo(env_backend='list', agent_backend='numpy', env_device=None, agent_device=None,
                       horizon=np.inf, gamma=0.9, state_shape=(1,), state_dtype=np.float64,
                       action_shape=(1,), action_dtype=np.float64, policy_state_shape=(1,), n_envs=2)


def append_stateful_steps(dataset, n_steps):
    mask = np.array([True, True])
    for t in range(n_steps):
        values = 10. * np.arange(2)[:, None] + t
        step = (values, np.zeros((2, 1)), np.ones(2), values + 1, np.zeros(2, dtype=bool), np.zeros(2, dtype=bool),
                values, values + 0.5)
        dataset.append_vectorized(step, [{}, {}], mask)


def test_infinite_horizon_flatten_keeps_the_policy_state_of_every_row():
    dataset = VectorizedDataset(make_infinite_horizon_info(), n_steps=10)
    append_stateful_steps(dataset, 3)

    flat = dataset.flatten()
    policy_state, policy_next_state = flat.parse_policy_state()

    assert np.array_equal(np.asarray(flat.state)[:, 0], np.array([0., 1., 2., 10., 11., 12.]))
    assert np.array_equal(policy_state[:, 0], np.array([0., 1., 2., 10., 11., 12.]))
    assert np.array_equal(policy_next_state[:, 0], np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]))


def test_join_with_an_empty_block_is_a_no_op():
    dataset = VectorizedDataset(make_infinite_horizon_info(), n_steps=10)
    append_stateful_steps(dataset, 3)
    flat = dataset.flatten()
    empty = VectorizedDataset(make_infinite_horizon_info(), n_steps=10).flatten()

    accumulated = empty
    accumulated += flat
    extended = flat.copy()
    extended += empty

    for joined in (flat + empty, empty + flat, accumulated, extended):
        assert np.array_equal(np.asarray(joined.state), np.asarray(flat.state))
        assert np.array_equal(joined.parse_policy_state()[0], flat.parse_policy_state()[0])
        assert np.array_equal(joined.parse()[5], flat.parse()[5])
