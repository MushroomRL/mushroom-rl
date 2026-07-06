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

    n_carry = dataset.clear(n_steps_per_fit=4)

    assert int(n_carry) == 2
    assert len(dataset) == 1
    assert dataset._policy_data is not None
    assert dataset.mask.sum() == 2
