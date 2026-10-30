from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator, TorchApproximator

import numpy as np
import pytest
import torch
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        self._h = nn.Linear(input_shape[0], output_shape[0])

    def forward(self, state, **kwargs):
        return self._h(state.float())


def test_deterministic_policy():
    np.random.seed(42)

    n_dims = 5

    approximator = LinearApproximator(input_shape=(n_dims,), output_shape=(2,))

    pi = DeterministicPolicy(approximator)

    w_new = np.random.rand(pi.weights_size)

    w_old = pi.get_weights()
    pi.set_weights(w_new)

    assert np.array_equal(w_new, approximator.get_weights())
    assert not np.array_equal(w_old, w_new)
    assert np.array_equal(w_new, pi.get_weights())

    s_test_1 = np.random.randn(5)
    s_test_2 = np.random.randn(5)
    a_test = approximator.predict(s_test_1)

    assert pi.get_regressor() == approximator

    assert pi(s_test_1, a_test) == 1
    assert pi(s_test_2, a_test) == 0

    a_stored = np.array([-0.24029878, -0.55175323])
    action = pi.draw_action(s_test_1)
    assert np.allclose(action, a_stored)

    assert np.allclose(pi.draw_action_greedy(s_test_1), action)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_deterministic_policy_torch(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires cuda')

    torch.manual_seed(42)

    approximator = TorchApproximator(network=LinearNetwork, input_shape=(5,), output_shape=(2,), device=device)

    pi = DeterministicPolicy(approximator)

    assert pi.backend_name == 'torch'
    assert pi.device == device

    s_test_1 = torch.randn(5, device=device)
    s_test_2 = torch.randn(5, device=device)
    a_test = approximator.predict(s_test_1)

    assert pi(s_test_1, a_test) == 1
    assert pi(s_test_2, a_test) == 0

    action = pi.draw_action(s_test_1)

    assert isinstance(action, torch.Tensor)
    assert action.device.type == device
    assert torch.equal(action, a_test)
    assert torch.equal(pi.draw_action_greedy(s_test_1), action)
