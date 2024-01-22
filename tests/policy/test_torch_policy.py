import torch
import torch.nn as nn

import numpy as np

from mushroom_rl.policy.torch_policy import TorchPolicy, GaussianTorchPolicy, BoltzmannTorchPolicy
from mushroom_rl.rl_utils.parameters import Parameter


def abstract_method_tester(f, *args):
    try:
        f(*args)
    except NotImplementedError:
        pass
    else:
        assert False


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, -1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)

        return a


def test_torch_policy():
    tmp = TorchPolicy(False)
    abstract_method_tester(tmp.draw_action_t, None)
    abstract_method_tester(tmp.log_prob_t, None, None)
    abstract_method_tester(tmp.entropy_t, None)
    abstract_method_tester(tmp.distribution_t, None)
    abstract_method_tester(tmp.set_weights, None)
    abstract_method_tester(tmp.get_weights)
    abstract_method_tester(tmp.parameters)
    tmp.reset()


def test_gaussian_torch_policy():
    np.random.seed(88)
    torch.manual_seed(88)
    pi = GaussianTorchPolicy(Network, (3,), (2,), n_features=50)

    state = torch.as_tensor(np.random.rand(3))
    action, _ = pi.draw_action(state)
    action_test = np.array([-0.21276927,  0.27437747])
    assert np.allclose(action.detach().cpu().numpy(), action_test)

    p_sa = pi(state, torch.as_tensor(action))
    p_sa_test = 0.07710557966732147
    assert np.allclose(p_sa.detach().cpu().numpy(), p_sa_test)

    entropy = pi.entropy()
    entropy_test = 2.837877
    assert np.allclose(entropy, entropy_test)


def test_boltzmann_torch_policy():
    np.random.seed(88)
    torch.manual_seed(88)
    beta = Parameter(1.0)
    pi = BoltzmannTorchPolicy(Network, (3,), (2,), beta, n_features=50)

    state = torch.as_tensor(np.random.rand(3, 3))
    action, _ = pi.draw_action(state)
    action_test = np.array([1, 0, 0])
    assert np.allclose(action.detach().cpu().numpy(), action_test)

    p_sa = pi(state[0], action[0])
    p_sa_test = 0.24054041611818922
    assert np.allclose(p_sa.detach(), p_sa_test)

    states = np.random.rand(1000, 3)
    entropy = pi.entropy(states)
    entropy_test = 0.5428627133369446
    assert np.allclose(entropy.detach().cpu().numpy(), entropy_test)
