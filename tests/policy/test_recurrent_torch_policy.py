import numpy as np
import torch
import torch.nn as nn

from mushroom_rl.policy import RecurrentGaussianTorchPolicy


class SimpleGRUNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_hidden=8, **kwargs):
        super().__init__()
        self._n_hidden = n_hidden
        self._rnn = nn.GRU(input_shape[0], n_hidden, batch_first=True)
        self._fc = nn.Linear(n_hidden, output_shape[0])

    def forward(self, state, policy_state, lengths):
        h0 = policy_state.float().view(1, -1, self._n_hidden)
        out, next_h = self._rnn(state.float(), h0)
        next_h = next_h.squeeze(0)

        if isinstance(lengths, torch.Tensor) and lengths.numel() > 1:
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
            last_out = out.gather(1, idx).squeeze(1)
        else:
            last_out = out[:, -1, :]

        return self._fc(last_out), next_h


def make_policy(n_state=4, n_action=2, n_hidden=8):
    np.random.seed(42)
    torch.manual_seed(42)
    return RecurrentGaussianTorchPolicy(
        network=SimpleGRUNetwork,
        input_shape=(n_state,),
        output_shape=(n_action,),
        policy_state_shape=(n_hidden,),
        n_hidden=n_hidden
    )


def test_recurrent_policy_reset():
    n_hidden = 8
    policy = make_policy(n_hidden=n_hidden)

    ps = policy.reset()

    assert isinstance(ps, torch.Tensor)
    assert ps.shape == (n_hidden,)
    assert torch.all(ps == 0.0)


def test_recurrent_policy_draw_action():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    state = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    policy_state = policy.reset()

    action, new_policy_state = policy.draw_action(state, policy_state)

    action_test = np.array([1.2770035, 0.48390746], dtype=np.float32)
    new_ps_test = np.array([[-0.13474981,  0.16390274,  0.04836516, -0.00375814,
                              -0.05680789,  0.06481361,  0.22528732, -0.00479783]], dtype=np.float32)

    assert action.shape == (n_action,)
    assert new_policy_state.shape == (1, n_hidden)
    assert np.allclose(action.detach().numpy(), action_test, atol=1e-5)
    assert np.allclose(new_policy_state.detach().numpy(), new_ps_test, atol=1e-5)


def test_recurrent_policy_draw_action_t():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    state = torch.zeros(1, 1, n_state)
    policy_state = torch.zeros(1, n_hidden)
    lengths = torch.tensor([1])

    torch.manual_seed(42)
    action, new_policy_state = policy.draw_action_t(state, policy_state)

    action_test = np.array([[0.596062, 0.41514623]], dtype=np.float32)

    assert action.shape == (1, n_action)
    assert new_policy_state.shape == (1, n_hidden)
    assert np.allclose(action.detach().numpy(), action_test, atol=1e-5)


def test_recurrent_policy_entropy():
    policy = make_policy()

    entropy = policy.entropy_t()

    assert isinstance(entropy, torch.Tensor)
    assert entropy.shape == ()
    assert np.isclose(entropy.item(), 2.837877, atol=1e-5)


def test_recurrent_policy_distribution_t():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 2
    state = torch.zeros(batch, 3, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([3, 3])

    dist = policy.distribution_t(state, policy_state, lengths)

    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.loc.shape == (batch, n_action)
    assert torch.allclose(dist.loc[0], dist.loc[1])


def test_recurrent_policy_distribution_and_policy_state_t():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 2
    state = torch.zeros(batch, 3, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([3, 3])

    dist, new_ps = policy.distribution_and_policy_state_t(state, policy_state, lengths)

    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.loc.shape == (batch, n_action)
    assert new_ps.shape == (batch, n_hidden)
    assert not torch.all(new_ps == 0.0)


def test_recurrent_policy_log_prob_t():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 3
    state = torch.zeros(batch, 1, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.ones(batch, dtype=torch.long)
    action = torch.zeros(batch, n_action)

    log_prob = policy.log_prob_t(state, action, policy_state, lengths)

    log_prob_test = np.array([[-1.9125082],
                               [-1.9125082],
                               [-1.9125082]], dtype=np.float32)

    assert log_prob.shape == (batch, 1)
    assert torch.allclose(log_prob[0], log_prob[1])
    assert torch.allclose(log_prob[0], log_prob[2])
    assert np.allclose(log_prob.detach().numpy(), log_prob_test, atol=1e-5)


def test_recurrent_policy_distribution_interface():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 2
    state = np.zeros((batch, 3, n_state), dtype=np.float32)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([3, 3])

    dist = policy.distribution(state, policy_state, lengths)

    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.loc.shape == (batch, n_action)