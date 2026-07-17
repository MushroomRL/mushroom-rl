import torch
import torch.nn as nn

from mushroom_rl.policy import RecurrentGaussianTorchPolicy


class SimpleGRUNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_hidden=8, **kwargs):
        super().__init__()
        assert isinstance(output_shape, list) and len(output_shape) == 2
        self._n_hidden = n_hidden
        self._rnn = nn.GRU(input_shape[0], n_hidden, batch_first=True)
        self._fc = nn.Linear(n_hidden, output_shape[0][0])

    def forward(self, state, policy_state, lengths, **kwargs):
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


def test_recurrent_policy_vectorized():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    ps = policy.reset_vectorized(torch.tensor([True, True, True]))
    assert ps.shape == (3, n_hidden)
    assert torch.all(ps == 0.0)

    action = policy.draw_action(torch.randn(3, n_state))
    assert action.shape == (3, n_action)
    assert policy.policy_state.shape == (3, n_hidden)

    before = policy.policy_state.clone()
    policy.reset_vectorized(torch.tensor([True, False, True]))
    assert torch.all(policy.policy_state[0] == 0.0)
    assert torch.all(policy.policy_state[2] == 0.0)
    assert torch.equal(policy.policy_state[1], before[1])


def test_recurrent_policy_draw_action():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    state = torch.tensor([0.1, -0.2, 0.3, -0.4])
    policy.reset()

    action = policy.draw_action(state)
    new_policy_state = policy.policy_state

    action_test = torch.tensor([1.2770035, 0.48390746])
    new_ps_test = torch.tensor([-0.13474981,  0.16390274,  0.04836516, -0.00375814,
                                -0.05680789,  0.06481361,  0.22528732, -0.00479783])

    assert action.shape == (n_action,)
    assert new_policy_state.shape == (n_hidden,)
    assert torch.allclose(action, action_test, atol=1e-5)
    assert torch.allclose(new_policy_state, new_ps_test, atol=1e-5)


def test_recurrent_policy_draw_action_greedy():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    state = torch.tensor([0.1, -0.2, 0.3, -0.4])
    policy.reset()

    action = policy.draw_action_greedy(state)
    new_policy_state = policy.policy_state

    action_test = torch.tensor([0.25209162, 0.26158917])
    new_ps_test = torch.tensor([-0.13474981,  0.16390274,  0.04836516, -0.00375814,
                                -0.05680789,  0.06481361,  0.22528732, -0.00479783])

    assert action.shape == (n_action,)
    assert new_policy_state.shape == (n_hidden,)
    assert torch.allclose(action, action_test, atol=1e-5)
    assert torch.allclose(new_policy_state, new_ps_test, atol=1e-5)


def test_recurrent_policy_entropy():
    policy = make_policy()

    entropy = policy.entropy()

    assert isinstance(entropy, torch.Tensor)
    assert entropy.shape == ()
    assert torch.isclose(entropy, torch.tensor(2.837877), atol=1e-5)


def test_recurrent_policy_distribution_and_policy_state():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 2
    state = torch.zeros(batch, 3, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([3, 3])

    dist, new_ps = policy.distribution_and_policy_state(state, policy_state, lengths)

    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.loc.shape == (batch, n_action)
    assert new_ps.shape == (batch, n_hidden)
    assert not torch.all(new_ps == 0.0)


def test_recurrent_policy_log_prob():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 3
    state = torch.zeros(batch, 1, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.ones(batch, dtype=torch.long)
    action = torch.zeros(batch, n_action)

    log_prob = policy.log_prob(state, action, policy_state, lengths)

    log_prob_test = torch.tensor([[-1.9125082],
                                  [-1.9125082],
                                  [-1.9125082]])

    assert log_prob.shape == (batch, 1)
    assert torch.allclose(log_prob[0], log_prob[1])
    assert torch.allclose(log_prob[0], log_prob[2])
    assert torch.allclose(log_prob, log_prob_test, atol=1e-5)


def test_recurrent_policy_draw_with_log_prob():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 3
    state = torch.zeros(batch, 1, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.ones(batch, dtype=torch.long)

    torch.manual_seed(42)
    action, log_prob, next_policy_state = policy.draw_with_log_prob(state, policy_state, lengths)

    action_test = torch.tensor([[0.5960621, 0.4151462],
                                [0.4938340, 0.5166699],
                                [-0.8634847, 0.1000085]])
    log_prob_test = torch.tensor([[-1.9028531],
                                  [-1.8918899],
                                  [-2.4856393]])

    assert action.shape == (batch, n_action)
    assert log_prob.shape == (batch, 1)
    assert next_policy_state.shape == (batch, n_hidden)
    assert action.requires_grad and log_prob.requires_grad
    assert torch.allclose(action, action_test, atol=1e-5)
    assert torch.allclose(log_prob, log_prob_test, atol=1e-5)


def test_recurrent_policy_distribution_interface():
    n_state, n_action, n_hidden = 4, 2, 8
    policy = make_policy(n_state, n_action, n_hidden)

    batch = 2
    state = torch.zeros(batch, 3, n_state)
    policy_state = torch.zeros(batch, n_hidden)
    lengths = torch.tensor([3, 3])

    dist = policy.distribution(state, policy_state, lengths)

    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.loc.shape == (batch, n_action)
