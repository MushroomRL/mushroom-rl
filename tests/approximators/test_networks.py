import numpy as np
import pytest
import torch
import torch.nn as nn

from mushroom_rl.approximators.parametric.networks import (
    MushroomNetwork,
    LinearNetwork,
    FeedForwardNetwork,
    ActorNetwork,
    QNetwork,
    CriticNetwork,
    AtariNetwork,
    AtariFeatureNetwork,
    RecurrentNetwork,
    RecurrentActorNetwork,
    RecurrentCriticNetwork,
    CategoricalNetwork,
    DuelingNetwork,
    NoisyNetwork,
    QuantileNetwork,
    RainbowNetwork,
)


def test_feed_forward():
    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_0.npy')
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=[64, 32])
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_1.npy')
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, n_layers=3)
    assert len(net._layers) == 4
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_2.npy')
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, activation='tanh')
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_3.npy')
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, gain_scale=0.1)
    w = net._layers[0].weight[0, :2].detach().numpy()
    expected = np.array([-0.00551182, -0.03212874])
    assert np.allclose(w, expected, atol=1e-7)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, weights_init='orthogonal')
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    w = net._layers[0].weight[0, :2].detach().numpy()
    expected_w = np.array([0.27986282, 0.06634405])
    assert np.allclose(w, expected_w, atol=1e-6)
    out = net(x).detach().numpy()
    expected_out = np.load('tests/approximators/data/test_feed_forward_out.npy')
    assert np.allclose(out, expected_out, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, bias_init='zeros')
    assert np.allclose(net._layers[0].bias.detach().numpy()[:4], np.zeros(4))
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_4.npy')
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = FeedForwardNetwork((4,), (2,), n_features=32, activation=nn.Tanh)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_5.npy')
    assert np.allclose(out, expected, atol=1e-6)

    assert issubclass(ActorNetwork, FeedForwardNetwork)
    assert ActorNetwork is not FeedForwardNetwork

    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_feed_forward_6.npy')
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)


def test_q_and_critic():
    torch.manual_seed(42)
    net = QNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.load('tests/approximators/data/test_q_and_critic.npy')
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = QNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    action = torch.tensor([[0], [1], [0]])
    out = net(x, action=action).detach().numpy()
    expected = np.array([0.5220006, -0.2754168, 0.5436677])
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    rng = np.random.RandomState(0)
    state = torch.as_tensor(rng.rand(3, 4), dtype=torch.float32)
    action = torch.as_tensor(rng.rand(3, 2), dtype=torch.float32)
    net = CriticNetwork([(4,), (2,)], (), n_features=32)
    out = net(state, action).detach().numpy()
    expected = np.array([0.41429877, 0.27793425, 0.3294595])
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)
    with pytest.raises(AssertionError):
        CriticNetwork([(4,), (2,)], (1,), n_features=32)

    torch.manual_seed(42)
    rng = np.random.RandomState(0)
    state = torch.as_tensor(rng.rand(3, 4), dtype=torch.float32)
    action = torch.as_tensor(rng.rand(3, 2), dtype=torch.float32)
    history = torch.as_tensor(rng.rand(3, 3, 2), dtype=torch.float32)
    net = CriticNetwork([(4,), (2,)], (), n_features=[8, 4], activation='gelu', action_history_shape=(3, 2))
    out = net(state, action, history).detach().numpy()

    assert [tuple(p.shape) for p in net.parameters()] == [(8, 12), (8,), (4, 8), (4,), (1, 4), (1,)]
    assert np.allclose(out, [-0.37403408, -0.7422491, -0.6758914], atol=1e-6)


def test_atari():
    torch.manual_seed(42)
    net = AtariNetwork((4, 84, 84), (6,))
    x = torch.zeros(2, 4, 84, 84)
    out = net(x).detach().numpy()
    expected_first3 = np.array([-0.0494135, 0.04909683, 0.0360373])
    assert out.shape == (2, 6)
    assert np.allclose(out[0, :3], expected_first3, atol=1e-6)

    torch.manual_seed(42)
    net = AtariNetwork((4, 84, 84), (6,))
    x = torch.zeros(2, 4, 84, 84)
    action = torch.tensor([[2], [4]])
    out = net(x, action=action).detach().numpy()
    expected = np.array([0.0360373, 0.09695501])
    assert out.shape == (2,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = AtariFeatureNetwork((4, 84, 84), (512,))
    x = torch.zeros(2, 4, 84, 84)
    out = net(x).detach().numpy()
    expected_first3 = np.array([0.00325448, 0., 0.])
    assert out.shape == (2, 512)
    assert np.allclose(out[0, :3], expected_first3, atol=1e-5)


def test_recurrent():
    torch.manual_seed(42)
    dim_env, dim_action = 4, 2
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    policy_state = torch.zeros(batch, 1, 16)
    lengths = torch.tensor([5, 3, 4])
    net = RecurrentActorNetwork(
        (dim_env,), [(dim_action,), (1, 16)],
        n_features=32,
        rnn_type='gru', n_hidden_features=16, num_hidden_layers=1
    )
    a, h = net(state, policy_state, lengths)
    a = a.detach().numpy()
    expected = np.load('tests/approximators/data/test_recurrent.npy')
    assert a.shape == (batch, dim_action)
    assert h.shape == (batch, 1, 16)
    assert np.allclose(a, expected, atol=1e-6)

    torch.manual_seed(42)
    dim_env = 4
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    policy_state = torch.zeros(batch, 1, 16)
    lengths = torch.tensor([5, 3, 4])
    net = RecurrentCriticNetwork(
        (dim_env,), (),
        rnn_type='gru', n_hidden_features=16, n_features=32, num_hidden_layers=1
    )
    q = net(state, policy_state, lengths).detach().numpy()
    expected = np.array([-1.7422979, -1.1620455, -1.4952478])
    assert q.shape == (batch,)
    assert np.allclose(q, expected, atol=1e-5)
    with pytest.raises(AssertionError):
        RecurrentCriticNetwork((dim_env,), (1,), rnn_type='gru')

    torch.manual_seed(42)
    dim_env, dim_action = 4, 2
    history_length, action_history_length = 3, 2
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, history_length, dim_env)
    action_history = torch.randn(batch, seq_len, action_history_length, dim_action)
    policy_state = torch.zeros(batch, 1, 16)
    lengths = torch.tensor([5, 3, 4])

    actor = RecurrentActorNetwork(
        (history_length, dim_env), [(dim_action,), (1, 16)],
        n_features=32, rnn_type='gru', n_hidden_features=16, num_hidden_layers=1,
        action_history_shape=(action_history_length, dim_action)
    )
    assert actor._h1_o.in_features == history_length * dim_env + action_history_length * dim_action
    a, h = actor(state, policy_state, lengths, action_history=action_history)
    assert a.shape == (batch, dim_action)
    assert h.shape == (batch, 1, 16)

    critic = RecurrentCriticNetwork(
        (history_length, dim_env), (),
        rnn_type='gru', n_hidden_features=16, n_features=32, num_hidden_layers=1,
        action_history_shape=(action_history_length, dim_action)
    )
    assert critic._h1_o.in_features == history_length * dim_env + action_history_length * dim_action
    q = critic(state, policy_state, lengths, action_history=action_history)
    assert q.shape == (batch,)

    assert RecurrentNetwork.compute_policy_state_shape('rnn', 16) == (1, 16)
    assert RecurrentNetwork.compute_policy_state_shape('gru', 16) == (1, 16)
    assert RecurrentNetwork.compute_policy_state_shape('lstm', 16) == (2, 1, 16)
    assert RecurrentNetwork.compute_policy_state_shape('gru', 16, 2) == (2, 16)
    assert RecurrentNetwork.compute_policy_state_shape('lstm', 16, 2) == (2, 2, 16)


def _make_input():
    return torch.from_numpy(np.random.RandomState(0).rand(3, 4).astype(np.float32))


def test_categorical():
    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_categorical_0.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.28390205, -0.58266914, 0.28076124], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    out = net(_make_input(), get_distribution=True).detach().numpy()
    expected_first = np.load('tests/approximators/data/test_categorical_first.npy')
    assert out.shape == (3, 3, 5)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)

    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action, get_distribution=True).detach().numpy()
    expected = np.load('tests/approximators/data/test_categorical_1.npy')
    assert out.shape == (3, 5)
    assert np.allclose(out, expected, atol=1e-6)


def test_dueling():
    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=True)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_dueling_0.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=True)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([1.6865444, 0.25348338, 1.6136137], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=False)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_dueling_1.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_noisy():
    torch.manual_seed(42)
    net = NoisyNetwork((4,), (3,), ActorNetwork, n_features=16)
    torch.manual_seed(7)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_noisy.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = NoisyNetwork((4,), (3,), ActorNetwork, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.02239636, -0.6771615, -0.19845623], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    nl = NoisyNetwork.NoisyLinear(8, 4)
    assert nl.mu_weight.shape == (4, 8)
    assert nl.sigma_weight.shape == (4, 8)
    assert nl.mu_bias.shape == (4,)
    assert nl.sigma_bias.shape == (4,)


def test_quantile():
    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_quantile_0.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([-0.05222861, 0.49662584, -0.03606144], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    out = net(_make_input(), get_quantiles=True).detach().numpy()
    expected_first = np.load('tests/approximators/data/test_quantile_first.npy')
    assert out.shape == (3, 3, 8)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)

    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action, get_quantiles=True).detach().numpy()
    expected = np.load('tests/approximators/data/test_quantile_1.npy')
    assert out.shape == (3, 8)
    assert np.allclose(out, expected, atol=1e-6)


def test_rainbow():
    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    torch.manual_seed(7)
    out = net(_make_input()).detach().numpy()
    expected = np.load('tests/approximators/data/test_rainbow_0.npy')
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.15971707, 0.29433823, 0.26661745], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)

    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    torch.manual_seed(7)
    out = net(_make_input(), get_distribution=True).detach().numpy()
    expected_first = np.load('tests/approximators/data/test_rainbow_first.npy')
    assert out.shape == (3, 3, 5)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)

    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action, get_distribution=True).detach().numpy()
    expected = np.load('tests/approximators/data/test_rainbow_1.npy')
    assert out.shape == (3, 5)
    assert np.allclose(out, expected, atol=1e-6)


def test_recurrent_lstm():
    torch.manual_seed(42)
    dim_env, dim_action, n_hidden = 4, 2, 16
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    lengths = torch.tensor([5, 3, 4])
    policy_state_shape = RecurrentNetwork.compute_policy_state_shape('lstm', n_hidden)
    policy_state = torch.zeros(batch, *policy_state_shape)
    net = RecurrentActorNetwork(
        (dim_env,), [(dim_action,), policy_state_shape],
        n_features=32, rnn_type='lstm', n_hidden_features=n_hidden, num_hidden_layers=1
    )
    a, h = net(state, policy_state, lengths)
    expected = np.load('tests/approximators/data/test_recurrent_lstm_0.npy')
    assert a.shape == (batch, dim_action)
    assert h.shape == (batch, 2, 1, n_hidden)
    assert np.allclose(a.detach().numpy(), expected, atol=1e-6)

    torch.manual_seed(42)
    dim_env, n_hidden = 4, 16
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    lengths = torch.tensor([5, 3, 4])
    policy_state_shape = RecurrentNetwork.compute_policy_state_shape('lstm', n_hidden)
    policy_state = torch.zeros(batch, *policy_state_shape)
    net = RecurrentCriticNetwork(
        (dim_env,), (), rnn_type='lstm', n_hidden_features=n_hidden, n_features=32,
        num_hidden_layers=1, hidden_state_treatment='use_policy_hidden_state'
    )
    q = net(state, policy_state, lengths)
    expected = np.array([1.9283445, 0.17133498, 2.3065667], dtype=np.float32)
    assert q.shape == (batch,)
    assert np.allclose(q.detach().numpy(), expected, atol=1e-5)

    torch.manual_seed(42)
    dim_env, n_hidden = 4, 16
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    lengths = torch.tensor([5, 3, 4])
    policy_state = torch.zeros(batch, *RecurrentNetwork.compute_policy_state_shape('lstm', n_hidden))
    net = RecurrentCriticNetwork(
        (dim_env,), (), rnn_type='lstm', n_hidden_features=n_hidden, n_features=32,
        num_hidden_layers=1, hidden_state_treatment='zero_initial'
    )
    q = net(state, policy_state, lengths)
    assert q.shape == (batch,)

    torch.manual_seed(42)
    dim_env, dim_action, n_hidden, num_layers = 4, 2, 16, 2
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    lengths = torch.tensor([5, 3, 4])
    policy_state_shape = RecurrentNetwork.compute_policy_state_shape('lstm', n_hidden, num_layers)
    policy_state = torch.zeros(batch, *policy_state_shape)
    actor = RecurrentActorNetwork(
        (dim_env,), [(dim_action,), policy_state_shape],
        n_features=32, rnn_type='lstm', n_hidden_features=n_hidden, num_hidden_layers=num_layers
    )
    a, h = actor(state, policy_state, lengths)
    expected_a = np.load('tests/approximators/data/test_recurrent_lstm_a.npy')
    assert h.shape == (batch, 2, num_layers, n_hidden)
    assert np.allclose(a.detach().numpy(), expected_a, atol=1e-6)

    critic = RecurrentCriticNetwork(
        (dim_env,), (), rnn_type='lstm', n_hidden_features=n_hidden, n_features=32,
        num_hidden_layers=num_layers, hidden_state_treatment='use_policy_hidden_state'
    )
    q = critic(state, policy_state, lengths)
    expected_q = np.array([1.0538795, 0.7622827, -0.9266093], dtype=np.float32)
    assert q.shape == (batch,)
    assert np.allclose(q.detach().numpy(), expected_q, atol=1e-5)


def test_distributional_support():
    for net in (CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16),
                RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16,
                               sigma_coeff=0.5)):
        assert '_a_values' in dict(net.named_buffers())
        assert '_a_values' not in net.state_dict()
        assert torch.equal(net._a_values, torch.tensor([-2., -1., 0., 1., 2.]))

        if torch.cuda.is_available():
            net.to('cuda')
            assert net._a_values.device.type == 'cuda'


def _network_factories():
    features = dict(features_network=FeedForwardNetwork, n_features=8)
    image = (4, 36, 36)
    return {
        'linear': lambda: LinearNetwork((4,), (2,), use_bias=True, gain=.3),
        'feed_forward': lambda: FeedForwardNetwork((3, 4), (2,), n_features=[8, 6], activation='tanh',
                                                   weights_init='orthogonal', bias_init='zeros',
                                                   action_history_shape=(2,)),
        'actor': lambda: ActorNetwork((4,), (2,), n_features=8),
        'q': lambda: QNetwork((4,), (3,), n_features=8),
        'critic': lambda: CriticNetwork([(4,), (2,)], (), n_features=8),
        'atari': lambda: AtariNetwork(image, (3,), n_features=16),
        'atari_features': lambda: AtariFeatureNetwork(image, (16,)),
        'dueling': lambda: DuelingNetwork((4,), (3,), avg_advantage=True, **features),
        'categorical': lambda: CategoricalNetwork((4,), (3,), n_atoms=5, v_min=-1, v_max=1, **features),
        'quantile': lambda: QuantileNetwork((4,), (3,), n_quantiles=4, **features),
        'noisy': lambda: NoisyNetwork((4,), (3,), **features),
        'rainbow': lambda: RainbowNetwork((4,), (3,), n_atoms=5, v_min=-1, v_max=1, sigma_coeff=.5, **features),
        'recurrent_actor': lambda: RecurrentActorNetwork((4,), [(2,), (2, 2, 6)], n_features=8, rnn_type='lstm',
                                                         n_hidden_features=6, num_hidden_layers=2),
        'recurrent_critic': lambda: RecurrentCriticNetwork((4,), (), rnn_type='gru', n_hidden_features=6,
                                                           n_features=8),
    }


@pytest.mark.parametrize('make', _network_factories().values(), ids=_network_factories().keys())
def test_reset_parameters(make):
    torch.manual_seed(0)
    constructed = make()
    constructed_next = torch.rand(3)

    torch.manual_seed(1)
    net = make()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(1.)
    torch.manual_seed(0)
    net.reset_parameters()

    assert net.state_dict().keys() == constructed.state_dict().keys()
    assert all(torch.equal(value, constructed.state_dict()[key]) for key, value in net.state_dict().items())
    assert torch.equal(torch.rand(3), constructed_next)


def test_reset_parameters_error():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self._h = nn.Linear(4, 2)

    class Network(MushroomNetwork):
        def __init__(self):
            super().__init__()
            self._activation = nn.ReLU()
            self._encoder = Encoder()

    with pytest.raises(TypeError, match='Encoder'):
        Network().reset_parameters()
