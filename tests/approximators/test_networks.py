import numpy as np
import torch
import torch.nn as nn

from mushroom_rl.approximators.parametric.networks import (
    ActorNetwork,
    QNetwork,
    CriticNetwork,
    AtariNetwork,
    AtariFeatureNetwork,
    RecurrentActorNetwork,
    RecurrentCriticNetwork,
)


def test_actor_network_scalar_features():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[0.5220006, -0.20175238],
                         [0.44349343, -0.2754168],
                         [0.5436677, -0.1532542]])
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)


def test_actor_network_list_features():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=[64, 32])
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[-0.8190474, 0.40304738],
                         [-0.9595563, 0.28353268],
                         [-0.8024489, 0.5790019]])
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)


def test_actor_network_n_layers():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, n_layers=3)
    assert len(net._layers) == 4
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[-0.07683997, 0.02765154],
                         [-0.19581519, -0.09588787],
                         [-0.11350869, 0.11468533]])
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)


def test_actor_network_tanh():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, activation='tanh')
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[-1.261526, 0.84943783],
                         [-1.422259, 0.61180925],
                         [-1.3073331, 0.8145788]])
    assert np.allclose(out, expected, atol=1e-6)


def test_actor_network_gain_scale():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, gain_scale=0.1)
    w = net._layers[0].weight[0, :2].detach().numpy()
    expected = np.array([-0.00551182, -0.03212874])
    assert np.allclose(w, expected, atol=1e-7)


def test_actor_network_orthogonal_init():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, weights_init='orthogonal')
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    w = net._layers[0].weight[0, :2].detach().numpy()
    expected_w = np.array([0.27986282, 0.06634405])
    assert np.allclose(w, expected_w, atol=1e-6)
    out = net(x).detach().numpy()
    expected_out = np.array([[-0.49445415, 0.15632166],
                              [-0.39369154, 0.19581775],
                              [-0.48856848, 0.09151582]])
    assert np.allclose(out, expected_out, atol=1e-6)


def test_actor_network_zero_bias_init():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, bias_init='zeros')
    assert np.allclose(net._layers[0].bias.detach().numpy()[:4], np.zeros(4))
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[-0.03467284, -0.08134562],
                         [-0.19547892, -0.15192705],
                         [0.17799613, 0.07703486]])
    assert np.allclose(out, expected, atol=1e-6)


def test_actor_network_activation_class():
    torch.manual_seed(42)
    net = ActorNetwork((4,), (2,), n_features=32, activation=nn.Tanh)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[-1.261526, 0.84943783],
                         [-1.422259, 0.61180925],
                         [-1.3073331, 0.8145788]])
    assert np.allclose(out, expected, atol=1e-6)


def test_q_network_output():
    torch.manual_seed(42)
    net = QNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    out = net(x).detach().numpy()
    expected = np.array([[0.5220006, -0.20175238],
                         [0.44349343, -0.2754168],
                         [0.5436677, -0.1532542]])
    assert out.shape == (3, 2)
    assert np.allclose(out, expected, atol=1e-6)


def test_q_network_action_gathering():
    torch.manual_seed(42)
    net = QNetwork((4,), (2,), n_features=32)
    x = torch.as_tensor(np.random.RandomState(0).rand(3, 4), dtype=torch.float32)
    action = torch.tensor([[0], [1], [0]])
    out = net(x, action=action).detach().numpy()
    expected = np.array([0.5220006, -0.2754168, 0.5436677])
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_critic_network():
    torch.manual_seed(42)
    rng = np.random.RandomState(0)
    state = torch.as_tensor(rng.rand(3, 4), dtype=torch.float32)
    action = torch.as_tensor(rng.rand(3, 2), dtype=torch.float32)
    net = CriticNetwork((6,), (1,), n_features=32)
    out = net(state, action).detach().numpy()
    expected = np.array([0.41429877, 0.27793425, 0.3294595])
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_atari_network():
    torch.manual_seed(42)
    net = AtariNetwork((4, 84, 84), (6,))
    x = torch.zeros(2, 4, 84, 84)
    out = net(x).detach().numpy()
    expected_first3 = np.array([-0.04561843, -0.02100883, -0.07272205])
    assert out.shape == (2, 6)
    assert np.allclose(out[0, :3], expected_first3, atol=1e-6)


def test_atari_network_action_gathering():
    torch.manual_seed(42)
    net = AtariNetwork((4, 84, 84), (6,))
    x = torch.zeros(2, 4, 84, 84)
    action = torch.tensor([[2], [4]])
    out = net(x, action=action).detach().numpy()
    expected = np.array([-0.07272205, 0.04723251])
    assert out.shape == (2,)
    assert np.allclose(out, expected, atol=1e-6)


def test_atari_feature_network():
    torch.manual_seed(42)
    net = AtariFeatureNetwork((4, 84, 84), (512,))
    x = torch.zeros(2, 4, 84, 84)
    out = net(x).detach().numpy()
    expected_first3 = np.array([0.00325448, 0., 0.])
    assert out.shape == (2, 512)
    assert np.allclose(out[0, :3], expected_first3, atol=1e-5)


def test_recurrent_actor_network():
    torch.manual_seed(42)
    dim_env, dim_action = 4, 2
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    policy_state = torch.zeros(batch, 1, 16)
    lengths = torch.tensor([5, 3, 4])
    net = RecurrentActorNetwork(
        (dim_env,), (dim_action,),
        n_features=32, dim_env_state=dim_env,
        rnn_type='gru', n_hidden_features=16, num_hidden_layers=1
    )
    a, h = net(state, policy_state, lengths)
    a = a.detach().numpy()
    expected = np.array([[-0.0694935, -0.01662157],
                         [-0.06815533, -0.01592569],
                         [-0.07035983, -0.01833964]])
    assert a.shape == (batch, dim_action)
    assert h.shape == (batch, 1, 16)
    assert np.allclose(a, expected, atol=1e-6)


def test_recurrent_critic_network():
    torch.manual_seed(42)
    dim_env, dim_action = 4, 2
    batch, seq_len = 3, 5
    state = torch.randn(batch, seq_len, dim_env)
    policy_state = torch.zeros(batch, 1, 16)
    lengths = torch.tensor([5, 3, 4])
    net = RecurrentCriticNetwork(
        (dim_env,), (1,),
        dim_env_state=dim_env, dim_action=dim_action,
        rnn_type='gru', n_hidden_features=16, n_features=32, num_hidden_layers=1
    )
    q = net(state, policy_state, lengths).detach().numpy()
    expected = np.array([-1.7422979, -1.1620455, -1.4952478])
    assert q.shape == (batch,)
    assert np.allclose(q, expected, atol=1e-5)
