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
    CategoricalNetwork,
    DuelingNetwork,
    NoisyNetwork,
    QuantileNetwork,
    RainbowNetwork,
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
    net = CriticNetwork([(4,), (2,)], (1,), n_features=32)
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
        (dim_env,), [(dim_action,), (16,)],
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


def _make_input():
    return torch.from_numpy(np.random.RandomState(0).rand(3, 4).astype(np.float32))


def test_categorical_network_q_values():
    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[-0.51802474,  0.28390205,  0.58237356],
                         [-0.58266914,  0.33951604,  0.6862249 ],
                         [-0.5961043 ,  0.28076124,  0.56525505]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_categorical_network_action_gathering():
    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.28390205, -0.58266914, 0.28076124], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_categorical_network_distribution():
    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    out = net(_make_input(), get_distribution=True).detach().numpy()
    expected_first = np.array([0.23680492, 0.32668704, 0.24101478, 0.10871422, 0.08677896], dtype=np.float32)
    assert out.shape == (3, 3, 5)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)


def test_categorical_network_distribution_action():
    torch.manual_seed(42)
    net = CategoricalNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action, get_distribution=True).detach().numpy()
    expected = np.array([[0.15402764, 0.1284764 , 0.12098037, 0.47259742, 0.12391815],
                         [0.25159165, 0.34750757, 0.21702139, 0.09973714, 0.0841423 ],
                         [0.13659264, 0.15124373, 0.12118733, 0.47676253, 0.11421385]], dtype=np.float32)
    assert out.shape == (3, 5)
    assert np.allclose(out, expected, atol=1e-6)


def test_dueling_network_avg_advantage():
    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=True)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[0.3997159 , 1.6865444 , 1.3915153 ],
                         [0.25348338, 1.7411749 , 1.3661271 ],
                         [0.7645298 , 1.6136137 , 1.4106592 ]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_dueling_network_avg_action_gathering():
    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=True)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([1.6865444, 0.25348338, 1.6136137], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_dueling_network_max_advantage():
    torch.manual_seed(42)
    net = DuelingNetwork((4,), (3,), ActorNetwork, n_features=16, avg_advantage=False)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[-0.12756991,  1.1592586 ,  0.86422944],
                         [-0.36742973,  1.1202618 ,  0.745214  ],
                         [ 0.41385043,  1.2629343 ,  1.0599798 ]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_noisy_network_q_values():
    torch.manual_seed(42)
    net = NoisyNetwork((4,), (3,), ActorNetwork, n_features=16)
    torch.manual_seed(7)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[-0.5709262 ,  0.02239636,  0.23186919],
                         [-0.6771615 , -0.10194737,  0.27086547],
                         [-0.53037447, -0.19845623,  0.12485565]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_noisy_network_action_gathering():
    torch.manual_seed(42)
    net = NoisyNetwork((4,), (3,), ActorNetwork, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.02239636, -0.6771615, -0.19845623], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_noisy_linear_parameter_shapes():
    nl = NoisyNetwork.NoisyLinear(8, 4)
    assert nl.mu_weight.shape == (4, 8)
    assert nl.sigma_weight.shape == (4, 8)
    assert nl.mu_bias.shape == (4,)
    assert nl.sigma_bias.shape == (4,)


def test_quantile_network_mean_q():
    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[ 0.42744595, -0.05222861, -0.10590004],
                         [ 0.49662584, -0.00288994, -0.07939427],
                         [ 0.32411247, -0.03606144, -0.10962137]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_quantile_network_action_gathering():
    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([-0.05222861, 0.49662584, -0.03606144], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_quantile_network_quantiles():
    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    out = net(_make_input(), get_quantiles=True).detach().numpy()
    expected_first = np.array([-0.04285797,  0.8221601 , -0.14703047, -0.02267686,
                                 0.7050508 ,  1.1124958 ,  0.42521736,  0.56720877], dtype=np.float32)
    assert out.shape == (3, 3, 8)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)


def test_quantile_network_quantiles_action():
    torch.manual_seed(42)
    net = QuantileNetwork((4,), (3,), ActorNetwork, n_quantiles=8, n_features=16)
    action = torch.tensor([[1], [0], [1]])
    out = net(_make_input(), action=action, get_quantiles=True).detach().numpy()
    expected = np.array([[-1.2687702 ,  0.03837946,  0.53966427, -0.278058  , -0.03144249,
                           0.3364426 ,  0.8223977 , -0.57644224],
                         [-0.19876304,  1.1473985 , -0.21926138,  0.13285962,  0.6010809 ,
                           1.3654819 ,  0.5616405 ,  0.5825697 ],
                         [-1.3488653 , -0.06751481,  0.2811662 , -0.3109569 ,  0.49284717,
                           0.49418762,  1.0214559 , -0.85081136]], dtype=np.float32)
    assert out.shape == (3, 8)
    assert np.allclose(out, expected, atol=1e-6)


def test_rainbow_network_q_values():
    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    torch.manual_seed(7)
    out = net(_make_input()).detach().numpy()
    expected = np.array([[ 0.22634576,  0.15971707, -0.0171884 ],
                         [ 0.29433823,  0.25769186,  0.07344408],
                         [ 0.24938956,  0.26661745, -0.04909211]], dtype=np.float32)
    assert out.shape == (3, 3)
    assert np.allclose(out, expected, atol=1e-6)


def test_rainbow_network_action_gathering():
    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action).detach().numpy()
    expected = np.array([0.15971707, 0.29433823, 0.26661745], dtype=np.float32)
    assert out.shape == (3,)
    assert np.allclose(out, expected, atol=1e-6)


def test_rainbow_network_distribution():
    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    torch.manual_seed(7)
    out = net(_make_input(), get_distribution=True).detach().numpy()
    expected_first = np.array([0.11195683, 0.17408013, 0.17749025, 0.44860604, 0.08786676], dtype=np.float32)
    assert out.shape == (3, 3, 5)
    assert np.allclose(out[0, 0], expected_first, atol=1e-6)


def test_rainbow_network_distribution_action():
    torch.manual_seed(42)
    net = RainbowNetwork((4,), (3,), ActorNetwork, n_atoms=5, v_min=-2.0, v_max=2.0, n_features=16, sigma_coeff=0.5)
    action = torch.tensor([[1], [0], [1]])
    torch.manual_seed(7)
    out = net(_make_input(), action=action, get_distribution=True).detach().numpy()
    expected = np.array([[0.09477784, 0.1908969 , 0.32020912, 0.24806263, 0.14605351],
                         [0.10430492, 0.15184514, 0.1780674 , 0.47677174, 0.08901073],
                         [0.09133542, 0.14836444, 0.3161716 , 0.2906044 , 0.15352415]], dtype=np.float32)
    assert out.shape == (3, 5)
    assert np.allclose(out, expected, atol=1e-6)
