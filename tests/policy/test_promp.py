import numpy as np
from scipy.stats import multivariate_normal

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.policy.promps import ProMP


def _repeat_features(z):
    return np.repeat(z, 5, axis=-1)


def _make_promp(duration=10, action_dim=2, sigma=None, periodic=False):
    n_features = 5
    phi = Features(n_outputs=n_features, function=_repeat_features)
    mu = LinearApproximator(input_shape=(n_features,), output_shape=(action_dim,))
    return ProMP(mu, phi, duration, sigma=sigma, periodic=periodic)


def test_policy_state_shape():
    np.random.seed(1)
    pi = _make_promp()
    assert pi.policy_state_shape == (1,)


def test_reset_returns_initial_state():
    np.random.seed(1)
    pi = _make_promp(duration=10)

    pi.reset()
    for _ in range(4):
        pi.draw_action(np.zeros(2))
    assert not np.allclose(pi.policy_state, 0)

    pi.reset()
    assert np.allclose(pi.policy_state, 0)


def test_weights_size():
    np.random.seed(1)
    pi = _make_promp(action_dim=2)
    assert pi.weights_size == 10


def test_weights_roundtrip():
    np.random.seed(1)
    pi = _make_promp(action_dim=2)
    w = np.arange(pi.weights_size, dtype=float)
    pi.set_weights(w)
    assert np.allclose(pi.get_weights(), w)


def test_draw_action_deterministic():
    np.random.seed(1)
    pi = _make_promp(duration=10, action_dim=2, sigma=None)
    pi.policy_state = np.array([3])
    action = pi.draw_action(np.zeros(2))
    assert np.allclose(action, np.zeros(2))
    assert np.allclose(pi.policy_state, 4)


def test_draw_action_deterministic_with_weights():
    np.random.seed(1)
    pi = _make_promp(duration=10, action_dim=2, sigma=None)
    pi.set_weights(np.ones(pi.weights_size))
    pi.policy_state = np.array([5])
    action = pi.draw_action(np.zeros(2))
    assert np.allclose(action, np.array([2.5, 2.5]))
    assert np.allclose(pi.policy_state, 6)


def test_draw_action_stochastic():
    np.random.seed(42)
    pi = _make_promp(duration=10, action_dim=2, sigma=np.eye(2) * 0.01)
    pi.policy_state = np.array([0])
    action = pi.draw_action(np.zeros(2))
    assert np.allclose(action, np.array([0.04967142, -0.01382643]))
    assert np.allclose(pi.policy_state, 1)


def test_update_time_increments():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    assert pi.update_time(None, np.array([3])) == 4


def test_update_time_clamps_non_periodic():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    assert pi.update_time(None, np.array([9])) == 10
    assert pi.update_time(None, np.array([10])) == 10


def test_update_time_periodic_no_clamp():
    np.random.seed(1)
    pi = _make_promp(duration=10, periodic=True)
    assert pi.update_time(None, np.array([9])) == 10
    assert pi.update_time(None, np.array([10])) == 11


def test_update_time_does_not_mutate_input():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    policy_state = np.array([3])
    policy_state_before = policy_state.copy()
    pi.update_time(None, policy_state)
    assert np.array_equal(policy_state, policy_state_before)


def test_draw_action_explicit_state_does_not_mutate_or_advance():
    np.random.seed(1)
    pi = _make_promp(duration=10, action_dim=2, sigma=None)
    pi.reset()
    policy_state = np.array([3])
    policy_state_before = policy_state.copy()
    pi.draw_action(np.zeros(2), policy_state)
    assert np.array_equal(policy_state, policy_state_before)
    assert np.allclose(pi.policy_state, 0)


def test_compute_phase():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    assert np.allclose(pi._compute_phase(None, np.array([5])), 0.5)


def test_set_duration():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    pi.set_duration(6)
    assert pi._duration == 5


def test_draw_action_vectorized():
    np.random.seed(1)
    n_envs = 3
    pi = _make_promp(duration=10, action_dim=2, sigma=None)

    ps = pi.reset_vectorized(np.array([True, True, True]))
    assert ps.shape == (n_envs, 1)

    actions = pi.draw_action(np.zeros((n_envs, 2)))
    assert actions.shape == (n_envs, 2)
    assert np.allclose(pi.policy_state, 1)

    before = pi.policy_state.copy()
    pi.reset_vectorized(np.array([True, False, True]))
    assert np.allclose(pi.policy_state[0], 0)
    assert np.allclose(pi.policy_state[2], 0)
    assert np.allclose(pi.policy_state[1], before[1])


def test_policy_state_setter_copies():
    np.random.seed(1)
    pi = _make_promp(duration=10)
    external = np.array([4.0])
    pi.policy_state = external
    external[0] = 99.0
    assert pi.policy_state[0] == 4.0


def test_call_density():
    np.random.seed(1)
    pi = _make_promp(duration=10, action_dim=2, sigma=np.eye(2))
    pi.set_weights(np.zeros(pi.weights_size))
    value = pi(np.zeros(2), np.zeros(2), np.array([0]))
    assert np.allclose(value, multivariate_normal.pdf(np.zeros(2), np.zeros(2), np.eye(2)))


def test_serialization(tmpdir):
    np.random.seed(1)
    pi = _make_promp(duration=10, action_dim=2, sigma=None)
    pi.set_weights(np.arange(pi.weights_size, dtype=float))

    path = tmpdir / 'promp.msh'
    pi.save(path)
    loaded = ProMP.load(path)

    assert np.allclose(loaded.get_weights(), pi.get_weights())

    pi.reset()
    loaded.reset()
    for _ in range(4):
        assert np.allclose(pi.draw_action(np.zeros(2)), loaded.draw_action(np.zeros(2)))
