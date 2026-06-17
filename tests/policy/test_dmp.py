import numpy as np

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.policy.dmp import DMP


def _make_dmp(action_dim=1, goal=None):
    n_features = 4
    phi = Features(n_outputs=n_features, function=lambda z: np.repeat(z, n_features))
    mu = LinearApproximator(input_shape=(n_features,), output_shape=(action_dim,))
    if goal is None:
        goal = np.ones(action_dim)
    return DMP(mu, phi, goal=goal, dt=0.01, tau=1.0,
               alpha_v=np.ones(action_dim), beta_v=np.ones(action_dim) * 0.25,
               alpha_z=np.ones(action_dim), beta_z=np.ones(action_dim) * 0.25)


def test_policy_state_shape():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    assert pi.policy_state_shape == (4, 1)


def test_reset_returns_initial_state():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    state = np.zeros(1)

    # evolve the dynamical system away from the default before resetting
    ps = pi.reset()
    for _ in range(5):
        _, ps = pi.draw_action(state, ps)
    assert not np.allclose(ps, 0.0)

    ps_reset = pi.reset()
    assert ps_reset.shape == (4, 1)
    assert np.all(ps_reset == 0.0)


def test_weights_size_and_roundtrip():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    assert pi.weights_size == 4
    w = np.arange(pi.weights_size, dtype=float)
    pi.set_weights(w)
    assert np.allclose(pi.get_weights(), w)


def test_draw_action_dynamics():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    state = np.zeros(1)

    ps = pi.reset()
    a1, ps = pi.draw_action(state, ps)
    a2, ps = pi.draw_action(state, ps)
    a3, ps = pi.draw_action(state, ps)

    # zero-weight regressor -> forcing term is zero, dynamics driven only by the goal
    assert np.allclose(a1, np.array([0.0]))
    assert np.allclose(a2, np.array([2.5e-05]))
    assert np.allclose(a3, np.array([7.475e-05]))


def test_draw_action_does_not_mutate_input_policy_state():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    state = np.zeros(1)

    ps = np.full((4, 1), 0.5)
    ps_before = ps.copy()
    pi.draw_action(state, ps)
    assert np.allclose(ps, ps_before)


def test_set_get_goal():
    np.random.seed(1)
    pi = _make_dmp(action_dim=2)
    new_goal = np.array([3.0, 4.0])
    pi.set_goal(new_goal)
    assert np.allclose(pi.get_goal(state=None), new_goal)


def test_call_matches_deterministic_action():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    state = np.zeros(1)

    ps = pi.reset()
    action, _ = pi.draw_action(state, ps.copy())

    assert pi(state, action, ps.copy()) == 1.0
    assert pi(state, action + 1.0, ps.copy()) == 0.0