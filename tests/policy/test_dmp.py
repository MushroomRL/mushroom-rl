import numpy as np

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.policy.dmp import DMP


def _repeat_features(z):
    return np.repeat(z, 4, axis=-1)


def _make_dmp(action_dim=1, goal=None):
    n_features = 4
    phi = Features(n_outputs=n_features, function=_repeat_features)
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

    pi.reset()
    for _ in range(5):
        pi.draw_action(state)
    assert not np.allclose(pi.policy_state, 0.0)

    pi.reset()
    assert pi.policy_state.shape == (4, 1)
    assert np.all(pi.policy_state == 0.0)


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

    pi.reset()
    a1 = pi.draw_action(state)
    a2 = pi.draw_action(state)
    a3 = pi.draw_action(state)

    assert np.allclose(a1, np.array([0.0]))
    assert np.allclose(a2, np.array([2.5e-05]))
    assert np.allclose(a3, np.array([7.475e-05]))


def test_draw_action_explicit_state_does_not_mutate_or_advance():
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    state = np.zeros(1)
    pi.reset()

    ps = np.full((4, 1), 0.5)
    ps_before = ps.copy()
    pi.draw_action(state, ps)
    assert np.allclose(ps, ps_before)
    assert np.all(pi.policy_state == 0.0)


def test_draw_action_vectorized():
    np.random.seed(1)
    n_envs = 3
    pi = _make_dmp(action_dim=1)

    ps = pi.reset_vectorized(np.array([True, True, True]))
    assert ps.shape == (n_envs, 4, 1)

    a_single = _make_dmp(action_dim=1)
    a_single.reset()
    single_first = a_single.draw_action(np.zeros(1))

    actions = pi.draw_action(np.zeros((n_envs, 1)))
    assert actions.shape == (n_envs, 1)
    for i in range(n_envs):
        assert np.allclose(actions[i], single_first)

    before = pi.policy_state.copy()
    pi.reset_vectorized(np.array([True, False, True]))
    assert np.all(pi.policy_state[0] == 0.0)
    assert np.all(pi.policy_state[2] == 0.0)
    assert np.allclose(pi.policy_state[1], before[1])


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

    pi.reset()
    action = pi.draw_action(state, pi.policy_state)

    assert pi(state, action, pi.policy_state) == 1.0
    assert pi(state, action + 1.0, pi.policy_state) == 0.0


def test_serialization(tmpdir):
    np.random.seed(1)
    pi = _make_dmp(action_dim=1)
    pi.set_weights(np.arange(pi.weights_size, dtype=float))
    state = np.zeros(1)

    path = tmpdir / 'dmp.msh'
    pi.save(path)
    loaded = DMP.load(path)

    assert np.allclose(loaded.get_weights(), pi.get_weights())

    pi.reset()
    loaded.reset()
    for _ in range(5):
        assert np.allclose(pi.draw_action(state), loaded.draw_action(state))
