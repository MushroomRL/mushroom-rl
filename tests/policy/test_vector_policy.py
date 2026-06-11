import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy.gaussian_policy import GaussianPolicy
from mushroom_rl.policy.vector_policy import VectorPolicy
from mushroom_rl.policy.policy import ParametricPolicy


def _make_gaussian_policy(state_dim=3, action_dim=2):
    mu = Regressor(LinearApproximator, input_shape=(state_dim,), output_shape=(action_dim,))
    return GaussianPolicy(mu, np.eye(action_dim))


class _StatefulPolicy(ParametricPolicy):
    """Stateful policy that increments its policy state at each step, for testing the stateful paths."""
    def __init__(self, reset_value):
        super().__init__(policy_state_shape=(reset_value.shape[0],))
        self._reset_value = reset_value.copy()
        self._w = np.zeros(3)
        self._add_save_attr(_w='numpy')

    def draw_action(self, state, policy_state=None):
        return self._w.copy(), policy_state + 1.0

    def reset(self):
        return self._reset_value.copy()

    def set_weights(self, weights):
        self._w = weights.copy()

    def get_weights(self):
        return self._w.copy()

    @property
    def weights_size(self):
        return 3


def test_len():
    np.random.seed(1)
    pi = _make_gaussian_policy()
    vpi = VectorPolicy(pi, 4)
    assert len(vpi) == 4


def test_weights_size():
    np.random.seed(1)
    pi = _make_gaussian_policy(state_dim=3, action_dim=2)
    vpi = VectorPolicy(pi, 2)
    assert vpi.weights_size == (2, 6)


def test_get_flat_policy():
    np.random.seed(1)
    pi = _make_gaussian_policy()
    vpi = VectorPolicy(pi, 3)
    assert isinstance(vpi.get_flat_policy(), GaussianPolicy)


def test_set_get_weights_per_env():
    np.random.seed(1)
    n_envs = 3
    pi = _make_gaussian_policy(state_dim=3, action_dim=2)
    vpi = VectorPolicy(pi, n_envs)

    new_weights = np.arange(n_envs * 6, dtype=float).reshape(n_envs, 6)
    vpi.set_weights(new_weights)

    assert np.allclose(vpi.get_weights(), new_weights)
    for i in range(n_envs):
        assert np.allclose(vpi._policy_vector[i].get_weights(), new_weights[i])


def test_draw_action_stateless_uses_per_env_weights():
    np.random.seed(5)
    n_envs = 2
    pi = _make_gaussian_policy(state_dim=2, action_dim=1)
    vpi = VectorPolicy(pi, n_envs)

    vpi.set_weights(np.stack([np.array([5.0, 0.0]), np.array([-5.0, 0.0])]))

    actions, next_ps = vpi.draw_action(np.zeros((n_envs, 2)), None)

    assert next_ps is None
    assert np.allclose(actions, np.array([[0.44122749], [-0.33087015]]))


def test_draw_action_stateful_returns_stacked_next_state():
    np.random.seed(1)
    n_envs = 3
    pi = _StatefulPolicy(np.array([7.0, 8.0]))
    vpi = VectorPolicy(pi, n_envs)

    policy_state = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    actions, next_ps = vpi.draw_action(np.zeros((n_envs, 4)), policy_state)

    assert actions.shape == (n_envs, 3)
    assert np.allclose(next_ps, np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))


def test_set_n_grow():
    np.random.seed(1)
    pi = _make_gaussian_policy()
    vpi = VectorPolicy(pi, 3)
    vpi.set_n(5)
    assert len(vpi) == 5


def test_set_n_shrink():
    np.random.seed(1)
    pi = _make_gaussian_policy()
    vpi = VectorPolicy(pi, 4)
    vpi.set_n(2)
    assert len(vpi) == 2


def test_set_n_grow_copies_are_independent():
    np.random.seed(1)
    pi = _make_gaussian_policy(state_dim=3, action_dim=2)
    vpi = VectorPolicy(pi, 1)
    vpi.set_n(3)

    # Assign distinct weights per env; the grown copies must not alias each other
    new_weights = np.arange(3 * 6, dtype=float).reshape(3, 6)
    vpi.set_weights(new_weights)
    assert np.allclose(vpi.get_weights(), new_weights)


def test_reset_stateful_no_mask():
    np.random.seed(1)
    n_envs = 3
    reset_value = np.array([3.14, 2.72])
    pi = _StatefulPolicy(reset_value)
    vpi = VectorPolicy(pi, n_envs)

    # evolve the policy state away from the reset value before resetting
    policy_state = vpi.reset()
    _, policy_state = vpi.draw_action(np.zeros((n_envs, 4)), policy_state)
    assert not np.allclose(policy_state, reset_value)

    policy_states = vpi.reset()

    assert policy_states.shape == (n_envs, 2)
    for i in range(n_envs):
        assert np.allclose(policy_states[i], reset_value)


def test_reset_stateful_with_mask():
    np.random.seed(1)
    n_envs = 3
    reset_value = np.array([3.14, 2.72])
    pi = _StatefulPolicy(reset_value)
    vpi = VectorPolicy(pi, n_envs)

    # evolve the policy state away from the reset value before resetting
    policy_state = vpi.reset()
    _, policy_state = vpi.draw_action(np.zeros((n_envs, 4)), policy_state)
    assert not np.allclose(policy_state, reset_value)

    mask = np.array([True, False, True])
    policy_states = vpi.reset_vectorized(mask)

    assert policy_states.shape == (n_envs, 2)
    assert np.allclose(policy_states[0], reset_value)
    assert np.allclose(policy_states[2], reset_value)