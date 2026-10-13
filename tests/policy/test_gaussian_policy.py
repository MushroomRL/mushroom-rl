import numpy as np
from mushroom_rl.policy.gaussian_policy import (GaussianPolicy, DiagonalGaussianPolicy,
                                                StateStdGaussianPolicy, StateLogStdGaussianPolicy)
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.rl_utils.parameters import LinearParameter
from mushroom_rl.utils.numerical_gradient import numerical_diff_policy


def test_univariate_gaussian():
    np.random.seed(42)
    sigma = 1e-3 * np.eye(1)

    n_dims = 5

    approximator = LinearApproximator(
                             input_shape=(n_dims,),
                             output_shape=(1,))

    pi = GaussianPolicy(approximator, sigma)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_gaussian_greedy():
    np.random.seed(42)
    n_dims = 5
    sigma = np.eye(2)

    approximator = LinearApproximator(input_shape=(n_dims,), output_shape=(2,))

    pi = GaussianPolicy(approximator, sigma)
    pi.set_weights(np.random.rand(pi.weights_size))

    state = np.random.randn(n_dims)

    mean = pi.draw_action_greedy(state)

    assert np.allclose(mean, approximator.predict(state))
    assert np.allclose(pi.draw_action_greedy(state), mean)


def test_gaussian_std_schedule():
    np.random.seed(42)
    std = LinearParameter(np.array([1., .5]), np.array([.1, .1]), n=2)

    approximator = LinearApproximator(input_shape=(3,), output_shape=(2,))

    pi = GaussianPolicy(approximator, std)
    pi.set_weights(np.random.rand(pi.weights_size))

    state = np.random.randn(3)
    actions = [pi.draw_action(state) for _ in range(3)]
    actions_test = [[1.2758453495, 0.8528825339], [0.9308643501, 1.0161040689], [0.786109301, 0.8194160585]]

    assert np.allclose(actions, actions_test)
    assert np.allclose(pi(state, actions[0]), 0.07054699961870194)

    diff_log = pi.diff_log(state, actions[0])
    diff_log_test = [47.1249775707, 22.9008681054, -14.0094923812, -21.9550547827, -10.669284947, 6.5268821029]

    assert np.allclose(diff_log, diff_log_test)
    assert np.allclose(pi.diff(state, actions[0]), numerical_diff_policy(pi, state, actions[0]))


def test_gaussian_std_vector():
    np.random.seed(42)
    approximator = LinearApproximator(input_shape=(3,), output_shape=(2,))
    approximator.set_weights(np.random.rand(approximator.weights_size))
    state = np.random.randn(3)

    np.random.seed(42)
    pi = GaussianPolicy(approximator, np.array([.3, .2]))
    action = pi.draw_action(state)

    np.random.seed(42)
    pi_matrix = GaussianPolicy(approximator, np.diag([.09, .04]))
    action_matrix = pi_matrix.draw_action(state)

    assert np.allclose(action, [1.1264515714, 0.9642549815])
    assert np.allclose(action, action_matrix)
    assert np.allclose(pi.diff_log(state, action), pi_matrix.diff_log(state, action))

    pi.set_sigma(np.array([.6, .4]))
    pi_matrix.set_sigma(np.diag([.36, .16]))

    assert np.allclose(pi(state, action), pi_matrix(state, action))
    assert np.allclose(pi.diff_log(state, action), pi_matrix.diff_log(state, action))


def test_multivariate_gaussian():
    np.random.seed(42)
    n_dims = 5
    n_outs = 3

    random_matrix = np.random.rand(n_outs, n_outs)

    sigma = random_matrix.dot(random_matrix.T)

    approximator = LinearApproximator(
                             input_shape=(n_dims,),
                             output_shape=(n_outs,))

    pi = GaussianPolicy(approximator, sigma)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_diagonal_gaussian():
    np.random.seed(42)
    n_dims = 5
    n_outs = 3

    std = np.random.randn(n_outs)

    approximator = LinearApproximator(
                             input_shape=(n_dims,),
                             output_shape=(n_outs,))

    pi = DiagonalGaussianPolicy(approximator, std)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_state_std_gaussian():
    np.random.seed(42)
    n_dims = 5
    n_outs = 3

    mu_approximator = LinearApproximator(
                                input_shape=(n_dims,),
                                output_shape=(n_outs,))

    std_approximator = LinearApproximator(
                                 input_shape=(n_dims,),
                                 output_shape=(n_outs,))

    pi = StateStdGaussianPolicy(mu_approximator, std_approximator)
    weights = np.random.rand(pi.weights_size) + .1
    pi.set_weights(weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_state_log_std_gaussian():
    np.random.seed(42)
    n_dims = 5
    n_outs = 3

    mu_approximator = LinearApproximator(
                                input_shape=(n_dims,),
                                output_shape=(n_outs,))

    log_var_approximator = LinearApproximator(
                                     input_shape=(n_dims,),
                                     output_shape=(n_outs,))

    pi = StateLogStdGaussianPolicy(mu_approximator, log_var_approximator)
    weights = np.random.rand(pi.weights_size)
    pi.set_weights(weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_gaussian_logs_sigma():
    class FakeLogger:
        def __init__(self):
            self.calls = list()

        def log_training(self, prefix=None, **kwargs):
            self.calls.append({(prefix + '/' + name if prefix else name): value for name, value in kwargs.items()})

    np.random.seed(42)
    approximator = LinearApproximator(input_shape=(3,), output_shape=(2,))
    pi = GaussianPolicy(approximator, LinearParameter(np.array([1., .5]), np.array([.1, .1]), n=2))

    logger = FakeLogger()
    pi.set_logger(logger)
    pi.draw_action(np.ones(3))

    assert len(logger.calls) == 1
    assert np.allclose(logger.calls[0]['policy/sigma'], np.array([.55, .3]))

    pi.set_sigma(np.diag([.09, .04]))
    pi.draw_action(np.ones(3))

    assert len(logger.calls) == 2
    assert np.allclose(logger.calls[1]['policy/sigma'], np.array([.3, .2]))
