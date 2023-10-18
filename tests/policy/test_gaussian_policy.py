from mushroom_rl.policy.gaussian_policy import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.utils.numerical_gradient import numerical_diff_policy


def test_univariate_gaussian():
    np.random.seed(88)
    sigma = 1e-3 * np.eye(1)

    n_dims = 5

    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(1,))

    pi = GaussianPolicy(approximator, sigma)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action, _ = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_gaussian():
    np.random.seed(88)
    n_dims = 5
    n_outs = 3

    random_matrix = np.random.rand(n_outs, n_outs)

    sigma = random_matrix.dot(random_matrix.T)

    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(n_outs,))

    pi = GaussianPolicy(approximator, sigma)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action, _ = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_diagonal_gaussian():
    np.random.seed(88)
    n_dims = 5
    n_outs = 3

    std = np.random.randn(n_outs)

    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(n_outs,))

    pi = DiagonalGaussianPolicy(approximator, std)
    mu_weights = np.random.rand(pi.weights_size)
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action, _ = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_state_std_gaussian():
    np.random.seed(88)
    n_dims = 5
    n_outs = 3

    mu_approximator = Regressor(LinearApproximator,
                                input_shape=(n_dims,),
                                output_shape=(n_outs,))

    std_approximator = Regressor(LinearApproximator,
                                 input_shape=(n_dims,),
                                 output_shape=(n_outs,))

    pi = StateStdGaussianPolicy(mu_approximator, std_approximator)
    weights = np.random.rand(pi.weights_size) + .1
    pi.set_weights(weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action, _ = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)


def test_multivariate_state_log_std_gaussian():
    np.random.seed(88)
    n_dims = 5
    n_outs = 3

    mu_approximator = Regressor(LinearApproximator,
                                input_shape=(n_dims,),
                                output_shape=(n_outs,))

    log_var_approximator = Regressor(LinearApproximator,
                                     input_shape=(n_dims,),
                                     output_shape=(n_outs,))

    pi = StateLogStdGaussianPolicy(mu_approximator, log_var_approximator)
    weights = np.random.rand(pi.weights_size)
    pi.set_weights(weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action, _ = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)
