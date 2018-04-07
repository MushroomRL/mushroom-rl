from mushroom.policy.gaussian_policy import *
from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.utils.numerical_gradient import numerical_diff_policy


def univariate_gaussian():
    print('Testing univariate gaussian policy...')
    sigma = 1e-3*np.eye(1)

    n_dims = 5

    approximator_params = dict(input_dim=n_dims)
    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(1,),
                             params=approximator_params)

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


def multivariate_gaussian():
    print('Testing multivariate gaussian policy...')
    n_dims = 5
    n_outs = 3

    random_matrix = np.random.rand(n_outs, n_outs)

    sigma = random_matrix.dot(random_matrix.T)

    approximator_params = dict(input_dim=n_dims)
    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(n_outs,),
                             params=approximator_params)

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


def multivariate_diagonal_gaussian():
    print('Testing multivariate diagonal gaussian policy...')
    n_dims = 5
    n_outs = 3

    std = np.random.randn(n_outs)

    approximator_params = dict(input_dim=n_dims)
    approximator = Regressor(LinearApproximator,
                             input_shape=(n_dims,),
                             output_shape=(n_outs,),
                             params=approximator_params)

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


def multivariate_state_std_gaussian():
    print('Testing multivariate state std gaussian policy...')
    n_dims = 5
    n_outs = 3

    std = np.random.randn(n_outs)

    approximator_params = dict(input_dim=n_dims)
    mu_approximator = Regressor(LinearApproximator,
                                input_shape=(n_dims,),
                                output_shape=(n_outs,),
                                params=approximator_params)

    std_approximator = Regressor(LinearApproximator,
                                 input_shape=(n_dims,),
                                 output_shape=(n_outs,),
                                 params=approximator_params)

    pi = StateStdGaussianPolicy(mu_approximator, std_approximator)
    mu_weights = np.random.rand(pi.weights_size)+0.1
    pi.set_weights(mu_weights)

    x = np.random.randn(20, n_dims)

    for x_i in x:
        state = np.atleast_1d(x_i)
        action = pi.draw_action(state)
        exact_diff = pi.diff(state, action)
        numerical_diff = numerical_diff_policy(pi, state, action)

        assert np.allclose(exact_diff, numerical_diff)

if __name__ == '__main__':
    print('Executing policy test...')

    univariate_gaussian()
    multivariate_gaussian()
    multivariate_diagonal_gaussian()
    multivariate_state_std_gaussian()
