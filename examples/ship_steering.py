import numpy as np

from mushroom.algorithms.policy_search import REINFORCE
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core.core import Core
from mushroom.environments import ShipSteering
from mushroom.features.basis import GaussianRBF
from mushroom.features.features import Features
from mushroom.features.tensors import gaussian_tensor
from mushroom.policy import GaussianPolicy, MultivariateGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter, AdaptiveParameter


def experiment(n_iterations, n_runs, ep_per_run, use_tensorflow):
    np.random.seed()

    # MDP
    mdp = ShipSteering()

    # Policy
    if use_tensorflow:
        tensor_list = gaussian_tensor.generate([3, 3, 6, 2],
                                               [[0., 150.],
                                                [0., 150.],
                                                [-np.pi, np.pi],
                                                [-np.pi / 12, np.pi / 12]])

        phi = Features(tensor_list=tensor_list, name='phi',
                       input_dim=mdp.observation_space.shape[0])
    else:
        basis = GaussianRBF.generate([3, 3, 6, 2],
                                     [[0., 150.],
                                      [0., 150.],
                                      [-np.pi, np.pi],
                                      [-np.pi / 12, np.pi / 12]])

        phi = Features(basis_list=basis)

    input_shape = (phi.size,)
    shape = input_shape + mdp.action_space.shape

    approximator_params = dict(params_shape=shape)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.action_space.shape,
                             params=approximator_params)
    #sigma = Parameter(value=.05)
    #policy = GaussianPolicy(mu=approximator, sigma=sigma)

    sigma = np.array([[.05]])
    policy = MultivariateGaussianPolicy(mu=approximator, sigma=sigma)

    # Agent
    learning_rate = AdaptiveParameter(value=.01)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = REINFORCE(policy, mdp.gamma, agent_params, phi)

    # Train
    core = Core(agent, mdp)
    for i in xrange(n_runs):
        core.learn(n_iterations=n_iterations, how_many=ep_per_run,
                   n_fit_steps=1, iterate_over='episodes')
        dataset_eval = core.evaluate(how_many=ep_per_run,
                                     iterate_over='episodes')
        J = compute_J(dataset_eval, gamma=mdp.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))

    np.save('ship_steering.npy', dataset_eval)


if __name__ == '__main__':
    experiment(n_iterations=40, n_runs=10, ep_per_run=100, use_tensorflow=True)
