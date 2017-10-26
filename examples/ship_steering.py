import numpy as np
from mushroom.core.core import Core
from mushroom.environments import ShipSteering
from mushroom.algorithms.policy_search import REINFORCE
from mushroom.policy import GaussianPolicy
from mushroom.approximators.regressor import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.features import Features
from mushroom.approximators.basis import GaussianRBF
from mushroom.utils.parameters import Parameter
from mushroom.utils.dataset import compute_J


def experiment(n_iterations, n_runs, ep_per_run):
    np.random.seed()

    # MDP
    mdp = ShipSteering()

    # Policy
    sigma = Parameter(value=0.05)
    policy = GaussianPolicy(sigma=sigma)


    # Agent
    basis = GaussianRBF.generate([3, 3, 6, 2],[[0.0, 150.0],[0.0, 150.0], [-np.pi, np.pi], [-np.pi/12, np.pi/12]])
    phi = Features(basis_list=basis)

    input_shape = (phi.size,)
    shape = input_shape + mdp.action_space.shape

    approximator_params = dict(params_shape=shape)
    approximator = Regressor(LinearApproximator, input_shape=input_shape, output_shape=mdp.action_space.shape,
                             params=approximator_params)

    learning_rate = Parameter(value=.001)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = REINFORCE(approximator, policy, mdp.gamma, agent_params, phi)

    # Train
    core = Core(agent, mdp)
    for i in xrange(n_runs):
        core.learn(n_iterations=n_iterations, how_many=ep_per_run, n_fit_steps=1,
               iterate_over='episodes')
        dataset_eval = core.evaluate(how_many=ep_per_run, iterate_over='episodes')
        J = compute_J(dataset_eval, gamma=mdp.gamma)
        print 'iteration ', i, ' J ', np.mean(J)

    np.save('ship_r', dataset_eval)

if __name__ == '__main__':
    #experiment(n_iterations=40, n_runs=10, ep_per_run=100)
    experiment(n_iterations=40, n_runs=2, ep_per_run=100)
