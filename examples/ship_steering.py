from mushroom.approximators.regressor import Regressor
from mushroom.policy import GaussianPolicy
from mushroom.utils.parameters import Parameter
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core.core import Core


import numpy as np


def experiment():
    np.random.seed()

    # MDP
    mdp = None

    # Policy
    sigma = Parameter(value=1)
    pi = GaussianPolicy(sigma=sigma)

    approximator_params = dict(
        params=np.array([1, 1]))
        #params_shape=(2,1))
    approximator = Regressor(LinearApproximator, input_shape=(2,), output_shape=(1,), params=approximator_params)

    s = np.array([-1.1288, -1.4633])
    a = np.array([-1.8027])

    g = pi.diff(approximator, s, a)

    print a
    print g

    a = pi(approximator, s)

    # Agent
    """
    shape = mdp.observation_space.size + mdp.action_space.size
    learning_rate = Parameter(value=.2)
    algorithm_params = dict(learning_rate=learning_rate)
    fit_params = dict()
    agent_params = {'algorithm_params': algorithm_params,
                    'fit_params': fit_params}
    agent = None

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_iterations=10000, how_many=1, n_fit_steps=1,
               iterate_over='samples')
    """


if __name__ == '__main__':
    experiment()
