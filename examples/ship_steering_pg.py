import numpy as np

from mushroom.algorithms.policy_search import REINFORCE, GPOMDP, eNAC
from mushroom.approximators.parametric import LinearApproximator
from mushroom.approximators.regressor import Regressor
from mushroom.core import Core
from mushroom.environments import ShipSteering
from mushroom.features.tiles import Tiles
from mushroom.features.features import Features
from mushroom.policy import MultivariateDiagonalGaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import Parameter
from tqdm import tqdm


"""
This script aims to replicate the experiments on the Ship Steering MDP 
using policy gradient algorithms.

"""

tqdm.monitor_interval = 0

def experiment(alg, learning_rate, n_runs, n_iterations, ep_per_run):
    np.random.seed()

    # MDP
    mdp = ShipSteering()

    # Policy
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 6]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 2

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    phi = Features(tilings=tilings)
    input_shape = (phi.size,)

    approximator_params = dict(input_dim=phi.size)
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape,
                             params=approximator_params)

    std = np.array([3e-1])
    policy = MultivariateDiagonalGaussianPolicy(mu=approximator, std=std)

    # Agent
    algorithm_params = dict(learning_rate=learning_rate)
    agent = alg(policy, mdp.info, features=phi, **algorithm_params)

    # Train
    print(alg.__name__)
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_runs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))


if __name__ == '__main__':

    algs_params =[
        (REINFORCE, Parameter(1e-4)),
        #(GPOMDP, Parameter(1e-4)),
        #(eNAC, Parameter(1e-2))
    ]


    for alg, learning_rate in algs_params:
        experiment(alg, learning_rate, n_runs=100, n_iterations=5, ep_per_run=40)
