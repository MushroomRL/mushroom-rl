import numpy as np

from mushroom_rl.algorithms.policy_search import REPS, RWR, PGPE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import ShipSteering
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.features.features import Features
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.lr_optimizers import AdaptiveParameterOptimizer

from tqdm import tqdm


"""
This script aims to replicate the experiments on the Ship Steering MDP 
using policy gradient algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, n_iterations, ep_per_run):
    np.random.seed()

    # MDP
    mdp = ShipSteering()

    # Policy
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    n_tiles = [5, 5, 6]
    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    phi = Features(tilings=tilings)
    input_shape = (phi.size,)

    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 4e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, features=phi, **params)

    # Train
    print(alg.__name__)
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_run)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset_eval = core.evaluate(n_episodes=ep_per_run)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))


if __name__ == '__main__':

    algs_params = [
        (REPS, {'eps': 1.0}),
        (RWR, {'beta': 0.7}),
        (PGPE, {'grad_optimizer': AdaptiveParameterOptimizer(value=1.5)}),
        ]

    for alg, params in algs_params:
        experiment(alg, params, n_epochs=25, n_iterations=10, ep_per_run=20)
