import numpy as np

from mushroom_rl.core import Core
from mushroom_rl.environments.segway import Segway
from mushroom_rl.algorithms.policy_search import *
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.parameters import AdaptiveParameter

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, n_episodes, n_ep_per_fit):
    np.random.seed()

    # MDP
    mdp = Segway()

    # Policy
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    n_weights = approximator.weights_size
    mu = np.zeros(n_weights)
    sigma = 2e-0 * np.ones(n_weights)
    policy = DeterministicPolicy(approximator)
    dist = GaussianDiagonalDistribution(mu, sigma)

    agent = alg(mdp.info, dist, policy, **params)

    # Train
    print(alg.__name__)
    dataset_callback = CollectDataset()
    core = Core(agent, mdp, callbacks_fit=[dataset_callback])

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=n_ep_per_fit, render=False)
        J = compute_J(dataset_callback.get(), gamma=mdp.info.gamma)
        dataset_callback.clean()

        p = dist.get_parameters()

        print('mu:    ', p[:n_weights])
        print('sigma: ', p[n_weights:])
        print('Reward at iteration ' + str(i) + ': ' +
              str(np.mean(J)))

    print('Press a button to visualize the segway...')
    input()
    core.evaluate(n_episodes=3, render=True)


if __name__ == '__main__':
    algs_params = [
        (REPS, {'eps': 0.05}),
        (RWR, {'beta': 0.01}),
        (PGPE, {'learning_rate':  AdaptiveParameter(value=0.3)}),
        ]
    for alg, params in algs_params:
        experiment(alg, params, n_epochs=20, n_episodes=100, n_ep_per_fit=25)
