import os

from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.callbacks import PlotDataset

import numpy as np

from mushroom_rl.algorithms.policy_search import REINFORCE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import LQR
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveParameterOptimizer

from tqdm import tqdm


"""
This script shows how to use preprocessors and plot callback.

"""

tqdm.monitor_interval = 0


def experiment(n_epochs, n_iterations, ep_per_run, save_states_to_disk):
    np.random.seed()

    # MDP
    mdp = LQR.generate(dimensions=2, max_pos=10., max_action=5., episodic=True)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    sigma = Regressor(LinearApproximator,
                      input_shape=mdp.info.observation_space.shape,
                      output_shape=mdp.info.action_space.shape)

    sigma_weights = 2 * np.ones(sigma.weights_size)
    sigma.set_weights(sigma_weights)

    policy = StateStdGaussianPolicy(approximator, sigma)

    # Agent
    optimizer = AdaptiveParameterOptimizer(value=.01)
    algorithm_params = dict(learning_rate=optimizer)
    agent = REINFORCE(mdp.info, policy, **algorithm_params)

    # normalization callback
    prepro = MinMaxPreprocessor(mdp_info=mdp.info)

    # plotting callback
    plotter = PlotDataset(mdp.info, obs_normalized=True)

    # Train
    core = Core(agent, mdp, callback_step=plotter, preprocessors=[prepro])

    # training loop
    for n in range(n_epochs):
        core.learn(n_episodes=n_iterations * ep_per_run,
                   n_episodes_per_fit=ep_per_run)
        dataset = core.evaluate(n_episodes=ep_per_run, render=False)
        print('Epoch: ', n, '  J: ', np.mean(compute_J(dataset,
                                                       mdp.info.gamma)))

    if save_states_to_disk:
        # save normalization / plot states to disk path
        os.makedirs("./temp/", exist_ok=True)
        prepro.save_state("./temp/normalization_state")
        plotter.save_state("./temp/plotting_state")

        # load states from disk path
        prepro.load_state("./temp/normalization_state")
        plotter.load_state("./temp/plotting_state")


if __name__ == '__main__':
    experiment(n_epochs=10, n_iterations=10, ep_per_run=100,
               save_states_to_disk=False)
