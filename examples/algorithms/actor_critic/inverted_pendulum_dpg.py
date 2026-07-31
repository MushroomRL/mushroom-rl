"""
Simple script to solve the Inverted Pendulum problem with the deterministic policy gradient algorithm
COPDAC-Q and tile coding.

The value function and the mean of the policy are drawn at every epoch, so that their shaping over the state
space can be followed during the run.

"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import COPDAC_Q
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.utils.callbacks import CollectDataset


class Display:
    def __init__(self, V, mu, low, high, psi):
        plt.ion()

        self._V = V
        self._mu = mu
        self._psi = psi

        self._fig = plt.figure(figsize=(10, 5))
        ax1 = self._fig.add_subplot(1, 2, 1)
        ax2 = self._fig.add_subplot(1, 2, 2)

        self._theta = np.linspace(low[0], high[0], 100)
        self._omega = np.linspace(low[1], high[1], 100)

        vv, mm = self._compute_data()

        ext = [low[0], high[0],
               low[1], high[1]]

        ax1.set_title('V')
        im1 = ax1.imshow(vv, cmap=cm.coolwarm, extent=ext, aspect='auto')
        self._fig.colorbar(im1, ax=ax1)

        ax2.set_title('mean')
        im2 = ax2.imshow(mm, cmap=cm.coolwarm, extent=ext, aspect='auto')
        self._fig.colorbar(im2, ax=ax2)

        self._im = [im1, im2]

        self._counter = 0

        plt.draw()
        plt.pause(0.1)

    def pump(self, *args, **kwargs):
        self._counter += 1
        if self._counter % 100 == 0:
            self._fig.canvas.flush_events()

    def __call__(self, *args, **kwargs):
        vv, mm = self._compute_data()

        self._im[0].set_data(vv)
        self._im[0].autoscale()
        self._im[1].set_data(mm)
        self._im[1].autoscale()

        self._counter = 0

        plt.draw()
        plt.pause(.1)

    def _compute_data(self):
        n_points = len(self._theta) * len(self._omega)
        vv = np.empty(n_points)
        mm = np.empty(n_points)

        c = 0
        for y in self._omega:
            for x in self._theta:
                s = np.array([x, y])
                vv[c] = self._V(self._psi(s)).item()
                mm[c] = self._mu(s).item()
                c += 1

        shape = (len(self._theta), len(self._omega))
        vv = vv.reshape(shape)
        mm = mm.reshape(shape)

        return vv, mm


def experiment(n_epochs, n_episodes, render=True, seed=None):
    np.random.seed(seed)

    # MDP
    n_steps = 5000
    mdp = InvertedPendulum(horizon=n_steps)

    logger = Logger(COPDAC_Q.name(), results_dir=None)
    logger.log_experiment_info(COPDAC_Q, mdp, n_epochs=n_epochs, n_episodes=n_episodes)

    # Agent
    n_tilings = 10
    alpha_theta = Parameter(5e-3 / n_tilings)
    alpha_omega = Parameter(0.5 / n_tilings)
    alpha_v = Parameter(0.5 / n_tilings)
    tilings = Tiles.generate(n_tilings, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings)

    input_shape = mdp.info.observation_space.shape

    mu = LinearApproximator(input_shape=input_shape,
                            output_shape=mdp.info.action_space.shape,
                            phi=phi)

    sigma = 1e-1 * np.eye(1)
    policy = GaussianPolicy(mu, sigma)

    agent = COPDAC_Q(mdp.info, policy, mu,
                     alpha_theta, alpha_omega, alpha_v,
                     value_function_features=phi)

    # Train
    dataset_callback = CollectDataset()
    visualization_callback = Display(agent._V, mu,
                                     mdp.info.observation_space.low,
                                     mdp.info.observation_space.high,
                                     phi)
    core = Core(agent, mdp, logger=logger, callbacks_fit=[dataset_callback],
                callback_step=visualization_callback.pump)

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes,
                   n_steps_per_fit=1, render=False)
        R = dataset_callback.get().undiscounted_return.sum() / n_steps / n_episodes
        dataset_callback.clean()
        visualization_callback()

        logger.log_evaluation(i + 1, R_mean=R)

    if render:
        logger.info('Press a button to visualize the pendulum')
        input()
        sigma = 1e-8 * np.eye(1)
        policy.set_sigma(sigma)
        core.evaluate(n_steps=n_steps, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=24, n_episodes=5, render=args.render)
