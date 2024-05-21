import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mushroom_rl.algorithms.actor_critic import COPDAC_Q
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import GaussianPolicy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.utils.callbacks import CollectDataset

from tqdm import tqdm, trange

tqdm.monitor_interval = 0


class Display:
    def __init__(self, V, mu, low, high, phi, psi):
        plt.ion()

        self._V = V
        self._mu = mu
        self._phi = phi
        self._psi = psi

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        self._theta = np.linspace(low[0], high[0], 100)
        self._omega = np.linspace(low[1], high[1], 100)

        vv, mm = self._compute_data()

        ext = [low[0], high[0],
               low[1], high[1]]

        ax1.set_title('V')
        im1 = ax1.imshow(vv, cmap=cm.coolwarm, extent=ext, aspect='auto')
        fig.colorbar(im1, ax=ax1)

        ax2.set_title('mean')
        im2 = ax2.imshow(mm, cmap=cm.coolwarm, extent=ext, aspect='auto')
        fig.colorbar(im2, ax=ax2)

        self._im = [im1, im2]

        self._counter = 0

        plt.draw()
        plt.pause(0.1)

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
                s = self._phi(np.array([x, y]))
                s_v = self._psi(np.array([x, y]))
                vv[c] = self._V(s_v)
                mm[c] = self._mu(s)
                c += 1

        shape = (len(self._theta), len(self._omega))
        vv = vv.reshape(shape)
        mm = mm.reshape(shape)

        return vv, mm


def experiment(n_epochs, n_episodes):
    np.random.seed()

    logger = Logger(COPDAC_Q.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + COPDAC_Q.__name__)

    # MDP
    n_steps = 5000
    mdp = InvertedPendulum(horizon=n_steps)

    # Agent
    n_tilings = 10
    alpha_theta = Parameter(5e-3 / n_tilings)
    alpha_omega = Parameter(0.5 / n_tilings)
    alpha_v = Parameter(0.5 / n_tilings)
    tilings = Tiles.generate(n_tilings, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)

    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator,
                   input_shape=input_shape,
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
                                     phi, phi)
    core = Core(agent, mdp, callbacks_fit=[dataset_callback])

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes,
                   n_steps_per_fit=1, render=False)
        J = dataset_callback.get().undiscounted_return
        dataset_callback.clean()
        visualization_callback()
        logger.epoch_info(i+1, R_mean=np.sum(J) / n_steps/n_episodes)

    logger.info('Press a button to visualize the pendulum...')
    input()
    sigma = 1e-8 * np.eye(1)
    policy.set_sigma(sigma)
    core.evaluate(n_steps=n_steps, render=True)


if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 5

    experiment(n_epochs, n_episodes)
