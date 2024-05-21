import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from mushroom_rl.algorithms.actor_critic import StochasticAC_AVG
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import StateLogStdGaussianPolicy
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.rl_utils.parameters import Parameter

from tqdm import tqdm, trange
tqdm.monitor_interval = 0


class Display:
    def __init__(self, V, mu, std, low, high, phi, psi):
        plt.ion()

        self._V = V
        self._mu = mu
        self._std = std
        self._phi = phi
        self._psi = psi

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        self._theta = np.linspace(low[0], high[0], 100)
        self._omega = np.linspace(low[1], high[1], 100)

        vv, mm, ss = self._compute_data()

        ext = [low[0], high[0],
               low[1], high[1]]

        ax1.set_title('V')
        im1 = ax1.imshow(vv, cmap=cm.coolwarm, extent=ext, aspect='auto')
        fig.colorbar(im1, ax=ax1)

        ax2.set_title('mean')
        im2 = ax2.imshow(mm, cmap=cm.coolwarm, extent=ext, aspect='auto')
        fig.colorbar(im2, ax=ax2)

        ax3.set_title('sigma')
        im3 = ax3.imshow(ss, cmap=cm.coolwarm, extent=ext, aspect='auto')
        fig.colorbar(im3, ax=ax3)

        self._im = [im1, im2, im3]

        self._counter = 0

        plt.draw()
        plt.pause(.1)

    def __call__(self, *args, **kwargs):
        vv, mm, ss = self._compute_data()

        self._im[0].set_data(vv)
        self._im[0].autoscale()
        self._im[1].set_data(mm)
        self._im[1].autoscale()
        self._im[2].set_data(ss)
        self._im[2].autoscale()

        self._counter = 0

        plt.draw()
        plt.pause(.1)

    def _compute_data(self):
        n_points = len(self._theta) * len(self._omega)
        vv = np.empty(n_points)
        mm = np.empty(n_points)
        ss = np.empty(n_points)

        c = 0
        for y in self._omega:
            for x in self._theta:
                s = self._phi(np.array([x, y]))
                s_v = self._psi(np.array([x, y]))
                vv[c] = self._V(s_v)
                mm[c] = self._mu(s)
                ss[c] = np.exp(self._std(s)) ** 2
                c += 1

        shape = (len(self._theta), len(self._omega))
        vv = vv.reshape(shape)
        mm = mm.reshape(shape)
        ss = ss.reshape(shape)

        return vv, mm, ss


def experiment(n_epochs, n_episodes):
    np.random.seed()

    logger = Logger(StochasticAC_AVG.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + StochasticAC_AVG.__name__)

    # MDP
    n_steps = 5000
    mdp = InvertedPendulum(horizon=n_steps)

    # Agent
    n_tilings = 11
    alpha_r = Parameter(.0001)
    alpha_theta = Parameter(.001 / n_tilings)
    alpha_v = Parameter(.1 / n_tilings)
    tilings = Tiles.generate(n_tilings-1, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)

    tilings_v = tilings + Tiles.generate(1, [1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)
    psi = Features(tilings=tilings_v)

    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator,
                   input_shape=input_shape,
                   output_shape=mdp.info.action_space.shape,
                   phi=phi)

    std = Regressor(LinearApproximator,
                    input_shape=input_shape,
                    output_shape=mdp.info.action_space.shape,
                    phi=phi)

    std_0 = np.sqrt(1.)
    std.set_weights(np.log(std_0) / n_tilings * np.ones(std.weights_size))

    policy = StateLogStdGaussianPolicy(mu, std)

    agent = StochasticAC_AVG(mdp.info, policy,
                             alpha_theta, alpha_v, alpha_r,
                             lambda_par=.5,
                             value_function_features=psi)

    # Train
    dataset_callback = CollectDataset()
    display_callback = Display(agent._V, mu, std,
                               mdp.info.observation_space.low,
                               mdp.info.observation_space.high,
                               phi, psi)
    core = Core(agent, mdp, callbacks_fit=[dataset_callback])

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=n_episodes,
                   n_steps_per_fit=1, render=False)
        J = dataset_callback.get().undiscounted_return
        dataset_callback.clean()
        display_callback()
        logger.epoch_info(i+1, R_mean=np.sum(J) / n_steps/n_episodes)

    logger.info('Press a button to visualize the pendulum...')
    input()
    core.evaluate(n_steps=n_steps, render=True)


if __name__ == '__main__':
    n_epochs = 24
    n_episodes = 5

    experiment(n_epochs, n_episodes)
