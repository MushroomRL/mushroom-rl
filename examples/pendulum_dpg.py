import numpy as np

from mushroom.algorithms.actor_critic import COPDAC_Q
from mushroom.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.policy import GaussianPolicy
from mushroom.utils.dataset import compute_J
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter

from tqdm import tqdm
tqdm.monitor_interval = 0


def experiment(n_epochs, n_steps, n_eval_episodes):
    np.random.seed()

    # MDP
    mdp = InvertedPendulum()

    # Agent
    n_tilings = 10
    alpha_theta = ExponentialDecayParameter(1, decay_exp=1.0)
    alpha_omega = ExponentialDecayParameter(1.5/n_tilings, decay_exp=2/3)
    alpha_v = ExponentialDecayParameter(1/n_tilings, decay_exp=2/3)
    tilings = Tiles.generate(n_tilings, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)

    phi = Features(tilings=tilings)
    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator, input_shape=input_shape,
                   output_shape=mdp.info.action_space.shape)

    sigma = 1e-3*np.eye(1)
    policy = GaussianPolicy(mu, sigma)

    agent = COPDAC_Q(policy, mu, mdp.info,
                     alpha_theta, alpha_omega, alpha_v,
                     value_function_features=phi,
                     policy_features=phi)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=n_eval_episodes)
    J = compute_J(dataset_eval, gamma=1.0)
    print('Total Reward per episode at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_steps=n_steps,
                   n_steps_per_fit=1)
        dataset_eval = core.evaluate(n_episodes=n_eval_episodes, render=False)
        J = compute_J(dataset_eval, gamma=1.0)
        print('Total Reward per episode at iteration ' + str(i) + ': ' +
              str(np.mean(J)))

    #core.evaluate(n_episodes=1, render=True)

if __name__ == '__main__':
    n_epochs = 50
    n_steps = 25000
    n_eval_episodes = 10
    experiment(n_epochs, n_steps, n_eval_episodes)

