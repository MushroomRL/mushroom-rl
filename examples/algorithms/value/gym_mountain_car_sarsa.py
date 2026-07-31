"""
Simple script to solve the Mountain Car problem with True Online SARSA(lambda) and tile coding.

The environment, the tile coding and the trace decay follow the mountain car setup used in:
"True Online TD(lambda)". Seijen H. V. et al. 2014.

"""
import numpy as np

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(alpha, n_episodes, seed=0):
    np.random.seed(seed)

    # MDP
    mdp = Gymnasium(name='MountainCar-v0', horizon=int(1e4), gamma=1., headless=False)
    mdp.seed(seed)

    logger = Logger(TrueOnlineSARSALambda.name(), results_dir=None)
    logger.log_experiment_info(TrueOnlineSARSALambda, mdp, alpha=alpha, n_episodes=n_episodes)

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    n_tilings = 10
    tilings = Tiles.generate(n_tilings, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings)

    learning_rate = Parameter(alpha / n_tilings)

    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda_coeff': .9}

    agent = TrueOnlineSARSALambda(mdp.info, pi,
                                  approximator_params=approximator_params,
                                  **algorithm_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Train
    core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False)

    # Visualize the final policy
    dataset = core.evaluate(n_episodes=1, render=True)

    logger.info(f'R: {dataset.undiscounted_return.mean()}')


if __name__ == '__main__':
    experiment(alpha=.1, n_episodes=40)
