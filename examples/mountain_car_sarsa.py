import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

"""
This script aims to replicate the experiments on the Mountain Car MDP as
presented in:
"True Online TD(lambda)". Seijen H. V. et al.. 2014.

"""


def experiment(alpha):
    np.random.seed()

    # MDP
    mdp = Gymnasium(name='MountainCar-v0', horizon=int(1e4), gamma=1., headless=False)

    # Policy
    epsilon = Parameter(value=0.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    n_tilings = 10
    tilings = Tiles.generate(n_tilings, [10, 10],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high)
    features = Features(tilings=tilings)

    learning_rate = Parameter(alpha / n_tilings)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n,
                               phi=features)
    algorithm_params = {'learning_rate': learning_rate,
                        'lambda_coeff': .9}

    agent = TrueOnlineSARSALambda(mdp.info, pi,
                                  approximator_params=approximator_params,
                                  **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=40, n_steps_per_fit=1, render=False)
    dataset = core.evaluate(n_episodes=1, render=True)

    return np.mean(dataset.undiscounted_return)


if __name__ == '__main__':
    n_experiment = 1

    logger = Logger(TrueOnlineSARSALambda.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + TrueOnlineSARSALambda.__name__)

    alpha = .1
    Js = Parallel(
        n_jobs=-1)(delayed(experiment)(alpha) for _ in range(n_experiment))

    logger.info('J: %f' % np.mean(Js))
