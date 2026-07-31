"""
Simple script to solve the Puddle World problem with True Online SARSA(lambda) and tile coding.

"""
import numpy as np

from tqdm import trange

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import PuddleWorld
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(alpha, n_epochs, n_steps, seed=None):
    np.random.seed(seed)

    # MDP
    mdp = PuddleWorld(horizon=1000)

    logger = Logger(TrueOnlineSARSALambda.name(), results_dir=None)
    logger.log_experiment_info(TrueOnlineSARSALambda, mdp, alpha=alpha, n_epochs=n_epochs, n_steps=n_steps)

    # Policy
    epsilon = Parameter(value=0.1)
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

    agent = TrueOnlineSARSALambda(mdp.info, pi, approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # Train
    dataset = core.evaluate(n_episodes=5, render=False)
    J = dataset.discounted_return.mean()

    logger.log_evaluation(0, J=J)

    for epoch in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1, render=False)
        dataset = core.evaluate(n_episodes=5, render=False)
        J = dataset.discounted_return.mean()

        logger.log_evaluation(epoch + 1, J=J)

    # Visualize the final policy
    core.evaluate(n_episodes=5, render=True, quiet=True)


if __name__ == '__main__':
    experiment(alpha=.1, n_epochs=10, n_steps=5000)
