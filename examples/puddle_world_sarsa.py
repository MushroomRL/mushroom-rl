import numpy as np
from joblib import Parallel, delayed

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import PuddleWorld
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

from tqdm import trange


def experiment(alpha, quiet):
    np.random.seed()

    # Logger
    logger = Logger('PuddleWorld_TrueOnlineSARSALambda', results_dir=None)
    logger.strong_line()
    logger.info('Environment: PuddleWorld')
    logger.info('Experiment Algorithm: TrueOnlineSARSALambda')


    # MDP
    mdp = PuddleWorld(horizon=1000)


    # Policy
    epsilon = Parameter(value=0.1)
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

    agent = TrueOnlineSARSALambda(mdp.info, pi, approximator_params=approximator_params, **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    dataset = core.evaluate(n_episodes=5, render=False)
    J = np.mean(dataset.discounted_return)

    logger.epoch_info(0, J=J)

    for epoch in trange(10, leave=False, disable=quiet):
        core.learn(n_steps=5000, n_steps_per_fit=1, render=False)
        dataset = core.evaluate(n_episodes=5, render=False, quiet=quiet)
        J = np.mean(dataset.discounted_return)
        logger.epoch_info(epoch + 1, J=J)

    if not quiet:
        core.evaluate(n_episodes=5, render=True, quiet=True)


    return np.mean(dataset.undiscounted_return)


if __name__ == '__main__':
    n_experiment = 1

    alpha = .1
    Js = Parallel(n_jobs=-1)(delayed(experiment)(alpha, n_experiment > 1) for _ in range(n_experiment))
