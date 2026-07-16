import numpy as np

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter

"""
This script runs the Mountain Car experiment on the ``'list'`` dataset backend.

When the environment horizon is not finite the collected dataset cannot be pre-allocated, so MushroomRL
automatically switches its storage to the ``'list'`` backend, which grows the dataset one step at a time. No
change to the agent is required.

An environment can also request this backend directly by declaring ``'list'`` in its ``MDPInfo``, which is
convenient when its states or actions are structured, non-array objects.

"""


def experiment(alpha):
    logger = Logger('ListBackendExperiment', results_dir=None)
    logger.strong_line()
    logger.info('Mountain Car with infinite horizon and list dataset backend')
    logger.weak_line()

    np.random.seed(0)

    # MDP
    mdp = Gymnasium(name='MountainCar-v0', horizon=np.inf, gamma=1.)

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
    core = Core(agent, mdp)

    # Evaluate
    logger.info('- Evaluating random policy for 10000 steps')
    dataset = core.evaluate(n_steps=10000, render=False)
    logger.info(f'R: {dataset.undiscounted_return.mean()}')
    episode_length = dataset.episodes_length
    if len(episode_length) > 0:
        logger.info(f'completed episodes: {len(episode_length)}, mean episode length: {episode_length.mean()}')
    else:
        logger.info(f'episode length: {len(dataset)}, episode not completed')

    # Train
    logger.info('- Learning for 100 episodes')
    core.learn(n_episodes=100, n_steps_per_fit=1, render=False)
    logger.info('- Evaluating for one episode')
    dataset = core.evaluate(n_episodes=1, render=True)
    logger.info(f'R: {dataset.undiscounted_return.mean()}')
    logger.info(f'episode length: {dataset.episodes_length.item()}')

    backend = dataset.array_backend.get_backend_name()
    logger.weak_line()
    logger.info(f'dataset backend: {backend}')


if __name__ == '__main__':
    experiment(alpha=.1)
