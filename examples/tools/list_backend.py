"""
This script runs the Mountain Car experiment on the ``'list'`` dataset backend.

When the environment horizon is not finite the collected dataset cannot be pre-allocated, so MushroomRL
automatically switches its storage to the ``'list'`` backend, which grows the dataset one step at a time. No
change to the agent is required.

An environment can also request this backend directly by declaring ``'list'`` in its ``MDPInfo``, which is
convenient when its states or actions are structured, non-array objects.

"""
import numpy as np

from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def experiment(alpha, n_episodes, n_steps_test, seed=0):
    np.random.seed(seed)

    # MDP
    mdp = Gymnasium(name='MountainCar-v0', horizon=np.inf, gamma=1.)
    mdp.seed(seed)

    logger = Logger('list_backend', results_dir=None)
    logger.log_experiment_info(TrueOnlineSARSALambda, mdp, alpha=alpha, n_episodes=n_episodes,
                               n_steps_test=n_steps_test)

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

    # Evaluate
    logger.info(f'- Evaluating the random policy for {n_steps_test} steps')
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    R = dataset.undiscounted_return.mean()
    logger.info(f'R: {R}')

    episode_length = dataset.episodes_length
    if len(episode_length) > 0:
        logger.info(f'completed episodes: {len(episode_length)}, mean episode length: {episode_length.mean()}')
    else:
        logger.info(f'episode length: {len(dataset)}, episode not completed')

    # Train
    logger.info(f'- Learning for {n_episodes} episodes')
    core.learn(n_episodes=n_episodes, n_steps_per_fit=1, render=False)

    # Visualize the final policy
    logger.info('- Evaluating the learned policy for one episode')
    dataset = core.evaluate(n_episodes=1, render=True)
    R = dataset.undiscounted_return.mean()
    logger.info(f'R: {R}')
    logger.info(f'episode length: {dataset.episodes_length.item()}')

    logger.weak_line()
    logger.info(f'dataset backend: {dataset.array_backend.get_backend_name()}')


if __name__ == '__main__':
    experiment(alpha=.1, n_episodes=100, n_steps_test=10000)
