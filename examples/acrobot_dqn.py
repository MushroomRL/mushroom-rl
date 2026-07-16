import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.approximators.parametric.networks import QNetwork

from tqdm import trange


def experiment(n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(DQN.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + DQN.__name__)

    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = Gymnasium('Acrobot-v1', horizon, gamma, headless=False)

    # Policy
    epsilon = LinearParameter(value=1., threshold_value=.01, n=5000)
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random, backend='torch')

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=QNetwork,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n)

    # Agent
    agent = DQN(mdp.info, pi, TorchApproximator,
                approximator_params=approximator_params, batch_size=batch_size,
                initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                target_update_frequency=target_update_frequency)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False, greedy=True)
    R = np.mean(dataset.undiscounted_return)
    logger.epoch_info(0, R=R)

    pi.set_epsilon(epsilon)
    for n in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, greedy=True)
        R = np.mean(dataset.undiscounted_return)
        logger.epoch_info(n + 1, R=R)

    logger.info('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True, greedy=True)


if __name__ == '__main__':
    experiment(n_epochs=20, n_steps=1000, n_steps_test=2000)
