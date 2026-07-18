import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DSORDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter


def experiment(n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(DSORDQN.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + DSORDQN.__name__)

    mdp = Gymnasium('Acrobot-v1', horizon=1000, gamma=.99,
                    headless=False)

    epsilon = LinearParameter(value=1., threshold_value=.01, n=5000)
    policy = EpsGreedy(Parameter(1.), backend='torch')

    initial_replay_size = 500
    approximator_params = dict(
        network=QNetwork,
        optimizer={'class': optim.Adam, 'params': {'lr': .001}},
        loss=F.smooth_l1_loss,
        n_features=80,
        input_shape=mdp.info.observation_space.shape,
        output_shape=mdp.info.action_space.size,
        n_actions=mdp.info.action_space.n
    )

    agent = DSORDQN(
        mdp.info, policy, TorchApproximator,
        approximator_params=approximator_params,
        relaxation_factor=1.2,
        batch_size=200,
        initial_replay_size=initial_replay_size,
        max_replay_size=5000,
        target_update_frequency=100
    )
    core = Core(agent, mdp)

    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size)
    dataset = core.evaluate(n_steps=n_steps_test, greedy=True)
    logger.epoch_info(0, R=np.mean(dataset.undiscounted_return))

    policy.set_epsilon(epsilon)
    for epoch in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, greedy=True)
        logger.epoch_info(
            epoch + 1, R=np.mean(dataset.undiscounted_return))


if __name__ == '__main__':
    experiment(n_epochs=20, n_steps=1000, n_steps_test=2000)
