"""
Simple script to solve the Acrobot problem with A2C and a Boltzmann policy.

"""
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import A2C
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import BoltzmannTorchPolicy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(n_epochs, n_steps, n_steps_per_fit, n_steps_test, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = Gymnasium('Acrobot-v1', horizon, gamma, headless=False)
    mdp.seed(seed)

    logger = Logger(A2C.name(), results_dir=None)
    logger.log_experiment_info(A2C, mdp, n_epochs=n_epochs, n_steps=n_steps,
                               n_steps_per_fit=n_steps_per_fit, n_steps_test=n_steps_test)

    # Policy
    beta = Parameter(1e0)
    pi = BoltzmannTorchPolicy(ActorNetwork,
                              mdp.info.observation_space.shape,
                              (mdp.info.action_space.n,),
                              beta=beta,
                              n_features=32)

    # Agent
    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 1e-3,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = A2C(mdp.info, pi,
                actor_optimizer={'class': optim.RMSprop,
                                 'params': {'lr': 1e-3, 'eps': 3e-3}},
                critic_params=critic_params,
                ent_coeff=0.01)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    R = dataset.undiscounted_return.mean()

    logger.log_evaluation(0, R=R)

    for n in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        R = dataset.undiscounted_return.mean()

        logger.log_evaluation(n + 1, R=R)

    if render:
        logger.info('Press a button to visualize acrobot')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=40, n_steps=1000, n_steps_per_fit=5, n_steps_test=2000,
               render=args.render, use_cuda=args.use_cuda, seed=args.seed)
