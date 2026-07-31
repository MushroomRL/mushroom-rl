"""
Simple script to solve the Pendulum problem with A2C and a Gaussian policy.

"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import A2C
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils


def experiment(env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_steps_test,
               alg_params, policy_params, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = Gymnasium(env_id, horizon, gamma, headless=False)
    mdp.seed(seed)

    logger = Logger(A2C.name(), results_dir=None)
    logger.log_experiment_info(A2C, mdp, n_epochs=n_epochs, n_steps=n_steps,
                               n_steps_per_fit=n_steps_per_fit, n_steps_test=n_steps_test,
                               **alg_params, **policy_params)

    # Policy
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    # Agent
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=64,
                         activation='tanh',
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = A2C(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E)

    if render:
        logger.info('Press a button to visualize the pendulum')
        input()
        core.evaluate(n_episodes=5, render=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    policy_params = dict(std_0=1., n_features=64, activation='tanh')

    a2c_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 7e-4, 'eps': 3e-3}},
                      max_grad_norm=0.5,
                      ent_coeff=0.0)

    experiment(env_id='Pendulum-v1', horizon=200, gamma=.99, n_epochs=40, n_steps=30000,
               n_steps_per_fit=5, n_steps_test=5000, alg_params=a2c_params,
               policy_params=policy_params, render=args.render, use_cuda=args.use_cuda)
