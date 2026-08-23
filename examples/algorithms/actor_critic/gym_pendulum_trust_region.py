"""
Simple script to solve the Pendulum problem with the trust region algorithms PPO and TRPO.

Both algorithms log their diagnostics at the debug level, so the console log level is lowered to see them.

"""
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.experiments import select_class


def get_algorithms():
    return [PPO, TRPO]


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params, debug=True, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = Gymnasium(env_id, horizon, gamma, headless=False)
    mdp.seed(seed)

    console_log_level = logging.DEBUG if debug else logging.INFO
    logger = Logger(alg.name(), results_dir=None, console_log_level=console_log_level)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_steps=n_steps,
                               n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test,
                               **alg_params, **policy_params)

    # Policy
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    # Agent
    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

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
    parser.add_argument('--alg', choices=[alg.name() for alg in get_algorithms()], default=PPO.name(),
                        help='the trust region algorithm to run')
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')
    parser.add_argument('--debug', action='store_true', help='show debug information in the terminal')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    alg = select_class(args.alg, get_algorithms())

    policy_params = dict(std_0=1., n_features=32)

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 3e-4}},
                      n_epochs_policy=4,
                      batch_size=64,
                      eps_ppo=.2,
                      lam=.95)

    trpo_params = dict(ent_coeff=0.0,
                       max_kl=.01,
                       lam=.95,
                       n_epochs_line_search=10,
                       n_epochs_cg=100,
                       cg_damping=1e-2,
                       cg_residual_tol=1e-10)

    alg_params = {PPO: ppo_params, TRPO: trpo_params}[alg]

    experiment(alg=alg, env_id='Pendulum-v1', horizon=200, gamma=.99,
               n_epochs=40, n_steps=30000, n_steps_per_fit=3000,
               n_episodes_test=25, alg_params=alg_params, policy_params=policy_params,
               debug=args.debug, render=args.render, use_cuda=args.use_cuda, seed=args.seed)
