"""
This script shows how to run the Isaac Sim Cartpole environment, solving it with PPO.

IsaacSim environments are natively vectorized: a single environment object simulates many copies of the task
on the GPU and returns batched observations, so no wrapper is needed to parallelize them.

"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.isaac_sim import IsaacLauncher
from mushroom_rl.approximators.parametric.networks import ActorNetwork

# Isaac Sim has to be running before its environments can be imported
IsaacLauncher.launch(headless=True)

from mushroom_rl.environments.isaacsim_envs import CartPoleIsaac


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, alg_params, policy_params,
               n_envs=64, render=True, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # MDP
    mdp = CartPoleIsaac(n_envs)

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_steps=n_steps,
                               n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test,
                               n_envs=n_envs, **alg_params, **policy_params)

    # Policy
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    # Agent
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=100,
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()
    E = agent.policy.entropy().item()
    A = dataset.absorbing.sum().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E, absorbing=A)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        A = dataset.absorbing.sum().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E, absorbing=A)

    if render:
        logger.info('Press a button to visualize the cartpole')
        input()
        core.evaluate(n_episodes=5, render=True, record=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-render', action='store_false', dest='render', help='skip the final visualization')

    parser.add_argument('--seed', type=int, default=None,
                        help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    TorchUtils.set_default_device('cuda:0')

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 3e-4}},
                      n_epochs_policy=4,
                      batch_size=100,
                      eps_ppo=.2,
                      lam=.95)

    policy_params = dict(std_0=1., n_features=32, use_cuda=True)

    experiment(alg=PPO, n_epochs=20, n_steps=30000, n_steps_per_fit=3000, n_episodes_test=64,
               alg_params=ppo_params, policy_params=policy_params, render=args.render, seed=args.seed)
