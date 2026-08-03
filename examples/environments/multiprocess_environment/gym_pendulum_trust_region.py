"""
This script shows how to run a trust region experiment on many copies of the Pendulum environment at once,
with MultiprocessEnvironment.

The environment is not vectorized itself: MultiprocessEnvironment runs one copy of it per worker process and
batches their observations, so the experiment collects several transitions per step of the core loop. The
torch thread count is lowered accordingly, since the parallelism is already in the environment.

"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.core import Core, Logger, MultiprocessEnvironment
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import select_class
from mushroom_rl.approximators.parametric.networks import ActorNetwork
from mushroom_rl.utils.torch_utils import TorchUtils

torch.set_num_threads(1)
torch.set_num_interop_threads(4)


def get_algorithms():
    return [PPO, TRPO]


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params, n_envs, render=True, use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = MultiprocessEnvironment(Gymnasium, env_id, horizon, gamma, n_envs=n_envs)
    mdp.seed(seed)

    logger = Logger(alg.name(), results_dir=None)
    logger.log_experiment_info(alg, mdp, env_id=env_id, n_epochs=n_epochs, n_steps=n_steps,
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
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    alg = select_class(args.alg, get_algorithms())

    policy_params = dict(std_0=1., n_features=32)

    parameters = dict()

    parameters[PPO] = dict(actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}},
                           n_epochs_policy=4,
                           batch_size=64,
                           eps_ppo=.2,
                           lam=.95)

    parameters[TRPO] = dict(ent_coeff=0.0,
                            max_kl=.01,
                            lam=.95,
                            n_epochs_line_search=10,
                            n_epochs_cg=100,
                            cg_damping=1e-2,
                            cg_residual_tol=1e-10)

    experiment(alg=alg, env_id='Pendulum-v1', horizon=200, gamma=.99,
               n_epochs=40, n_steps=30000, n_steps_per_fit=3000,
               n_episodes_test=25, alg_params=parameters[alg],
               policy_params=policy_params, n_envs=15, render=args.render, use_cuda=args.use_cuda)
