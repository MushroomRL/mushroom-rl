"""
This script shows how to run the Isaac Sim quadruped locomotion environments, solving them with RudinPPO.

IsaacSim environments are natively vectorized: a single environment object simulates many copies of the task
on the GPU and returns batched observations, so no wrapper is needed to parallelize them.

The three quadrupeds share the same reward structure and training setup, and differ only in the robot: the
Unitree A1, the Honey Badger and the Silver Badger, the last of which adds a spine joint.

"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import RudinPPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.isaac_sim import IsaacLauncher
from mushroom_rl.utils import TorchUtils
from mushroom_rl.utils.experiments import get_log_dir
from mushroom_rl.approximators.parametric.networks import ActorNetwork

# Isaac Sim has to be running before its environments can be imported
IsaacLauncher.launch(headless=True)

from mushroom_rl.environments.isaacsim_envs import A1Walking, HoneyBadgerWalking, SilverBadgerWalking

ROBOTS = dict(a1=A1Walking, honey_badger=HoneyBadgerWalking, silver_badger=SilverBadgerWalking)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=list(ROBOTS), default='a1', help='the quadruped to train')

    return parser.parse_args()


def experiment(alg, robot, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, alg_params, policy_params,
               n_envs=4096, horizon=1000, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # MDP
    mdp = ROBOTS[robot](n_envs, horizon, True)

    logger = Logger(f'RudinPPO_{robot}', results_dir=get_log_dir(__file__), log_console=True,
                    use_timestamp=True)
    logger.log_experiment_info(alg, mdp, n_epochs=n_epochs, n_steps=n_steps,
                               n_steps_per_fit=n_steps_per_fit, n_episodes_test=n_episodes_test,
                               n_envs=n_envs, horizon=horizon, robot=robot, **alg_params, **policy_params)

    # Policy
    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    # Agent
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((4096 * 24) / 16),
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
    V = agent._V(dataset.get_init_states()).mean().detach().item()

    logger.log_evaluation(0, J=J, R=R, entropy=E, V=V)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

        # record the run only once every five epochs, to keep the videos to a manageable size
        record = (it + 1) % 5 == 0 or it == n_epochs - 1
        dataset = core.evaluate(n_episodes=n_episodes_test, render=record, record=record)

        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        V = agent._V(dataset.get_init_states()).mean().detach().item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E, V=V)

        del dataset


if __name__ == '__main__':
    args = parse_args()

    TorchUtils.set_default_device('cuda:0')

    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 1e-3}},
                      n_epochs_policy=5,
                      batch_size=int((4096 * 24) / 16),
                      eps_ppo=.2,
                      lam=.95,
                      ent_coeff=0.01)

    policy_params = dict(std_0=1., n_features=[512, 256, 128], use_cuda=True)

    experiment(alg=RudinPPO, robot=args.robot, n_epochs=40, n_steps=4096 * 24 * 50, n_steps_per_fit=4096 * 24,
               n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params)
