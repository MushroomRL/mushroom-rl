"""
This script shows how to log an experiment on Weights & Biases.

The hyperparameters are passed to the wandb run configuration when the logger is built, so that the runs can
be compared and filtered on them. From there on the ordinary logging methods also reach wandb: metrics
through ``log_evaluation``, and the recorded evaluation videos through ``log_video``.

"""
import argparse

from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.utils import get_log_dir
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork


def build_agent(mdp, batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                n_features):
    """
    Build the SAC agent, with a Gaussian actor and a pair of critics.

    """
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    return SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,
               batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha)


def experiment(n_epochs, n_steps, n_steps_test, save_agent=False, agent_dir=None, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gymnasium('Pendulum-v1', horizon, gamma, headless=True)
    mdp.seed(seed)

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    # wandb run configuration
    hyperparams = dict(gamma=gamma, horizon=horizon, batch_size=batch_size,
                       n_features=n_features, warmup_transitions=warmup_transitions,
                       tau=tau, lr_alpha=lr_alpha, max_replay_size=max_replay_size)

    wandb_kwargs = Logger.default_wandb_kwargs('mushroom_rl_wandb_example',
                                               config=hyperparams,
                                               name=SAC.name())

    logger = Logger('wandb_example', results_dir=get_log_dir(__file__), use_timestamp=True,
                    wandb_kwargs=wandb_kwargs)
    logger.log_experiment_info(SAC, mdp, n_epochs=n_epochs, n_steps=n_steps, n_steps_test=n_steps_test,
                               **hyperparams)

    # Agent
    if agent_dir is not None:
        agent = SAC.load(agent_dir / 'agent-best.msh')
    else:
        agent = build_agent(mdp, batch_size, initial_replay_size, max_replay_size, warmup_transitions,
                            tau, lr_alpha, n_features)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # Record an evaluation video before training and upload it to wandb
    core.evaluate(n_episodes=1, render=True, record=True)
    logger.log_video(0)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

        logger.log_evaluation(it + 1, J=J, R=R, entropy=E)

        if save_agent:
            logger.log_best_agent(agent, J)

        if it + 1 == 20:
            core.evaluate(n_episodes=1, render=True, record=True)
            logger.log_video(it + 1)

    # Record a final evaluation video and upload it to wandb
    core.evaluate(n_episodes=1, render=True, record=True)
    logger.log_video(n_epochs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', type=Path, default=None, metavar='DIR',
                        help='log directory of a previous run, whose best agent is the one to start from')
    parser.add_argument('--save', action='store_true',
                        help='save the best agent of the run, as agent-best.msh in the log directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(n_epochs=40, n_steps=1000, n_steps_test=2000, save_agent=args.save, agent_dir=args.load)
