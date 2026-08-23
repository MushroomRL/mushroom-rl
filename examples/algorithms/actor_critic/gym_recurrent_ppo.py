"""
Simple script to solve a Gymnasium MuJoCo task with the recurrent version of PPO, PPO_BPTT.

The policy and the critic are recurrent networks trained with truncated backpropagation through time, and the
previous action can be fed back to them as part of the history.

"""
import argparse

import numpy as np
import torch
import torch.optim as optim

from tqdm import trange

from mushroom_rl.core import Logger, Core
from mushroom_rl.environments import Gymnasium
from mushroom_rl.algorithms.actor_critic import PPO_BPTT
from mushroom_rl.policy import RecurrentGaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import RecurrentActorNetwork, RecurrentCriticNetwork
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.utils.experiments import get_log_dir


def experiment(env, horizon, gamma, n_epochs, n_steps_per_epoch, n_steps_per_fit, n_episodes_test,
               lr_actor=0.001, lr_critic=0.001, batch_size_actor=32, batch_size_critic=32,
               n_epochs_policy=10, clip_eps_ppo=0.05, gae_lambda=0.95, std_0=0.5, rnn_type='gru',
               n_hidden_features=128, num_hidden_layers=1, truncation_length=5, use_prev_action=True,
               use_cuda=False, seed=None):
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        assert torch.cuda.is_available(), 'CUDA was requested, but it is not available on this machine.'

    TorchUtils.set_default_device('cuda:0' if use_cuda else 'cpu')

    # MDP
    mdp = Gymnasium(env, horizon=horizon, gamma=gamma)
    mdp.seed(seed)

    logger = Logger('gym_recurrent_ppo', results_dir=get_log_dir(__file__), seed=seed)
    logger.log_experiment_info(PPO_BPTT, mdp, n_epochs=n_epochs, n_steps_per_epoch=n_steps_per_epoch,
                               n_steps_per_fit=n_steps_per_fit, rnn_type=rnn_type,
                               n_hidden_features=n_hidden_features, num_hidden_layers=num_hidden_layers,
                               truncation_length=truncation_length, use_prev_action=use_prev_action)

    # Policy
    dim_env_state = mdp.info.observation_space.shape[0]
    dim_action = mdp.info.action_space.shape[0]

    policy = RecurrentGaussianTorchPolicy(network=RecurrentActorNetwork,
                                          input_shape=(dim_env_state,),
                                          output_shape=(dim_action,),
                                          n_features=128,
                                          rnn_type=rnn_type,
                                          n_hidden_features=n_hidden_features,
                                          num_hidden_layers=num_hidden_layers,
                                          action_history_shape=(dim_action,) if use_prev_action else None,
                                          std_0=std_0)

    # Critic
    critic_params = dict(network=RecurrentCriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic,
                                               'weight_decay': 0.0}},
                         loss=torch.nn.MSELoss(),
                         batch_size=batch_size_critic,
                         input_shape=(dim_env_state,),
                         output_shape=(1,),
                         n_features=128,
                         n_hidden_features=n_hidden_features,
                         rnn_type=rnn_type,
                         num_hidden_layers=num_hidden_layers,
                         action_history_shape=(dim_action,) if use_prev_action else None
                         )

    # Agent
    agent = PPO_BPTT(mdp.info, policy, critic_params=critic_params,
                     actor_optimizer={'class': optim.Adam,
                                      'params': {'lr': lr_actor, 'weight_decay': 0.0}},
                     n_epochs_policy=n_epochs_policy,
                     batch_size=batch_size_actor,
                     dim_env_state=dim_env_state,
                     eps_ppo=clip_eps_ppo,
                     lam=gae_lambda,
                     truncation_length=truncation_length,
                     action_history_length=1 if use_prev_action else 0)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_episodes=5)

    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    L = dataset.episodes_length.mean()

    logger.log_evaluation(0, J=J, R=R, L=L)

    for i in trange(1, n_epochs + 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test)

        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        L = dataset.episodes_length.mean()

        logger.log_evaluation(i, J=J, R=R, L=L)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-cuda', action='store_true', help='run on the GPU instead of the CPU')
    parser.add_argument('--seed', type=int, default=None, help='seed of the experiment, random when not given')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment(env='HalfCheetah-v5', horizon=1000, gamma=0.99, n_epochs=300, n_steps_per_epoch=50000,
               n_steps_per_fit=2000, n_episodes_test=10, use_cuda=args.use_cuda, seed=args.seed)
