import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import TRPO, PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs.cartpole import CartPole
from mushroom_rl.utils import TorchUtils


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = CartPole(64, True)
    
    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=100,
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)
    #agent.set_logger(logger)

    core = VectorCore(agent, mdp)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    E = agent.policy.entropy().item()
    A = torch.sum(dataset.absorbing).item()

    logger.epoch_info(0, J=J, R=R, entropy=E, absorbing=A)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        E = agent.policy.entropy().item()
        A = torch.sum(dataset.absorbing).item()

        logger.epoch_info(it+1, J=J, R=R, entropy=E, absorbing=A)

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True, record=True)


if __name__ == '__main__':
    ppo_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': 3e-4}},
        n_epochs_policy=4,
        batch_size=100,
        eps_ppo=.2,
        lam=.95
    )
    policy_params = dict(
        std_0=1.,
        n_features=32,
        use_cuda=True

    )
    experiment(alg=PPO, n_epochs=20, n_steps=30000, n_steps_per_fit=3000,
                   n_episodes_test=64, alg_params=ppo_params, policy_params=policy_params)
