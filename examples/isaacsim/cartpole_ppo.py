import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import TRPO, PPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs.cartpole import CartPole
from mushroom_rl.utils import TorchUtils
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = CartPole(64, True)
    
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=100,
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = VectorCore(agent, mdp, logger=logger)

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
