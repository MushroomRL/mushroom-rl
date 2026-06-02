import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import RudinPPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments.isaacsim_envs import HoneyBadgerWalking
from mushroom_rl.utils import TorchUtils
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__ + "_honey_batcher", results_dir="./logs/", log_console=True, use_timestamp=True)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = HoneyBadgerWalking(4096, 1000, True, True)
    
    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=[512, 256, 128],
                         batch_size=int((4096*24) / 32),
                         use_cuda=True,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = VectorCore(agent, mdp)
    
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True, record=True)

    J = torch.mean(dataset.discounted_return).item()
    R = torch.mean(dataset.undiscounted_return).item()
    E = agent.policy.entropy().item()
    
    logger.epoch_info(0, J=J, R=R, entropy=E)
    del dataset

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True)

        J = torch.mean(dataset.discounted_return).item()
        R = torch.mean(dataset.undiscounted_return).item()
        E = agent.policy.entropy().item()
    
        logger.epoch_info(it + 1, J=J, R=R, entropy=E)
        del dataset

if __name__ == '__main__':
    TorchUtils.set_default_device('cuda:0')
    ppo_params = dict(
        actor_optimizer={'class': optim.Adam,
        'params': {'lr': 1e-3}},#changed from 1e-3
        n_epochs_policy=5,
        batch_size=int((4096*24) / 32),
        eps_ppo=.2,
        lam=.95,
        ent_coeff=0.00
    )
    policy_params = dict(
        std_0=1.,
        n_features=[512, 256, 128],
        use_cuda=True
    )
    experiment(alg=RudinPPO, n_epochs=30, n_steps=4096*24*50*2, n_steps_per_fit=4096*24,
                   n_episodes_test=256, alg_params=ppo_params, policy_params=policy_params)
