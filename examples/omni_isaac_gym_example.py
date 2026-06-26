import hydra
from omegaconf import DictConfig
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.algorithms.actor_critic import TRPO, PPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.environments import OmniIsaacGymEnv
from mushroom_rl.utils import TorchUtils
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def experiment(cfg_dict, headless, alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params):

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = OmniIsaacGymEnv(cfg_dict, headless=True)
    mdp.render_all()


    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
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

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = torch.mean(dataset.discounted_return)
    R = torch.mean(dataset.undiscounted_return)
    E = agent.policy.entropy().item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = torch.mean(dataset.discounted_return)
        R = torch.mean(dataset.undiscounted_return)
        E = agent.policy.entropy().item()

        logger.epoch_info(it+1, J=J, R=R, entropy=E)

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=False)


@hydra.main(config_name="config", config_path="./cfg")
def parse_hydra_configs(cfg: DictConfig):
    TorchUtils.set_default_device('cuda')
    headless = cfg.headless
    cfg_dict = omegaconf_to_dict(cfg)

    max_kl = .015

    policy_params = dict(
        std_0=1.,
        n_features=32,
        use_cuda=True

    )

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

    algs_params = [
        (PPO, 'ppo', ppo_params)
    ]

    for alg, alg_name, alg_params in algs_params:
        experiment(cfg_dict=cfg_dict, headless=headless, alg=alg, n_epochs=40, n_steps=30000, n_steps_per_fit=3000,
                   n_episodes_test=512, alg_params=alg_params, policy_params=policy_params)


if __name__ == '__main__':
    parse_hydra_configs()
