import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import logging
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params, debug=True):

    console_logging_level = logging.DEBUG if debug else logging.INFO
    logger = Logger(alg.__name__, results_dir=None, console_log_level=console_logging_level)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    mdp = Gymnasium(env_id, horizon, gamma, headless=False)

    critic_params = dict(network=ActorNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = Core(agent, mdp, logger=logger)

    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy().item()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy().item()

        logger.epoch_info(it+1, J=J, R=R, entropy=E)

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    debug = True
    max_kl = .015

    policy_params = dict(
        std_0=1.,
        n_features=32
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
        (PPO, 'ppo', ppo_params),
        (TRPO, 'trpo', trpo_params)
     ]

    for alg, alg_name, alg_params in algs_params:
        experiment(alg=alg, env_id='Pendulum-v1', horizon=200, gamma=.99,
                   n_epochs=40, n_steps=30000, n_steps_per_fit=3000,
                   n_episodes_test=25, alg_params=alg_params,
                   policy_params=policy_params, debug=debug)
