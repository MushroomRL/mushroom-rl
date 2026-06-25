import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork

from tqdm import trange


def experiment(n_epochs, n_steps, n_steps_test, save_agent, load_agent):
    np.random.seed()

    # MDP
    horizon = 200
    gamma = 0.99
    mdp = Gymnasium('Pendulum-v1', horizon, gamma, headless=True)

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
                                               name=SAC.__name__)

    logger = Logger(SAC.__name__, results_dir='./logs', wandb_kwargs=wandb_kwargs)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + SAC.__name__)

    if load_agent:
        agent = SAC.load('logs/SAC/agent-best.msh')
    else:
        # Approximator
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

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
        critic_params = dict(network=CriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 3e-4}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,))

        # Agent
        agent = SAC(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp, logger=logger)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

    logger.log_evaluation(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # Record an evaluation video before training and upload it to wandb
    core.evaluate(n_episodes=1, render=True, record=True)
    logger.log_video(0)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(torch.from_numpy(dataset.state)).item()

        logger.log_evaluation(it+1, J=J, R=R, entropy=E)

        if save_agent:
            logger.log_best_agent(agent, J)

        if it + 1 == 20:
            core.evaluate(n_episodes=1, render=True, record=True)
            logger.log_video(it+1)

    # Record a final evaluation video and upload it to wandb
    core.evaluate(n_episodes=1, render=True, record=True)
    logger.log_video(n_epochs)



if __name__ == '__main__':
    experiment(n_epochs=40, n_steps=1000, n_steps_test=2000, save_agent=False, load_agent=False)
