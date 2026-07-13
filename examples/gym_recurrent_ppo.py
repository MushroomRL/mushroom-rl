import numpy as np
import torch
import torch.optim as optim

from mushroom_rl.core import Logger, Core
from mushroom_rl.environments import Gymnasium

from mushroom_rl.algorithms.actor_critic import PPO_BPTT
from mushroom_rl.policy import RecurrentGaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import RecurrentActorNetwork, RecurrentCriticNetwork

from tqdm import trange


def experiment(
        env: str = 'HalfCheetah-v5',
        horizon: int = 1000,
        gamma: float = 0.99,
        n_epochs: int = 300,
        n_steps_per_epoch: int = 50000,
        n_steps_per_fit: int = 2000,
        n_episode_eval: int = 10,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
        batch_size_actor: int = 32,
        batch_size_critic: int = 32,
        n_epochs_policy: int = 10,
        clip_eps_ppo: float = 0.05,
        gae_lambda: float = 0.95,
        seed: int = 0,
        results_dir: str = './logs',
        std_0: float = 0.5,
        rnn_type: str = "gru",
        n_hidden_features: int = 128,
        num_hidden_layers: int = 1,
        truncation_length: int = 5,
        use_prev_action: bool = True
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # prepare logging
    logger = Logger(results_dir=results_dir, log_name="gym_recurrent_ppo", seed=seed)

    # MDP
    mdp = Gymnasium(env, horizon=horizon, gamma=gamma)

    # create the policy
    dim_env_state = mdp.info.observation_space.shape[0]
    dim_action = mdp.info.action_space.shape[0]

    policy = RecurrentGaussianTorchPolicy(network=RecurrentActorNetwork,
                                          policy_state_shape=(n_hidden_features,),
                                          input_shape=(dim_env_state, ),
                                          output_shape=(dim_action,),
                                          n_features=128,
                                          rnn_type=rnn_type,
                                          n_hidden_features=n_hidden_features,
                                          num_hidden_layers=num_hidden_layers,
                                          use_prev_action=use_prev_action,
                                          std_0=std_0)

    # setup critic
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
                         dim_action=dim_action,
                         use_prev_action=use_prev_action
                         )

    alg_params = dict(actor_optimizer={'class':  optim.Adam,
                                       'params': {'lr': lr_actor,
                                                  'weight_decay': 0.0}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=batch_size_actor,
                      dim_env_state=dim_env_state,
                      eps_ppo=clip_eps_ppo,
                      lam=gae_lambda,
                      truncation_length=truncation_length,
                      action_history_length=1 if use_prev_action else 0
                      )

    # Create the agent
    agent = PPO_BPTT(mdp_info=mdp.info, policy=policy, critic_params=critic_params, **alg_params)

    # Create Core
    core = Core(agent, mdp)

    # Evaluation
    dataset = core.evaluate(n_episodes=5)
    J = dataset.discounted_return.mean()
    R = dataset.undiscounted_return.mean()
    L = dataset.episodes_length.mean()
    logger.log_numpy(R=R, J=J, L=L)
    logger.epoch_info(0, R=R, J=J, L=L)

    for i in trange(1, n_epochs+1, 1, leave=False):
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit)

        # Evaluation
        dataset = core.evaluate(n_episodes=n_episode_eval)
        J = dataset.discounted_return.mean()
        R = dataset.undiscounted_return.mean()
        L = dataset.episodes_length.mean()
        logger.log_numpy(R=R, J=J, L=L)
        logger.epoch_info(i, R=R, J=J, L=L)


if __name__ == '__main__':
    experiment()
