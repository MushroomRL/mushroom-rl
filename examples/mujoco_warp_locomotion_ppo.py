import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.environments.mujoco_warp_envs import HopperWarp
from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu") / 10)
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu") / 10)
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear") / 10)

    def forward(self, state, **kwargs):
        x = F.relu(self._h1(torch.squeeze(state, 1).float()))
        x = F.relu(self._h2(x))
        return self._h3(x)


def experiment(env_class, num_envs, n_epochs, n_steps, n_steps_per_fit, n_episodes_test):
    np.random.seed(0)

    logger = Logger(PPO.__name__ + "_HopperWarp", results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + PPO.__name__ + "  env: HopperWarp")

    mdp = env_class(num_envs=num_envs)

    actor_lr = 3e-4
    critic_lr = 3e-4
    n_features = 64
    batch_size = 64
    n_epochs_policy = 10
    eps_ppo = 0.2
    lam = 0.95
    std_0 = 1.0

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.Adam, "params": {"lr": critic_lr}},
        loss=F.mse_loss,
        n_features=n_features,
        batch_size=batch_size,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(1,),
    )

    alg_params = dict(
        actor_optimizer={"class": optim.Adam, "params": {"lr": actor_lr}},
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps_ppo,
        lam=lam,
        critic_params=critic_params,
    )

    policy = GaussianTorchPolicy(
        Network,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=std_0,
        n_features=n_features
    )

    agent = PPO(mdp.info, policy, **alg_params)
    core = VectorCore(agent, mdp)

    # Initial evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=True)
    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()
    E = agent.policy.entropy().item()
    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=True)

        J = dataset.discounted_return.mean().item()
        R = dataset.undiscounted_return.mean().item()
        E = agent.policy.entropy().item()
        logger.epoch_info(it + 1, J=J, R=R, entropy=E)

    mdp.stop()

if __name__ == "__main__":
    envs = [HopperWarp]
    for env in envs:
        experiment(env_class=env, num_envs=4096, n_epochs=50, n_steps=300000,
                   n_steps_per_fit=30000, n_episodes_test=10)