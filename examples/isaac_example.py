import hydra
from omegaconf import DictConfig

from tqdm import trange

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.core import VectorCore, Logger
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.environments import IsaacEnv
from omniisaacgymenvs.tasks.cartpole import CartpoleTask


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

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

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


def experiment(env, n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(DQN.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + DQN.__name__)

    # MDP
    horizon = 1000
    gamma = 0.99
    gamma_eval = 1.

    # Policy
    epsilon = LinearParameter(value=1., threshold_value=.01, n=5000)
    epsilon_test = Parameter(value=0.)
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

    # Settings
    initial_replay_size = 500
    max_replay_size = 5000
    target_update_frequency = 100
    batch_size = 200
    n_features = 80
    train_frequency = 1

    # Approximator
    input_shape = env.info.observation_space.shape
    approximator_params = dict(network=Network,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=env.info.action_space.size,
                               n_actions=env.info.action_space.n)

    # Agent
    agent = DQN(env.info, pi, TorchApproximator,
                approximator_params=approximator_params, batch_size=batch_size,
                initial_replay_size=initial_replay_size,
                max_replay_size=max_replay_size,
                target_update_frequency=target_update_frequency)

    # Algorithm
    core = VectorCore(agent, env)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = dataset.discounted_return
    logger.epoch_info(0, J=np.mean(J))

    for n in trange(n_epochs):
        pi.set_epsilon(epsilon)
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = dataset.discounted_return
        logger.epoch_info(n+1, J=np.mean(J))

    logger.info('Press a button to visualize acrobot')
    input()
    core.evaluate(n_episodes=5, render=True)


def omegaconf_to_dict(d: DictConfig):
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless
    sim_app_cfg_path = cfg.sim_app_cfg_path
    cfg_dict = omegaconf_to_dict(cfg)

    return sim_app_cfg_path, headless, cfg_dict


if __name__ == '__main__':
    sim_app_cfg_path, headless, cfg_dict = parse_hydra_configs()
    env = IsaacEnv(sim_app_cfg_path, CartpoleTask, cfg_dict, headless=headless)
    experiment(env, n_epochs=20, n_steps=1000, n_steps_test=2000)