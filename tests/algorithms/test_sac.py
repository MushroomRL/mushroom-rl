import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork


def learn_sac():
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    warmup_transitions = 10
    tau = 0.005
    lr_alpha = 3e-4

    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=None,
                           n_layers=0,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=None,
                              n_layers=0,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer,
                critic_params, batch_size, initial_replay_size, max_replay_size,
                warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    core = Core(agent, mdp)

    core.learn(n_steps=4 * initial_replay_size,
               n_steps_per_fit=initial_replay_size)

    return agent


def test_sac():
    policy = learn_sac().policy
    w = policy.get_weights()
    w_test = torch.tensor([1.34026611, 0.93226570, 0.19634973, 1.24291027, -0.23446862, -0.33998108])

    assert torch.allclose(w, w_test)


def test_sac_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_sac()

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)