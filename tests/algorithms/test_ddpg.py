import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic import DDPG, TD3
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork


def learn(alg):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=torch.ones(1) * .2, theta=.15, dt=1e-2)

    initial_replay_size = 100
    max_replay_size = 5000
    batch_size = 200
    tau = .001

    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=None,
                        n_layers=0,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = [actor_input_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=critic_input_shape,
                         output_shape=(1,))

    agent = alg(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    assert agent._fit_count > 0

    return agent


def test_ddpg():
    policy = learn(DDPG).policy
    w = policy.get_weights()
    w_test = torch.tensor([-0.00764989, 1.23425770, 0.10049075])

    assert torch.allclose(w, w_test)


def test_ddpg_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(DDPG)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_td3():
    policy = learn(TD3).policy
    w = policy.get_weights()
    w_test = torch.tensor([1.34045994, 0.93246555, 0.19654009])

    assert torch.allclose(w, w_test)


def test_td3_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn(TD3)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
