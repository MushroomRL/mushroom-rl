import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pytest

from helper.dummies import FakeLogger
from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.algorithms.actor_critic import A2C

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork
from mushroom_rl.utils import TorchUtils


def learn_a2c(logger=None, **extra_params):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    policy_params = dict(std_0=1., n_features=None, n_layers=0)

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 device=extra_params.get('device'),
                                 **policy_params)

    agent = A2C(mdp.info, policy, **algorithm_params, **extra_params)
    if logger is not None:
        agent.set_logger(logger)

    core = Core(agent, mdp)
    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


def learn_a2c_history(history_length):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    window_shape = (history_length,) + mdp.info.observation_space.shape

    policy_params = dict(std_0=1., n_features=None, n_layers=0)

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         input_shape=window_shape,
                         n_features=None,
                         n_layers=0,
                         output_shape=(1,))

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01,
                            history_length=history_length)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 window_shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = A2C(mdp.info, policy, **algorithm_params)

    core = Core(agent, mdp)
    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


def learn_a2c_action_history(action_history_length):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    action_history_shape = mdp.info.action_space.shape

    policy_params = dict(std_0=1., n_features=None, n_layers=0, action_history_shape=action_history_shape)

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         action_history_shape=action_history_shape)

    algorithm_params = dict(critic_params=critic_params,
                            actor_optimizer={'class': optim.RMSprop,
                                             'params': {'lr': 7e-4,
                                                        'eps': 3e-3}},
                            max_grad_norm=0.5,
                            ent_coeff=0.01,
                            action_history_length=action_history_length)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = A2C(mdp.info, policy, **algorithm_params)

    core = Core(agent, mdp)
    core.learn(n_episodes=10, n_episodes_per_fit=5)

    return agent


@pytest.mark.parametrize('device', devices())
def test_a2c(device, tmpdir):
    agent = learn_a2c()

    w = agent.policy.get_weights()

    tu.assert_recorded(w, 'test_a2c', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_a2c_options(device):
    TorchUtils.set_default_device('cpu')
    logger = FakeLogger()
    agent = learn_a2c(logger, device=device)

    tu.assert_recorded(agent.policy.get_weights(), 'test_a2c', device)
    assert logger.keys() == {'actor/loss', 'actor/entropy', 'critic/loss'}


@pytest.mark.parametrize('device', devices())
def test_a2c_history_length(device):
    agent = learn_a2c_history(history_length=3)

    w = agent.policy.get_weights()

    assert agent.history_length == 3
    tu.assert_recorded(w, 'test_a2c_history_length', device)


@pytest.mark.parametrize('device', devices())
def test_a2c_action_history_length(device):
    agent = learn_a2c_action_history(action_history_length=1)

    w = agent.policy.get_weights()

    assert agent._history_manager.uses_action
    tu.assert_recorded(w, 'test_a2c_action_history_length', device)
