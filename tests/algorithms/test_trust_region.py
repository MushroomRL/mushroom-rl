import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim

from helper.dummies import FakeLogger
from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import FeedForwardNetwork, ActorNetwork


def learn(alg, alg_params, logger=None):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy_params = dict(std_0=1., n_features=None, n_layers=0)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)
    if logger is not None:
        agent.set_logger(logger)

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return agent


def learn_history(alg, alg_params, history_length):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    window_shape = (history_length,) + mdp.info.observation_space.shape

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=window_shape,
                         output_shape=(1,))

    policy_params = dict(std_0=1., n_features=None, n_layers=0)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 window_shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params
    alg_params['history_length'] = history_length

    agent = alg(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return agent


def learn_action_history(alg, alg_params, action_history_length):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    action_history_shape = mdp.info.action_space.shape

    critic_params = dict(network=FeedForwardNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         action_history_shape=action_history_shape)

    policy_params = dict(std_0=1., n_features=None, n_layers=0, action_history_shape=action_history_shape)

    policy = GaussianTorchPolicy(ActorNetwork,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params
    alg_params['action_history_length'] = action_history_length

    agent = alg(mdp.info, policy, **alg_params)

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return agent


@pytest.mark.parametrize('device', devices())
def test_PPO(device, tmpdir):
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    agent = learn(PPO, params)
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_PPO', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


@pytest.mark.parametrize('device', devices())
def test_TRPO(device, tmpdir):
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    agent = learn(TRPO, params)
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_TRPO', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh')


@pytest.mark.parametrize('device', devices())
def test_logger(device):
    ppo_params = dict(actor_optimizer={'class': optim.Adam, 'params': {'lr': 3e-4}}, n_epochs_policy=4,
                      batch_size=64, eps_ppo=.2, lam=.95, device=device)
    trpo_params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10, n_epochs_cg=10,
                       cg_damping=1e-2, cg_residual_tol=1e-10, device=device)

    for alg, params in [(PPO, ppo_params), (TRPO, trpo_params)]:
        logger = FakeLogger()
        w = learn(alg, params, logger=logger).policy.get_weights()

        tu.assert_recorded(w, f'test_{alg.__name__}', device)
        assert logger.keys() == {'R', 'actor/entropy', 'actor/kl', 'critic/loss'}
        assert logger.messages == ['Iteration 1', 'Iteration 2']


def test_TRPO_rejected_step():
    torch.manual_seed(1)
    w_initial = GaussianTorchPolicy(ActorNetwork, (2,), (1,), std_0=1., n_features=None, n_layers=0).get_weights()
    params = dict(n_epochs_line_search=0, cg_residual_tol=1e10)

    assert torch.equal(learn(TRPO, params).policy.get_weights(), w_initial)


@pytest.mark.parametrize('device', devices())
def test_PPO_history_length(device):
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    agent = learn_history(PPO, params, history_length=3)
    w = agent.policy.get_weights()

    assert agent.history_length == 3
    tu.assert_recorded(w, 'test_PPO_history_length', device)


@pytest.mark.parametrize('device', devices())
def test_TRPO_history_length(device):
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    agent = learn_history(TRPO, params, history_length=3)
    w = agent.policy.get_weights()

    assert agent.history_length == 3
    tu.assert_recorded(w, 'test_TRPO_history_length', device)


@pytest.mark.parametrize('device', devices())
def test_PPO_action_history_length(device):
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    agent = learn_action_history(PPO, params, action_history_length=1)
    w = agent.policy.get_weights()

    assert agent._history_manager.uses_action
    tu.assert_recorded(w, 'test_PPO_action_history_length', device)


@pytest.mark.parametrize('device', devices())
def test_TRPO_action_history_length(device):
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    agent = learn_action_history(TRPO, params, action_history_length=1)
    w = agent.policy.get_weights()

    assert agent._history_manager.uses_action
    tu.assert_recorded(w, 'test_TRPO_action_history_length', device)
