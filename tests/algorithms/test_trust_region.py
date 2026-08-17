import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork


def learn(alg, alg_params):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    critic_params = dict(network=ActorNetwork,
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

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return agent


def learn_history(alg, alg_params, history_length):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    window_shape = (history_length,) + mdp.info.observation_space.shape

    critic_params = dict(network=ActorNetwork,
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

    critic_params = dict(network=ActorNetwork,
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


def test_PPO():
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    policy = learn(PPO, params).policy
    w = policy.get_weights()
    w_test = torch.tensor([0.6613755, -1.333808, -0.13946329, -0.00241474])

    assert torch.allclose(w, w_test)


def test_PPO_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)

    agent_save = learn(PPO, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        tu.assert_eq(save_attr, load_attr)


def test_TRPO():
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    policy = learn(TRPO, params).policy
    w = policy.get_weights()
    w_test = torch.tensor([0.53987426, -1.3105278, 0.02826479, -0.02005163])

    assert torch.allclose(w, w_test, rtol=1e-4), f"actual={w}, expected={w_test}, diff={w - w_test}"


def test_TRPO_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)

    agent_save = learn(TRPO, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        tu.assert_eq(save_attr, load_attr)


def test_PPO_history_length():
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    agent = learn_history(PPO, params, history_length=3)
    w = agent.policy.get_weights()
    w_test = torch.tensor([0.4689, 0.1265, -0.1155, 0.2546, 0.0435, 0.3359, -0.0862, -0.0023])

    assert agent.history_length == 3
    assert torch.allclose(w, w_test, atol=1e-4), f"actual={w}, expected={w_test}, diff={w - w_test}"


def test_TRPO_history_length():
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    agent = learn_history(TRPO, params, history_length=3)
    w = agent.policy.get_weights()
    w_test = torch.tensor([0.4595, 0.1035, -0.1322, 0.2626, -0.0210, 0.3771, 0.0137, -0.0177])

    assert agent.history_length == 3
    assert torch.allclose(w, w_test, atol=1e-4), f"actual={w}, expected={w_test}, diff={w - w_test}"


def test_PPO_action_history_length():
    params = dict(actor_optimizer={'class': optim.Adam,
                                   'params': {'lr': 3e-4}},
                  n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
    agent = learn_action_history(PPO, params, action_history_length=1)
    w = agent.policy.get_weights()
    w_test = torch.tensor([-1.1554, 0.7321, -0.2500, 0.2686, -0.0023])

    assert agent._history_manager.uses_action
    assert torch.allclose(w, w_test, atol=1e-4), f"actual={w}, expected={w_test}, diff={w - w_test}"


def test_TRPO_action_history_length():
    params = dict(ent_coeff=0.0, max_kl=.001, lam=.98, n_epochs_line_search=10,
                  n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10)
    agent = learn_action_history(TRPO, params, action_history_length=1)
    w = agent.policy.get_weights()
    w_test = torch.tensor([-1.2469, 0.7433, -0.2348, 0.3901, -0.0191])

    assert agent._history_manager.uses_action
    assert torch.allclose(w, w_test, atol=1e-4), f"actual={w}, expected={w_test}, diff={w - w_test}"
