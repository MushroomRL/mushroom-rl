import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent

from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, **kwargs):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


def learn(alg, alg_params):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy_params = dict(std_0=1.)

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

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
    w_test = np.array([0.9378777, -1.8841006 , -0.13794397, -0.00241548])

    assert np.allclose(w, w_test)


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
    w_test = np.array([9.5286590e-01, -1.9460459e+00, -1.2838534e-01, 8.5962377e-04])

    assert np.allclose(w, w_test)


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
