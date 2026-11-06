import numpy as np
import pytest

import torch
import torch.optim as optim
import torch.nn.functional as F

from helper.dummies import FakeLogger
from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.rl_utils.preprocessors import Preprocessor
from mushroom_rl.utils import TorchUtils


class RecordingPreprocessor(Preprocessor):
    def __init__(self, scale):
        super().__init__('torch')
        self.scale = scale
        self.shapes = list()

    def __call__(self, state):
        self.shapes.append(tuple(state.shape))

        return state * self.scale


def learn_sac(setup=None, critic_n_models=None, **sac_params):
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
    if critic_n_models is not None:
        critic_params['n_models'] = critic_n_models

    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer,
                critic_params, batch_size, initial_replay_size, max_replay_size,
                warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, **sac_params)

    if setup is not None:
        setup(agent)

    core = Core(agent, mdp)

    core.learn(n_steps=4 * initial_replay_size,
               n_steps_per_fit=initial_replay_size)

    return agent


@pytest.mark.parametrize('device', devices())
def test_sac(device, tmpdir):
    agent = learn_sac()
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_sac', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_sac_options(device):
    TorchUtils.set_default_device('cpu')
    logger = FakeLogger()
    agent = learn_sac(setup=lambda agent: agent.set_logger(logger), critic_n_models=2, device=device)

    tu.assert_recorded(agent.policy.get_weights(), 'test_sac', device)
    assert {key for call in logger.calls for key in call} == {'actor/entropy', 'actor/loss', 'alpha/loss',
                                                              'alpha/value', 'critic/loss_0', 'critic/loss_1'}

    with pytest.raises(AssertionError):
        learn_sac(critic_n_models=3, device=device)

    agent = learn_sac(use_log_alpha_loss=True, target_entropy=-2., device=device)
    log_prob = torch.tensor([-1., -2.], device=device)
    log_alpha = agent._log_alpha.detach().clone()

    assert torch.allclose(agent._update_alpha(log_prob), -(log_alpha * (log_prob - 2.)).mean())


@pytest.mark.parametrize('device', devices())
def test_sac_fit_preprocessor_identity(device):
    preprocessor = RecordingPreprocessor(1.)
    policy = learn_sac(lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_sac', device)
    assert preprocessor.shapes == [(64, 2)] * 8


@pytest.mark.parametrize('device', devices())
def test_sac_fit_preprocessor(device):
    preprocessor = RecordingPreprocessor(0.)
    policy = learn_sac(lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_sac_fit_preprocessor', device)
    assert preprocessor.shapes == [(64, 2)] * 8
