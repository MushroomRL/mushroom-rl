import numpy as np
import pytest
import torch
import torch.optim as optim
import torch.nn.functional as F

from helper.dummies import FakeLogger
from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.actor_critic import DDPG, TD3, DeepAC
from mushroom_rl.core import Core
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.rl_utils.preprocessors import Preprocessor


class RecordingPreprocessor(Preprocessor):
    def __init__(self, scale):
        super().__init__('torch')
        self.scale = scale
        self.shapes = list()

    def __call__(self, state):
        self.shapes.append(tuple(state.shape))

        return state * self.scale


def learn(alg, setup=None, **alg_params):
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
                initial_replay_size, max_replay_size, tau, **alg_params)

    if setup is not None:
        setup(agent)

    core = Core(agent, mdp)

    core.learn(n_episodes=10, n_episodes_per_fit=5)

    assert agent._fit_count > 0

    return agent


@pytest.mark.parametrize('device', devices())
def test_ddpg(device, tmpdir):
    agent = learn(DDPG)
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_ddpg', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_ddpg_options(device):
    logger = FakeLogger()
    agent = learn(DDPG, setup=lambda agent: agent.set_logger(logger), device=device)

    tu.assert_recorded(agent.policy.get_weights(), 'test_ddpg', device)
    assert logger.keys() == {'actor/loss', 'critic/loss'}

    with pytest.raises(NotImplementedError):
        DeepAC.fit(None, None)
    with pytest.raises(NotImplementedError):
        DeepAC._post_load(None, None)


@pytest.mark.parametrize('device', devices())
def test_td3(device, tmpdir):
    agent = learn(TD3)
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_td3', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_ddpg_fit_preprocessor_identity(device):
    preprocessor = RecordingPreprocessor(1.)
    policy = learn(DDPG, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_ddpg', device)
    assert preprocessor.shapes == [(200, 2)] * 4


@pytest.mark.parametrize('device', devices())
def test_ddpg_fit_preprocessor(device):
    preprocessor = RecordingPreprocessor(0.)
    policy = learn(DDPG, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_ddpg_fit_preprocessor', device)
    assert preprocessor.shapes == [(200, 2)] * 4


@pytest.mark.parametrize('device', devices())
def test_td3_fit_preprocessor(device):
    preprocessor = RecordingPreprocessor(0.)
    policy = learn(TD3, lambda agent: agent.add_fit_preprocessor(preprocessor)).policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_td3_fit_preprocessor', device)
    assert preprocessor.shapes == [(200, 2)] * 4


@pytest.mark.parametrize('device', devices())
def test_ddpg_overwrite_actor_parameters(device):
    parameters = dict()

    def exclude_bias(agent):
        parameters['default'] = agent.actor_parameters
        weight, bias = agent.actor_parameters
        parameters['weight'], parameters['bias'] = weight.detach().clone(), bias.detach().clone()
        agent.overwrite_actor_parameters([weight])

    agent = learn(DDPG, exclude_bias)

    weight, bias = agent.policy._approximator.parameters()

    assert len(parameters['default']) == 2
    assert parameters['default'][0] is weight and parameters['default'][1] is bias
    assert len(agent.actor_parameters) == 1 and agent.actor_parameters[0] is weight
    assert torch.equal(bias, parameters['bias'])
    assert not torch.equal(weight, parameters['weight'])

    w = agent.policy.get_weights()

    tu.assert_recorded(w, 'test_ddpg_overwrite_actor_parameters', device)
