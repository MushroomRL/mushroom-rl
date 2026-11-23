from functools import partial

import numpy as np
import pytest
import torch
import torch.optim as optim
import torch.nn.functional as F

from helper.dummies import FakeLogger, RecordingPreprocessor
from helper.cases import devices, needs_cuda
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms.actor_critic import DDPG, TD3, OffPolicyDeepAC
from mushroom_rl.core import Agent, Batch, Core
from mushroom_rl.core.asymmetric_history_manager import AsymmetricHistoryManager
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy, GaussianTorchPolicy
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import ActorNetwork, CriticNetwork
from mushroom_rl.utils import TorchUtils


def learn(alg, setup=None, policy=None, critic_n_models=None, asymmetric=None, **alg_params):
    mdp = InvertedPendulum(horizon=50)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    initial_replay_size = 100
    max_replay_size = 5000
    batch_size = 200
    tau = .001

    actor_input_shape = mdp.info.observation_space.shape
    critic_state_shape = actor_input_shape
    if asymmetric is not None:
        actor_input_shape = (len(asymmetric['actor_indices']),)
        critic_state_shape = (len(asymmetric['critic_indices']),)
    actor_params = dict(network=ActorNetwork,
                        n_features=None,
                        n_layers=0,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape)

    if policy is None:
        device = alg_params.get('device')
        policy = OrnsteinUhlenbeckPolicy(TorchApproximator(**actor_params, device=device), sigma=torch.ones(1) * .2,
                                         theta=.15, dt=1e-2, device=device)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': .001}}

    critic_input_shape = [critic_state_shape, mdp.info.action_space.shape]
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': .001}},
                         loss=F.mse_loss,
                         n_features=None,
                         n_layers=0,
                         input_shape=critic_input_shape,
                         output_shape=())
    if critic_n_models is not None:
        critic_params['n_models'] = critic_n_models

    make = alg if asymmetric is None else partial(alg.asymmetric, **asymmetric)
    agent = make(mdp.info, policy, actor_optimizer, critic_params, batch_size, initial_replay_size, max_replay_size,
                 tau, **alg_params)

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
    TorchUtils.set_default_device('cpu')
    logger = FakeLogger()
    agent = learn(DDPG, setup=lambda agent: agent.set_logger(logger), device=device)

    tu.assert_recorded(agent.policy.get_weights(), 'test_ddpg', device)
    assert logger.keys() == {'actor/loss', 'critic/loss'}

    with pytest.raises(NotImplementedError):
        OffPolicyDeepAC.fit(None, None)


@pytest.mark.parametrize('device', devices())
def test_td3(device, tmpdir):
    agent = learn(TD3)
    policy = agent.policy
    w = policy.get_weights()

    tu.assert_recorded(w, 'test_td3', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)

    with pytest.raises(AssertionError):
        learn(TD3, critic_n_models=1)


@needs_cuda()
def test_ddpg_policy_device():
    actor = TorchApproximator(network=ActorNetwork, n_features=None, n_layers=0, input_shape=(2,), output_shape=(1,),
                              device='cpu')
    policy = OrnsteinUhlenbeckPolicy(actor, sigma=torch.ones(1) * .2, theta=.15, dt=1e-2, device='cpu')

    with pytest.raises(AssertionError):
        learn(DDPG, policy=policy, device='cuda')


@pytest.mark.parametrize('device', devices())
def test_ddpg_gaussian_policy(device):
    torch.manual_seed(1)
    policy = GaussianTorchPolicy(ActorNetwork, (2,), (1,), std_0=.2, n_features=None, n_layers=0)
    weights = policy.get_weights().clone()

    agent = learn(DDPG, policy=policy)
    new_weights = agent.policy.get_weights()

    assert not torch.equal(new_weights[:-1], weights[:-1])
    assert torch.equal(new_weights[-1:], weights[-1:])


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
def test_ddpg_asymmetric(device, tmpdir):
    asymmetric = dict(actor_indices=[1], critic_indices=[1, 0])
    for alg in (DDPG, TD3):
        preprocessor = RecordingPreprocessor(1.)
        agent = learn(alg, lambda agent: agent.add_fit_preprocessor(preprocessor), asymmetric=asymmetric,
                      device=device)

        tu.assert_recorded(agent.policy.get_weights(), f'test_{alg.__name__.lower()}_asymmetric', device)
        assert type(agent.history_manager) is AsymmetricHistoryManager
        assert agent.history_manager.critic_indices == [1, 0] and '_asymmetric_options' not in vars(agent)
        assert preprocessor.shapes == [(200, 1)] * 4
        tu.assert_save_load(agent, tmpdir / f'{alg.__name__}.msh', full_save=True)
        loaded = Agent.load(tmpdir / f'{alg.__name__}.msh')
        assert loaded._replay_memory._history_manager is loaded.history_manager

    agent.add_fit_preprocessor(RecordingPreprocessor(2.))
    state = torch.ones(3, 1)
    reward = torch.zeros(3)
    shared = agent._augment_batch(Batch(state, None, reward, state + 1, None, None))
    separate = agent._augment_batch(Batch(state, None, reward, state + 1, None, None, critic_state=state + 2,
                                          critic_next_state=state + 3))

    assert shared.critic_state is shared.state and shared.critic_next_state is shared.next_state
    assert torch.equal(shared.state, 2 * state) and torch.equal(shared.next_state, 2 * state + 2)
    assert shared.reward is reward and type(shared) is Batch
    assert torch.equal(separate.state, 2 * state) and torch.equal(separate.critic_state, state + 2)
    assert torch.equal(separate.critic_next_state, state + 3)


@pytest.mark.parametrize('device', devices())
def test_ddpg_overwrite_actor_parameters(device):
    parameters = dict()

    def exclude_bias(agent):
        parameters['default'] = agent.actor_parameters
        weight, bias = agent.actor_parameters
        parameters['weight'], parameters['bias'] = weight.detach().clone(), bias.detach().clone()
        agent.overwrite_actor_parameters([weight])

    agent = learn(DDPG, exclude_bias)

    weight, bias = agent.policy.parameters()

    assert len(parameters['default']) == 2
    assert parameters['default'][0] is weight and parameters['default'][1] is bias
    assert len(agent.actor_parameters) == 1 and agent.actor_parameters[0] is weight
    assert torch.equal(bias, parameters['bias'])
    assert not torch.equal(weight, parameters['weight'])

    w = agent.policy.get_weights()

    tu.assert_recorded(w, 'test_ddpg_overwrite_actor_parameters', device)
