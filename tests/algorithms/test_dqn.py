import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from helper.cases import devices
from helper.utils import TestUtils as tu

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN, \
    MaxminDQN, DuelingDQN, CategoricalDQN, QuantileDQN, NoisyDQN, Rainbow
from mushroom_rl.environments import CartPole
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.utils import TorchUtils


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

    def forward(self, state, action=None):
        return torch.squeeze(state, 1).float()


class ScaledCartPole(CartPole):
    def step(self, action):
        state, reward, absorbing, info = super().step(action)

        return state, 10. * reward, absorbing, info


def get_variant_params(alg):
    variant_params = {CategoricalDQN: dict(n_atoms=2, v_min=-1, v_max=1),
                      Rainbow: dict(n_atoms=2, v_min=-1, v_max=1),
                      QuantileDQN: dict(n_quantiles=2)}

    return variant_params.get(alg, dict())


def learn(alg, alg_params, logger=None, n_models=None, randomness=None, mdp=None):
    # MDP
    mdp = CartPole() if mdp is None else mdp
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random, backend='torch', device=alg_params.get('device'))

    # Approximator
    input_shape = mdp.info.observation_space.shape
    approximator_params = dict(network=QNetwork if alg not in [CategoricalDQN, Rainbow] else FeatureNetwork,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss if alg not in [CategoricalDQN, Rainbow] else None,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=2,
                               n_layers=0
                               )

    if n_models is not None:
        approximator_params['n_models'] = n_models

    if randomness is not None:
        approximator_params['randomness'] = randomness

    # Agent
    agent = alg(mdp.info, pi, approximator_params=approximator_params,
                **get_variant_params(alg), **alg_params)

    if logger is not None:
        agent.set_logger(logger)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=500, n_steps_per_fit=5)

    assert agent._n_updates > 0

    return agent


@pytest.mark.parametrize('device', devices())
def test_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    agent = learn(DQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_dqn_logger(device, tmpdir):
    logger = Logger(f'dqn_logger_{device}', results_dir=tmpdir, use_timestamp=True, force_numpy=True)

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    learn(DQN, params, logger)

    loss_file = np.load(logger.path / 'training' / 'critic_loss.npy')

    assert loss_file.shape == (81,)

    loss_test = dict(cpu=[0.6862927079200745, 0.6281029582023621], cuda=[0.6408013105392456, 0.3261929154396057])

    assert np.allclose(loss_file[[0, -1]], loss_test[device], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('device', devices())
def test_options(device):
    TorchUtils.set_default_device('cpu')
    params = dict(batch_size=50, initial_replay_size=100, max_replay_size=500, target_update_frequency=50,
                  clip_reward=True, fit_params=dict(n_epochs=2), device=device)
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}

    w = learn(DQN, params, mdp=ScaledCartPole()).approximator.get_weights()
    w_prioritized = learn(DQN, dict(params, replay_memory=replay_memory, predict_params=dict(prediction='min')),
                          n_models=2, mdp=ScaledCartPole()).approximator.get_weights()

    tu.assert_recorded(w, 'test_dqn_options', device)
    tu.assert_recorded(w_prioritized, 'test_dqn_options_prioritized', device)


@pytest.mark.parametrize('device', devices())
def test_prioritized_dqn(device, tmpdir):
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent = learn(DQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_prioritized_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_double_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    agent = learn(DoubleDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_double_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_averaged_dqn(device, tmpdir):
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(AveragedDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_averaged_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_maxmin_dqn(device, tmpdir):
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(MaxminDQN, params)
    approximator = agent.approximator

    w = approximator[0].get_weights()

    tu.assert_recorded(w, 'test_maxmin_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_dueling_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(DuelingDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_dueling_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_dueling_dqn_max_advantage(device):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50,
                  avg_advantage=False)
    approximator = learn(DuelingDQN, params).approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_dueling_dqn_max_advantage', device)


@pytest.mark.parametrize('device', devices())
def test_categorical_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(CategoricalDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_categorical_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_categorical_dqn_ensemble(device):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params, n_models=2).approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_categorical_dqn_ensemble', device)


@pytest.mark.parametrize('device', devices())
def test_categorical_dqn_prioritized(device):
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent = learn(CategoricalDQN, params)

    w = agent.approximator.get_weights()

    tu.assert_recorded(w, 'test_categorical_dqn_prioritized', device)

    leaves = agent._replay_memory._tree._tree[-agent._replay_memory._max_size:]
    assert len(np.unique(leaves)) == dict(cpu=492, cuda=493)[device]


@pytest.mark.parametrize('device', devices())
def test_quantile_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(QuantileDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_quantile_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_quantile_dqn_prioritized(device):
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent = learn(QuantileDQN, params)

    w = agent.approximator.get_weights()

    tu.assert_recorded(w, 'test_quantile_dqn_prioritized', device)

    leaves = agent._replay_memory._tree._tree[-agent._replay_memory._max_size:]
    assert len(np.unique(leaves)) == dict(cpu=485, cuda=489)[device]


@pytest.mark.parametrize('device', devices())
def test_noisy_dqn(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent = learn(NoisyDQN, params)
    approximator = agent.approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_noisy_dqn', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_noisy_dqn_ensemble(device):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(NoisyDQN, params, n_models=2).approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_noisy_dqn_ensemble', device)


def test_ensemble_randomness_error():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)

    with pytest.raises(RuntimeError):
        learn(NoisyDQN, params, n_models=2, randomness='error')


@pytest.mark.parametrize('device', devices())
def test_rainbow(device, tmpdir):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    agent = learn(Rainbow, params)

    w = agent.approximator.get_weights()

    tu.assert_recorded(w, 'test_rainbow', device)

    tu.assert_save_load(agent, tmpdir / 'agent.msh', full_save=True)


@pytest.mark.parametrize('device', devices())
def test_rainbow_ensemble(device):
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    approximator = learn(Rainbow, params, n_models=2).approximator

    w = approximator.get_weights()

    tu.assert_recorded(w, 'test_rainbow_ensemble', device)
