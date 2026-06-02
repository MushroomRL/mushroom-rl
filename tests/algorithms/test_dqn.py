import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Core, Agent, AgentInfo, Logger
from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN,\
    MaxminDQN, DuelingDQN, CategoricalDQN, QuantileDQN, NoisyDQN, Rainbow
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.approximators.parametric.networks import QNetwork


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

    def forward(self, state, action=None):
        return torch.squeeze(state, 1).float()


def learn(alg, alg_params, logger=None):
    # MDP
    mdp = CartPole()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Policy
    epsilon_random = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon_random)

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

    # Agent
    if alg not in [DuelingDQN, QuantileDQN, CategoricalDQN, NoisyDQN, Rainbow]:
        agent = alg(mdp.info, pi, NumpyTorchApproximator,
                    approximator_params=approximator_params, **alg_params)
    elif alg in [CategoricalDQN, Rainbow]:
        agent = alg(mdp.info, pi, approximator_params=approximator_params,
                    n_atoms=2, v_min=-1, v_max=1, **alg_params)
    elif alg in [QuantileDQN]:
        agent = alg(mdp.info, pi, approximator_params=approximator_params,
                    n_quantiles=2, **alg_params)
    else:
        agent = alg(mdp.info, pi, approximator_params=approximator_params,
                    **alg_params)

    if logger is not None:
        agent.set_logger(logger)

    # Algorithm
    core = Core(agent, mdp)

    core.learn(n_steps=500, n_steps_per_fit=5)

    return agent


def test_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.20050035, 0.24002515, -0.02295898, 0.32232404, -0.36275005,
                       -0.01453803, -0.05687293, 0.43410516, 0.1590765])

    assert np.allclose(w, w_test)


def test_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    agent_save = learn(DQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_dqn_logger(tmpdir):
    logger = Logger('dqn_logger', results_dir=tmpdir, use_timestamp=True)

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    learn(DQN, params, logger)

    loss_file = np.load(logger.path / 'loss_Q.npy')

    assert loss_file.shape == (90,)
    assert loss_file[0] == 0.7991676926612854 and loss_file[-1] == 0.5159794688224792


def test_prioritized_dqn():

    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.19866402, 0.2357937, -0.02972439, 0.31267533, -0.35202834,
                       -0.00553649, -0.05840667, 0.43584204, 0.16416478])

    assert np.allclose(w, w_test)


def test_prioritized_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent_save = learn(DQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_double_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DoubleDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.20049825, 0.24002622, -0.02295567, 0.3223236, -0.3627546,
                       -0.01453804, -0.05687486, 0.43409595, 0.1590613])

    assert np.allclose(w, w_test)


def test_double_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50)
    agent_save = learn(DoubleDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_averaged_dqn():
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(AveragedDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.20050386, 0.24001993, -0.0221192, 0.32318467, -0.36512366,
                       -0.0166209, -0.0568868, 0.43519753, 0.16197258])

    assert np.allclose(w, w_test)


def test_averaged_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, n_approximators=5, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(AveragedDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_maxmin_dqn():
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(MaxminDQN, params).approximator

    w = approximator[0].get_weights()
    w_test = np.array([-0.15172458, 0.28573582, 0.03817964, 0.38416952, -0.41005808,
                       -0.06267934, -0.16421957, 0.3418669, 0.07960468])

    assert np.allclose(w, w_test)


def test_maxmin_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, n_approximators=5, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(MaxminDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_dueling_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(DuelingDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.314546, 0.54604924, 0.20915823, -0.07975413, -0.74092555,
                       0.5050177, -0.14456294, -0.5389542, -0.3534887, 0.07831666,
                       0.61038506, 0.6548378, 0.02511988, 0.04343156, 0.6825224,
                       -1.3261368, 0.9239293, -0.3491529])

    assert np.allclose(w, w_test)


def test_dueling_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(DuelingDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_categorical_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.99196255, 0.3011091, -0.37743387, -0.6596791, -0.7362115, 0.49457926,
                       -0.11459535, -0.4379756, -0.48315868, -0.03166399, 0.1229288, 0.10807777,
                       0.6754166, 0.61144304, -1.1425636, 0.8134951, 0.03522068, 0.06444537])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_categorical_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(CategoricalDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_quantile_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(QuantileDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.33921587, 0.5343955, 0.24983048, -0.06784417, -0.6075502,
                       0.3601502, -1.115198, 0.6865875, 0.29323563, 1.2454914,
                       -0.12360247, 0.02286462, 0.7213106, -1.0360246, -1.121014,
                       -0.5056948, -0.18354908, -0.3344422, 0.99689376, -0.13668007,
                       0.44865605, -0.49715987, 0.4467747, 0.4644443])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_quantile_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(QuantileDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_noisy_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(NoisyDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.30852458, 0.5647726, 0.13873313, -0.17767607, -0.59714687,
                       0.45167482, 0.14077583, 0.10388352, 0.1845346, -0.19802122,
                       -0.00943174, -0.13583986, 0.30453312, 0.41494054, 0.28900382,
                       0.42479482, 0.3897369, 0.28961122, 0.17465961, 0.07124078,
                       0.53694916, 0.42173594, 0.42799577, 0.2785571])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_noisy_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(NoisyDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_rainbow():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    approximator = learn(Rainbow, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.41471523, -0.24608319, -0.18744999, 0.26587564, 0.39882535, 0.412821,
                       0.30898723, 0.29745516, -0.5973996, 0.35576734, 0.41858765, 0.2911771,
                       -0.09666843, 0.32220146, 0.04949852, -0.04904625, 0.3972141, 0.32487455,
                       0.3105287, 0.38326296, 0.15647355, 0.07453305, 0.31577617, 0.38884395,
                       0.30908346, -0.20951316, -0.1023823, -0.12970605, 0.40118366, 0.41426662,
                       0.30691648, 0.2924496, 0.08292492, 0.01674112, 0.33560023, 0.3732411,
                       0.5594649, 0.17095159, -0.20466673, -0.37797216, 0.29877642, 0.3118145,
                       0.40977645, 0.39796302, -0.0712048, -0.35232118, 0.40097338, 0.3074576])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_rainbow_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=1,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    agent_save = learn(Rainbow, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
