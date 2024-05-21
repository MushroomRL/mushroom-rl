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


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action=None):
        q = F.relu(self._h1(torch.squeeze(state, 1).float()))

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


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
    approximator_params = dict(network=Network if alg not in [CategoricalDQN, Rainbow] else FeatureNetwork,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss if alg not in [CategoricalDQN, Rainbow] else None,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=2
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
    w_test = np.array([-0.20857571, 0.4301014, 0.09157596, 0.56593966, -0.573920,
                       -0.07434221, -0.07043041, 0.42729577, 0.15255776])

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
    assert loss_file[0] == 0.9765409231185913 and loss_file[-1] == 0.6936992406845093


def test_prioritized_dqn():

    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.2410347, 0.39138362, 0.12457055, 0.60612524, -0.54973847,
                       -0.06486652, -0.07349031, 0.4376623, 0.14254288])

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
    w_test = np.array([-0.20857571, 0.4301014, 0.09157596, 0.56593966, -0.5739204,
                       -0.07434221, -0.07043041, 0.42729577, 0.15255776])

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
    w_test = np.array([-0.20855692, 0.4300971, 0.09070298, 0.56503105, -0.57473886,
                       -0.07523372, -0.07045465, 0.42477432, 0.15313861])

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
    w_test = np.array([-0.20750952, 0.41153884, 0.06031952, 0.54991245, -0.58597267,
                       -0.09532283, -0.1639097, 0.34269238, 0.08022686])

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
    w_test = np.array([-0.44132388, 0.79595834, 0.23078996, -0.17289384,
                       -0.7490091, 0.5055381, -0.14357203, -0.4858748,
                       -0.38062495, 0.10331012, 0.62843525, 0.5607314,
                       0.05413188, 0.07322324, 0.56302196, -1.3005875,
                       0.94485873, -0.34308702])

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
    w_test = np.array([-0.445598, 0.7921833, 0.3127064, -0.13804975, -0.7560823,
                       0.35417593, -1.1218646, 0.7265262, 0.40201563, 1.2316055,
                       0.02598637, 0.02116407, 0.76916, -1.0395582, -1.0759451,
                       -0.5113829, -0.18624172, -0.33754084, 1.005778, -0.2562586,
                       0.5079987, -0.5034418, 0.3462327, 0.45655805])

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
    w_test = np.array([-0.40481162, 0.84183586, 0.29812846, -0.15331453, -0.6233022,
                       0.44782484, 0.17155018, 0.07006463, 0.23487908, -0.23030677,
                       -0.10514411, -0.13489397, 0.32838345, 0.37297514, 0.32157022,
                       0.38325936, 0.30015582, 0.28873885, 0.16997868, 0.06498576,
                       0.5568779, 0.4157398, 0.4247934, 0.2948213 ])

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
