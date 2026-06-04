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

    assert agent._n_updates > 0

    return agent


def test_dqn():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.19848235, 0.23930363, -0.00657551, 0.33505076, -0.36853665,
                       -0.02079613, -0.06618375, 0.43131915, 0.16657214])

    assert np.allclose(w, w_test)


def test_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
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

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    learn(DQN, params, logger)

    loss_file = np.load(logger.path / 'loss_Q.npy')

    assert loss_file.shape == (81,)
    assert np.allclose(loss_file[0], 0.7500651478767395)
    assert np.allclose(loss_file[-1], 0.5382087826728821)


def test_prioritized_dqn():

    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.2036822, 0.23392732, -0.0123152, 0.3307111, -0.36101586,
                       -0.0121725, -0.06685782, 0.4360569, 0.15306984])

    assert np.allclose(w, w_test)


def test_prioritized_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
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
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    approximator = learn(DoubleDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.19848235, 0.23930365, -0.00657496, 0.33505082, -0.36853364,
                       -0.02080254, -0.06618375, 0.43131968, 0.1665634])

    assert np.allclose(w, w_test)


def test_double_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    agent_save = learn(DoubleDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_averaged_dqn():
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(AveragedDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.19846025, 0.2393236, -0.00552265, 0.33593556, -0.3700093,
                       -0.02234051, -0.06621848, 0.4320244, 0.16846392])

    assert np.allclose(w, w_test)


def test_averaged_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(AveragedDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_maxmin_dqn():
    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(MaxminDQN, params).approximator

    w = approximator[0].get_weights()
    w_test = np.array([-0.14637606, 0.29144737, 0.04357848, 0.38892528, -0.41530567,
                       -0.06814317, -0.15791543, 0.34688082, 0.08587101])

    assert np.allclose(w, w_test)


def test_maxmin_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, n_approximators=5, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(MaxminDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_dueling_dqn():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(DuelingDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.3061806, 0.56389284, 0.20562603, -0.10358601, -0.7341211,
                       0.50224435, -0.12417892, -0.5442874, -0.37345916, 0.09038441,
                       0.59898126, 0.66412216, 0.02598017, 0.04493077, 0.6922575,
                       -1.3478152, 0.91464853, -0.35326177])

    assert np.allclose(w, w_test)


def test_dueling_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(DuelingDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_categorical_dqn():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.9961103, 0.3062344, -0.38158175, -0.66480434, -0.6700932, 0.42846054,
                       -0.13206097, -0.45175707, -0.46569303, -0.01788258, 0.12564264, 0.10536393,
                       0.68053454, 0.61753774, -1.1476808, 0.80740047, 0.02384322, 0.07582285])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_categorical_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(CategoricalDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_quantile_dqn():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(QuantileDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.32990956, 0.5447867, 0.24215372, -0.07598333, -0.6022206,
                       0.34981713, -1.1227207, 0.6894482, 0.29463118, 1.2344137,
                       -0.12729159, 0.01054376, 0.73561853, -1.0448822, -1.1211058,
                       -0.5167815, -0.19001749, -0.35105047, 0.999933, -0.13281494,
                       0.44891715, -0.4945631, 0.43864432, 0.4653612])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_quantile_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(QuantileDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_noisy_dqn():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(NoisyDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([-0.3010259, 0.57106143, 0.1527097, -0.1720791, -0.60167384,
                       0.4341214, 0.1494948, 0.09190126, 0.19013157, -0.2041249,
                       0.00164998, -0.13452633, 0.31217825, 0.40318996, 0.29261786,
                       0.42166942, 0.39590827, 0.29078802, 0.16835015, 0.06620326,
                       0.5465707, 0.41377932, 0.4227871, 0.28850415])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_noisy_dqn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    agent_save = learn(NoisyDQN, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_rainbow():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    approximator = learn(Rainbow, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.40122062, -0.2713766, -0.17395529, 0.29116902, 0.3863779, 0.39097774,
                       0.31873056, 0.3150636, -0.6239894, 0.38235706, 0.39467114, 0.3141591,
                       -0.11718756, 0.36295494, 0.07001767, -0.08979987, 0.37919402, 0.3576465,
                       0.32715875, 0.347632, 0.17648698, 0.05451963, 0.33272108, 0.3706946,
                       0.2915318, -0.2385303, -0.08483079, -0.10068879, 0.38783127, 0.3902555,
                       0.32132867, 0.3168995, 0.11075438, -0.01108828, 0.36168385, 0.34621114,
                       0.58059007, 0.18328355, -0.22579168, -0.39030415, 0.32016888, 0.32173228,
                       0.3869971, 0.38787156, -0.1020159, -0.32151017, 0.36618155, 0.33658257])

    assert np.allclose(w, w_test, rtol=1e-4)


def test_rainbow_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=1,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    agent_save = learn(Rainbow, params)

    agent_save.save(agent_path, full_save=True)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
