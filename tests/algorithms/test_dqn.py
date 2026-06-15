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
from mushroom_rl.approximators.parametric import TorchApproximator
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
    pi = EpsGreedy(epsilon=epsilon_random, backend='torch')

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
        agent = alg(mdp.info, pi, TorchApproximator,
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
    w_test = torch.tensor([-0.19200534, 0.24389042, -0.00848553, 0.3319371, -0.36971146,
                           -0.0203344, -0.07255276, 0.43228102, 0.17088246])

    assert torch.allclose(w, w_test)


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
    assert np.allclose(loss_file[0], 0.6862927079200745)
    assert np.allclose(loss_file[-1], 0.6511316895484924)


def test_prioritized_dqn():

    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([-0.1949389, 0.24404983, -0.02552294, 0.32007575, -0.3530751,
                           -0.00599685, -0.07104248, 0.4382063, 0.16815822])

    assert torch.allclose(w, w_test)


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
    w_test = torch.tensor([-0.19200534, 0.24389042, -0.00847956, 0.33193585, -0.36971503,
                           -0.02033439, -0.07255276, 0.43226764, 0.17087594])

    assert torch.allclose(w, w_test)


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
    w_test = torch.tensor([-0.19112705, 0.24443369, -0.00192504, 0.3365183, -0.36826974,
                           -0.02310302, -0.06729474, 0.43261513, 0.17063351])

    assert torch.allclose(w, w_test)


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
    w_test = torch.tensor([-0.14843702, 0.29001242, 0.03975038, 0.38547155, -0.41254845,
                           -0.06506725, -0.15931359, 0.34374514, 0.0826805])

    assert torch.allclose(w, w_test)


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
    w_test = torch.tensor([-0.29283327, 0.55574864, 0.22541833, -0.0875193, -0.73231137,
                           0.5023717, -0.18161395, -0.53954935, -0.33074442, 0.07271275,
                           0.63677883, 0.65162057, 0.02478503, 0.0478552, 0.68452716,
                           -1.3329868, 0.9104391, -0.35360464])

    assert torch.allclose(w, w_test)


def test_dueling_dqn_max_advantage():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50,
                  avg_advantage=False)
    approximator = learn(DuelingDQN, params).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([-0.2719743, 0.5613536, 0.2201571, -0.09746218, -0.71286273,
                           0.50259686, -0.17259271, -0.528971, -0.3208862, 0.03267539,
                           0.6074301, 0.6597853, 0.03088126, 0.00268685, 0.68195426,
                           -1.341527, 0.8975734, -0.360234])

    assert torch.allclose(w, w_test)


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
    w_test = torch.tensor([0.9995616, 0.30595043, -0.38503268, -0.66452056, -0.7088026, 0.46717066,
                           -0.13178714, -0.45272827, -0.46596685, -0.01691139, 0.12424235, 0.10676417,
                           0.6806235, 0.6191608, -1.1477692, 0.8057775, 0.03099489, 0.0686712])

    assert torch.allclose(w, w_test, rtol=1e-4)


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
    w_test = torch.tensor([-0.32854617, 0.545965, 0.23684339, -0.08290148, -0.6057714,
                           0.3587803, -1.1263039, 0.6958708, 0.32324034, 1.2375141,
                           -0.11570817, 0.00793537, 0.73455185, -1.0388892, -1.1009699,
                           -0.51682687, -0.18576333, -0.34831285, 1.0020312, -0.13552995,
                           0.4542169, -0.49865612, 0.47680384, 0.4611693])

    assert torch.allclose(w, w_test, rtol=1e-4)


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
    w_test = torch.tensor([-0.30767578, 0.56588894, 0.14080539, -0.17563716, -0.60002732,
                           0.46806017, 0.13817938, 0.09937383, 0.18325211, -0.19386475,
                           0.00053908, -0.12810652, 0.30581218, 0.40659443, 0.28652766,
                           0.4319292, 0.39860207, 0.28698882, 0.16350792, 0.0747801,
                           0.54863387, 0.41095498, 0.43024299, 0.28844392])

    assert torch.allclose(w, w_test, rtol=1e-4)


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
    w_test = torch.tensor([0.42586854, -0.25041503, -0.19860311, 0.27020743, 0.4059107, 0.40676644,
                           0.29740217, 0.29967862, -0.59745735, 0.3558251, 0.4162725, 0.28836426,
                           -0.09792814, 0.40676, 0.05075819, -0.13360481, 0.39370537, 0.38923103,
                           0.30976593, 0.3126058, 0.2209986, 0.01000803, 0.37632477, 0.32897794,
                           0.31132254, -0.22169225, -0.10462158, -0.11752684, 0.4013015, 0.40268973,
                           0.29772234, 0.30463547, 0.07883958, 0.02082647, 0.3304029, 0.37447345,
                           0.5573451, 0.16025995, -0.20254706, -0.3672807, 0.29669714, 0.29901275,
                           0.41084516, 0.4121757, -0.12198, -0.30154607, 0.3476741, 0.35078502])

    assert torch.allclose(w, w_test, rtol=1e-4)


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
