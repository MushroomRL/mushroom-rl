import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Core, Agent, Logger
from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN, \
    MaxminDQN, DuelingDQN, CategoricalDQN, QuantileDQN, NoisyDQN, Rainbow
from mushroom_rl.environments import CartPole
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.approximators.parametric.networks import QNetwork


class FeatureNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

    def forward(self, state, action=None):
        return torch.squeeze(state, 1).float()


def get_variant_params(alg):
    variant_params = {CategoricalDQN: dict(n_atoms=2, v_min=-1, v_max=1),
                      Rainbow: dict(n_atoms=2, v_min=-1, v_max=1),
                      QuantileDQN: dict(n_quantiles=2)}

    return variant_params.get(alg, dict())


def learn(alg, alg_params, logger=None, n_models=None, randomness=None):
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
    logger = Logger('dqn_logger', results_dir=tmpdir, use_timestamp=True, force_numpy=True)

    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50)
    learn(DQN, params, logger)

    loss_file = np.load(logger.path / 'training' / 'critic_loss.npy')

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


def test_categorical_dqn_ensemble():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params, n_models=2).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([[0.99830014, 0.30862880, -0.38377142, -0.66719884, -0.72498733,
                            0.48335490, -0.13292903, -0.45644495, -0.46482500, -0.01319471,
                            0.13840525, 0.09260134, 0.68031740, 0.62151283, -1.14746261,
                            0.80342537, 0.03018998, 0.06947609],
                           [-0.42118350, -0.20138900, -1.10992527, -0.51682109, -0.73422748,
                            -0.27915561, 0.73996288, -0.22297402, 1.14453173, 0.90700752,
                            0.63015109, -0.23550124, 0.01245439, -0.74631345, 0.67602193,
                            0.30892509, -0.44027597, 0.27289289]])

    assert torch.allclose(w, w_test, rtol=1e-4)


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


def test_noisy_dqn_ensemble():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(NoisyDQN, params, n_models=2).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([[-0.31128243, 0.56742239, 0.14685135, -0.16481605, -0.61745375,
                            0.41514882, 0.14010701, 0.09320547, 0.18906650, -0.19923376,
                            0.00755294, -0.13527167, 0.30086264, 0.40413558, 0.29461730,
                            0.41600201, 0.40639064, 0.29105210, 0.14933082, 0.07377291,
                            0.54131001, 0.39827541, 0.42120215, 0.28398669],
                           [0.06294403, 0.76217669, 0.69013655, -1.15468574, -0.24574728,
                            -0.16705115, 0.39663973, 0.20313065, 0.71676534, 0.42320797,
                            -0.66667509, -0.64505261, 0.31736046, 0.33818042, 0.39665458,
                            0.30461195, 0.33025485, 0.36893621, -0.27435872, 0.59338033,
                            -0.16370605, 0.41478497, 0.31813756, 0.30946082]])

    assert torch.allclose(w, w_test, rtol=1e-4)


def test_ensemble_randomness_error():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50)

    with pytest.raises(RuntimeError):
        learn(NoisyDQN, params, n_models=2, randomness='error')


def test_rainbow():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    approximator = learn(Rainbow, params).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([0.42845434, -0.24809416, -0.20118916, 0.26788655, 0.41274157, 0.40684506,
                           0.29550439, 0.30127576, -0.59203315, 0.35040084, 0.42223322, 0.28629413,
                           -0.11818635, 0.38609692, 0.07101646, -0.11294184, 0.37428197, 0.37702212,
                           0.32961598, 0.3276788, 0.18715654, 0.04385009, 0.34483591, 0.36077037,
                           0.30864027, -0.22007941, -0.10193933, -0.1191397, 0.39863941, 0.40309867,
                           0.30896536, 0.30512568, 0.09705524, 0.0026108, 0.34753001, 0.36029631,
                           0.55655897, 0.15793894, -0.20176075, -0.36495951, 0.2986218, 0.29928797,
                           0.41159186, 0.41410273, -0.10305242, -0.32047361, 0.36886129, 0.33652306])

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


def test_rainbow_ensemble():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50, n_steps_return=3,
                  alpha_coeff=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))
    approximator = learn(Rainbow, params, n_models=2).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([[0.43343487, -0.24157998, -0.20616950, 0.26137239, 0.41191459,
                            0.41725552, 0.29342473, 0.28889030, -0.58864588, 0.34701362,
                            0.42389163, 0.28162596, -0.12972784, 0.37265050, 0.08255789,
                            -0.09949530, 0.36673626, 0.36744103, 0.33211154, 0.33214206,
                            0.19507907, 0.03592752, 0.35160932, 0.35001242, 0.32032481,
                            -0.20727010, -0.11362395, -0.13194911, 0.41250587, 0.42320812,
                            0.29844806, 0.28985339, 0.10554177, -0.00587574, 0.35386574,
                            0.35319650, 0.55448818, 0.15291481, -0.19969003, -0.35993519,
                            0.29050288, 0.28959396, 0.41448954, 0.41392374, -0.12040639,
                            -0.30311960, 0.35375032, 0.35277450],
                           [-0.16890033, 0.09132323, 0.36383507, 0.32687274, 0.40145576,
                            0.40328500, 0.30427882, 0.30017641, -0.61432797, 0.36144501,
                            0.42256466, 0.28260523, 0.12719125, 0.59734005, 0.53708172,
                            -0.57088166, 0.29141295, 0.28852940, 0.41864398, 0.42291117,
                            -0.60810453, -0.40527865, 0.40775761, 0.29062709, 0.68221945,
                            -0.05159308, 0.24131249, -0.39478955, 0.40739870, 0.41463229,
                            0.29969510, 0.29482204, 0.63608932, -0.24143942, 0.29797107,
                            0.40838188, 0.55214882, 0.03277899, -0.39438787, 0.33905768,
                            0.37543702, 0.37882078, 0.33214301, 0.32873160, -0.39521536,
                            0.22783215, 0.33246726, 0.37022114]])

    assert torch.allclose(w, w_test, rtol=1e-4)
