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
    w_test = torch.tensor([-0.192002, 0.24389659, -0.00820864, 0.33209562, -0.3692264,
                           -0.01977555, -0.07253735, 0.43182647, 0.17043342])

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
    assert np.allclose(loss_file[-1], 0.6281029582023621)


def test_prioritized_dqn():

    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    approximator = learn(DQN, params).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([-0.1997187, 0.23901853, -0.02347298, 0.32159477, -0.35603976,
                           -0.00871724, -0.06996532, 0.4368977, 0.16555829])

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
    w_test = torch.tensor([-0.19200195, 0.24389662, -0.00819397, 0.33209211, -0.36922678,
                           -0.01977558, -0.07253741, 0.43180698, 0.17043276])

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
    w_test = torch.tensor([-0.19113483, 0.24442753, -0.00193313, 0.33645973, -0.36852428,
                           -0.02368798, -0.06728531, 0.43250641, 0.17080002])

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
    w_test = torch.tensor([-0.14844167, 0.29000255, 0.03975032, 0.38547155, -0.41255197,
                           -0.06506653, -0.1593183, 0.34374532, 0.08268036])

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
    w_test = torch.tensor([-0.29178128, 0.55463171, 0.22557013, -0.08695056, -0.73201102,
                           0.50301647, -0.18293807, -0.53910583, -0.33002707, 0.07241649,
                           0.63779509, 0.65126824, 0.02539234, 0.04835916, 0.68402201,
                           -1.33267581, 0.90949267, -0.35305756])

    assert torch.allclose(w, w_test)


def test_dueling_dqn_max_advantage():
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=5000, target_update_frequency=50,
                  avg_advantage=False)
    approximator = learn(DuelingDQN, params).approximator

    w = approximator.get_weights()
    w_test = torch.tensor([-0.27162558, 0.55981463, 0.22016869, -0.09739858, -0.71308094,
                           0.50259709, -0.17256802, -0.52853751, -0.32069674, 0.03300169,
                           0.60702407, 0.6589216, 0.03202675, 0.00282796, 0.68064702,
                           -1.34077501, 0.89765507, -0.3596119])

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


def test_categorical_dqn_prioritized():
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent = learn(CategoricalDQN, params)

    w = agent.approximator.get_weights()
    w_test = torch.tensor([0.99492681, 0.30372638, -0.38039818, -0.6622963, -0.70050967, 0.45887721,
                           -0.12906331, -0.45170408, -0.46869081, -0.01793555, 0.13370971, 0.09729694,
                           0.66298205, 0.59981567, -1.13012791, 0.82512254, 0.02841161, 0.0712545])

    assert torch.allclose(w, w_test, rtol=1e-4)

    leaves = agent._replay_memory._tree._tree[-agent._replay_memory._max_size:]
    assert len(np.unique(leaves)) == 492


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


def test_quantile_dqn_prioritized():
    replay_memory = {"class": PrioritizedReplayMemory,
                     "params": dict(alpha=.6, beta=LinearParameter(.4, threshold_value=1, n=500 // 5))}
    params = dict(batch_size=50, initial_replay_size=100,
                  max_replay_size=500, target_update_frequency=50,
                  replay_memory=replay_memory)
    agent = learn(QuantileDQN, params)

    w = agent.approximator.get_weights()
    w_test = torch.tensor([-0.32953724, 0.54472113, 0.2416994, -0.07950477, -0.61509699,
                           0.35881355, -1.12328517, 0.69162154, 0.36370972, 1.23972535,
                           -0.11985977, 0.01031534, 0.73459566, -1.04432285, -1.09142709,
                           -0.51700956, -0.19440261, -0.34549186, 0.99952668, -0.13364422,
                           0.45050237, -0.49673495, 0.46826643, 0.46266884])

    assert torch.allclose(w, w_test, rtol=1e-4)

    leaves = agent._replay_memory._tree._tree[-agent._replay_memory._max_size:]
    assert len(np.unique(leaves)) == 485


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
    w_test = torch.tensor([-0.30684721, 0.56686342, 0.14087328, -0.17538731, -0.60078925,
                           0.46820533, 0.13880453, 0.09881935, 0.18416016, -0.19479546,
                           0.00113559, -0.12921767, 0.30625999, 0.40629369, 0.28734064,
                           0.43081486, 0.39919993, 0.28532007, 0.16328686, 0.07399006,
                           0.54874563, 0.41084078, 0.429299, 0.28834155])

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
    w_test = torch.tensor([[-0.30898598, 0.56966251, 0.14830196, -0.16330893, -0.61583519,
                            0.41710711, 0.14109634, 0.09312337, 0.18974943, -0.19954745,
                            0.00823867, -0.13557191, 0.30167395, 0.40422878, 0.29510283,
                            0.41567451, 0.4070498, 0.29075122, 0.15064237, 0.07357291,
                            0.54149288, 0.39983037, 0.42101696, 0.28405854],
                           [0.06439685, 0.76319039, 0.69043452, -1.15463042, -0.24546584,
                            -0.166899, 0.3968434, 0.2026937, 0.71823609, 0.42156425,
                            -0.66523796, -0.64652872, 0.31756073, 0.33784986, 0.3988432,
                            0.30376759, 0.33161837, 0.36747232, -0.27429023, 0.59443343,
                            -0.16265249, 0.41489264, 0.31877431, 0.31075183]])

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
