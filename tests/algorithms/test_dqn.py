import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN, MaxminDQN, CategoricalDQN
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory


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


def learn(alg, alg_params):
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
    approximator_params = dict(network=Network if alg is not CategoricalDQN else FeatureNetwork,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               n_features=2, use_cuda=False)

    # Agent
    if alg is not CategoricalDQN:
        agent = alg(mdp.info, pi, TorchApproximator,
                    approximator_params=approximator_params, **alg_params)
    else:
        agent = alg(mdp.info, pi, approximator_params=approximator_params,
                    n_atoms=2, v_min=-1, v_max=1, **alg_params)

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


def test_prioritized_dqn():
    replay_memory = PrioritizedReplayMemory(
        50, 500, alpha=.6,
        beta=LinearParameter(.4, threshold_value=1, n=500 // 5)
    )
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

    replay_memory = PrioritizedReplayMemory(
        50, 500, alpha=.6,
        beta=LinearParameter(.4, threshold_value=1, n=500 // 5)
    )
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


def test_categorical_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([0.98373884, 0.2899274, -0.36921054, -0.6484974, -0.74544126,
                       0.5038091, -0.11945444, -0.4370291, -0.47829974, -0.03261064,
                       0.11510377, 0.1159029, 0.68508214, 0.6178692, -1.1522279,
                       0.807069, 0.02732106, 0.07234504])

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
