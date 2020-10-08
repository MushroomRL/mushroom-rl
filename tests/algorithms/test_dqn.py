import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.algorithms.value import DQN, DoubleDQN, AveragedDQN, CategoricalDQN
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
    w_test = np.array([-0.15894288, 0.47257397, 0.05482405, 0.5442066,
                       -0.56469935, -0.07374532, -0.0706185, 0.40790945,
                       0.12486243])

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
    w_test = np.array([-0.1384063, 0.48957556, 0.02254359, 0.50994426,
                       -0.56277484, -0.075808, -0.06829552, 0.3642576,
                       0.15519235])

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
    w_test = np.array([-0.15894286, 0.47257394, 0.05482561, 0.54420704,
                       -0.5646987, -0.07374918, -0.07061853, 0.40789905,
                       0.12482855])

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
    w_test = np.array([-0.15889995, 0.47253257, 0.05424322, 0.5434766,
                       -0.56529117, -0.0743931, -0.07070775, 0.4055584,
                       0.12588869])

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


def test_categorical_dqn():
    params = dict(batch_size=50, initial_replay_size=50,
                  max_replay_size=5000, target_update_frequency=50)
    approximator = learn(CategoricalDQN, params).approximator

    w = approximator.get_weights()
    w_test = np.array([1.0035713, 0.30592525, -0.38904265, -0.66449565,
                       -0.71816885, 0.47653696, -0.12593754, -0.44365975,
                       -0.47181657, -0.02598009, 0.11935875, 0.11164782,
                       0.659329, 0.5941985, -1.1264751, 0.8307397, 0.01681535,
                       0.08285073])

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
