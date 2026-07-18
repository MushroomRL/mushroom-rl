from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mushroom_rl.algorithms.value import DoubleSORQLearning, DSORDQN
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import CartPole, GridWorld
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter, LinearParameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric.networks import QNetwork


class DummyApproximator:
    def __init__(self, values):
        self._values = values

    def predict(self, state, action=None, **kwargs):
        values = self._values[state.squeeze(1).long()]
        if action is not None:
            values = values.gather(1, action.long()).squeeze(1)
        return values


def test_dsor_dqn_target():
    agent = DSORDQN.__new__(DSORDQN)
    agent.approximator = DummyApproximator(torch.tensor([
        [1., 3.], [5., 2.], [7., 4.], [6., 9.]
    ]))
    agent.target_approximator = DummyApproximator(torch.tensor([
        [10., 20.], [30., 40.], [50., 60.], [70., 80.]
    ]))
    agent._predict_params = dict()
    agent._relaxation_factor = 1.2
    agent.mdp_info = SimpleNamespace(gamma=.9)

    state = torch.tensor([[0], [1]])
    next_state = torch.tensor([[2], [3]])
    reward = torch.tensor([2., 4.])
    absorbing = torch.tensor([False, True])

    target = agent._compute_target(state, reward, next_state, absorbing)

    assert torch.allclose(target, torch.tensor([52.4, -1.2]))


def test_dsor_dqn_with_unit_relaxation_is_double_dqn_target():
    agent = DSORDQN.__new__(DSORDQN)
    agent.approximator = DummyApproximator(torch.tensor([
        [1., 3.], [7., 4.]
    ]))
    agent.target_approximator = DummyApproximator(torch.tensor([
        [10., 20.], [50., 60.]
    ]))
    agent._predict_params = dict()
    agent._relaxation_factor = 1.
    agent.mdp_info = SimpleNamespace(gamma=.9)

    target = agent._compute_target(
        torch.tensor([[0]]), torch.tensor([2.]), torch.tensor([[1]]),
        torch.tensor([False]))

    assert torch.allclose(target, torch.tensor([47.]))


def test_double_sor_q_learning_update(monkeypatch):
    mdp = GridWorld(2, 2, start=(0, 0), goal=(1, 1))
    policy = EpsGreedy(Parameter(1.))
    agent = DoubleSORQLearning(
        mdp.info, policy, Parameter(.5), relaxation_factor=1.2)

    agent.Q[0].table[0] = [1., 4., 0., 0.]
    agent.Q[0].table[1] = [5., 3., 0., 0.]
    agent.Q[1].table[0] = [10., 20., 0., 0.]
    agent.Q[1].table[1] = [7., 8., 0., 0.]
    monkeypatch.setattr(np.random, 'uniform', lambda: 0.)

    agent._update(
        np.array([0]), np.array([1]), 2., np.array([1]), False)

    assert agent.Q[0][np.array([0]), np.array([1])] == pytest.approx(4.98)


def _make_dsor_dqn(replay_memory=None):
    mdp = CartPole()
    policy = EpsGreedy(Parameter(1.), backend='torch')
    approximator_params = dict(
        network=QNetwork,
        optimizer={'class': torch.optim.Adam, 'params': {'lr': .001}},
        loss=torch.nn.functional.smooth_l1_loss,
        input_shape=mdp.info.observation_space.shape,
        output_shape=mdp.info.action_space.size,
        n_actions=mdp.info.action_space.n,
        n_features=8,
        n_layers=1
    )
    agent = DSORDQN(
        mdp.info, policy, TorchApproximator,
        approximator_params=approximator_params,
        relaxation_factor=1.1,
        batch_size=2,
        initial_replay_size=2,
        max_replay_size=10,
        target_update_frequency=2,
        replay_memory=replay_memory
    )
    return agent, mdp


def test_dsor_dqn_prioritized_replay():
    replay_memory = {
        'class': PrioritizedReplayMemory,
        'params': dict(alpha=.6, beta=LinearParameter(.4, 1., 10))
    }
    agent, mdp = _make_dsor_dqn(replay_memory)

    Core(agent, mdp).learn(n_steps=4, n_steps_per_fit=1, quiet=True)

    assert agent._n_updates == 4


def test_dsor_dqn_save(tmpdir):
    agent, _ = _make_dsor_dqn()
    path = tmpdir / 'dsor_dqn.msh'

    agent.save(path, full_save=True)
    loaded_agent = Agent.load(path)

    assert loaded_agent._relaxation_factor == 1.1
    assert isinstance(loaded_agent, DSORDQN)


@pytest.mark.parametrize('algorithm', [DSORDQN, DoubleSORQLearning])
def test_non_positive_relaxation_factor(algorithm):
    with pytest.raises(ValueError, match='relaxation factor must be positive'):
        algorithm(None, None, None, relaxation_factor=0.)
