import numpy as np

import torch

import pytest

from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent, HasNextAction
from mushroom_rl.algorithms.value import (QLearning, QLambda, DoubleQLearning, WeightedQLearning, MaxminQLearning,
                                          SpeedyQLearning, RLearning, RQLearning, RQLearningOnPolicy, SARSA,
                                          SARSALambda, SARSALambdaContinuous, ExpectedSARSA, TrueOnlineSARSALambda)
from mushroom_rl.algorithms.value.td.td import TD
from mushroom_rl.approximators.parametric import LinearApproximator, NumpyTorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import GridWorld, PuddleWorld
from mushroom_rl.approximators.parametric.networks import QNetwork
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy.td_policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def assert_properly_loaded(agent_save, agent_load):
    for att in vars(agent_save):
        method = agent_save._save_attributes.get(att, '')
        if att != 'next_action' and method != 'none':
            save_attr = getattr(agent_save, att)
            load_attr = getattr(agent_load, att)
            tu.assert_eq(save_attr, load_attr)


def initialize():
    np.random.seed(1)
    torch.manual_seed(1)
    return (EpsGreedy(Parameter(1)), GridWorld.from_size(2, 2, start=(0, 0), goal=(1, 1), goal_reward=10.),
            PuddleWorld(horizon=1000))


def test_q_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = QLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_q_learning.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_double_q_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = DoubleQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q_0 = np.load('tests/algorithms/data/test_double_q_learning_q_0.npy')
    test_q_1 = np.load('tests/algorithms/data/test_double_q_learning_q_1.npy')

    assert np.allclose(agent.Q[0].table, test_q_0)
    assert np.allclose(agent.Q[1].table, test_q_1)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_weighted_q_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = WeightedQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_weighted_q_learning.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_maxmin_q_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = MaxminQLearning(mdp.info, pi, Parameter(.5), n_tables=4)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=500, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_maxmin_q_learning.npy')

    assert np.allclose(agent.Q[0].table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_speedy_q_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = SpeedyQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_speedy_q_learning.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_sarsa(tmpdir):
    pi, mdp, _ = initialize()
    agent = SARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_sarsa.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_q_lambda(tmpdir):
    pi, mdp, _ = initialize()
    agent = QLambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_q_lambda.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_sarsa_lambda_discrete(tmpdir):
    pi, mdp, _ = initialize()
    agent = SARSALambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_sarsa_lambda_discrete.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_sarsa_lambda_continuous_linear(tmpdir):
    pi, _, mdp_continuous = initialize()

    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings)

    approximator_params = dict(
        input_shape=mdp_continuous.info.observation_space.shape,
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n,
        phi=features
    )
    agent = SARSALambdaContinuous(mdp_continuous.info, pi, LinearApproximator,
                                  Parameter(.1), .9,  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.load('tests/algorithms/data/test_sarsa_lambda_continuous_linear.npy')

    assert np.allclose(agent.Q.get_weights(), test_w)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_sarsa_lambda_continuous_nn(tmpdir):
    pi, _, mdp_continuous = initialize()

    approximator_params = dict(
        input_shape=mdp_continuous.info.observation_space.shape,
        output_shape=(mdp_continuous.info.action_space.n,),
        network=QNetwork,
        n_features=None,
        n_layers=0,
        n_actions=mdp_continuous.info.action_space.n,
    )
    agent = SARSALambdaContinuous(mdp_continuous.info, pi, NumpyTorchApproximator, Parameter(.1), .9,
                                  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.load('tests/algorithms/data/test_sarsa_lambda_continuous_nn.npy')

    assert np.allclose(agent.Q.get_weights(), test_w)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_expected_sarsa(tmpdir):
    pi, mdp, _ = initialize()
    agent = ExpectedSARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_expected_sarsa.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_true_online_sarsa_lambda(tmpdir):
    pi, _, mdp_continuous = initialize()

    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings)

    approximator_params = dict(
        input_shape=mdp_continuous.info.observation_space.shape,
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n,
        phi=features,
    )
    agent = TrueOnlineSARSALambda(mdp_continuous.info, pi, Parameter(.1), .9,
                                  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.load('tests/algorithms/data/test_true_online_sarsa_lambda.npy')

    assert np.allclose(agent.Q.get_weights(), test_w)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_r_learning(tmpdir):
    pi, mdp, _ = initialize()
    agent = RLearning(mdp.info, pi, Parameter(.1), Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_r_learning.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_rq_learning(tmpdir):
    pi, mdp, _ = initialize()

    agent = RQLearningOnPolicy(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_rq_learning_q_0.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent_on_policy.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent_on_policy.msh'))

    agent = RQLearningOnPolicy(mdp.info, pi, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_rq_learning_q_1.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_rq_learning_q_2.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.load('tests/algorithms/data/test_rq_learning_q_3.npy')

    assert np.allclose(agent.Q.table, test_q)

    agent.save(tmpdir / 'agent.msh')
    assert_properly_loaded(agent, Agent.load(tmpdir / 'agent.msh'))


def test_has_next_action_reset():
    pi, mdp, _ = initialize()

    agent = SARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent, mdp)

    core.learn(n_steps=50, n_steps_per_fit=1, quiet=True)

    assert agent._next_action is None

    agent._next_action = np.array([1])
    agent.episode_start(mdp.reset()[0], {})
    assert agent._next_action is None

    agent._next_action = np.array([1])
    agent.stop()
    assert agent._next_action is None

    agent._next_action = np.array([1])
    agent.episode_start_vectorized(np.zeros((4, 1)), {}, np.ones(4, dtype=bool))
    assert agent._next_action is None

    dataset = core.evaluate(n_episodes=1, quiet=True, greedy=True)

    assert len(dataset) > 0


def test_has_next_action_requires_agent():
    with pytest.raises(TypeError):
        class NotAnAgent(HasNextAction):
            pass


def test_has_next_action_requires_mixin_first():
    with pytest.raises(TypeError):
        class MixinLast(TD, HasNextAction):
            pass
