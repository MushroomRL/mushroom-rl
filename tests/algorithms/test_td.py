import numpy as np

import torch

import pytest

from datetime import datetime
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


def test_q_learning():
    pi, mdp, _ = initialize()
    agent = QLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.82981873, 8.1436511, 7.90657196, 8.80224609],
                       [8.63085938, 9.921875, 7.09541016, 7.5234375],
                       [5.05047656, 7.59268474, 7.09104243, 9.84375],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_q_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = QLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_double_q_learning():
    pi, mdp, _ = initialize()
    agent = DoubleQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q_0 = np.array([[1.771875, 1.27575, 2.53125, 2.925],
                         [3.375, 5., 3.19570313, 3.54375],
                         [2.9109375, 1.28144531, 1.02515625, 5.],
                         [0., 0., 0., 0.]])
    test_q_1 = np.array([[2.61984375, 0., 2.3540625, 4.21875],
                         [3.41015625, 8.75, 1.8225, 3.375],
                         [2.075625, 0., 0., 5.],
                         [0., 0., 0., 0.]])

    assert np.allclose(agent.Q[0].table, test_q_0)
    assert np.allclose(agent.Q[1].table, test_q_1)


def test_double_q_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = DoubleQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_weighted_q_learning():
    pi, mdp, _ = initialize()
    agent = WeightedQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[6.56851727, 8.58758824, 7.36142507, 6.661079],
                       [6.40888819, 9.921875, 2.35106874, 5.68743581],
                       [7.53484628, 6.67026108, 8.28982735, 9.6875],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_weighted_q_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = WeightedQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_maxmin_q_learning():
    pi, mdp, _ = initialize()
    agent = MaxminQLearning(mdp.info, pi, Parameter(.5), n_tables=4)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=500, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[6.94511719, 8.578125, 7.64850311, 8.4123008],
                       [7.875, 9.9609375, 7.55222168, 7.453125],
                       [6.35976563, 5.90625, 8.859375, 9.98046875],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q[0].table, test_q)


def test_maxmin_q_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = MaxminQLearning(mdp.info, pi, Parameter(.5), n_tables=5)

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_speedy_q_learning():
    pi, mdp, _ = initialize()
    agent = SpeedyQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.82981873, 8.1436511, 7.90657196, 8.80224609],
                       [8.63085938, 9.921875, 7.09541016, 7.5234375],
                       [5.05047656, 7.59268474, 7.09104243, 9.84375],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_speedy_q_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = SpeedyQLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_sarsa():
    pi, mdp, _ = initialize()
    agent = SARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.27595873e-3, 6.46813482e-1, 5.34533718e-2, 5.30549419e-1],
                       [4.16370330e-1, 4.68559, 2.17458448e-2, 2.13375103e-2],
                       [8.01728098e-2, 1.21547824e-2, 8.10008084e-2, 3.439],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_sarsa_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = SARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_q_lambda():
    pi, mdp, _ = initialize()
    agent = QLambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[4.59011145, 4.45577162, 4.477737, 5.42726693],
                       [3.69380053, 5.217031, 1.81896314, 2.23973815],
                       [2.88362079, 3.3933483, 3.45751059, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_q_lambda_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = QLambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_sarsa_lambda_discrete():
    pi, mdp, _ = initialize()
    agent = SARSALambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[8.00084239e-1, 2.78998556, 8.49461088e-1, 3.08124553],
                       [2.04980969, 4.68559, 9.58512559e-1, 8.28891198e-1],
                       [2.12515718, 1.19486219, 1.88503624, 3.439],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_sarsa_lambda_discrete_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = SARSALambda(mdp.info, pi, Parameter(.1), .9)

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_sarsa_lambda_continuous_linear():
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

    test_w = np.array([-82.31759493, 0., -82.67048958, 0., -107.74658538,
                       0., -105.56482617, 0., -72.24653201, 0.,
                       -73.05283658, 0., -116.89230496, 0., -106.48877521,
                       0., -99.50640198, 0., -92.73162587, 0.])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_sarsa_lambda_continuous_linear_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

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
    agent_save = SARSALambdaContinuous(mdp_continuous.info, pi, LinearApproximator, Parameter(.1), .9,
                                       approximator_params=approximator_params)

    core = Core(agent_save, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_sarsa_lambda_continuous_nn():
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

    test_w = np.array([-43.621815, -79.27821, -45.31944, -89.00731, -31.043007,
                       -77.56836, -47.811436, -88.486305, -42.081253, -82.90734,
                       -158.01721, -178.69487, -151.41151, -185.12704, -167.8663])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_sarsa_lambda_continuous_nn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, _, mdp_continuous = initialize()

    approximator_params = dict(
        input_shape=mdp_continuous.info.observation_space.shape,
        output_shape=(mdp_continuous.info.action_space.n,),
        network=QNetwork,
        n_features=None,
        n_layers=0,
        n_actions=mdp_continuous.info.action_space.n
    )
    agent_save = SARSALambdaContinuous(mdp_continuous.info, pi, NumpyTorchApproximator, Parameter(.1), .9,
                                       approximator_params=approximator_params)

    core = Core(agent_save, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_expected_sarsa():
    pi, mdp, _ = initialize()
    agent = ExpectedSARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[9.14381458e-2, 3.21586373e-1, 1.25796943e-1, 5.84265739e-1],
                       [4.43277909e-1, 5.217031, 3.79675179e-2, 2.43461065e-1],
                       [1.07048288e-2, 2.92411568e-1, 1.87598670e-1, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_expected_sarsa_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = ExpectedSARSA(mdp.info, pi, Parameter(.1))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_true_online_sarsa_lambda():
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

    test_w = np.array([-75.40322828, 0., -82.05694011, 0., -102.60400109,
                       0., -104.14404304, 0., -67.59137525, 0.,
                       -72.77565331, 0., -111.60368847, 0., -108.15358127,
                       0., -95.09502145, 0., -93.86466772, 0.])

    print(agent.Q.get_weights())

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_true_online_sarsa_lambda_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

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
    agent_save = TrueOnlineSARSALambda(mdp_continuous.info, pi, Parameter(.1), .9,
                                       approximator_params=approximator_params)

    core = Core(agent_save, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_r_learning():
    pi, mdp, _ = initialize()
    agent = RLearning(mdp.info, pi, Parameter(.1), Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[-6.83038063, -4.28087029, -7.42147162, -3.21661291],
                       [-2.63817524, 2.41451035, -2.96275568, -9.77443411e-1],
                       [-3.43487698, -3.70095282, -2.7904181, 1.64107139],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_r_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()
    agent_save = RLearning(mdp.info, pi, Parameter(.1), Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)


def test_rq_learning():
    pi, mdp, _ = initialize()

    agent = RQLearningOnPolicy(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.92803867e-1, 2.64425388, 1.58659331, 1.03653182],
                       [2.46563831, 5.217031, 4.77272892e-1, 1.11994842],
                       [1.19424375e-1, 1.51291318, 7.77071865e-1, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearningOnPolicy(mdp.info, pi, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[3.93802071e-2, 1.99166837e-1, 5.11401164e-3, 1.86700624e-1],
                       [3.65512500e-3, 4.0951, 4.53973748e-3, 1.94970864e-1],
                       [5.30271828e-3, 7.19718797e-3, 3.29102499e-1, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[3.20876697, 2.71184803, 3.2068956, 3.89599706],
                       [2.801295, 5.217031, 2.6313255, 3.6840555],
                       [2.77349948, 0., 2.47646238, 3.439],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[3.44856016e-1, 8.63331136e-1, 2.70012766e-1, 1.53378035],
                       [1.2952935, 5.217031, 2.13557880e-1, 1.70021182],
                       [9.70524996e-2, 3.77295750e-1, 3.53847094e-1, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


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


def test_rq_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()

    agent_save = RQLearningOnPolicy(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)
