import numpy as np

from mushroom.algorithms.value import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.environments.grid_world import GridWorld
from mushroom.environments.gym_env import Gym
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy.td_policy import EpsGreedy
from mushroom.utils.parameters import Parameter

pi = EpsGreedy(Parameter(1))
mdp = GridWorld(2, 2, start=(0, 0), goal=(1, 1))
mdp_continuous = Gym(name='MountainCar-v0', horizon=np.inf, gamma=1.)


def test_q_learning():
    alg = QLearning(pi, mdp.info, Parameter(.5))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 53, 2, 3],
                       [7, 5, 6, 7],
                       [8, 9, -45, 11],
                       [12, 38, 14, 15]])

    assert np.array_equal(alg.Q.table, test_q)


def test_double_q_learning():
    np.random.seed(1)

    alg = DoubleQLearning(pi, mdp.info, Parameter(.5))

    alg.Q[0].table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)
    alg.Q[1].table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q_1 = np.array([[0, 53, 2, 3],
                         [4, 5, 6, 7],
                         [8, 9, -45, 11],
                         [12, 38, 14, 15]])

    test_q_2 = np.array([[0, 1, 2, 3],
                         [7, 5, 6, 7],
                         [8, 9, 10, 11],
                         [12, 13, 14, 15]])

    assert np.array_equal(alg.Q[0].table, test_q_1)
    assert np.array_equal(alg.Q[1].table, test_q_2)


def test_weighted_q_learning():
    alg = WeightedQLearning(pi, mdp.info, Parameter(.5))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 52, 2, 3],
                       [7, 5, 6, 7],
                       [8, 9, -45, 11],
                       [12, 37, 14, 15]])

    assert np.array_equal(alg.Q.table, test_q)


def test_speedy_q_learning():
    alg = SpeedyQLearning(pi, mdp.info, Parameter(.1))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 16, 2, 3],
                       [4, 5, 6, 7],
                       [8, 9, -1, 11],
                       [12, 18, 14, 15]])

    assert np.array_equal(alg.Q.table, test_q)


def test_sarsa():
    alg = SARSA(pi, mdp.info, Parameter(.1))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(mdp.info.size)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 11, 2, 3],
                       [4, 5, 6, 7],
                       [8, 9, -1, 11],
                       [12, 17, 14, 15]])

    assert np.array_equal(alg.Q.table, test_q)


def test_sarsa_lambda_discrete():
    alg = SARSALambdaDiscrete(pi, mdp.info, Parameter(.1), .9)

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 9.334405, 2, 3],
                       [1.4005, 5, 6, 7],
                       [8, 9, -1, 11],
                       [12, 9.05, 14, 15]])

    assert np.allclose(alg.Q.table, test_q)


def test_sarsa_lambda_continuous():
    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n
    )
    alg = SARSALambdaContinuous(LinearApproximator, pi, mdp_continuous.info,
                                Parameter(.1), .9, features=features,
                                approximator_params=approximator_params)

    s_1 = np.linspace(mdp_continuous.info.observation_space.low[0],
                      mdp_continuous.info.observation_space.high[0], 10)
    s_2 = np.linspace(mdp_continuous.info.observation_space.low[1],
                      mdp_continuous.info.observation_space.high[1], 10)
    for i in s_1:
        for j in s_2:
            alg._update(np.array([i, j]), np.array([1]), 100, np.array([0, 0]),
                        0)

    test_w = np.array([[0, 0, 0, 0],
                       [320.43, 399.8616, 340.397, 417.218],
                       [0, 0, 0, 0]])

    assert np.allclose(alg.Q.get_weights(), test_w.ravel())


def test_expected_sarsa():
    alg = ExpectedSARSA(pi, mdp.info, Parameter(.1))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 11.395, 2, 3],
                       [4.6, 5, 6, 7],
                       [8, 9, -1, 11],
                       [12, 17.915, 14, 15]])

    assert np.allclose(alg.Q.table, test_q)


def test_true_online_sarsa_lambda():
    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n
    )
    alg = TrueOnlineSARSALambda(pi, mdp_continuous.info,
                                Parameter(.1), .9, features=features,
                                approximator_params=approximator_params)

    s_1 = np.linspace(mdp_continuous.info.observation_space.low[0],
                      mdp_continuous.info.observation_space.high[0], 10)
    s_2 = np.linspace(mdp_continuous.info.observation_space.low[1],
                      mdp_continuous.info.observation_space.high[1], 10)
    for i in s_1:
        for j in s_2:
            alg._update(np.array([i, j]), np.array([1]), 100, np.array([0, 0]),
                        0)

    test_w = np.array([[0, 0, 0, 0],
                       [927.0283, 798.57594, 876.8018, 705.227],
                       [0, 0, 0, 0]])

    assert np.allclose(alg.Q.get_weights(), test_w.ravel())


def test_r_learning():
    alg = RLearning(pi, mdp.info, Parameter(.1), Parameter(.5))

    alg.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg._update(0, 1, 100, 1, 0)
    alg._update(1, 0, 10, 3, 1)
    alg._update(3, 1, 50, 3, 0)
    alg._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 11.6, 2, 3],
                       [-.17, 5, 6, 7],
                       [8, 9, -5.77, 11],
                       [12, 13.43, 14, 15]])

    assert np.allclose(alg.Q.table, test_q)


def test_rq_learning():
    alg_on_policy_beta = RQLearning(pi, mdp.info, Parameter(.1),
                                    beta=Parameter(.5))
    alg_on_policy_delta = RQLearning(pi, mdp.info, Parameter(.1),
                                     delta=Parameter(.5))
    alg_off_policy_beta = RQLearning(pi, mdp.info, Parameter(.1),
                                     off_policy=True, beta=Parameter(.5))
    alg_off_policy_delta = RQLearning(pi, mdp.info, Parameter(.1),
                                      off_policy=True, delta=Parameter(.5))

    alg_on_policy_beta.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg_on_policy_beta._update(0, 1, 100, 1, 0)
    alg_on_policy_beta._update(1, 0, 10, 3, 1)
    alg_on_policy_beta._update(3, 1, 50, 3, 0)
    alg_on_policy_beta._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 11.8, 2, 3],
                       [1, 5, 6, 7],
                       [8, 9, -10, 11],
                       [12, 10.4, 14, 15]])

    assert np.allclose(alg_on_policy_beta.Q.table, test_q)

    alg_on_policy_delta.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg_on_policy_delta._update(0, 1, 100, 1, 0)
    alg_on_policy_delta._update(1, 0, 10, 3, 1)
    alg_on_policy_delta._update(3, 1, 50, 3, 0)
    alg_on_policy_delta._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 10.27, 2, 3],
                       [1, 5, 6, 7],
                       [8, 9, -10, 11],
                       [12, 5.63, 14, 15]])

    assert np.allclose(alg_on_policy_delta.Q.table, test_q)

    alg_off_policy_beta.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg_off_policy_beta._update(0, 1, 100, 1, 0)
    alg_off_policy_beta._update(1, 0, 10, 3, 1)
    alg_off_policy_beta._update(3, 1, 50, 3, 0)
    alg_off_policy_beta._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 13.15, 2, 3],
                       [1, 5, 6, 7],
                       [8, 9, -10, 11],
                       [12, 11.75, 14, 15]])

    assert np.allclose(alg_off_policy_beta.Q.table, test_q)

    alg_off_policy_delta.Q.table = np.arange(np.prod(mdp.info.size)).reshape(
        mdp.info.size).astype(np.float)

    alg_off_policy_delta._update(0, 1, 100, 1, 0)
    alg_off_policy_delta._update(1, 0, 10, 3, 1)
    alg_off_policy_delta._update(3, 1, 50, 3, 0)
    alg_off_policy_delta._update(2, 2, -100, 3, 1)

    test_q = np.array([[0, 10.315, 2, 3],
                       [1, 5, 6, 7],
                       [8, 9, -10, 11],
                       [12, 5.675, 14, 15]])

    assert np.allclose(alg_off_policy_delta.Q.table, test_q)
