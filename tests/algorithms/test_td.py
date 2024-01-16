import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.value import *
from mushroom_rl.approximators.parametric import LinearApproximator, NumpyTorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import GridWorld, PuddleWorld
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.policy.td_policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def assert_properly_loaded(agent_save, agent_load):
    for att, method in vars(agent_save).items():
        if att != 'next_action':
            save_attr = getattr(agent_save, att)
            load_attr = getattr(agent_load, att)
            tu.assert_eq(save_attr, load_attr)


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


def initialize():
    np.random.seed(1)
    torch.manual_seed(1)
    return EpsGreedy(Parameter(1)), GridWorld(2, 2, start=(0, 0), goal=(1, 1)),\
           PuddleWorld(horizon=1000)


def test_q_learning():
    pi, mdp, _ = initialize()
    agent = QLearning(mdp.info, pi, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.82042542, 8.40151978, 7.64961548, 8.82421875],
                       [8.77587891, 9.921875, 7.29316406, 8.68359375],
                       [7.7203125, 7.69921875, 4.5, 9.84375],
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

    test_q_0 = np.array([[2.6578125, 6.94757812, 3.73359375, 7.171875],
                         [2.25, 7.5, 3.0375, 3.375],
                         [3.0375, 5.4140625, 2.08265625, 8.75],
                         [0., 0., 0., 0.]])
    test_q_1 = np.array([[2.72109375, 4.5, 4.36640625, 6.609375],
                         [4.5, 9.375, 4.49296875, 4.5],
                         [1.0125, 5.0625, 5.625, 8.75],
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

    test_q = np.array([[8.00815525, 4.09343205, 7.94406811, 8.96270031],
                       [8.31597686, 9.99023438, 6.42921521, 7.70471909],
                       [7.26069091, 0.87610663, 3.70440836, 9.6875],
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
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0., 0., 0., 0.],
                       [0., 7.5, 0., 0.],
                       [0., 0., 0., 5.],
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

    test_q = np.array([[7.82042542, 8.40151978, 7.64961548, 8.82421875],
                       [8.77587891, 9.921875, 7.29316406, 8.68359375],
                       [7.7203125, 7.69921875, 4.5, 9.84375],
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

    test_q = np.array([[4.31368701e-2, 3.68037689e-1, 4.14040445e-2, 1.64007642e-1],
                       [6.45491436e-1, 4.68559000, 8.07603735e-2, 1.67297938e-1],
                       [4.21445838e-2, 3.71538042e-3, 0., 3.439],
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

    test_q = np.array([[5.07310744, 5.6013244, 3.42130445, 5.90556511],
                       [3.4410511, 5.217031, 2.51555213, 4.0616156],
                       [3.76728025, 2.17726915, 1.0955066, 4.68559],
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

    test_q = np.array([[1.88093529, 2.42467354, 1.07390687, 2.39288988],
                       [2.46058746, 4.68559, 1.5661933, 2.56586018],
                       [1.24808966, 0.91948465, 0.47734152, 3.439],
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
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
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
                        0., -99.50640198,  0., -92.73162587, 0.])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_sarsa_lambda_continuous_linear_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, _, mdp_continuous = initialize()

    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
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
        network=Network,
        n_actions=mdp_continuous.info.action_space.n,
    )
    agent = SARSALambdaContinuous(mdp_continuous.info, pi, NumpyTorchApproximator, Parameter(.1), .9,
                                  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.array([-1.8319136, -1.5731438, -2.9321826, -3.5566466, -1.282013,
                       -2.7563045, -0.790771, -0.2194604, -0.5647575, -0.4195656,
                       -4.46288, -11.742587, -5.3095326, -0.27556023, -0.05155428])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_sarsa_lambda_continuous_nn_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, _, mdp_continuous = initialize()

    approximator_params = dict(
        input_shape=mdp_continuous.info.observation_space.shape,
        output_shape=(mdp_continuous.info.action_space.n,),
        network=Network,
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

    test_q = np.array([[0.10221208, 0.48411449, 0.07688765, 0.64002317],
                       [0.58525881, 5.217031, 0.06047094, 0.48214145],
                       [0.08478224, 0.28873536, 0.06543094, 4.68559],
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
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
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
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
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

    test_q = np.array([[-6.19137991, -3.9368055, -5.11544257, -3.43673781],
                       [-2.52319391, 1.92201829, -2.77602918, -2.45972955],
                       [-5.38824415, -2.43019918, -1.09965936, 2.04202511],
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

    agent = RQLearning(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0.32411217, 2.9698436, 0.46474438, 1.10269504],
                       [2.99505139, 5.217031, 0.40933461, 0.37687883],
                       [0.41942675, 0.32363486, 0., 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[1.04081115e-2, 5.14662188e-1, 1.73951634e-2, 1.24081875e-01],
                       [0., 2.71, 1.73137500e-4, 4.10062500e-6],
                       [0., 4.50000000e-2, 0., 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), off_policy=True,
                       beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[3.55204022, 4.54235939, 3.42601165, 2.95170908],
                       [2.73877031, 3.439, 2.42031528, 2.86634531],
                       [3.43274708, 3.8592342, 3.72637395, 5.217031],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(mdp.info, pi, Parameter(.1), off_policy=True,
                       delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0.18947806, 1.57782254, 0.21911489, 1.05197011],
                       [0.82309759, 5.217031, 0.04167492, 0.61472604],
                       [0.23620541, 0.59828262, 1.25299991, 5.217031],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_rq_learning_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    pi, mdp, _ = initialize()

    agent_save = RQLearning(mdp.info, pi, Parameter(.1), beta=Parameter(.5))

    core = Core(agent_save, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    assert_properly_loaded(agent_save, agent_load)
