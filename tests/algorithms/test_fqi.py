import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.value import BoostedFQI, DoubleFQI, FQI
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


def learn(alg, alg_params):
    mdp = CarOnHill()
    np.random.seed(1)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               n_actions=mdp.info.action_space.n,
                               n_models=1 if alg is not BoostedFQI else alg_params['n_iterations'],
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)
    approximator = ExtraTreesRegressor

    # Agent
    agent = alg(mdp.info, pi, approximator, approximator_params=approximator_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=5, n_episodes_per_fit=5)

    test_epsilon = Parameter(0.75)
    agent.policy.set_epsilon(test_epsilon)
    dataset = core.evaluate(n_episodes=2)

    return agent, np.mean(dataset.compute_J(mdp.info.gamma))


def test_fqi():
    params = dict(n_iterations=10)
    _, j = learn(FQI, params)
    j_test = -0.06763797713952796

    assert j == j_test


def test_fqi_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(n_iterations=10)
    agent_save, _ = learn(FQI, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_fqi_boosted():
    params = dict(n_iterations=10)
    _, j = learn(BoostedFQI, params)
    j_test = -0.04487241596542538

    assert j == j_test


def test_fqi_boosted_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(n_iterations=10)
    agent_save, _ = learn(BoostedFQI, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)


def test_double_fqi():
    params = dict(n_iterations=10)
    _, j = learn(DoubleFQI, params)
    j_test = -0.19933233708925654

    assert j == j_test


def test_double_fqi_save(tmpdir):
    agent_path = tmpdir / 'agent_{}'.format(datetime.now().strftime("%H%M%S%f"))

    params = dict(n_iterations=10)
    agent_save, _ = learn(DoubleFQI, params)

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    for att, method in vars(agent_save).items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)

        tu.assert_eq(save_attr, load_attr)
