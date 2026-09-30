import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.core import Agent
from mushroom_rl.algorithms.value import BoostedFQI, DoubleFQI, FQI
from mushroom_rl.core import Core
from mushroom_rl.environments import CarOnHill
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import Parameter


class FlattenExtraTrees:
    def __init__(self, input_shape, output_shape, **params):
        self._models = [ExtraTreesRegressor(**params) for _ in range(output_shape[0])]
        self.fit_state = None

    def fit(self, x, a, q):
        self.fit_state = x
        x = x.reshape(len(x), -1)
        for i, m in enumerate(self._models):
            mask = a[:, 0] == i
            m.fit(x[mask], q[mask])

    def predict(self, x):
        x = x.reshape(len(x), -1)
        return np.stack([m.predict(x) for m in self._models], axis=-1)


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


def test_double_fqi_history():
    np.random.seed(1)

    mdp = CarOnHill()
    pi = EpsGreedy(epsilon=Parameter(1.))
    n_actions = mdp.info.action_space.n
    approximator_params = dict(input_shape=(2,) + mdp.info.observation_space.shape, output_shape=(n_actions,),
                               n_actions=n_actions, n_estimators=10, min_samples_split=5, min_samples_leaf=2)
    agent = DoubleFQI(mdp.info, pi, FlattenExtraTrees, approximator_params=approximator_params, n_iterations=3,
                      quiet=True, history_length=2)

    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=4, quiet=True)
    agent.fit(dataset)

    state = dataset.state
    last = dataset.last
    previous_state = np.zeros_like(state)
    previous_state[1:] = state[:-1]
    previous_state[1:][last[:-1]] = 0.
    windows = np.stack([previous_state, state], axis=1)

    half = len(dataset) // 2
    assert not last[half - 1]
    for i in range(2):
        assert np.array_equal(agent.approximator[i].model.fit_state, windows[i * half:(i + 1) * half])

    q_0 = agent.approximator.predict(windows[:2], idx=0)
    q_1 = agent.approximator.predict(windows[:2], idx=1)
    assert np.allclose(q_0, np.array([[-7.990885416666666e-05, -0.00026949652777777774],
                                      [-0.0012900318287037036, 0.0]]))
    assert np.allclose(q_1, np.array([[-0.007060182291666666, -0.0020644687500000003],
                                      [-0.009422225347222223, -0.0008288585069444445]]))

    q_single = agent.approximator.predict(windows[1])
    assert q_single.shape == (n_actions,)
    assert np.allclose(q_single, (q_0[1] + q_1[1]) / 2)

    pi.set_epsilon(Parameter(0.))
    dataset = core.evaluate(n_episodes=2, quiet=True)
    assert np.all((dataset.action >= 0) & (dataset.action < n_actions))
