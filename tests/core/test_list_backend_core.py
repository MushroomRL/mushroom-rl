import numpy as np

from mushroom_rl.core import Agent, Core, Environment, VectorizedEnvironment, MDPInfo, Box
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, action):
        self._action = action
        super().__init__()

    def draw_action(self, state):
        return self._action


class ConstantAgent(Agent):
    def __init__(self, mdp_info, action, backend='numpy'):
        super().__init__(mdp_info, ConstantPolicy(action), backend=backend)

    def fit(self, dataset):
        pass


class RaggedChainEnv(Environment):
    def __init__(self, horizon=5):
        observation_space = Box(-np.inf, np.inf, shape=(1,))
        action_space = Box(-1.0, 1.0, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.9, horizon=horizon, backend='list')
        self._t = 0
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._t = 0
        return [0.0], {'v': 0}

    def step(self, action):
        self._t += 1
        absorbing = self._t >= 4
        return [float(i) for i in range(self._t + 1)], 1.0, absorbing, {'v': self._t}

    def render(self, record=False):
        pass

    def stop(self):
        pass


class ListVecEnv(VectorizedEnvironment):
    def __init__(self):
        n_envs = 3
        observation_space = Box(-np.inf, np.inf, shape=(2,))
        action_space = Box(-1.0, 1.0, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.9, horizon=10, backend='list')
        self._t = np.zeros(n_envs, dtype=int)
        super().__init__(mdp_info, n_envs)

    def _obs(self):
        return [np.array([float(t), float(t)]) for t in self._t]

    def reset_all(self, env_mask, state=None):
        self._t[np.asarray(env_mask)] = 0
        return self._obs(), [{'v': float(t)} for t in self._t]

    def step_all(self, env_mask, action):
        self._t[np.asarray(env_mask)] += 1
        reward = np.ones(self._n_envs)
        absorbing = (self._t >= 4) & np.asarray(env_mask)
        return self._obs(), reward, absorbing, [{'v': float(t)} for t in self._t]

    def render_all(self, env_mask, record=False):
        pass

    def stop(self):
        pass


class WeirdObjectEnv(Environment):
    def __init__(self):
        observation_space = Box(-np.inf, np.inf, shape=(1,))
        action_space = Box(-1.0, 1.0, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, gamma=0.9, horizon=5, backend='list')
        self._t = 0
        super().__init__(mdp_info)

    def _obs(self):
        return {'id': self._t, 'items': list(range(self._t + 1))}

    def reset(self, state=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        absorbing = self._t >= 3
        return self._obs(), 1.0, absorbing, {}

    def render(self, record=False):
        pass

    def stop(self):
        pass


class ObjectRecordingPolicy(Policy):
    def __init__(self):
        self.seen = []
        super().__init__()

    def draw_action(self, state):
        self.seen.append(state)
        return {'move': 'noop'}


class ObjectAgent(Agent):
    def __init__(self, mdp_info):
        super().__init__(mdp_info, ObjectRecordingPolicy(), backend='list')

    def fit(self, dataset):
        pass


def test_sequential_list_backend_ragged():
    mdp = RaggedChainEnv()
    agent = ConstantAgent(mdp.info, np.array([0.0]))
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=2, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert len(dataset) == 8
    assert dataset.state[0] == [0.0]
    assert dataset.state[3] == [0.0, 1.0, 2.0, 3.0]
    assert dataset.info['v'] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert dataset[:4].info['v'] == [1, 2, 3, 4]
    assert np.array_equal(dataset.compute_J(), np.array([4.0, 4.0]))


def test_vectorized_list_backend():
    mdp = ListVecEnv()
    agent = ConstantAgent(mdp.info, [np.array([0.0])] * 3, backend='list')
    core = Core(agent, mdp)

    dataset = core.evaluate(n_steps=30, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert len(dataset) == 30
    assert np.array_equal(np.array(dataset.state[0]), np.array([0.0, 0.0]))
    assert np.array_equal(np.array(dataset.state[3]), np.array([3.0, 3.0]))
    assert np.array_equal(dataset.compute_J(), np.array([4., 4., 2., 4., 4., 2., 4., 4., 2.]))

    core.learn(n_steps=40, n_steps_per_fit=7, quiet=True)
    core.learn(n_episodes=6, n_episodes_per_fit=2, quiet=True)


def test_forced_list_backend_infinite_horizon():
    mdp = Gymnasium(name='MountainCar-v0', horizon=np.inf, gamma=1.)
    mdp.seed(1)

    class HeuristicPolicy(Policy):
        def draw_action(self, state):
            return np.array([2 if state[1] >= 0 else 0])

    agent = Agent(mdp.info, HeuristicPolicy())
    agent.fit = lambda dataset: None

    core = Core(agent, mdp, dataset_backend='list')
    dataset = core.evaluate(n_episodes=2, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert np.all(np.isfinite(dataset.compute_J()))

    numpy_core = Core(agent, mdp)
    raised = False
    try:
        numpy_core.evaluate(n_episodes=1, quiet=True)
    except AssertionError:
        raised = True
    assert raised


def test_weird_object_env():
    mdp = WeirdObjectEnv()
    agent = ObjectAgent(mdp.info)
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=1, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 1
    assert len(dataset) == 3
    assert dataset.state[0] == {'id': 0, 'items': [0]}
    assert dataset.state[2] == {'id': 2, 'items': [0, 1, 2]}
    assert dataset[:2].state[1] == {'id': 1, 'items': [0, 1]}
    assert agent.policy.seen[0] == {'id': 0, 'items': [0]}
