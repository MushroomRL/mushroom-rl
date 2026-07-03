import numpy as np

from mushroom_rl.core import Agent, Core, Environment, VectorizedEnvironment, MDPInfo, Box
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import Policy


def dummy_agent(mdp_info, action_fn):
    agent = Agent(mdp_info, Policy())
    agent.draw_action = action_fn
    agent.fit = lambda dataset: None
    return agent


class VariableLengthEnv(Environment):
    def __init__(self):
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(1,)), Box(-1.0, 1.0, shape=(1,)),
                           gamma=0.9, horizon=5, backend='list')
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


class ObjectEnv(Environment):
    def __init__(self):
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(1,)), Box(-1.0, 1.0, shape=(1,)),
                           gamma=0.9, horizon=5, backend='list')
        self._t = 0
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._t = 0
        return {'id': 0}, {}

    def step(self, action):
        self._t += 1
        absorbing = self._t >= 3
        return {'id': self._t}, 1.0, absorbing, {}

    def render(self, record=False):
        pass

    def stop(self):
        pass


class ListVecEnv(VectorizedEnvironment):
    def __init__(self, with_info=True):
        n_envs = 3
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(2,)), Box(-1.0, 1.0, shape=(1,)),
                           gamma=0.9, horizon=10, backend='list')
        self._with_info = with_info
        self._t = np.zeros(n_envs, dtype=int)
        super().__init__(mdp_info, n_envs)

    def _obs(self):
        return [np.array([float(t), float(t)]) for t in self._t]

    def _step_info(self):
        return [{'v': float(t)} for t in self._t] if self._with_info else [{} for _ in self._t]

    def reset_all(self, env_mask, state=None):
        self._t[np.asarray(env_mask)] = 0
        return self._obs(), self._step_info()

    def step_all(self, env_mask, action):
        self._t[np.asarray(env_mask)] += 1
        reward = np.ones(self._n_envs)
        absorbing = (self._t >= 4) & np.asarray(env_mask)
        return self._obs(), reward, absorbing, self._step_info()

    def render_all(self, env_mask, record=False):
        pass

    def stop(self):
        pass


def test_variable_length_state_action():
    mdp = VariableLengthEnv()
    agent = dummy_agent(mdp.info, lambda state: list(range(len(state))))
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=2, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert len(dataset) == 8
    assert dataset.state[0] == [0.0]
    assert dataset.state[3] == [0.0, 1.0, 2.0, 3.0]
    assert dataset.action[0] == [0]
    assert dataset.action[3] == [0, 1, 2, 3]
    assert dataset.info['v'] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert dataset[:4].info['v'] == [1, 2, 3, 4]
    assert np.array_equal(dataset.compute_J(), np.array([4.0, 4.0]))


def test_object_state_action():
    mdp = ObjectEnv()
    agent = dummy_agent(mdp.info, lambda state: {'move': state['id']})
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=1, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 1
    assert len(dataset) == 3
    assert dataset.state[0] == {'id': 0}
    assert dataset.state[2] == {'id': 2}
    assert dataset.action[0] == {'move': 0}
    assert dataset.action[2] == {'move': 2}
    assert dataset[:2].state[1] == {'id': 1}
    assert list(dataset.info.keys()) == []


def test_vectorized_extra_info():
    mdp = ListVecEnv(with_info=True)
    agent = dummy_agent(mdp.info, lambda state: [np.array([0.0]) for _ in state])
    core = Core(agent, mdp)

    dataset = core.evaluate(n_steps=30, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert len(dataset) == 30
    assert dataset.info['v'] == [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
                                 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
                                 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]
    assert np.array_equal(dataset.compute_J(), np.array([4., 4., 2., 4., 4., 2., 4., 4., 2.]))

    core.learn(n_steps=40, n_steps_per_fit=7, quiet=True)
    core.learn(n_episodes=6, n_episodes_per_fit=2, quiet=True)


def test_vectorized_no_extra_info():
    mdp = ListVecEnv(with_info=False)
    agent = dummy_agent(mdp.info, lambda state: [np.array([0.0]) for _ in state])
    core = Core(agent, mdp)

    dataset = core.evaluate(n_steps=30, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert len(dataset) == 30
    assert list(dataset.info.keys()) == []


def test_infinite_horizon_uses_list_backend():
    mdp = Gymnasium(name='MountainCar-v0', horizon=np.inf, gamma=1.)
    mdp.seed(1)

    agent = dummy_agent(mdp.info, lambda state: np.array([2 if state[1] >= 0 else 0]))
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=2, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert np.all(np.isfinite(dataset.compute_J()))
