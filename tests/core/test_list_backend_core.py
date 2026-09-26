import numpy as np
import pytest

from mushroom_rl.core import Agent, AgentInfo, Core, Dataset, Environment, VectorizedEnvironment, MDPInfo, Box
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.preprocessors import Preprocessor

from mushroom_rl.core._impl.history_state import HistoryState


def dummy_agent(mdp_info, action_fn):
    agent = Agent(mdp_info, Policy())
    agent.draw_action = action_fn
    agent.fit = lambda dataset: None
    return agent


class CountingListPreprocessor(Preprocessor):
    def __init__(self):
        self.batch_sizes = list()
        self._offset = 0
        super().__init__()

    def __call__(self, obs):
        return [o + self._offset for o in obs]

    def update(self, obs):
        self.batch_sizes.append(len(obs))
        self._offset += 1


class ListStepRecorder(object):
    def __init__(self):
        self.states = list()
        self.next_states = list()
        self.lasts = list()

    def __call__(self, samples):
        state, _, _, next_state, _, last = samples[:6]
        self.states.append(np.array(state))
        self.next_states.append(np.array(next_state))
        self.lasts.append(np.array(last))


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

    def _obs(self):
        return [np.array([float(t), float(t)]) for t in self._t]

    def _step_info(self):
        return [{'v': float(t)} for t in self._t] if self._with_info else [{} for _ in self._t]


class StaggeredListVecEnv(VectorizedEnvironment):
    def __init__(self):
        n_envs = 3
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(2,)), Box(-1.0, 1.0, shape=(1,)),
                           gamma=0.9, horizon=10, backend='list')
        self._episode_lengths = np.array([2, 3, 5])
        self._t = np.zeros(n_envs, dtype=int)
        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._t[np.asarray(env_mask)] = 0
        return self._obs(), [{} for _ in self._t]

    def step_all(self, env_mask, action):
        self._t[np.asarray(env_mask)] += 1
        reward = np.ones(self._n_envs)
        absorbing = (self._t >= self._episode_lengths) & np.asarray(env_mask)
        return self._obs(), reward, absorbing, [{} for _ in self._t]

    def render_all(self, env_mask, record=False):
        pass

    def stop(self):
        pass

    def _obs(self):
        return [np.array([float(t), float(t)]) for t in self._t]


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
    assert np.array_equal(dataset.compute_J(), np.array([4., 4., 4., 4., 4., 4.]))

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


def test_vectorized_empty_dataset():
    mdp = ListVecEnv(with_info=True)
    agent = dummy_agent(mdp.info, lambda state: [np.array([0.0]) for _ in state])
    core = Core(agent, mdp)

    dataset = core.evaluate(n_steps=0, quiet=True)

    assert len(dataset) == 0
    assert dataset.n_episodes == 0
    assert dataset.array_backend.get_backend_name() == 'list'


def test_vectorized_reset_preprocessing():
    np.random.seed(42)

    mdp = StaggeredListVecEnv()
    agent = dummy_agent(mdp.info, lambda state: [np.array([0.0]) for _ in state])
    preprocessor = CountingListPreprocessor()
    agent.add_core_preprocessor(preprocessor)

    recorder = ListStepRecorder()
    core = Core(agent, mdp, callback_step=recorder)

    core.evaluate(n_steps=90, quiet=True)

    assert sum(preprocessor.batch_sizes) == 121
    assert min(preprocessor.batch_sizes) > 0

    for t in range(1, len(recorder.states)):
        for e in range(mdp.number):
            if not recorder.lasts[t - 1][e]:
                assert np.array_equal(recorder.next_states[t - 1][e], recorder.states[t][e])


def test_infinite_horizon_uses_list_backend():
    mdp = Gymnasium(name='MountainCar-v0', horizon=np.inf, gamma=1.)
    mdp.seed(1)

    agent = dummy_agent(mdp.info, lambda state: np.array([2 if state[1] >= 0 else 0]))
    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=2, quiet=True)

    assert dataset.array_backend.get_backend_name() == 'list'
    assert dataset.n_episodes == 2
    assert np.array_equal(dataset.compute_J(), np.array([-124., -122.]))


class ListActionPolicy(Policy):
    def draw_action(self, state, **kwargs):
        return [np.array([float(s[0])]) for s in state]


class FitRecorderAgent(Agent):
    def __init__(self, mdp_info, policy, backend='list', history_length=None):
        super().__init__(mdp_info, policy, backend=backend, history_length=history_length)
        self.fit_actions = list()

    def fit(self, dataset):
        self.fit_actions.append([float(a[0]) for a in dataset.action])


def test_vectorized_list_agent_learns_through_episode_resets():
    mdp = ListVecEnv(with_info=False)
    agent = FitRecorderAgent(mdp.info, ListActionPolicy())
    core = Core(agent, mdp)

    core.learn(n_steps=30, n_steps_per_fit=6, quiet=True)

    assert agent.fit_actions == [[0., 1., 0., 1., 0., 1.], [2., 3., 2., 3., 2., 3.], [0., 1., 0., 1., 0., 1.],
                                 [2., 3., 2., 3., 2., 3.], [0., 1., 0., 1., 0., 1.]]


def test_list_agent_history_stacking_is_not_supported():
    mdp = ListVecEnv(with_info=False)

    with pytest.raises(NotImplementedError):
        FitRecorderAgent(mdp.info, ListActionPolicy(), history_length=2)


def test_list_agent_nstep_parsing_is_not_supported():
    mdp = ListVecEnv(with_info=False)
    agent = FitRecorderAgent(mdp.info, ListActionPolicy())
    dataset = Dataset.from_array([[0.], [1.]], [[0.], [0.]], [1., 1.], [[1.], [2.]], [False, False], [False, True],
                                 gamma=0.9, backend='list')

    with pytest.raises(NotImplementedError):
        agent.history_manager.parse_nstep_history(dataset, gamma=0.9, n_steps_return=1)


def test_agent_data_conversion_to_list_is_not_supported():
    states = np.array([[0.], [1.]])
    actions = np.array([[0.], [0.]])
    rewards = np.array([1., 1.])
    absorbings = np.array([False, False])
    lasts = np.array([False, True])
    stateful = Dataset.from_array(states, actions, rewards, states + 1, absorbings, lasts,
                                  policy_state=np.zeros((2, 1)), policy_next_state=np.zeros((2, 1)), gamma=0.9)
    with_history = Dataset.from_array(states, actions, rewards, states + 1, absorbings, lasts, gamma=0.9)
    with_history._history_state = HistoryState('numpy', None, np.array([0]), {'obs_history': np.array([[1.]])})
    plain = Dataset.from_array(states, actions, rewards, states + 1, absorbings, lasts, gamma=0.9)

    with pytest.raises(NotImplementedError):
        stateful.to_backend('list')
    with pytest.raises(NotImplementedError):
        with_history.to_backend('list')
    assert plain.to_backend('list').state == [[0.], [1.]]


def test_list_agent_reset_clears_the_last_action_of_the_reset_envs_only():
    history_manager = HistoryManager(AgentInfo(is_episodic=False, policy_state_shape=None, backend='list'))
    action = [{'move': 0}, {'move': 1}, {'move': 2}]

    history_manager.reset_vectorized(np.array([True, True, True]))
    history_manager.record_action(action)
    history_manager.reset_vectorized(np.array([False, True, False]))

    assert history_manager._last_action == [{'move': 0}, None, {'move': 2}]
    assert action == [{'move': 0}, {'move': 1}, {'move': 2}]
