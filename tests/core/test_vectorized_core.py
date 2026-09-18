import numpy as np
import pytest
import torch

from mushroom_rl.core import Agent, ArrayBackend, Core, VectorizedEnvironment, MDPInfo, Box
from mushroom_rl.core._impl import VectorizedCoreLogic
from mushroom_rl.core.core import SequentialCore
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.preprocessors import Preprocessor
from mushroom_rl.utils import TorchUtils


class DummyPolicy(Policy):
    def __init__(self, action_shape, backend):
        self._dim = action_shape[0]
        self._backend = backend
        super().__init__()

    def draw_action(self, state):
        shape = (self._dim,) if len(state.shape) == 1 else (state.shape[0], self._dim)
        if self._backend == 'torch':
            return torch.randn(*shape)
        elif self._backend == 'numpy':
            return np.random.randn(*shape)
        else:
            raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, mdp_info, backend):
        policy = DummyPolicy(mdp_info.action_space.shape, backend)
        super().__init__(mdp_info, policy, backend=backend)

    def fit(self, dataset):
        print(f'\t* samples={len(dataset)}, episodes={len(dataset.episodes_length)}')
        assert len(dataset.episodes_length) in (1, 20) or len(dataset) == 150


class DummyEpisodicAgent(Agent):
    def __init__(self, mdp_info, backend):
        self._backend = backend
        policy = DummyPolicy(mdp_info.action_space.shape, backend)
        super().__init__(mdp_info, policy, is_episodic=True, backend=backend)
        self._counter = 0

    def fit(self, dataset):
        assert len(dataset.theta_list) == 5

    def episode_start_vectorized(self, initial_states, episode_info, start_mask, greedy=False):
        n_envs = len(start_mask)
        current_count = self._counter
        self._counter += 1
        if self._backend == 'torch':
            return None, torch.ones(n_envs, 2) * current_count
        elif self._backend == 'numpy':
            return None, np.ones((n_envs, 2)) * current_count
        else:
            raise NotImplementedError


class GreedyDummyPolicy(Policy):
    def __init__(self, action_shape, backend):
        self._dim = action_shape[0]
        self._backend = backend
        super().__init__()

    def draw_action(self, state):
        if self._backend == 'torch':
            return torch.randn(state.shape[0], self._dim)
        elif self._backend == 'numpy':
            return np.random.randn(state.shape[0], self._dim)
        else:
            raise NotImplementedError

    def draw_action_greedy(self, state):
        if self._backend == 'torch':
            return torch.ones(state.shape[0], self._dim)
        elif self._backend == 'numpy':
            return np.ones((state.shape[0], self._dim))
        else:
            raise NotImplementedError


class GreedyDummyAgent(Agent):
    def __init__(self, mdp_info, backend):
        policy = GreedyDummyPolicy(mdp_info.action_space.shape, backend)
        super().__init__(mdp_info, policy, backend=backend)

    def fit(self, dataset):
        pass


class CountingPreprocessor(Preprocessor):
    def __init__(self):
        self.batch_sizes = list()
        self._offset = 0
        super().__init__()

    def __call__(self, obs):
        return obs + self._offset

    def update(self, obs):
        self.batch_sizes.append(len(obs))
        self._offset += 1


class StepRecorder(object):
    def __init__(self, array_backend):
        self.states = list()
        self.next_states = list()
        self.lasts = list()
        self._array_backend = array_backend

    def __call__(self, samples):
        state, _, _, next_state, _, last = samples[:6]
        self.states.append(np.array(self._array_backend.to_numpy(state)))
        self.next_states.append(np.array(self._array_backend.to_numpy(next_state)))
        self.lasts.append(np.array(self._array_backend.to_numpy(last)))


class DummyVecEnv(VectorizedEnvironment):
    def __init__(self, backend, device=None, n_envs=10):
        state_dim = 3

        horizon = 100
        gamma = 0.99

        observation_space = Box(0, 200, shape=(3,))
        action_space = Box(0, 200, shape=(2,))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, backend=backend, device=device)

        if backend == 'torch':
            self._device = TorchUtils.get_device() if device is None else device
            self._state = torch.empty(n_envs, state_dim).to(self._device)
        elif backend == 'numpy':
            self._state = np.empty((n_envs, state_dim))
        else:
            raise NotImplementedError

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        if state is not None:
            self._state[env_mask] = state[env_mask]
        elif self.info.backend == 'torch':
            self._state[env_mask] = torch.randint(size=(env_mask.sum(), self._state.shape[1]),
                                                  low=2, high=200).float().to(self._device)
        elif self.info.backend == 'numpy':
            self._state[env_mask] = np.random.randint(size=(env_mask.sum(), self._state.shape[1]),
                                                      low=2, high=200).astype(float)

        next_state = self._state.clone() if self.info.backend == 'torch' else self._state.copy()

        return next_state, [{}] * self._n_envs

    def step_all(self, env_mask, action):
        self._state[env_mask] -= 1

        if self.info.backend == 'torch':
            reward = torch.zeros(self._state.shape[0]).to(self._device)
        elif self.info.backend == 'numpy':
            reward = np.zeros(self._state.shape[0])
        else:
            raise NotImplementedError

        done = (self._state == 0).any(1)

        next_state = self._state.clone() if self.info.backend == 'torch' else self._state.copy()

        return next_state, reward, done & env_mask, [{}] * self._n_envs


def run_exp(env_backend, agent_backend):
    torch.random.manual_seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)

    core = Core(agent, env)

    print('- evaluate n_steps=2000')
    dataset = core.evaluate(n_steps=2000)
    assert len(dataset) == 2000

    print('- evaluate n_episodes=20')
    dataset = core.evaluate(n_episodes=20)
    assert len(dataset.episodes_length) == 20

    print('- evaluate n_episodes=1')
    dataset = core.evaluate(n_episodes=1)
    assert len(dataset.episodes_length) == 1

    print('- evaluate single initial state')
    expected_states = np.array([[4., 5., 6.]], dtype=np.float32)
    initial_states = torch.from_numpy(expected_states).to(TorchUtils.get_device()) \
        if env_backend == 'torch' else expected_states
    dataset = core.evaluate(initial_states=initial_states)
    assert len(dataset.episodes_length) == 1
    init_states = dataset.array_backend.to_numpy(dataset.get_init_states())
    assert sorted(tuple(row) for row in init_states) == sorted(tuple(row) for row in expected_states)

    print('- learn n_episodes=5 n_episodes_per_fit=1')
    core.learn(n_episodes=5, n_episodes_per_fit=1)

    print('- learn n_steps=10000 n_episodes_per_fit=20')
    core.learn(n_steps=10000, n_episodes_per_fit=20)

    print('- learn n_steps=10000 n_steps_per_fit=150')
    core.learn(n_steps=10000, n_steps_per_fit=150)

    print('- learn n_episode=100 n_steps_per_fit=150')
    core.learn(n_episodes=100, n_steps_per_fit=150)

    print('- learn n_episode=100 n_episodes_per_fit=20')
    core.learn(n_episodes=100, n_episodes_per_fit=20)


def run_exp_episodic(env_backend, agent_backend):
    torch.random.manual_seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyEpisodicAgent(env.info, agent_backend)

    core = Core(agent, env)

    print('- evaluate n_episodes=20')
    dataset = core.evaluate(n_episodes=20)
    assert len(dataset.episodes_length) == 20
    assert len(dataset.theta_list) == 20

    print('- learn n_episodes=25 n_episodes_per_fit=5')
    core.learn(n_episodes=25, n_episodes_per_fit=5)


def run_exp_initial_states(env_backend, agent_backend):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)

    core = Core(agent, env)

    expected_states = np.array([[5., 9., 7.],
                                [3., 8., 6.],
                                [10., 4., 11.],
                                [12., 7., 5.],
                                [6., 6., 9.],
                                [8., 5., 4.],
                                [9., 3., 10.]], dtype=np.float32)

    initial_states = torch.from_numpy(expected_states).to(TorchUtils.get_device()) \
        if env_backend == 'torch' else expected_states

    dataset = core.evaluate(initial_states=initial_states)

    assert len(dataset.episodes_length) == 7

    init_states = dataset.array_backend.to_numpy(dataset.get_init_states())

    assert len(init_states) == 7
    assert sorted(tuple(row) for row in init_states) == sorted(tuple(row) for row in expected_states)


def run_exp_greedy(env_backend, agent_backend):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = GreedyDummyAgent(env.info, agent_backend)

    core = Core(agent, env)

    dataset = core.evaluate(n_steps=50, quiet=True, greedy=True)
    actions = dataset.array_backend.to_numpy(dataset.action)
    assert np.all(actions == 1.0)

    dataset_stochastic = core.evaluate(n_steps=50, quiet=True)
    actions_stochastic = dataset_stochastic.array_backend.to_numpy(dataset_stochastic.action)
    assert not np.all(actions_stochastic == 1.0)


def run_exp_default_env(env_backend):
    env = DummyVecEnv(env_backend)

    default_env = 3
    env.set_default_env(default_env)

    array_backend = ArrayBackend.get_array_backend(env_backend)
    initial_state = array_backend.from_list([4., 5., 6.])
    action = array_backend.from_list([1., 1.])

    state, _ = env.reset(initial_state)
    next_state, _, _, _ = env.step(action)

    assert np.array_equal(array_backend.to_numpy(state), np.array([4., 5., 6.]))
    assert np.array_equal(array_backend.to_numpy(next_state), np.array([3., 4., 5.]))


def run_exp_reset_preprocessing(env_backend, agent_backend, n_observations):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)
    preprocessor = CountingPreprocessor()
    agent.add_core_preprocessor(preprocessor)

    recorder = StepRecorder(ArrayBackend.get_array_backend(env_backend))
    core = Core(agent, env, callback_step=recorder)

    core.evaluate(n_steps=1000, quiet=True)

    assert sum(preprocessor.batch_sizes) == n_observations
    assert min(preprocessor.batch_sizes) > 0

    for t in range(1, len(recorder.states)):
        for e in range(env.number):
            if not recorder.lasts[t - 1][e]:
                assert np.array_equal(recorder.next_states[t - 1][e], recorder.states[t][e])

    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)
    preprocessor = CountingPreprocessor()
    agent.add_core_preprocessor(preprocessor)

    core = Core(agent, env)
    core.learn(n_episodes=20, n_episodes_per_fit=1, quiet=True)

    assert min(preprocessor.batch_sizes) > 0


def run_exp_step_preprocessing(env_backend, agent_backend, n_observations):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)
    preprocessor = CountingPreprocessor()
    agent.add_core_preprocessor(preprocessor)

    core = Core(agent, env)

    core.evaluate(n_episodes=13, quiet=True)

    assert sum(preprocessor.batch_sizes) == n_observations
    assert min(preprocessor.batch_sizes) > 0


def run_exp_partial_first_batch_preprocessing(env_backend, agent_backend, n_observations):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)
    preprocessor = CountingPreprocessor()
    agent.add_core_preprocessor(preprocessor)

    core = Core(agent, env)

    core.evaluate(n_episodes=7, quiet=True)

    assert preprocessor.batch_sizes[0] == 7
    assert sum(preprocessor.batch_sizes) == n_observations
    assert min(preprocessor.batch_sizes) > 0


def test_vectorized_core_partial_first_batch_preprocessing():
    run_exp_partial_first_batch_preprocessing(env_backend='torch', agent_backend='torch', n_observations=400)
    run_exp_partial_first_batch_preprocessing(env_backend='torch', agent_backend='numpy', n_observations=400)
    run_exp_partial_first_batch_preprocessing(env_backend='numpy', agent_backend='torch', n_observations=418)
    run_exp_partial_first_batch_preprocessing(env_backend='numpy', agent_backend='numpy', n_observations=418)


def test_vectorized_core_step_preprocessing():
    run_exp_step_preprocessing(env_backend='torch', agent_backend='torch', n_observations=630)
    run_exp_step_preprocessing(env_backend='torch', agent_backend='numpy', n_observations=786)
    run_exp_step_preprocessing(env_backend='numpy', agent_backend='torch', n_observations=674)
    run_exp_step_preprocessing(env_backend='numpy', agent_backend='numpy', n_observations=681)


def test_vectorized_core_reset_preprocessing():
    run_exp_reset_preprocessing(env_backend='torch', agent_backend='torch', n_observations=1025)
    run_exp_reset_preprocessing(env_backend='torch', agent_backend='numpy', n_observations=1022)
    run_exp_reset_preprocessing(env_backend='numpy', agent_backend='torch', n_observations=1028)
    run_exp_reset_preprocessing(env_backend='numpy', agent_backend='numpy', n_observations=1025)


class StaleAbsorbingVecEnv(VectorizedEnvironment):
    def __init__(self):
        self._episode_lengths = np.array([2, 3, 5, 7])
        self._t = np.zeros(4, dtype=int)
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(1,)), Box(-1., 1., shape=(1,)),
                           gamma=.99, horizon=100, backend='numpy')
        super().__init__(mdp_info, 4)

    def reset_all(self, env_mask, state=None):
        self._t[env_mask] = 0
        return self._observation(), [{} for _ in range(self._n_envs)]

    def step_all(self, env_mask, action):
        self._t[env_mask] += 1
        absorbing = self._t >= self._episode_lengths
        return self._observation(), np.zeros(self._n_envs), absorbing, [{} for _ in range(self._n_envs)]

    def render_all(self, env_mask, record=False):
        pass

    def stop(self):
        pass

    def _observation(self):
        return self._t.reshape(-1, 1).astype(float)


class DictInfoVecEnv(VectorizedEnvironment):
    def __init__(self, n_envs):
        self._t = np.zeros(n_envs, dtype=int)
        mdp_info = MDPInfo(Box(-np.inf, np.inf, shape=(1,)), Box(-1., 1., shape=(1,)),
                           gamma=.99, horizon=10, backend='numpy')
        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._t[np.asarray(env_mask)] = 0
        return self._observation(), {'k': self._t.copy()}

    def step_all(self, env_mask, action):
        self._t[np.asarray(env_mask)] += 1
        absorbing = (self._t >= 3) & np.asarray(env_mask)
        return self._observation(), np.ones(self._n_envs), absorbing, {'k': self._t.copy()}

    def render_all(self, env_mask, record=False):
        pass

    def stop(self):
        pass

    def _observation(self):
        return self._t.reshape(-1, 1).astype(float)


def test_vectorized_env_info_as_dict_of_arrays():
    env = DictInfoVecEnv(3)
    env.set_default_env(1)

    state, episode_info = env.reset()
    next_state, reward, absorbing, step_info = env.step(np.zeros(1))

    assert np.array_equal(state, np.array([0.]))
    assert np.array_equal(next_state, np.array([1.]))
    assert episode_info == {'k': 0}
    assert step_info == {'k': 1}

    env = DictInfoVecEnv(1)
    agent = dict_info_agent(env.info)

    dataset = Core(agent, env).evaluate(n_steps=6, quiet=True)

    assert len(dataset) == 6
    assert np.array_equal(dataset.info['k'], np.array([1, 2, 3, 1, 2, 3]))


def dict_info_agent(mdp_info):
    agent = Agent(mdp_info, Policy())
    agent.draw_action = lambda state: np.zeros(1) if state.ndim == 1 else np.zeros((state.shape[0], 1))
    agent.fit = lambda dataset: None

    return agent


def test_core_routes_single_env_vectorized_to_sequential():
    np.random.seed(42)

    env = DummyVecEnv('numpy', n_envs=1)
    agent = DummyAgent(env.info, 'numpy')
    core = Core(agent, env)

    assert isinstance(core, SequentialCore)

    dataset = core.evaluate(n_steps=20, quiet=True)

    assert len(dataset) == 20
    assert dataset.state.shape == (20, 3)


def test_vectorized_core_absorbing_of_inactive_envs():
    for n_episodes, n_samples in [(7, 24), (12, 43), (14, 48)]:
        env = StaleAbsorbingVecEnv()
        agent = DummyAgent(env.info, 'numpy')
        core = Core(agent, env)

        dataset = core.evaluate(n_episodes=n_episodes, quiet=True)

        assert dataset.n_episodes == n_episodes
        assert len(dataset) == n_samples


def test_vectorized_core_empty_dataset():
    for env_backend in ['numpy', 'torch']:
        np.random.seed(42)
        torch.random.manual_seed(42)

        env = DummyVecEnv(env_backend)
        agent = DummyAgent(env.info, env_backend)
        core = Core(agent, env)

        dataset = core.evaluate(n_steps=0, quiet=True)

        assert len(dataset) == 0
        assert dataset.n_episodes == 0
        assert len(dataset.episodes_length) == 0
        assert len(dataset.compute_J()) == 0
        assert dataset.compute_metrics() == (0, 0, 0, 0, 0)


def test_vectorized_core_moves_exactly_n_steps():
    for n_steps in [1, 7, 13, 51, 100]:
        np.random.seed(42)

        env = DummyVecEnv('numpy')
        agent = DummyAgent(env.info, 'numpy')
        core = Core(agent, env)

        dataset = core.evaluate(n_steps=n_steps, quiet=True)

        assert len(dataset) == n_steps


def test_vectorized_core_no_fit_below_n_steps_per_fit():
    np.random.seed(42)

    env = DummyVecEnv('numpy')
    agent = GreedyDummyAgent(env.info, 'numpy')
    fits = list()

    core = Core(agent, env, callbacks_fit=[lambda dataset: fits.append(1)])

    core.learn(n_steps=4, n_steps_per_fit=10, quiet=True)

    assert fits == []

    core.learn(n_steps=30, n_steps_per_fit=10, quiet=True)

    assert len(fits) == 3


def test_vectorized_core_steps_per_fit_below_n_envs():
    env = DummyVecEnv('numpy')
    agent = DummyAgent(env.info, 'numpy')
    core = Core(agent, env)

    with pytest.raises(AssertionError):
        core.learn(n_steps=200, n_steps_per_fit=env.number - 1, quiet=True)

    core.learn(n_steps=300, n_steps_per_fit=150, quiet=True)


def test_vectorized_core_env_device_differs_from_default():
    if not torch.cuda.is_available():
        return

    torch.random.manual_seed(42)

    env = DummyVecEnv('torch', device='cuda')
    agent = DummyAgent(env.info, 'torch')
    core = Core(agent, env)

    dataset = core.evaluate(n_steps=100, quiet=True)

    assert len(dataset) == 100
    assert dataset.state.device.type == 'cuda'


def test_vectorized_core_logic_reset_count():
    logic = VectorizedCoreLogic('numpy', 10)
    last = logic.converter.ones(10, dtype=bool)

    assert logic.n_reset_envs == 0

    logic.initialize_evaluate()
    logic.initialize_run(n_steps=100, n_episodes=None, initial_states=None, quiet=True)

    assert logic.n_reset_envs == 0

    mask = logic.get_mask(last)

    assert logic.n_reset_envs == 10
    assert logic.n_reset_envs == int((last & mask).sum())

    logic.initialize_learn(None, 3)
    logic.initialize_run(n_steps=None, n_episodes=6, initial_states=None, quiet=True)

    assert logic.n_reset_envs == 0

    mask = logic.get_mask(last)

    assert logic.n_reset_envs == 3
    assert logic.n_reset_envs == int((last & mask).sum())

    logic.after_fit_vectorized(last, 0)

    assert logic.n_reset_envs == 0

    logic.terminate_run()


def run_exp_evaluate_after_learn(env_backend, agent_backend):
    torch.random.manual_seed(42)
    np.random.seed(42)

    env = DummyVecEnv(env_backend)
    agent = DummyAgent(env.info, agent_backend)

    core = Core(agent, env)

    core.learn(n_episodes=20, n_episodes_per_fit=1, quiet=True)

    assert len(core.evaluate(n_steps=200, quiet=True)) == 200

    core.learn(n_steps=300, n_steps_per_fit=150, quiet=True)

    assert len(core.evaluate(n_steps=200, quiet=True)) == 200


def test_vectorized_core_evaluate_after_learn():
    run_exp_evaluate_after_learn(env_backend='torch', agent_backend='torch')
    run_exp_evaluate_after_learn(env_backend='torch', agent_backend='numpy')
    run_exp_evaluate_after_learn(env_backend='numpy', agent_backend='torch')
    run_exp_evaluate_after_learn(env_backend='numpy', agent_backend='numpy')


def test_vectorized_env_default_env_interface():
    run_exp_default_env(env_backend='torch')
    run_exp_default_env(env_backend='numpy')


def test_vectorized_core_greedy_evaluation():
    run_exp_greedy(env_backend='torch', agent_backend='torch')
    run_exp_greedy(env_backend='torch', agent_backend='numpy')
    run_exp_greedy(env_backend='numpy', agent_backend='torch')
    run_exp_greedy(env_backend='numpy', agent_backend='numpy')


def test_vectorized_core():
    print('# CPU test')
    run_exp(env_backend='torch', agent_backend='torch')
    run_exp(env_backend='torch', agent_backend='numpy')
    run_exp(env_backend='numpy', agent_backend='torch')
    run_exp(env_backend='numpy', agent_backend='numpy')

    run_exp_episodic(env_backend='torch', agent_backend='torch')
    run_exp_episodic(env_backend='torch', agent_backend='numpy')
    run_exp_episodic(env_backend='numpy', agent_backend='torch')
    run_exp_episodic(env_backend='numpy', agent_backend='numpy')

    run_exp_initial_states(env_backend='torch', agent_backend='torch')
    run_exp_initial_states(env_backend='torch', agent_backend='numpy')
    run_exp_initial_states(env_backend='numpy', agent_backend='torch')
    run_exp_initial_states(env_backend='numpy', agent_backend='numpy')

    if torch.cuda.is_available():
        print('# Testing also cuda')
        TorchUtils.set_default_device('cuda')
        run_exp(env_backend='torch', agent_backend='torch')
        run_exp(env_backend='torch', agent_backend='numpy')
        run_exp_episodic(env_backend='torch', agent_backend='torch')
        run_exp_episodic(env_backend='torch', agent_backend='numpy')
        TorchUtils.set_default_device('cpu')
