import numpy as np
import torch

from mushroom_rl.core import Core, MDPInfo, Box, Environment, VectorizedEnvironment, Agent, Dataset
from mushroom_rl.policy import Policy
from mushroom_rl.utils.callbacks import CollectDataset


class RecordingPolicy(Policy):
    def __init__(self, history_length):
        super().__init__()
        self._history_length = history_length
        self.windows = dict()
        self.action_windows = dict()

    def draw_action(self, state, **kwargs):
        if self._history_length > 1:
            windows = state if state.ndim == 3 else state[None]
            frames = windows[:, -1]
        else:
            windows = state if state.ndim == 2 else state[None]
            frames = windows
        action_history = kwargs.get('action_history')
        for i, frame in enumerate(frames):
            self.windows[frame.tobytes()] = windows[i].copy()
            if action_history is not None:
                self.action_windows[frame.tobytes()] = np.asarray(action_history)[i].copy()
        action = frames[:, 1:2].copy()
        return action if state.ndim == windows.ndim else action[0]


class SeamAgent(Agent):
    def __init__(self, mdp_info, history_length=1, action_history_length=0):
        super().__init__(mdp_info, RecordingPolicy(history_length), backend='numpy',
                         history_length=history_length, action_history_length=action_history_length)
        self.blocks = list()

    def fit(self, dataset):
        state = self._history_manager.parse_state(dataset)
        extra = self._history_manager.parse_history(dataset)[6]
        self.blocks.append(dict(dataset=dataset.copy(), parse_state=np.asarray(state),
                                action_history=extra.get('action_history')))


class CountingEnv(Environment):
    def __init__(self, horizon):
        mdp_info = MDPInfo(Box(0, 1e6, shape=(3,)), Box(-1e6, 1e6, shape=(1,)), 0.99, horizon)
        self._t = -1
        self._ep_t = 0
        super().__init__(mdp_info)

    def reset(self, state=None):
        self._t += 1
        self._ep_t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        self._ep_t += 1
        return self._obs(), 0., False, {}

    def _obs(self):
        return np.array([0., self._t, self._ep_t])


class CountingVecEnv(VectorizedEnvironment):
    def __init__(self, n_envs, horizon):
        mdp_info = MDPInfo(Box(0, 1e6, shape=(3,)), Box(-1e6, 1e6, shape=(1,)), 0.99, horizon)
        self._t = np.full(n_envs, -1.)
        self._ep_t = np.zeros(n_envs)
        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._t[env_mask] += 1
        self._ep_t[env_mask] = 0
        return self._obs(), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        self._t[env_mask] += 1
        self._ep_t[env_mask] += 1
        return self._obs(), np.zeros(self._n_envs), np.zeros(self._n_envs, dtype=bool), [{}] * self._n_envs

    def _obs(self):
        return np.stack([np.arange(self._n_envs), self._t, self._ep_t], axis=1).astype(float)


def mismatching_rows(agent):
    mismatches = list()
    for k, block in enumerate(agent.blocks):
        states = np.asarray(block['dataset'].state)
        for row in range(len(states)):
            key = states[row].tobytes()
            if not np.array_equal(agent.policy.windows[key], block['parse_state'][row]):
                mismatches.append((k, row))
            if block['action_history'] is not None and \
                    not np.array_equal(agent.policy.action_windows[key], block['action_history'][row]):
                mismatches.append((k, row))
    return mismatches


def test_sequential_parse_state_matches_the_online_window_across_fit_blocks():
    env = CountingEnv(horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_steps=40, n_steps_per_fit=8, quiet=True)

    assert len(agent.blocks) == 5
    assert mismatching_rows(agent) == []
    assert np.array_equal(agent.blocks[1]['parse_state'][0], np.array([[0., 6., 6.], [0., 7., 7.], [0., 8., 8.]]))
    assert np.array_equal(agent.blocks[1]['parse_state'][2], np.array([[0., 0., 0.], [0., 0., 0.], [0., 11., 0.]]))
    assert len(agent.blocks[0]['dataset'].history_state) == 0
    assert np.array_equal(agent.blocks[1]['dataset'].history_state.positions, np.array([0]))


def test_sequential_one_step_blocks_continue_across_clear():
    env = CountingEnv(horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_steps=12, n_steps_per_fit=1, quiet=True)

    assert len(agent.blocks) == 12
    assert mismatching_rows(agent) == []
    assert np.array_equal(agent.blocks[5]['parse_state'][0], np.array([[0., 3., 3.], [0., 4., 4.], [0., 5., 5.]]))
    assert np.array_equal(agent.blocks[10]['parse_state'][0], np.array([[0., 0., 0.], [0., 0., 0.], [0., 11., 0.]]))


def test_vectorized_parse_state_matches_the_online_window_across_fit_blocks():
    env = CountingVecEnv(3, horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_steps=81, n_steps_per_fit=27, quiet=True)

    assert len(agent.blocks) == 3
    assert mismatching_rows(agent) == []
    assert np.array_equal(agent.blocks[1]['parse_state'][9], np.array([[1., 7., 7.], [1., 8., 8.], [1., 9., 9.]]))
    assert np.array_equal(agent.blocks[1]['parse_state'][10], np.array([[0., 0., 0.], [0., 0., 0.], [1., 11., 0.]]))
    assert np.array_equal(agent.blocks[1]['dataset'].history_state.positions, np.array([0, 9, 18]))


def test_ragged_consume_gives_the_leftover_row_its_own_window():
    env = CountingVecEnv(3, horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_steps=100, n_steps_per_fit=25, quiet=True)

    assert len(agent.blocks) == 4
    assert mismatching_rows(agent) == []


def test_episode_budget_leaves_environments_idle_without_breaking_the_windows():
    env = CountingVecEnv(3, horizon=4)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_episodes=5, n_episodes_per_fit=2, quiet=True)

    assert len(agent.blocks) == 2
    assert mismatching_rows(agent) == []


def test_action_history_window_at_a_block_start():
    env = CountingVecEnv(2, horizon=10)
    agent = SeamAgent(env.info, action_history_length=2)
    core = Core(agent, env)

    core.learn(n_steps=40, n_steps_per_fit=10, quiet=True)

    assert len(agent.blocks) == 4
    assert mismatching_rows(agent) == []
    assert np.array_equal(agent.blocks[1]['action_history'][0], np.array([[3.], [4.]]))
    assert np.array_equal(agent.blocks[1]['action_history'][5], np.array([[3.], [4.]]))


def test_next_state_window_at_a_block_start():
    env = CountingEnv(horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)

    core.learn(n_steps=16, n_steps_per_fit=8, quiet=True)

    dataset = agent.blocks[1]['dataset']
    state, _, _, next_state, _, _, _ = agent.history_manager.parse_history(dataset)

    assert np.array_equal(next_state[0], np.array([[0., 7., 7.], [0., 8., 8.], [0., 9., 9.]]))
    assert np.array_equal(next_state[1], np.array([[0., 8., 8.], [0., 9., 9.], [0., 10., 10.]]))
    assert np.array_equal(state[1], next_state[0])


def test_attachment_survives_to_backend_views_concatenation_and_save(tmpdir):
    env = CountingVecEnv(3, horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)
    core.learn(n_steps=81, n_steps_per_fit=27, quiet=True)

    block = agent.blocks[1]['dataset']
    window = block.history_state.windows('obs_history')

    converted = block.to_backend('torch')
    assert torch.equal(converted.history_state.positions, torch.tensor([0, 9, 18]))
    assert torch.equal(converted.history_state.windows('obs_history'), torch.from_numpy(window))

    view = block[1:]
    assert np.array_equal(view.history_state.positions, np.array([8, 17]))
    assert np.array_equal(view.history_state.windows('obs_history'), window[1:])

    both = block + agent.blocks[2]['dataset']
    assert np.array_equal(both.history_state.positions, np.array([0, 9, 18, 27, 36, 45]))

    path = tmpdir / 'block.msh'
    block.save(path)
    loaded = Dataset.load(path)
    assert np.array_equal(loaded.history_state.positions, block.history_state.positions)
    assert np.array_equal(loaded.history_state.windows('obs_history'), window)
    assert np.array_equal(agent.history_manager.parse_state(loaded), agent.blocks[1]['parse_state'])


def test_stitching_concatenation_drops_the_attached_entry():
    env = CountingEnv(horizon=10)
    agent = SeamAgent(env.info, history_length=3)
    core = Core(agent, env)
    core.learn(n_steps=16, n_steps_per_fit=8, quiet=True)

    first, second = agent.blocks[0]['dataset'], agent.blocks[1]['dataset']
    both = first + second

    assert len(second.history_state) == 1
    assert len(both.history_state) == 0
    assert np.array_equal(agent.history_manager.parse_state(both)[8], agent.blocks[1]['parse_state'][0])


def test_fit_callbacks_receive_the_flat_dataset():
    env = CountingVecEnv(3, horizon=10)
    agent = SeamAgent(env.info)
    kinds = list()
    collector = CollectDataset(initial_capacity=8)
    core = Core(agent, env, callbacks_fit=[lambda dataset: kinds.append(type(dataset)), collector])

    core.learn(n_steps=60, n_steps_per_fit=30, quiet=True)
    collected = collector.get()

    assert kinds == [Dataset, Dataset]
    assert len(collected) == 60
    assert collected.n_episodes == 6
    assert np.array_equal(collected.episodes_length, np.full(6, 10))
