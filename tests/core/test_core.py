import cv2
import numpy as np
import pytest
from pathlib import Path

from mushroom_rl.core import Agent, Core, Environment, Logger, MDPInfo, AgentInfo, VectorizedEnvironment
from mushroom_rl.core.spaces import Box, Discrete
from mushroom_rl.core.history_manager import HistoryManager
from mushroom_rl.environments import Atari

from mushroom_rl.policy import Policy


class RandomDiscretePolicy(Policy):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def draw_action(self, state):
        return [np.random.randint(self._n)]


class DummyAgent(Agent):
    def __init__(self, mdp_info):
        policy = RandomDiscretePolicy(mdp_info.action_space.n)
        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        pass


def test_agent_history_manager_mutually_exclusive():
    obs_space = Box(np.full((4,), -1.0), np.full((4,), 1.0), (4,))
    act_space = Discrete(3)
    mdp_info = MDPInfo(obs_space, act_space, gamma=0.99, horizon=100, backend='numpy')
    policy = RandomDiscretePolicy(act_space.n)

    agent_info = AgentInfo(is_episodic=False, policy_state_shape=None, backend='numpy')
    history_manager = HistoryManager.default_streams(mdp_info, agent_info, history_length=3)

    agent = Agent(mdp_info, policy, history_manager=history_manager)
    assert agent.history_manager is history_manager

    with pytest.raises(AssertionError):
        Agent(mdp_info, policy, history_length=3, history_manager=history_manager)


class GreedyDummyPolicy(Policy):
    def draw_action(self, state):
        return np.array([np.random.randint(4)])

    def draw_action_greedy(self, state):
        return np.array([1])


class GreedyDummyAgent(Agent):
    def __init__(self, mdp_info):
        super().__init__(mdp_info, GreedyDummyPolicy())

    def fit(self, dataset):
        pass


def test_core_greedy_evaluation():
    from mushroom_rl.environments import GridWorld

    mdp = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    core = Core(GreedyDummyAgent(mdp.info), mdp)

    dataset = core.evaluate(n_steps=10, quiet=True, greedy=True)
    assert np.all(np.array(dataset.action) == 1)

    np.random.seed(0)
    dataset_stochastic = core.evaluate(n_steps=10, quiet=True)
    assert not np.all(np.array(dataset_stochastic.action) == 1)


def test_core_greedy_evaluation_unsupported():
    from mushroom_rl.environments import GridWorld

    mdp = GridWorld.from_size(height=3, width=3, goal=(2, 2), start=(0, 0))

    core = Core(DummyAgent(mdp.info), mdp)

    with pytest.raises(NotImplementedError):
        core.evaluate(n_steps=5, quiet=True, greedy=True)


def test_core():
    mdp = Atari(name='ALE/Breakout-v5', repeat_action_probability=0.0)

    agent = DummyAgent(mdp.info)

    core = Core(agent, mdp)

    np.random.seed(2)
    mdp.seed(2)

    core.learn(n_steps=100, n_steps_per_fit=1)

    dataset = core.evaluate(n_steps=20)

    assert 'lives' in dataset.info
    assert 'episode_frame_number' in dataset.info
    assert 'frame_number' in dataset.info

    info_lives = np.array(dataset.info['lives'])

    print(info_lives)
    lives_gt = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
    assert len(info_lives) == 20
    assert np.all(info_lives == lives_gt)
    assert len(dataset) == 20


class RenderingEnv(Environment):
    def __init__(self, horizon):
        observation_space = Box(0, 1, shape=(1,))
        action_space = Discrete(2)
        mdp_info = MDPInfo(observation_space, action_space, 0.99, horizon, dt=0.02, backend='numpy')

        self.render_flags = list()

        super().__init__(mdp_info)

    def reset(self, state=None):
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, False, {}

    def render(self, record=False):
        self.render_flags.append(record)

        return np.full((2, 2, 3), len(self.render_flags), dtype=np.uint8) if record else None


class FrameRecorder:
    def __init__(self, fps=None, **kwargs):
        self.fps = fps
        self.frames = list()
        self.masks = list()
        self.n_stops = 0

    def __call__(self, frame, mask=None):
        self.frames.append(frame)
        self.masks.append(mask)

    def stop(self):
        self.n_stops += 1


def test_core_record_evaluate():
    mdp = RenderingEnv(horizon=5)
    logger = Logger('test_core_record_evaluate', results_dir=None, recorder_class=FrameRecorder)
    core = Core(DummyAgent(mdp.info), mdp, logger=logger)

    assert core.agent.logger is logger

    core.evaluate(n_episodes=1, render=True, record=True, quiet=True)

    assert mdp.render_flags == [True] * 5
    assert len(logger.video_recorder.frames) == 5
    assert np.all(logger.video_recorder.frames[0] == 1)
    assert np.all(logger.video_recorder.frames[4] == 5)
    assert logger.video_recorder.fps == 50
    assert logger.video_recorder.n_stops == 1


def test_core_record_learn():
    mdp = RenderingEnv(horizon=4)
    logger = Logger('test_core_record_learn', results_dir=None, recorder_class=FrameRecorder)
    core = Core(DummyAgent(mdp.info), mdp, logger=logger)

    core.learn(n_steps=4, n_steps_per_fit=4, render=True, record=True, quiet=True)

    assert len(logger.video_recorder.frames) == 4
    assert logger.video_recorder.n_stops == 1


def test_core_record_without_logger():
    mdp = RenderingEnv(horizon=5)
    core = Core(DummyAgent(mdp.info), mdp)

    assert core.agent.logger is None

    with pytest.raises(AssertionError):
        core.evaluate(n_episodes=1, render=True, record=True, quiet=True)

    with pytest.raises(AssertionError):
        core.learn(n_steps=5, n_steps_per_fit=5, render=True, record=True, quiet=True)


def test_core_record_without_render():
    mdp = RenderingEnv(horizon=5)
    logger = Logger('test_core_record_without_render', results_dir=None, recorder_class=FrameRecorder)
    core = Core(DummyAgent(mdp.info), mdp, logger=logger)

    with pytest.raises(AssertionError):
        core.evaluate(n_episodes=1, record=True, quiet=True)

    with pytest.raises(AssertionError):
        core.learn(n_steps=5, n_steps_per_fit=5, record=True, quiet=True)


class RenderingVecEnv(VectorizedEnvironment):
    def __init__(self, n_envs, horizon):
        observation_space = Box(0, 1, shape=(1,))
        action_space = Box(0, 1, shape=(1,))
        mdp_info = MDPInfo(observation_space, action_space, 0.99, horizon, dt=0.02, backend='numpy')

        self._steps = np.zeros(n_envs, dtype=int)

        super().__init__(mdp_info, n_envs)

    def reset_all(self, env_mask, state=None):
        self._steps[env_mask] = 0

        return np.zeros((self._n_envs, 1)), [{}] * self._n_envs

    def step_all(self, env_mask, action):
        self._steps[env_mask] += 1

        return (np.zeros((self._n_envs, 1)), np.zeros(self._n_envs), np.zeros(self._n_envs, dtype=bool),
                [{}] * self._n_envs)

    def render_all(self, env_mask, record=False):
        if not record:
            return None

        return np.stack([np.full((100, 100, 3), 10 * (self._n_envs * env + self._steps[env]), dtype=np.uint8)
                         for env in range(self._n_envs) if env_mask[env]])


class VecRandomPolicy(Policy):
    def draw_action(self, state):
        return np.zeros((state.shape[0], 1))


class VecDummyAgent(Agent):
    def __init__(self, mdp_info):
        super().__init__(mdp_info, VecRandomPolicy(), backend='numpy')

    def fit(self, dataset):
        pass


def test_vectorized_core_record(tmpdir):
    mdp = RenderingVecEnv(n_envs=3, horizon=3)
    logger = Logger('test_vectorized_core_record', results_dir=tmpdir, fps=30)
    core = Core(VecDummyAgent(mdp.info), mdp, logger=logger)

    core.evaluate(n_episodes=3, render=True, record=True, quiet=True)

    video_dir = Path(tmpdir) / 'test_vectorized_core_record' / 'videos'
    videos = list(video_dir.rglob('*.mp4'))

    assert videos == [video_dir / 'recording.mp4']
    assert not list(video_dir.rglob('*.mkv'))
    assert logger.recorded_videos == videos
    assert videos[0].stat().st_size > 0

    capture = cv2.VideoCapture(str(videos[0]))
    values = list()

    while True:
        read, frame = capture.read()

        if not read:
            break

        values.append(frame.mean())

    capture.release()

    assert len(values) == 9
    assert all(value > 0 for value in values)


def test_vectorized_core_record_passes_mask():
    mdp = RenderingVecEnv(n_envs=3, horizon=3)
    logger = Logger('test_vectorized_core_record_single', results_dir=None, recorder_class=FrameRecorder)
    core = Core(VecDummyAgent(mdp.info), mdp, logger=logger)

    core.evaluate(n_episodes=1, render=True, record=True, quiet=True)

    assert len(logger.video_recorder.frames) == 3
    assert all(frame.shape == (100, 100, 3) for frame in logger.video_recorder.frames)
    assert all(np.all(mask == np.array([True, False, False])) for mask in logger.video_recorder.masks)
    assert logger.video_recorder.n_stops == 1

    mdp = RenderingVecEnv(n_envs=3, horizon=3)
    logger = Logger('test_vectorized_core_record_multi', results_dir=None, recorder_class=FrameRecorder)
    core = Core(VecDummyAgent(mdp.info), mdp, logger=logger)

    core.evaluate(n_episodes=2, render=True, record=True, quiet=True)

    assert len(logger.video_recorder.frames) == 3
    assert all(frame.shape == (2, 100, 100, 3) for frame in logger.video_recorder.frames)
    assert all(np.all(mask == np.array([True, True, False])) for mask in logger.video_recorder.masks)
    assert logger.video_recorder.n_stops == 1
